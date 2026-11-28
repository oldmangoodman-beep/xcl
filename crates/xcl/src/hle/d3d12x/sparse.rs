//! The title's own memory, seen by the GPU: sparse windows.
//!
//! A console title keeps GPU data in memory it reserves and commits itself (XDefiant: one
//! 8 GB reservation of 64 KiB pages it commits in steps and places buffers in) – on the
//! console all of it is GPU-visible at the same addresses. XCL gives such reservations
//! addresses in the *sparse windows* of the unified-memory block (`gpu::uma`: windows 0..2
//! in `for_runtime`), so translated shaders reach them through `mem`…`mem2` like any unified
//! memory. On the GPU each sparse window is a reserved (tiled) buffer of 4 GiB whose 64 KiB
//! tiles are mapped onto video-memory heaps as the title commits pages.
//! The CPU's view stays the title's memory: what the CPU wrote (write-watch) is copied into
//! the tiles before the title's next ExecuteCommandLists via an asynchronous ring-buffered
//! staging upload queue, and the title's queues GPU-wait on the upload fence (no CPU stall).
//! Only the bytes the CPU changed go (64-byte chunks against a hash of what was sent last,
//! `changed_chunks`): the rest of a written page may hold what the GPU wrote there (counters,
//! indirect arguments) and must not be overwritten with the CPU's stale view of it.
//! Buffers the title places there get reserved resources over the same tiles, so PC-side
//! copies, views and barriers see the same memory.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_DECOMMIT, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_READWRITE,
};

const TILE: usize = 0x1_0000;
/// tiles of a video-memory pool heap (4 MiB). A heap goes back only when every tile of it is
/// free (`State::retire_free_pools`), and a title frees its tiles in runs spread over all its
/// heaps: with heaps of 64 MiB a retail title's world load unmapped 1.5 GB of its 4.6 GB of
/// tiles and 3 of 78 heaps went back (0.2 GB); with heaps of 4 MiB 1.25 GB went back.
const POOL_TILES: u32 = 64;
/// tiles decommitted while no title work was submitted that stay mapped (`State::parked`) before
/// they are unmapped together: 256 MB of video memory
const PARKED_MAX: usize = 4096;
/// bytes of the ring-buffered upload staging buffer
const STAGING: usize = 64 << 20;
/// alignment of sub-allocations inside the upload staging ring (write-combined cacheline)
const RING_ALIGN: usize = 64;
/// number of in-flight upload command allocators / lists
const UPLOAD_SLOTS: usize = 16;
/// bytes of the dedicated pre-zeroed buffer for tile/hole clears (1 MiB)
const ZERO_BUF: usize = 1 << 20;
/// bytes of the GPU→CPU readback staging buffer
const READBACK_STAGING: usize = 16 << 20;

const PAGE: usize = 0x1000;
const PAGES_PER_TILE: usize = TILE / PAGE;
const CHUNK: usize = 64;
const CHUNKS: usize = PAGE / CHUNK;

/// Fast non-cryptographic integer hasher for page/tile lookup maps on the hot submit path.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn finish(&self) -> u64 {
        // Tile addresses are 64 KiB aligned. Move mixed bits into the low bucket-index bits.
        self.0.rotate_right(23)
    }
}

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

struct TileEntry {
    pool: (u32, u32),
    /// bit `i` set => 4 KiB page `i` of this 64 KiB tile is uncommitted (a hole)
    holes: u16,
    /// bit `i` set => the title gave page `i` a protection other than writable
    not_rw: u16,
    /// per-4 KiB-page hashes of the 64-byte chunks last sent to the GPU (`None` = all zero)
    sent: [Option<Box<[u64; CHUNKS]>>; PAGES_PER_TILE],
}

impl TileEntry {
    fn with_holes(pool: (u32, u32), holes: u16) -> Self {
        Self { pool, holes, not_rw: 0, sent: std::array::from_fn(|_| None) }
    }
}

/// Query actual OS page commitment across the 64 KiB tile at `tile`.
/// Bit `i` in the returned mask is set whenever 4 KiB page `tile + i * PAGE` is not committed
/// and accessible (`MEM_RESERVE`, `MEM_FREE`, `PAGE_NOACCESS`, `PAGE_GUARD`, etc.).
unsafe fn query_tile_holes(tile: usize) -> u16 {
    let mut holes = 0xFFFFu16;
    let end = tile.saturating_add(TILE);
    let mut cur = tile;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        // SAFETY: `VirtualQuery` inspects process virtual memory metadata without dereferencing `cur`.
        let n = VirtualQuery(cur as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>());
        if n == 0 || mbi.RegionSize == 0 {
            break;
        }
        let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).min(end);
        let accessible = mbi.State == MEM_COMMIT && mbi.Protect != 0 && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
        if accessible {
            let mut pg = cur;
            while pg < region_end {
                holes &= !(1u16 << ((pg - tile) / PAGE));
                pg += PAGE;
            }
        }
        if region_end <= cur {
            break;
        }
        cur = region_end;
    }
    holes
}

/// Update `mapped`, `free`, and `to_zero` for a decommit of `[a, a + n)`.
/// Returns the base addresses of tiles that became completely uncommitted (`holes == 0xFFFF`)
/// and were removed from `mapped` (ready to be unmapped from their sparse window buffers).
fn decommit_tiles(
    mapped: &mut FxHashMap<usize, TileEntry>,
    pinned: &FxHashMap<(u32, u32), u32>,
    free: &mut Vec<(u32, u32)>,
    to_zero: &mut Vec<(usize, usize)>,
    a: usize,
    n: usize,
) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut unmapped: Vec<usize> = Vec::new();
    let mut pg = a & !(PAGE - 1);
    let end = a.saturating_add(n);
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = t.saturating_add(TILE).min(end);
        if let Some(entry) = mapped.get_mut(&t) {
            let mut newly_holed = 0u16;
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                entry.sent[idx] = None;
                let bit = 1u16 << idx;
                if entry.holes & bit == 0 {
                    entry.holes |= bit;
                    newly_holed |= bit;
                }
                p += PAGE;
            }
            if entry.holes == 0xFFFF {
                unmapped.push(t);
            } else if newly_holed != 0 {
                for idx in 0..PAGES_PER_TILE {
                    if newly_holed & (1u16 << idx) != 0 {
                        to_zero.push((t + idx * PAGE, PAGE));
                    }
                }
            }
        }
        if tile_end <= pg {
            break;
        }
        pg = tile_end;
    }
    for &t in &unmapped {
        if let Some(entry) = mapped.remove(&t) {
            if !pinned.contains_key(&entry.pool) {
                free.push(entry.pool);
            }
        }
    }
    if !unmapped.is_empty() {
        to_zero.retain(|&(za, _)| mapped.contains_key(&(za & !(TILE - 1))));
    }
    unmapped
}

struct UploadSlot {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence_val: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SubmitResult {
    Submitted,
    NotSubmitted,
    Unordered,
}

enum UploadStep {
    Done(PreparationOutcome),
    Wait(WaitTicket),
}

enum BeginSlot {
    Ready(usize),
    Wait(WaitTicket),
    RetryRequired,
}

#[derive(Clone, Copy)]
enum WaitKind {
    Slot { index: usize, next_slot: usize, fence: u64 },
    Ring { front: (u64, u64), write: u64, submitted: u64, free: u64 },
}

struct WaitTicket {
    state_generation: u64,
    mapping_epoch: u64,
    dev: ID3D12Device,
    fence: ID3D12Fence,
    target: u64,
    kind: WaitKind,
}

/// Sources which can require one CPU page to be considered by a future immutable upload plan.
/// Multiple sources collapse into one page owner without losing why the page must be revisited.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct WorkReasons(u8);

impl WorkReasons {
    const ZERO: Self = Self(1);
    const RETRY: Self = Self(2);
    const DIRTY: Self = Self(4);

    fn insert(&mut self, other: Self) { self.0 |= other.0; }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WorkClaimToken {
    state_generation: u64,
    serial: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkClaimResolution {
    Committed,
    Restored,
    AlreadyFinished,
    StaleGeneration,
}

struct ClaimedWork {
    token: WorkClaimToken,
    pages: Vec<(usize, WorkReasons)>,
}

/// Inert foundation for splitting sparse planning from its serialized commit phase. Work added
/// while a page is owned stays journalled for a later claim; cancelling an older claim merges its
/// exact reasons back without rewinding or duplicating a newer claim. Tokens include the sparse
/// state generation, so a late completion from replaced state cannot modify the replacement.
#[allow(dead_code)]
struct WorkClaimLedger {
    state_generation: u64,
    next_serial: u64,
    available: BTreeMap<usize, WorkReasons>,
    owners: BTreeMap<usize, u64>,
    claims: BTreeMap<u64, Vec<(usize, WorkReasons)>>,
}

#[allow(dead_code)]
impl WorkClaimLedger {
    fn new(state_generation: u64) -> Self {
        Self {
            state_generation,
            next_serial: 0,
            available: BTreeMap::new(),
            owners: BTreeMap::new(),
            claims: BTreeMap::new(),
        }
    }

    fn add_pages(&mut self, pages: impl IntoIterator<Item = usize>, reason: WorkReasons) {
        for page in pages {
            self.available.entry(page & !(PAGE - 1)).or_default().insert(reason);
        }
    }

    fn add_zero_ranges(&mut self, ranges: &[(usize, usize)]) {
        for &(start, bytes) in ranges {
            if bytes == 0 {
                continue;
            }
            let end = start.saturating_add(bytes);
            let mut page = start & !(PAGE - 1);
            while page < end {
                self.available.entry(page).or_default().insert(WorkReasons::ZERO);
                let Some(next) = page.checked_add(PAGE) else { break };
                page = next;
            }
        }
    }

    fn add_retry_pages(&mut self, pages: impl IntoIterator<Item = usize>) {
        self.add_pages(pages, WorkReasons::RETRY);
    }

    fn add_dirty_pages(&mut self, pages: impl IntoIterator<Item = usize>) {
        self.add_pages(pages, WorkReasons::DIRTY);
    }

    /// Claim at most `limit` pages in address order. A page has one owner even when zero, retry,
    /// and fresh dirty work overlap; reasons added after this snapshot remain for the next claim.
    fn claim(&mut self, limit: usize) -> Option<ClaimedWork> {
        if limit == 0 {
            return None;
        }
        let pages: Vec<_> = self.available
            .iter()
            .filter(|(page, _)| !self.owners.contains_key(page))
            .take(limit)
            .map(|(&page, &reasons)| (page, reasons))
            .collect();
        if pages.is_empty() {
            return None;
        }
        self.next_serial = self.next_serial.checked_add(1).expect("sparse work claim serial exhausted");
        let serial = self.next_serial;
        for &(page, _) in &pages {
            self.available.remove(&page);
            self.owners.insert(page, serial);
        }
        self.claims.insert(serial, pages.clone());
        Some(ClaimedWork {
            token: WorkClaimToken { state_generation: self.state_generation, serial },
            pages,
        })
    }

    fn commit(&mut self, token: WorkClaimToken) -> WorkClaimResolution {
        self.finish(token, false)
    }

    fn restore(&mut self, token: WorkClaimToken) -> WorkClaimResolution {
        self.finish(token, true)
    }

    fn cancel(&mut self, token: WorkClaimToken) -> WorkClaimResolution {
        self.restore(token)
    }

    fn finish(&mut self, token: WorkClaimToken, restore: bool) -> WorkClaimResolution {
        if token.state_generation != self.state_generation {
            return WorkClaimResolution::StaleGeneration;
        }
        let Some(pages) = self.claims.remove(&token.serial) else {
            return WorkClaimResolution::AlreadyFinished;
        };
        for (page, reasons) in pages {
            if self.owners.get(&page) != Some(&token.serial) {
                continue;
            }
            self.owners.remove(&page);
            if restore {
                self.available.entry(page).or_default().insert(reasons);
            }
        }
        if restore { WorkClaimResolution::Restored } else { WorkClaimResolution::Committed }
    }

    fn replace_generation(&mut self, state_generation: u64) {
        self.state_generation = state_generation;
        self.available.clear();
        self.owners.clear();
        self.claims.clear();
    }
}

/// Identity of one future sparse batch while it crosses planning and queue acceptance. `lease`
/// changes on takeover, so an old owner cannot publish or discard the new owner's batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct ClaimFrontierToken {
    state_generation: u64,
    epoch: u64,
    lease: u64,
    owner: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClaimFrontierState {
    Pending,
    Accepted,
    Receipted(u64),
    Obsolete,
}

#[allow(dead_code)]
struct ClaimFrontierRecord {
    lease: u64,
    owner: u64,
    state: ClaimFrontierState,
    prefix_fence: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum ClaimFrontierAction {
    Ready { through: u64, required_fence: Option<u64> },
    Progress(ClaimFrontierToken),
    TakeOver(ClaimFrontierToken),
    WaitForReceipt(ClaimFrontierToken),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum ClaimFrontierError {
    StaleGeneration,
    UnknownEpoch,
    StaleToken,
    WrongState,
    Fence,
    Exhausted,
}

/// Inert coordination model for future asynchronous sparse planning. Every observed epoch remains
/// visible until it is receipted or explicitly consumed as obsolete; a foreground queue therefore
/// cannot mistake work owned by another planner for an empty journal. There are no callers yet.
#[allow(dead_code)]
struct ClaimFrontier {
    state_generation: u64,
    next_epoch: u64,
    next_lease: u64,
    retired_through: u64,
    records: BTreeMap<u64, ClaimFrontierRecord>,
}

#[allow(dead_code)]
impl ClaimFrontier {
    fn new(state_generation: u64) -> Self {
        Self {
            state_generation,
            next_epoch: 0,
            next_lease: 0,
            retired_through: 0,
            records: BTreeMap::new(),
        }
    }

    fn claim(&mut self, owner: u64) -> Result<ClaimFrontierToken, ClaimFrontierError> {
        let epoch = self.next_epoch.checked_add(1).ok_or(ClaimFrontierError::Exhausted)?;
        let lease = self.next_lease.checked_add(1).ok_or(ClaimFrontierError::Exhausted)?;
        let token = ClaimFrontierToken { state_generation: self.state_generation, epoch, lease, owner };
        self.records.insert(epoch, ClaimFrontierRecord {
            lease,
            owner,
            state: ClaimFrontierState::Pending,
            prefix_fence: None,
        });
        self.next_epoch = epoch;
        self.next_lease = lease;
        Ok(token)
    }

    fn observe(
        &self, state_generation: u64, through: u64, requester: u64,
    ) -> Result<ClaimFrontierAction, ClaimFrontierError> {
        if state_generation != self.state_generation {
            return Err(ClaimFrontierError::StaleGeneration);
        }
        if through > self.next_epoch {
            return Err(ClaimFrontierError::UnknownEpoch);
        }
        if through <= self.retired_through {
            let required_fence = (through != 0)
                .then(|| self.records.get(&through).and_then(|record| record.prefix_fence))
                .flatten();
            return Ok(ClaimFrontierAction::Ready { through, required_fence });
        }
        let epoch = self.retired_through.checked_add(1).ok_or(ClaimFrontierError::Exhausted)?;
        let record = self.records.get(&epoch).ok_or(ClaimFrontierError::UnknownEpoch)?;
        let token = ClaimFrontierToken {
            state_generation: self.state_generation,
            epoch,
            lease: record.lease,
            owner: record.owner,
        };
        Ok(match record.state {
            ClaimFrontierState::Pending if record.owner == requester => ClaimFrontierAction::Progress(token),
            ClaimFrontierState::Pending => ClaimFrontierAction::TakeOver(token),
            ClaimFrontierState::Accepted => ClaimFrontierAction::WaitForReceipt(token),
            ClaimFrontierState::Receipted(_) | ClaimFrontierState::Obsolete => unreachable!("terminal frontier did not advance"),
        })
    }

    fn take_over(
        &mut self, token: ClaimFrontierToken, owner: u64,
    ) -> Result<ClaimFrontierToken, ClaimFrontierError> {
        self.validate(token, ClaimFrontierState::Pending)?;
        let lease = self.next_lease.checked_add(1).ok_or(ClaimFrontierError::Exhausted)?;
        let record = self.records.get_mut(&token.epoch).unwrap();
        record.lease = lease;
        record.owner = owner;
        self.next_lease = lease;
        Ok(ClaimFrontierToken { state_generation: self.state_generation, epoch: token.epoch, lease, owner })
    }

    fn accept(&mut self, token: ClaimFrontierToken) -> Result<(), ClaimFrontierError> {
        self.validate(token, ClaimFrontierState::Pending)?;
        self.records.get_mut(&token.epoch).unwrap().state = ClaimFrontierState::Accepted;
        Ok(())
    }

    fn receipt(&mut self, token: ClaimFrontierToken, fence: u64) -> Result<(), ClaimFrontierError> {
        self.validate(token, ClaimFrontierState::Accepted)?;
        if fence == 0 {
            return Err(ClaimFrontierError::Fence);
        }
        self.records.get_mut(&token.epoch).unwrap().state = ClaimFrontierState::Receipted(fence);
        self.advance();
        Ok(())
    }

    fn consume_obsolete(&mut self, token: ClaimFrontierToken) -> Result<(), ClaimFrontierError> {
        self.validate(token, ClaimFrontierState::Pending)?;
        self.records.get_mut(&token.epoch).unwrap().state = ClaimFrontierState::Obsolete;
        self.advance();
        Ok(())
    }

    fn validate(
        &self, token: ClaimFrontierToken, expected: ClaimFrontierState,
    ) -> Result<(), ClaimFrontierError> {
        if token.state_generation != self.state_generation {
            return Err(ClaimFrontierError::StaleGeneration);
        }
        let record = self.records.get(&token.epoch).ok_or(ClaimFrontierError::StaleToken)?;
        if record.lease != token.lease || record.owner != token.owner {
            return Err(ClaimFrontierError::StaleToken);
        }
        if record.state != expected {
            return Err(ClaimFrontierError::WrongState);
        }
        Ok(())
    }

    fn advance(&mut self) {
        let mut prefix_fence = if self.retired_through == 0 {
            0
        } else {
            self.records[&self.retired_through].prefix_fence.unwrap_or(0)
        };
        loop {
            let Some(next) = self.retired_through.checked_add(1) else { return };
            let Some(record) = self.records.get_mut(&next) else { return };
            match record.state {
                ClaimFrontierState::Receipted(fence) => prefix_fence = prefix_fence.max(fence),
                ClaimFrontierState::Obsolete => {}
                ClaimFrontierState::Pending | ClaimFrontierState::Accepted => return,
            }
            record.prefix_fence = (prefix_fence != 0).then_some(prefix_fence);
            self.retired_through = next;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum SparseBatchError {
    StaleGeneration,
    Fence,
    Poisoned,
    Unsealed,
    Exhausted,
    Corrupt,
    Frontier(ClaimFrontierError),
}

/// One atomic owner for work claims and their visibility frontier. This remains inert: no live
/// sparse state, command list, queue or fence uses it yet. Keeping both ledgers under one mutex is
/// what makes a lease rotation revoke the old planner without restoring or duplicating its work.
#[allow(dead_code)]
struct SparseBatchLedger {
    work: WorkClaimLedger,
    frontier: ClaimFrontier,
    fence: Option<SparseFenceIdentity>,
    bindings: BTreeMap<u64, WorkClaimToken>,
    poisoned: bool,
}

#[allow(dead_code)]
impl SparseBatchLedger {
    fn new(state_generation: u64) -> Self {
        Self {
            work: WorkClaimLedger::new(state_generation),
            frontier: ClaimFrontier::new(state_generation),
            fence: None,
            bindings: BTreeMap::new(),
            poisoned: false,
        }
    }

    fn check(&self, state_generation: u64) -> Result<(), SparseBatchError> {
        if state_generation != self.work.state_generation {
            return Err(SparseBatchError::StaleGeneration);
        }
        if self.poisoned {
            return Err(SparseBatchError::Poisoned);
        }
        Ok(())
    }

    fn seal(
        &mut self, state_generation: u64, owner: u64, limit: usize,
    ) -> Result<Option<(ClaimFrontierToken, Vec<(usize, WorkReasons)>)>, SparseBatchError> {
        self.check(state_generation)?;
        if self.work.next_serial == u64::MAX {
            return Err(SparseBatchError::Exhausted);
        }
        let Some(claimed) = self.work.claim(limit) else { return Ok(None) };
        let token = match self.frontier.claim(owner) {
            Ok(token) => token,
            Err(error) => {
                let _ = self.work.restore(claimed.token);
                return Err(SparseBatchError::Frontier(error));
            }
        };
        self.bindings.insert(token.epoch, claimed.token);
        Ok(Some((token, claimed.pages)))
    }

    fn pages(&self, token: ClaimFrontierToken) -> Result<Vec<(usize, WorkReasons)>, SparseBatchError> {
        self.check(token.state_generation)?;
        self.frontier.validate(token, ClaimFrontierState::Pending).map_err(SparseBatchError::Frontier)?;
        let work = self.bindings.get(&token.epoch).ok_or(SparseBatchError::Corrupt)?;
        self.work.claims.get(&work.serial).cloned().ok_or(SparseBatchError::Corrupt)
    }

    fn take_over(
        &mut self, token: ClaimFrontierToken, owner: u64,
    ) -> Result<(ClaimFrontierToken, Vec<(usize, WorkReasons)>), SparseBatchError> {
        let pages = self.pages(token)?;
        let token = self.frontier.take_over(token, owner).map_err(SparseBatchError::Frontier)?;
        Ok((token, pages))
    }

    fn observe(
        &self, state_generation: u64, through: u64, requester: u64,
    ) -> Result<ClaimFrontierAction, SparseBatchError> {
        self.check(state_generation)?;
        if !self.work.available.is_empty() {
            return Err(SparseBatchError::Unsealed);
        }
        self.frontier.observe(state_generation, through, requester).map_err(SparseBatchError::Frontier)
    }

    fn through(&self, state_generation: u64) -> Result<u64, SparseBatchError> {
        self.check(state_generation)?;
        if !self.work.available.is_empty() {
            return Err(SparseBatchError::Unsealed);
        }
        Ok(self.frontier.next_epoch)
    }

    fn accept(&mut self, token: ClaimFrontierToken) -> Result<WorkClaimToken, SparseBatchError> {
        self.check(token.state_generation)?;
        let work = self.binding(token, ClaimFrontierState::Pending)?;
        self.frontier.accept(token).map_err(SparseBatchError::Frontier)?;
        Ok(work)
    }

    fn receipt(&mut self, token: ClaimFrontierToken, fence: u64) -> Result<(), SparseBatchError> {
        self.check(token.state_generation)?;
        if fence == 0 {
            return Err(SparseBatchError::Frontier(ClaimFrontierError::Fence));
        }
        let work = self.binding(token, ClaimFrontierState::Accepted)?;
        if self.work.commit(work) != WorkClaimResolution::Committed {
            self.poisoned = true;
            return Err(SparseBatchError::Corrupt);
        }
        if let Err(error) = self.frontier.receipt(token, fence) {
            self.poisoned = true;
            return Err(SparseBatchError::Frontier(error));
        }
        self.bindings.remove(&token.epoch);
        Ok(())
    }

    fn obsolete(&mut self, token: ClaimFrontierToken) -> Result<(), SparseBatchError> {
        self.check(token.state_generation)?;
        let work = self.binding(token, ClaimFrontierState::Pending)?;
        if self.work.commit(work) != WorkClaimResolution::Committed {
            self.poisoned = true;
            return Err(SparseBatchError::Corrupt);
        }
        if let Err(error) = self.frontier.consume_obsolete(token) {
            self.poisoned = true;
            return Err(SparseBatchError::Frontier(error));
        }
        self.bindings.remove(&token.epoch);
        Ok(())
    }

    fn retry(&mut self, token: ClaimFrontierToken) -> Result<(), SparseBatchError> {
        self.check(token.state_generation)?;
        let work = self.binding(token, ClaimFrontierState::Pending)?;
        if self.work.restore(work) != WorkClaimResolution::Restored {
            self.poisoned = true;
            return Err(SparseBatchError::Corrupt);
        }
        if let Err(error) = self.frontier.consume_obsolete(token) {
            self.poisoned = true;
            return Err(SparseBatchError::Frontier(error));
        }
        self.bindings.remove(&token.epoch);
        Ok(())
    }

    fn poison_exact(
        &mut self, token: ClaimFrontierToken, work: WorkClaimToken,
    ) -> Result<(), SparseBatchError> {
        if token.state_generation != self.work.state_generation || work.state_generation != token.state_generation {
            return Err(SparseBatchError::StaleGeneration);
        }
        let resolution = self.work.commit(work);
        if !matches!(resolution, WorkClaimResolution::Committed | WorkClaimResolution::AlreadyFinished) {
            return Err(SparseBatchError::Corrupt);
        }
        if self.bindings.get(&token.epoch) == Some(&work) { self.bindings.remove(&token.epoch); }
        self.poisoned = true;
        Ok(())
    }

    fn binding(
        &self, token: ClaimFrontierToken, state: ClaimFrontierState,
    ) -> Result<WorkClaimToken, SparseBatchError> {
        self.frontier.validate(token, state).map_err(SparseBatchError::Frontier)?;
        let work = *self.bindings.get(&token.epoch).ok_or(SparseBatchError::Corrupt)?;
        self.work.claims.contains_key(&work.serial).then_some(work).ok_or(SparseBatchError::Corrupt)
    }

    fn replace_generation(
        &mut self, current: u64, replacement: u64, fence_allocation: Option<u64>,
    ) -> Result<(), SparseBatchError> {
        if current != self.work.state_generation {
            return Err(SparseBatchError::StaleGeneration);
        }
        // Generations are identities, not labels: reuse could resurrect an old epoch/lease tuple.
        if replacement <= current {
            return Err(SparseBatchError::StaleGeneration);
        }
        self.fence = match (self.fence, fence_allocation) {
            (None, None) => None,
            (Some(fence), Some(allocation)) if allocation > fence.allocation => {
                Some(SparseFenceIdentity { state_generation: replacement, allocation })
            }
            _ => return Err(SparseBatchError::Fence),
        };
        self.work.replace_generation(replacement);
        self.frontier = ClaimFrontier::new(replacement);
        self.bindings.clear();
        self.poisoned = false;
        Ok(())
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct SparseBatches {
    ledger: Arc<Mutex<SparseBatchLedger>>,
    state_generation: u64,
}

#[allow(dead_code)]
impl SparseBatches {
    fn new(state_generation: u64) -> Self {
        Self { ledger: Arc::new(Mutex::new(SparseBatchLedger::new(state_generation))), state_generation }
    }

    fn with_fence(state_generation: u64, fence_allocation: u64) -> Option<Self> {
        if state_generation == 0 || fence_allocation == 0 { return None }
        let batches = Self::new(state_generation);
        batches.ledger.lock().unwrap().fence = Some(SparseFenceIdentity { state_generation, allocation: fence_allocation });
        Some(batches)
    }

    fn add_pages(&self, pages: impl IntoIterator<Item = usize>, reason: WorkReasons) -> bool {
        // Iterator code is caller code: finish it before taking the non-reentrant ledger mutex.
        let mut pages: Vec<_> = pages.into_iter().map(|page| page & !(PAGE - 1)).collect();
        pages.sort_unstable();
        pages.dedup();
        let mut ledger = self.ledger.lock().unwrap();
        if ledger.check(self.state_generation).is_err() { return false }
        ledger.work.add_pages(pages, reason);
        true
    }

    fn add_dirty_pages(&self, pages: impl IntoIterator<Item = usize>) -> bool {
        self.add_pages(pages, WorkReasons::DIRTY)
    }

    fn add_retry_pages(&self, pages: impl IntoIterator<Item = usize>) -> bool {
        self.add_pages(pages, WorkReasons::RETRY)
    }

    fn add_zero_ranges(&self, ranges: &[(usize, usize)]) -> bool {
        let mut ledger = self.ledger.lock().unwrap();
        if ledger.check(self.state_generation).is_err() { return false }
        ledger.work.add_zero_ranges(ranges);
        true
    }

    /// Mirror one legacy upload attempt into an exact-generation batch. All three reason sets are
    /// added and claimed under one ledger lock so a future unlocked planner cannot observe only a
    /// prefix of the legacy journal.
    fn seal_legacy(
        &self,
        owner: u64,
        zero: &[(usize, usize)],
        retry: &[usize],
        dirty: &[(usize, usize)],
    ) -> Result<Option<SparseBatchLease>, SparseBatchError> {
        let mut dirty_pages = Vec::new();
        for &(start, bytes) in dirty {
            let end = start.saturating_add(bytes);
            let mut page = start & !(PAGE - 1);
            while page < end {
                dirty_pages.push(page);
                let Some(next) = page.checked_add(PAGE) else { break };
                page = next;
            }
        }
        dirty_pages.sort_unstable();
        dirty_pages.dedup();

        let mut ledger = self.ledger.lock().unwrap();
        ledger.check(self.state_generation)?;
        ledger.work.add_zero_ranges(zero);
        ledger.work.add_retry_pages(retry.iter().copied());
        ledger.work.add_dirty_pages(dirty_pages);
        let result = ledger.seal(self.state_generation, owner, usize::MAX)?;
        Ok(result.map(|(token, pages)| SparseBatchLease { ledger: self.ledger.clone(), token, pages }))
    }

    fn seal(&self, owner: u64, limit: usize) -> Result<Option<SparseBatchLease>, SparseBatchError> {
        let result = self.ledger.lock().unwrap().seal(self.state_generation, owner, limit)?;
        Ok(result.map(|(token, pages)| SparseBatchLease { ledger: self.ledger.clone(), token, pages }))
    }

    fn observe(&self, through: u64, requester: u64) -> Result<ClaimFrontierAction, SparseBatchError> {
        self.ledger.lock().unwrap().observe(self.state_generation, through, requester)
    }

    fn ready_receipt(&self, through: u64, requester: u64) -> Result<SparseReadyReceipt, ForegroundWaitError> {
        let ledger = self.ledger.lock().unwrap();
        let fence = ledger.fence.ok_or(ForegroundWaitError::Fence)?;
        match ledger.observe(self.state_generation, through, requester).map_err(ForegroundWaitError::Batch)? {
            ClaimFrontierAction::Ready { through, required_fence } => Ok(SparseReadyReceipt {
                state_generation: self.state_generation, fence, through, required_fence,
            }),
            _ => Err(ForegroundWaitError::NotReady),
        }
    }

    fn through(&self) -> Result<u64, SparseBatchError> {
        self.ledger.lock().unwrap().through(self.state_generation)
    }

    fn take_over(
        &self, token: ClaimFrontierToken, owner: u64,
    ) -> Result<SparseBatchLease, SparseBatchError> {
        let (token, pages) = self.ledger.lock().unwrap().take_over(token, owner)?;
        Ok(SparseBatchLease { ledger: self.ledger.clone(), token, pages })
    }

    fn replace_generation(&self, replacement: u64) -> Result<Self, SparseBatchError> {
        self.ledger.lock().unwrap().replace_generation(self.state_generation, replacement, None)?;
        Ok(Self { ledger: self.ledger.clone(), state_generation: replacement })
    }

    fn replace_generation_with_fence(
        &self, replacement: u64, fence_allocation: u64,
    ) -> Result<Self, SparseBatchError> {
        self.ledger.lock().unwrap().replace_generation(
            self.state_generation, replacement, Some(fence_allocation),
        )?;
        Ok(Self { ledger: self.ledger.clone(), state_generation: replacement })
    }
}

#[allow(dead_code)]
struct SparseBatchLease {
    ledger: Arc<Mutex<SparseBatchLedger>>,
    token: ClaimFrontierToken,
    pages: Vec<(usize, WorkReasons)>,
}

#[allow(dead_code)]
impl SparseBatchLease {
    fn pages(&self) -> &[(usize, WorkReasons)] { &self.pages }

    fn take_over(&self, owner: u64) -> Result<Self, SparseBatchError> {
        let (token, pages) = self.ledger.lock().unwrap().take_over(self.token, owner)?;
        Ok(Self { ledger: self.ledger.clone(), token, pages })
    }

    fn accept(self) -> Result<AcceptedSparseBatch, SparseBatchError> {
        let work = self.ledger.lock().unwrap().accept(self.token)?;
        Ok(AcceptedSparseBatch { ledger: self.ledger.clone(), token: Some((self.token, work)) })
    }

    fn obsolete(self) -> Result<(), SparseBatchError> {
        self.ledger.lock().unwrap().obsolete(self.token)
    }

    fn retry(self) -> Result<(), SparseBatchError> {
        self.ledger.lock().unwrap().retry(self.token)
    }
}

/// Accepted work is fail-closed: losing its owner consumes the claim and poisons the generation.
#[allow(dead_code)]
struct AcceptedSparseBatch {
    ledger: Arc<Mutex<SparseBatchLedger>>,
    token: Option<(ClaimFrontierToken, WorkClaimToken)>,
}

#[allow(dead_code)]
impl AcceptedSparseBatch {
    fn receipt(mut self, fence: u64) -> Result<(), SparseBatchError> {
        let result = self.ledger.lock().unwrap().receipt(self.token.unwrap().0, fence);
        if result.is_ok() { self.token.take(); }
        result
    }
}

impl Drop for AcceptedSparseBatch {
    fn drop(&mut self) {
        if let Some((token, work)) = self.token.take() {
            let _ = self.ledger.lock().unwrap().poison_exact(token, work);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[allow(dead_code)]
struct SparseFenceIdentity {
    state_generation: u64, allocation: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[allow(dead_code)]
struct TitleQueueIdentity {
    raw: usize, generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct SparseReadyReceipt {
    state_generation: u64, fence: SparseFenceIdentity, through: u64, required_fence: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct ForegroundWaitToken {
    state_generation: u64, fence: SparseFenceIdentity, queue: TitleQueueIdentity, serial: u64, value: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ForegroundWaitState { Reserved, Accepted }

struct ForegroundWaitRecord {
    fence: SparseFenceIdentity, installed: u64,
    pending: Option<(ForegroundWaitToken, ForegroundWaitState)>, uncertain: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum ForegroundWaitError {
    StaleGeneration, Fence, Queue, NotReady, Pending(ForegroundWaitToken),
    Uncertain, StaleToken, Exhausted,
    Batch(SparseBatchError),
}

#[allow(dead_code)]
enum ForegroundWaitDecision {
    NoWait { through: u64 }, Installed { through: u64, value: u64 },
    Reserve(ForegroundWaitReservation),
}

#[allow(dead_code)]
struct ForegroundWaitLedger {
    state_generation: u64, fence: SparseFenceIdentity,
    next_queue_generation: u64, next_serial: u64,
    queues: BTreeMap<usize, u64>,
    waits: BTreeMap<TitleQueueIdentity, ForegroundWaitRecord>,
}

#[allow(dead_code)]
impl ForegroundWaitLedger {
    fn new(state_generation: u64, fence_allocation: u64) -> Option<Self> {
        if state_generation == 0 || fence_allocation == 0 { return None }
        Some(Self {
            state_generation,
            fence: SparseFenceIdentity { state_generation, allocation: fence_allocation },
            next_queue_generation: 0,
            next_serial: 0,
            queues: BTreeMap::new(),
            waits: BTreeMap::new(),
        })
    }

    fn register_queue(&mut self, raw: usize) -> Result<TitleQueueIdentity, ForegroundWaitError> {
        if raw == 0 { return Err(ForegroundWaitError::Queue) }
        if let Some(&generation) = self.queues.get(&raw) {
            return Ok(TitleQueueIdentity { raw, generation });
        }
        let generation = self.next_queue_generation.checked_add(1).ok_or(ForegroundWaitError::Exhausted)?;
        self.next_queue_generation = generation;
        self.queues.insert(raw, generation);
        Ok(TitleQueueIdentity { raw, generation })
    }

    fn retire_queue(&mut self, queue: TitleQueueIdentity) -> Result<(), ForegroundWaitError> {
        if self.queues.get(&queue.raw) != Some(&queue.generation) { return Err(ForegroundWaitError::Queue) }
        self.queues.remove(&queue.raw);
        self.waits.remove(&queue);
        Ok(())
    }

    fn reserve(
        &mut self, queue: TitleQueueIdentity, ready: SparseReadyReceipt,
    ) -> Result<ForegroundWaitDecision, ForegroundWaitError> {
        if ready.state_generation != self.state_generation { return Err(ForegroundWaitError::StaleGeneration) }
        if ready.fence != self.fence { return Err(ForegroundWaitError::Fence) }
        if self.queues.get(&queue.raw) != Some(&queue.generation) { return Err(ForegroundWaitError::Queue) }
        if let Some(record) = self.waits.get(&queue) {
            if record.fence != ready.fence { return Err(ForegroundWaitError::Fence) }
            if record.uncertain { return Err(ForegroundWaitError::Uncertain) }
            if let Some((token, _)) = record.pending { return Err(ForegroundWaitError::Pending(token)) }
        }
        let Some(value) = ready.required_fence else { return Ok(ForegroundWaitDecision::NoWait { through: ready.through }) };
        if value == 0 { return Err(ForegroundWaitError::Fence) }
        let record = self.waits.entry(queue).or_insert(ForegroundWaitRecord {
            fence: ready.fence,
            installed: 0,
            pending: None,
            uncertain: false,
        });
        if record.installed >= value { return Ok(ForegroundWaitDecision::Installed { through: ready.through, value }) }
        let serial = self.next_serial.checked_add(1).ok_or(ForegroundWaitError::Exhausted)?;
        self.next_serial = serial;
        let token = ForegroundWaitToken { state_generation: self.state_generation, fence: ready.fence, queue, serial, value };
        record.pending = Some((token, ForegroundWaitState::Reserved));
        Ok(ForegroundWaitDecision::Reserve(ForegroundWaitReservation {
            ledger: None,
            token: Some(token),
            through: ready.through,
        }))
    }

    fn transition(
        &mut self, token: ForegroundWaitToken, expected: ForegroundWaitState,
        replacement: Option<ForegroundWaitState>,
    ) -> Result<(), ForegroundWaitError> {
        if token.state_generation != self.state_generation { return Err(ForegroundWaitError::StaleGeneration) }
        if token.fence != self.fence { return Err(ForegroundWaitError::Fence) }
        if self.queues.get(&token.queue.raw) != Some(&token.queue.generation) {
            return Err(ForegroundWaitError::Queue);
        }
        let record = self.waits.get_mut(&token.queue).ok_or(ForegroundWaitError::StaleToken)?;
        if record.pending != Some((token, expected)) { return Err(ForegroundWaitError::StaleToken) }
        record.pending = replacement.map(|state| (token, state));
        Ok(())
    }

    fn publish(&mut self, token: ForegroundWaitToken) -> Result<(), ForegroundWaitError> {
        self.transition(token, ForegroundWaitState::Accepted, None)?;
        self.waits.get_mut(&token.queue).unwrap().installed = token.value;
        Ok(())
    }

    fn uncertain(&mut self, token: ForegroundWaitToken) {
        if self.transition(token, ForegroundWaitState::Accepted, None).is_ok() {
            self.waits.get_mut(&token.queue).unwrap().uncertain = true;
        }
    }

    fn replace_generation(
        &mut self, current: u64, replacement: u64, fence_allocation: u64,
    ) -> Result<(), ForegroundWaitError> {
        if current != self.state_generation || replacement <= current {
            return Err(ForegroundWaitError::StaleGeneration);
        }
        if fence_allocation <= self.fence.allocation { return Err(ForegroundWaitError::Fence) }
        self.state_generation = replacement;
        self.fence = SparseFenceIdentity { state_generation: replacement, allocation: fence_allocation };
        self.waits.clear();
        Ok(())
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct ForegroundWaits(Arc<Mutex<ForegroundWaitLedger>>);

#[allow(dead_code)]
impl ForegroundWaits {
    fn new(state_generation: u64, fence_allocation: u64) -> Option<Self> {
        Some(Self(Arc::new(Mutex::new(ForegroundWaitLedger::new(state_generation, fence_allocation)?))))
    }

    fn register_queue(&self, raw: usize) -> Result<TitleQueueIdentity, ForegroundWaitError> {
        self.0.lock().unwrap().register_queue(raw)
    }

    fn fence(&self) -> SparseFenceIdentity { self.0.lock().unwrap().fence }

    fn retire_queue(&self, queue: TitleQueueIdentity) -> Result<(), ForegroundWaitError> {
        self.0.lock().unwrap().retire_queue(queue)
    }

    fn replace_generation(
        &self, current: u64, replacement: u64, fence_allocation: u64,
    ) -> Result<(), ForegroundWaitError> {
        self.0.lock().unwrap().replace_generation(current, replacement, fence_allocation)
    }

    fn reserve(
        &self, queue: TitleQueueIdentity, ready: SparseReadyReceipt,
    ) -> Result<ForegroundWaitDecision, ForegroundWaitError> {
        let result = self.0.lock().unwrap().reserve(queue, ready)?;
        Ok(match result {
            ForegroundWaitDecision::Reserve(mut reservation) => {
                reservation.ledger = Some(self.0.clone());
                ForegroundWaitDecision::Reserve(reservation)
            }
            ready => ready,
        })
    }
}

#[allow(dead_code)]
struct ForegroundWaitReservation {
    ledger: Option<Arc<Mutex<ForegroundWaitLedger>>>, token: Option<ForegroundWaitToken>, through: u64,
}

#[allow(dead_code)]
impl ForegroundWaitReservation {
    fn arm(mut self) -> Result<AcceptedForegroundWait, ForegroundWaitError> {
        let token = self.token.unwrap();
        self.ledger.as_ref().unwrap().lock().unwrap().transition(
            token, ForegroundWaitState::Reserved, Some(ForegroundWaitState::Accepted),
        )?;
        self.token.take();
        Ok(AcceptedForegroundWait { ledger: self.ledger.take().unwrap(), token: Some(token), through: self.through })
    }
}

impl Drop for ForegroundWaitReservation {
    fn drop(&mut self) {
        if let (Some(ledger), Some(token)) = (&self.ledger, self.token.take()) {
            let _ = ledger.lock().unwrap().transition(token, ForegroundWaitState::Reserved, None);
        }
    }
}

#[allow(dead_code)]
struct AcceptedForegroundWait {
    ledger: Arc<Mutex<ForegroundWaitLedger>>, token: Option<ForegroundWaitToken>, through: u64,
}

#[allow(dead_code)]
impl AcceptedForegroundWait {
    fn publish(mut self) -> Result<(u64, u64), ForegroundWaitError> {
        let token = self.token.unwrap();
        self.ledger.lock().unwrap().publish(token)?;
        self.token.take();
        Ok((self.through, token.value))
    }

    fn rollback(mut self) -> Result<(), ForegroundWaitError> {
        let token = self.token.unwrap();
        self.ledger.lock().unwrap().transition(token, ForegroundWaitState::Accepted, None)?;
        self.token.take();
        Ok(())
    }
}

impl Drop for AcceptedForegroundWait {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() { self.ledger.lock().unwrap().uncertain(token) }
    }
}

/// Point-in-time identities needed to prove that planning and a later serialized upload commit
/// still refer to the same sparse page. `source` names the observed CPU page by chunk hash rather
/// than retaining a raw pointer across the unlocked interval. Title writes are outside sparse
/// state/upload exclusion, so a future commit must own or revalidate the bytes it actually copies.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct UploadPageObservation {
    page: usize,
    pool: (u32, u32),
    hole: bool,
    sent: Option<[u64; CHUNKS]>,
    source: [u64; CHUNKS],
}

/// Owned point-in-time input for building a coordinated upload plan after releasing `STATE`.
/// Capture is the only step that needs the live mapping and title bytes; plan construction uses
/// these immutable observations, then commit revalidates them against the current State.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct UploadPlanningSnapshot {
    state_generation: u64,
    mapping_epoch: u64,
    claimed: Vec<(usize, WorkReasons)>,
    observations: Vec<UploadPageObservation>,
}

#[allow(dead_code)]
impl UploadPlanningSnapshot {
    fn capture(
        state_generation: u64,
        mapping_epoch: u64,
        claimed: &[(usize, WorkReasons)],
        mut observe: impl FnMut(usize) -> Option<UploadPageObservation>,
    ) -> Option<Self> {
        if state_generation == 0 { return None }
        let observations = claimed.iter().map(|&(page, _)| observe(page)).collect::<Option<Vec<_>>>()?;
        Some(Self { state_generation, mapping_epoch, claimed: claimed.to_vec(), observations })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct PlannedUploadPage {
    page: usize,
    tile: usize,
    pool: (u32, u32),
    reasons: WorkReasons,
    zero: bool,
    dirty: bool,
    dirty_ranges: Vec<(usize, usize)>,
    sent: Option<[u64; CHUNKS]>,
    source: [u64; CHUNKS],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum UploadPlanInvalid {
    StateGeneration,
    MappingEpoch,
    MissingPage,
    Mapping,
    SentHashes,
    SourceHashes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LegacyParityRevalidation {
    Ready,
    RetryCurrentGeneration,
    RestartForGeneration,
}

/// Private owned stand-in for a future mapped staging allocation. Keeping the copied bytes here
/// makes it impossible for the record/execute callback to retain a borrowed title-memory view.
#[allow(dead_code)]
struct OwnedUploadStaging {
    range: UploadRingRange,
    bytes: Box<[u8]>,
}

fn build_upload_pages(
    claimed: &[(usize, WorkReasons)], mut observe: impl FnMut(usize) -> Option<UploadPageObservation>,
) -> Option<Vec<PlannedUploadPage>> {
    let mut pages = Vec::with_capacity(claimed.len());
    for &(page, reasons) in claimed {
        let observation = observe(page)?;
        if observation.page != page || observation.hole { return None }
        let dirty = reasons.0 & (WorkReasons::DIRTY.0 | WorkReasons::RETRY.0) != 0;
        let mut dirty_ranges = Vec::new();
        if dirty {
            let sent = observation.sent.as_ref().unwrap_or(&ZERO_HASHES);
            for (chunk, (&source, &previous)) in observation.source.iter().zip(sent).enumerate() {
                if source != previous { push_run(&mut dirty_ranges, page + chunk * CHUNK, CHUNK) }
            }
        }
        pages.push(PlannedUploadPage {
            page,
            tile: page & !(TILE - 1),
            pool: observation.pool,
            reasons,
            zero: reasons.0 & WorkReasons::ZERO.0 != 0,
            dirty,
            dirty_ranges,
            sent: observation.sent,
            source: observation.source,
        });
    }
    Some(pages)
}

fn validate_upload_pages(
    pages: &[PlannedUploadPage], mut observe: impl FnMut(usize) -> Option<UploadPageObservation>,
) -> Result<(), UploadPlanInvalid> {
    for planned in pages {
        let current = observe(planned.page).ok_or(UploadPlanInvalid::MissingPage)?;
        if current.page != planned.page
            || current.page & !(TILE - 1) != planned.tile
            || current.pool != planned.pool
            || current.hole
        {
            return Err(UploadPlanInvalid::Mapping);
        }
        if current.sent != planned.sent { return Err(UploadPlanInvalid::SentHashes) }
        if current.source != planned.source { return Err(UploadPlanInvalid::SourceHashes) }
    }
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum CoordinatedSparseUploadError<C, E, S> {
    Invalid(UploadPlanInvalid),
    Reserve(UploadReservationError),
    Copy(C),
    StagingSize,
    Prepare(E),
    Accept(SparseBatchError),
    Signal(S),
    PublishReservation(UploadReservationError),
    PublishBatch(SparseBatchError),
}

/// New inert composition path: the plan owns a revocable batch lease rather than an independently
/// restoring work guard. Dropping before acceptance leaves the batch visible for foreground help.
#[allow(dead_code)]
struct CoordinatedSparseUploadPlan {
    lease: Option<SparseBatchLease>,
    state_generation: u64,
    mapping_epoch: u64,
    pages: Vec<PlannedUploadPage>,
}

#[allow(dead_code)]
impl CoordinatedSparseUploadPlan {
    fn build(
        lease: SparseBatchLease,
        state_generation: u64,
        mapping_epoch: u64,
        observe: impl FnMut(usize) -> Option<UploadPageObservation>,
    ) -> Option<Self> {
        let snapshot = UploadPlanningSnapshot::capture(
            state_generation, mapping_epoch, lease.pages(), observe,
        )?;
        Self::from_snapshot(lease, snapshot)
    }

    /// Build only from owned observations. A live caller captures `snapshot` under `STATE`, drops
    /// that lock, and performs this allocation/scanning work before reacquiring for `validate`.
    fn from_snapshot(lease: SparseBatchLease, snapshot: UploadPlanningSnapshot) -> Option<Self> {
        if lease.token.state_generation != snapshot.state_generation
            || lease.pages() != snapshot.claimed
            || snapshot.observations.len() != snapshot.claimed.len()
        {
            return None;
        }
        let mut observations = snapshot.observations.into_iter();
        let pages = build_upload_pages(&snapshot.claimed, |page| {
            let observation = observations.next()?;
            (observation.page == page).then_some(observation)
        })?;
        Some(Self {
            lease: Some(lease),
            state_generation: snapshot.state_generation,
            mapping_epoch: snapshot.mapping_epoch,
            pages,
        })
    }

    fn validate(
        &self,
        state_generation: u64,
        mapping_epoch: u64,
        observe: impl FnMut(usize) -> Option<UploadPageObservation>,
    ) -> Result<(), UploadPlanInvalid> {
        if state_generation != self.state_generation { return Err(UploadPlanInvalid::StateGeneration) }
        if mapping_epoch != self.mapping_epoch { return Err(UploadPlanInvalid::MappingEpoch) }
        validate_upload_pages(&self.pages, observe)
    }

    fn obsolete(mut self) -> Result<(), SparseBatchError> {
        self.lease.take().expect("unfinished coordinated sparse upload plan").obsolete()
    }

    fn retry(mut self) -> Result<(), SparseBatchError> {
        self.lease.take().expect("unfinished coordinated sparse upload plan").retry()
    }

    fn commit_reserved<C, E, S>(
        mut self,
        state_generation: u64,
        mapping_epoch: u64,
        observe: impl FnMut(usize) -> Option<UploadPageObservation>,
        reservations: &UploadReservations,
        slot: usize,
        bytes: usize,
        copy: impl FnOnce(UploadRingRange, &[PlannedUploadPage]) -> Result<Vec<u8>, C>,
        prepare: impl FnOnce(&OwnedUploadStaging, &[PlannedUploadPage]) -> Result<(), E>,
        execute: impl FnOnce(&OwnedUploadStaging, &[PlannedUploadPage]),
        signal: impl FnOnce() -> Result<u64, S>,
    ) -> Result<UploadSubmissionReceipt, CoordinatedSparseUploadError<C, E, S>> {
        self.validate(state_generation, mapping_epoch, observe)
            .map_err(CoordinatedSparseUploadError::Invalid)?;
        let reservation = reservations.reserve_for(state_generation, slot, bytes)
            .map_err(CoordinatedSparseUploadError::Reserve)?;
        let range = reservation.range();
        let copied = copy(range, &self.pages).map_err(CoordinatedSparseUploadError::Copy)?;
        if copied.len() != bytes { return Err(CoordinatedSparseUploadError::StagingSize) }
        let staging = OwnedUploadStaging { range, bytes: copied.into_boxed_slice() };
        prepare(&staging, &self.pages).map_err(CoordinatedSparseUploadError::Prepare)?;

        let batch = self.lease.take().expect("finished coordinated sparse upload plan").accept()
            .map_err(CoordinatedSparseUploadError::Accept)?;
        let accepted = AcceptedCoordinatedUpload::new(reservation, batch);
        execute(&staging, &self.pages);
        let fence = signal().map_err(CoordinatedSparseUploadError::Signal)?;
        match accepted.publish(fence) {
            Ok(receipt) => Ok(receipt),
            Err(CoordinatedPublishError::Reservation(error)) => {
                Err(CoordinatedSparseUploadError::PublishReservation(error))
            }
            Err(CoordinatedPublishError::Batch(error)) => Err(CoordinatedSparseUploadError::PublishBatch(error)),
        }
    }
}

enum LegacyParityUpload {
    Lease { lease: SparseBatchLease, state_generation: u64, mapping_epoch: u64 },
    Planned(CoordinatedSparseUploadPlan),
}

impl LegacyParityUpload {
    fn revalidate(
        &self,
        state_generation: u64,
        mapping_epoch: u64,
        observe: impl FnMut(usize) -> Option<UploadPageObservation>,
    ) -> LegacyParityRevalidation {
        let plan = match self {
            Self::Lease { state_generation: planned_generation, mapping_epoch: planned_epoch, .. } => {
                return if state_generation != *planned_generation {
                    LegacyParityRevalidation::RestartForGeneration
                } else if mapping_epoch != *planned_epoch {
                    LegacyParityRevalidation::RetryCurrentGeneration
                } else {
                    LegacyParityRevalidation::Ready
                };
            }
            Self::Planned(plan) => plan,
        };
        match plan.validate(state_generation, mapping_epoch, observe) {
            Ok(()) => LegacyParityRevalidation::Ready,
            Err(UploadPlanInvalid::StateGeneration) => LegacyParityRevalidation::RestartForGeneration,
            Err(_) => LegacyParityRevalidation::RetryCurrentGeneration,
        }
    }

    fn obsolete(self) -> Result<(), SparseBatchError> {
        match self {
            Self::Lease { lease, .. } => lease.obsolete(),
            Self::Planned(plan) => plan.obsolete(),
        }
    }

    fn retry(self) -> Result<(), SparseBatchError> {
        match self {
            Self::Lease { lease, .. } => lease.retry(),
            Self::Planned(plan) => plan.retry(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct UploadReservationToken {
    state_generation: u64,
    serial: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct UploadRingRange {
    start: u64,
    end: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct UploadSubmissionReceipt {
    token: UploadReservationToken,
    fence: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum UploadReservationState {
    Reserved,
    Submitted { fence: u64, retired: bool },
    Cancelled,
    Poisoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum UploadReservationError {
    Poisoned,
    StaleGeneration,
    Slot,
    SlotBusy,
    Empty,
    TooLarge,
    Full,
    Exhausted,
    Missing,
    State,
    Fence,
}

struct UploadReservationRecord {
    slot: usize,
    range: UploadRingRange,
    state: UploadReservationState,
}

/// Inert ownership model for a later short upload commit phase. Ring positions only advance;
/// cancellation leaves a retired interval in the ordered ledger instead of rewinding across a
/// newer reservation. Submitted intervals become reusable only after their exact fence retires and
/// every older interval has retired. A failed ordering signal poisons the whole generation.
#[allow(dead_code)]
struct UploadReservationLedger {
    state_generation: u64,
    capacity: u64,
    alignment: u64,
    next_serial: u64,
    cursor: u64,
    reclaimed: u64,
    poisoned: bool,
    slots: Vec<Option<u64>>,
    order: VecDeque<u64>,
    records: BTreeMap<u64, UploadReservationRecord>,
}

#[allow(dead_code)]
impl UploadReservationLedger {
    fn new(state_generation: u64, slots: usize, capacity: usize, alignment: usize) -> Self {
        assert!(slots > 0);
        assert!(capacity > 0);
        assert!(alignment > 0 && alignment.is_power_of_two() && capacity % alignment == 0);
        Self {
            state_generation,
            capacity: capacity as u64,
            alignment: alignment as u64,
            next_serial: 0,
            cursor: 0,
            reclaimed: 0,
            poisoned: false,
            slots: vec![None; slots],
            order: VecDeque::new(),
            records: BTreeMap::new(),
        }
    }

    fn reserve(
        &mut self, state_generation: u64, slot: usize, bytes: usize,
    ) -> Result<(UploadReservationToken, UploadRingRange), UploadReservationError> {
        if state_generation != self.state_generation {
            return Err(UploadReservationError::StaleGeneration);
        }
        if self.poisoned {
            return Err(UploadReservationError::Poisoned);
        }
        let owner = self.slots.get(slot).ok_or(UploadReservationError::Slot)?;
        if owner.is_some() {
            return Err(UploadReservationError::SlotBusy);
        }
        if bytes == 0 {
            return Err(UploadReservationError::Empty);
        }
        if bytes as u64 > self.capacity {
            return Err(UploadReservationError::TooLarge);
        }
        let (cursor, reclaimed) = if self.order.is_empty() {
            let offset = self.cursor % self.capacity;
            let boundary = if offset == 0 {
                self.cursor
            } else {
                self.cursor.checked_add(self.capacity - offset).ok_or(UploadReservationError::Exhausted)?
            };
            (boundary, boundary)
        } else {
            (self.cursor, self.reclaimed)
        };
        let mask = self.alignment - 1;
        let mut start = cursor.checked_add(mask).ok_or(UploadReservationError::Exhausted)? & !mask;
        let offset = start % self.capacity;
        if offset.checked_add(bytes as u64).ok_or(UploadReservationError::Exhausted)? > self.capacity {
            start = start.checked_add(self.capacity - offset).ok_or(UploadReservationError::Exhausted)?;
        }
        let end = start.checked_add(bytes as u64).ok_or(UploadReservationError::Exhausted)?;
        if end.checked_sub(reclaimed).ok_or(UploadReservationError::Exhausted)? > self.capacity {
            return Err(UploadReservationError::Full);
        }
        let serial = self.next_serial.checked_add(1).ok_or(UploadReservationError::Exhausted)?;
        self.next_serial = serial;
        self.cursor = end;
        self.reclaimed = reclaimed;
        self.slots[slot] = Some(serial);
        let range = UploadRingRange { start, end };
        self.order.push_back(serial);
        self.records.insert(serial, UploadReservationRecord { slot, range, state: UploadReservationState::Reserved });
        Ok((UploadReservationToken { state_generation, serial }, range))
    }

    fn submit(&mut self, token: UploadReservationToken, fence: u64) -> Result<(), UploadReservationError> {
        let record = self.record_mut(token)?;
        if record.state != UploadReservationState::Reserved || fence == 0 {
            return Err(UploadReservationError::State);
        }
        record.state = UploadReservationState::Submitted { fence, retired: false };
        Ok(())
    }

    fn complete(&mut self, receipt: UploadSubmissionReceipt) -> Result<u64, UploadReservationError> {
        let record = self.record_mut(receipt.token)?;
        match record.state {
            UploadReservationState::Submitted { fence: expected, retired: false } if expected == receipt.fence => {
                record.state = UploadReservationState::Submitted { fence: receipt.fence, retired: true };
                let slot = record.slot;
                self.slots[slot] = None;
                Ok(self.reclaim())
            }
            UploadReservationState::Submitted { fence: expected, .. } if expected != receipt.fence => {
                Err(UploadReservationError::Fence)
            }
            _ => Err(UploadReservationError::State),
        }
    }

    fn cancel(&mut self, token: UploadReservationToken) -> Result<u64, UploadReservationError> {
        let record = self.record_mut(token)?;
        if record.state != UploadReservationState::Reserved {
            return Err(UploadReservationError::State);
        }
        record.state = UploadReservationState::Cancelled;
        let slot = record.slot;
        self.slots[slot] = None;
        Ok(self.reclaim())
    }

    fn poison(&mut self, token: UploadReservationToken) -> Result<(), UploadReservationError> {
        let record = self.record_mut(token)?;
        if record.state != UploadReservationState::Reserved {
            return Err(UploadReservationError::State);
        }
        record.state = UploadReservationState::Poisoned;
        self.poisoned = true;
        Ok(())
    }

    /// Fail closed even after submission publication: no slot or ring byte in this generation may
    /// be reused when the matching batch receipt could not be published.
    fn poison_generation(&mut self, token: UploadReservationToken) {
        if token.state_generation != self.state_generation { return }
        if let Some(record) = self.records.get_mut(&token.serial) {
            if matches!(record.state, UploadReservationState::Reserved | UploadReservationState::Submitted { .. }) {
                record.state = UploadReservationState::Poisoned;
            }
        }
        self.poisoned = true;
    }

    fn record_mut(
        &mut self, token: UploadReservationToken,
    ) -> Result<&mut UploadReservationRecord, UploadReservationError> {
        if token.state_generation != self.state_generation {
            return Err(UploadReservationError::StaleGeneration);
        }
        self.records.get_mut(&token.serial).ok_or(UploadReservationError::Missing)
    }

    fn reclaim(&mut self) -> u64 {
        while let Some(&serial) = self.order.front() {
            let retired = self.records.get(&serial).is_some_and(|record| match record.state {
                UploadReservationState::Cancelled => true,
                UploadReservationState::Submitted { retired, .. } => retired,
                UploadReservationState::Reserved | UploadReservationState::Poisoned => false,
            });
            if !retired {
                break;
            }
            let record = self.records.remove(&serial).unwrap();
            self.reclaimed = record.range.end;
            self.order.pop_front();
        }
        self.reclaimed
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct UploadReservations(Arc<Mutex<UploadReservationLedger>>);

#[allow(dead_code)]
impl UploadReservations {
    fn new(state_generation: u64, slots: usize, capacity: usize, alignment: usize) -> Self {
        Self(Arc::new(Mutex::new(UploadReservationLedger::new(state_generation, slots, capacity, alignment))))
    }

    fn reserve(&self, slot: usize, bytes: usize) -> Result<UploadReservation, UploadReservationError> {
        let mut ledger = self.0.lock().unwrap();
        let generation = ledger.state_generation;
        let (token, range) = ledger.reserve(generation, slot, bytes)?;
        Ok(UploadReservation { ledger: self.0.clone(), token: Some(token), range })
    }

    fn reserve_for(
        &self, state_generation: u64, slot: usize, bytes: usize,
    ) -> Result<UploadReservation, UploadReservationError> {
        let (token, range) = self.0.lock().unwrap().reserve(state_generation, slot, bytes)?;
        Ok(UploadReservation { ledger: self.0.clone(), token: Some(token), range })
    }

    fn complete(&self, receipt: UploadSubmissionReceipt) -> Result<u64, UploadReservationError> {
        self.0.lock().unwrap().complete(receipt)
    }
}

#[allow(dead_code)]
struct UploadReservation {
    ledger: Arc<Mutex<UploadReservationLedger>>,
    token: Option<UploadReservationToken>,
    range: UploadRingRange,
}

#[allow(dead_code)]
impl UploadReservation {
    fn range(&self) -> UploadRingRange { self.range }

    fn submit(mut self, fence: u64) -> Result<UploadSubmissionReceipt, UploadReservationError> {
        let token = self.token.unwrap();
        self.ledger.lock().unwrap().submit(token, fence)?;
        self.token.take();
        Ok(UploadSubmissionReceipt { token, fence })
    }

    fn poison(mut self) -> Result<(), UploadReservationError> {
        let token = self.token.unwrap();
        self.ledger.lock().unwrap().poison(token)?;
        self.token.take();
        Ok(())
    }

    /// Publish after execute acceptance. Publication failure is fail-closed: the exact record and
    /// its entire generation are poisoned instead of being cancelled by this guard's destructor.
    fn publish_after_accept(mut self, fence: u64) -> Result<UploadSubmissionReceipt, UploadReservationError> {
        let token = self.token.take().expect("finished upload reservation");
        let mut ledger = self.ledger.lock().unwrap();
        if fence == 0 {
            let _ = ledger.poison(token);
            return Err(UploadReservationError::State);
        }
        match ledger.submit(token, fence) {
            Ok(()) => Ok(UploadSubmissionReceipt { token, fence }),
            Err(error) => {
                let _ = ledger.poison(token);
                Err(error)
            }
        }
    }
}

impl Drop for UploadReservation {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            let _ = self.ledger.lock().unwrap().cancel(token);
        }
    }
}

enum CoordinatedPublishError {
    Reservation(UploadReservationError),
    Batch(SparseBatchError),
}

/// Own both accepted ledgers. Any exit before both receipts are published poisons both generation
/// domains, including the narrow boundary after reservation submission but before frontier receipt.
struct AcceptedCoordinatedUpload {
    reservation: Option<UploadReservation>,
    reservation_token: UploadReservationToken,
    reservation_ledger: Arc<Mutex<UploadReservationLedger>>,
    batch: Option<AcceptedSparseBatch>,
    batch_token: ClaimFrontierToken,
    batch_ledger: Arc<Mutex<SparseBatchLedger>>,
    armed: bool,
}

impl AcceptedCoordinatedUpload {
    fn new(reservation: UploadReservation, batch: AcceptedSparseBatch) -> Self {
        Self {
            reservation_token: reservation.token.unwrap(),
            reservation_ledger: reservation.ledger.clone(),
            batch_token: batch.token.unwrap().0,
            batch_ledger: batch.ledger.clone(),
            reservation: Some(reservation),
            batch: Some(batch),
            armed: true,
        }
    }

    fn publish(mut self, fence: u64) -> Result<UploadSubmissionReceipt, CoordinatedPublishError> {
        let receipt = self.reservation.take().unwrap().publish_after_accept(fence)
            .map_err(CoordinatedPublishError::Reservation)?;
        self.batch.take().unwrap().receipt(fence).map_err(CoordinatedPublishError::Batch)?;
        self.armed = false;
        Ok(receipt)
    }
}

impl Drop for AcceptedCoordinatedUpload {
    fn drop(&mut self) {
        if !self.armed { return }
        if let Some(reservation) = self.reservation.take() { let _ = reservation.poison(); }
        if let Some(batch) = self.batch.take() { drop(batch); }
        self.reservation_ledger.lock().unwrap().poison_generation(self.reservation_token);
        let mut batches = self.batch_ledger.lock().unwrap();
        if batches.work.state_generation == self.batch_token.state_generation { batches.poisoned = true }
    }
}

/// Coordination ownership attached to one live sparse state generation. Keeping the three
/// ledgers together prevents a later activation from pairing work, waits, or ring reservations
/// from different device/fence generations.
#[allow(dead_code)]
#[derive(Clone)]
struct SparseCoordination {
    batches: SparseBatches,
    waits: ForegroundWaits,
    reservations: UploadReservations,
}

impl SparseCoordination {
    fn new(state_generation: u64, fence_allocation: u64) -> Option<Self> {
        Some(Self {
            batches: SparseBatches::with_fence(state_generation, fence_allocation)?,
            waits: ForegroundWaits::new(state_generation, fence_allocation)?,
            reservations: UploadReservations::new(state_generation, UPLOAD_SLOTS, STAGING, RING_ALIGN),
        })
    }

    fn seal_legacy(
        &self,
        owner: u64,
        zero: &[(usize, usize)],
        retry: &[usize],
        dirty: &[(usize, usize)],
    ) -> Result<Option<SparseBatchLease>, SparseBatchError> {
        self.batches.seal_legacy(owner, zero, retry, dirty)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PreparationOutcome {
    Ready,
    RetryRequired,
    ExecutedUntracked,
}

impl From<SubmitResult> for PreparationOutcome {
    fn from(result: SubmitResult) -> Self {
        match result {
            SubmitResult::Submitted => Self::Ready,
            SubmitResult::NotSubmitted => Self::RetryRequired,
            SubmitResult::Unordered => Self::ExecutedUntracked,
        }
    }
}

struct RetiredTiles {
    tiles: Vec<(u32, u32)>,
    /// Fence marks on `self.queue` and every known title queue (`None` = signal failed, keep held).
    marks: Option<Vec<(ID3D12Fence, u64)>>,
}

struct State {
    generation: u64,
    mapping_epoch: u64,
    /// Exact-generation ownership for the staged prepare/accept/publish sparse path. The legacy
    /// submission path remains authoritative until activation PRs journal and consume its work.
    #[allow(dead_code)]
    coordination: SparseCoordination,
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    /// reusable auto-reset Win32 event for fence waits (no per-wait CreateEventW/CloseHandle)
    event: HANDLE,
    /// fence signalled on title queues so `queue` waits before copying GPU writes back to CPU
    rb_fence: ID3D12Fence,
    rb_value: u64,
    /// (window start, reserved buffer) of each sparse window
    windows: Vec<(usize, ID3D12Resource)>,
    /// video-memory pool heaps (None: given back, its index free for the next one)
    pools: Vec<Option<ID3D12Heap>>,
    free: Vec<(u32, u32)>,
    /// pools all of whose tiles came free: released once the GPU is past `value` and a moment
    /// went by (titles' queues may still read what was mapped there)
    retired: Vec<(ID3D12Heap, u64, std::time::Instant)>,
    /// tile address → pool tile + uncommitted page hole mask + per-page sent chunk hashes
    mapped: FxHashMap<usize, TileEntry>,
    /// pool tiles also used by placed buffers (not freed while they are)
    pinned: FxHashMap<(u32, u32), u32>,
    /// tile address → pool tile of tiles decommitted since the last title submit, still mapped
    /// in their window: unmapped together (`flush_parked`) before the next submit, when the free
    /// list runs out, or past `PARKED_MAX`; a commit there before that takes its tile back with
    /// no new mapping. (A retail title moving physical pages one at a time while it loads
    /// decommits and commits again thousands of tiles a second, and each `UpdateTileMappings`
    /// took ~0.15 ms.)
    parked: FxHashMap<usize, (u32, u32)>,
    /// unmapped parked pool tiles awaiting `self.queue` and in-flight title queues before entering `free`
    quarantined: VecDeque<RetiredTiles>,
    /// per-title-queue fences used to mark in-flight GPU work; raw keys are removed only by
    /// exact-generation queue retirement before that queue's address can be reused
    queue_fences: FxHashMap<usize, (ID3D12Fence, u64)>,
    /// placed buffers → the pool tiles they use
    buffers: FxHashMap<usize, Vec<(u32, u32)>>,
    /// reused tiles / decommitted partial pages to clear: (address, bytes)
    to_zero: Vec<(usize, usize)>,
    /// ring-buffered upload staging heap (`STAGING` bytes, persistently mapped)
    staging: Option<(ID3D12Resource, *mut u8)>,
    /// monotonic ring-buffer write/submitted/retired byte cursors
    ring_write: u64,
    ring_submitted: u64,
    ring_free: u64,
    /// submitted ring chunks awaiting GPU completion: `(fence_val, ring_end_pos)`
    inflight: VecDeque<(u64, u64)>,
    /// false after ExecuteCommandLists succeeded but its ordering Signal failed: no upload
    /// allocator or staging byte may be reused because completion can no longer be proved.
    uploads_ordered: bool,
    /// dedicated 1 MiB pre-zeroed buffer for tile/hole clears (zero staging bandwidth)
    zero_buf: Option<ID3D12Resource>,
    /// ring of async upload command allocators + lists
    slots: Vec<UploadSlot>,
    next_slot: usize,
    /// dedicated readback staging buffer + command allocator / list
    readback: Option<(ID3D12Resource, *mut u8)>,
    rb_alloc: ID3D12CommandAllocator,
    rb_list: ID3D12GraphicsCommandList,
    rb_last_fence: u64,
    /// the fence value each title queue last waited for
    waited: FxHashMap<usize, u64>,
    /// `mapped` as sorted (start, end) runs (None: to be made again)
    runs: Option<Vec<(usize, usize)>>,
    /// reusable buffer for dirty runs in `before_execute` (avoids per-submit heap alloc)
    scratch_runs: Vec<(usize, usize)>,
    /// Dirty CPU pages whose write-watch bits were reset before an upload failed.
    retry_pages: Vec<usize>,
}

impl Drop for State {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

const K0: u64 = 0x9E37_79B9_7F4A_7C15;
const K1: u64 = 0xBF58_476D_1CE4_E5B9;
const K2: u64 = 0x94D0_49BB_1331_11EB;
const K3: u64 = 0x243F_6A88_85A3_08D3;

/// 4-way parallel hash over 8 `u64` words (64 bytes): breaks serial multiply latency chain.
#[inline(always)]
const fn hash_words(w: [u64; 8]) -> u64 {
    let a = (w[0] ^ K3).wrapping_mul(K0) ^ w[4].wrapping_mul(K1);
    let b = (w[1] ^ K0).wrapping_mul(K1) ^ w[5].wrapping_mul(K2);
    let c = (w[2] ^ K1).wrapping_mul(K2) ^ w[6].wrapping_mul(K3);
    let d = (w[3] ^ K2).wrapping_mul(K3) ^ w[7].wrapping_mul(K0);
    let h = a.rotate_left(27).wrapping_add(b.rotate_left(13)) ^ c.rotate_left(41).wrapping_add(d.rotate_left(53));
    h.wrapping_mul(K0) ^ (h >> 32)
}

const ZERO_HASH: u64 = hash_words([0; 8]);
const ZERO_HASHES: [u64; CHUNKS] = [ZERO_HASH; CHUNKS];

#[inline(always)]
fn read_chunk_words(c: &[u8]) -> [u64; 8] {
    debug_assert_eq!(c.len(), CHUNK);
    let p = c.as_ptr() as *const [u8; 8];
    unsafe {
        [
            u64::from_le_bytes(p.add(0).read_unaligned()),
            u64::from_le_bytes(p.add(1).read_unaligned()),
            u64::from_le_bytes(p.add(2).read_unaligned()),
            u64::from_le_bytes(p.add(3).read_unaligned()),
            u64::from_le_bytes(p.add(4).read_unaligned()),
            u64::from_le_bytes(p.add(5).read_unaligned()),
            u64::from_le_bytes(p.add(6).read_unaligned()),
            u64::from_le_bytes(p.add(7).read_unaligned()),
        ]
    }
}

/// hash of 64 bytes
#[inline(always)]
fn chunk_hash(c: &[u8]) -> u64 {
    hash_words(read_chunk_words(c))
}

#[inline(always)]
fn push_run(out: &mut Vec<(usize, usize)>, addr: usize, len: usize) {
    match out.last_mut() {
        Some(last) if last.0 + last.1 == addr => last.1 += len,
        _ => out.push((addr, len)),
    }
}

/// Land GPU bytes in one CPU page, 64-byte chunk by chunk. A chunk still matching `sent` has
/// no newer CPU write and may be replaced; a changed chunk keeps its CPU bytes and its old hash,
/// so the next write-watch look uploads it. With `gpu_written`, `src` is bytes the GPU's own
/// commands wrote: they land in a changed chunk too (the CPU wrote other bytes of it, such as the
/// cells a wait before the write watched), and its old hash still makes the next look upload the
/// whole chunk, the GPU's bytes with the CPU's. The hash check and copy are not atomic against a
/// concurrent CPU store: callers rely on the title's fence discipline to exclude stores to the
/// same bytes while its synchronous copy-back lands.
fn land_page(slot: &mut Option<Box<[u64; CHUNKS]>>, page: &mut [u8], at: usize, src: &[u8], gpu_written: bool) {
    debug_assert_eq!(page.len(), PAGE);
    debug_assert!(at + src.len() <= page.len());
    let sent = slot.get_or_insert_with(|| Box::new(ZERO_HASHES));
    let end = at + src.len();
    let mut offset = at;
    while offset < end {
        let chunk = offset / CHUNK;
        let chunk_start = chunk * CHUNK;
        let copy_end = ((chunk + 1) * CHUNK).min(end);
        let source = offset - at;
        if chunk_hash(&page[chunk_start..chunk_start + CHUNK]) == sent[chunk] {
            page[offset..copy_end].copy_from_slice(&src[source..source + copy_end - offset]);
            sent[chunk] = chunk_hash(&page[chunk_start..chunk_start + CHUNK]);
        } else if gpu_written {
            page[offset..copy_end].copy_from_slice(&src[source..source + copy_end - offset]);
        }
        offset = copy_end;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HashUpdate {
    at: usize,
    hash: u64,
}

const MAX_BATCH_HASH_UPDATES: usize = STAGING / CHUNK;

/// One synchronously-recorded dirty upload batch. The slot, ring rollback point, and unpublished
/// hash plan move together so a failed submission cannot publish bytes that have no fence receipt.
struct PreparedDirtyBatch {
    slot: usize,
    ring_start: u64,
    updates: Vec<HashUpdate>,
}

impl PreparedDirtyBatch {
    fn new(slot: usize, ring_start: u64) -> Self {
        Self { slot, ring_start, updates: Vec::new() }
    }

    fn reopen(mut self, slot: usize, ring_start: u64) -> Self {
        debug_assert!(self.updates.is_empty());
        self.slot = slot;
        self.ring_start = ring_start;
        self
    }

    fn hash_plan_would_overflow(&self, bytes: usize) -> bool {
        batch_hash_plan_would_overflow(self.updates.len(), bytes)
    }

    fn append_hashes(&mut self, at: usize, bytes: &[u8]) {
        append_hash_updates(at, bytes, &mut self.updates);
    }

    fn clear_after_submit(&mut self) {
        self.updates.clear();
    }
}

fn batch_hash_plan_would_overflow(current: usize, bytes: usize) -> bool {
    bytes / CHUNK > MAX_BATCH_HASH_UPDATES.saturating_sub(current)
}

/// Plan changed 64-byte chunk runs in a 4 KiB page without publishing their hashes. The hashes
/// become `sent` only after the corresponding upload has a durable queue-fence receipt.
/// This keeps the fresh-zero-page fast path allocation-free.
#[inline]
fn scan_page(
    page: &[u8],
    pg: usize,
    slot: &Option<Box<[u64; CHUNKS]>>,
    out: &mut Vec<(usize, usize)>,
) {
    if let Some(sent) = slot.as_deref() {
        for (k, c) in page.chunks_exact(CHUNK).enumerate() {
            let h = chunk_hash(c);
            if h == sent[k] {
                continue;
            }
            push_run(out, pg + k * CHUNK, CHUNK);
        }
        return;
    }
    // Page has never had non-zero data sent: skip zero chunks with a cheap OR-reduction and
    // allocate the hash array only if at least one chunk is non-zero.
    let mut chunks = page.chunks_exact(CHUNK).enumerate();
    while let Some((k, c)) = chunks.next() {
        let w = read_chunk_words(c);
        if (w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0 {
            continue;
        }
        let h = hash_words(w);
        if h == ZERO_HASH {
            continue;
        }
        push_run(out, pg + k * CHUNK, CHUNK);
        for (k2, c2) in chunks {
            let h2 = chunk_hash(c2);
            if h2 == ZERO_HASH {
                continue;
            }
            push_run(out, pg + k2 * CHUNK, CHUNK);
        }
        break;
    }
}

fn append_hash_updates(at: usize, bytes: &[u8], updates: &mut Vec<HashUpdate>) {
    debug_assert_eq!(at % CHUNK, 0);
    debug_assert_eq!(bytes.len() % CHUNK, 0);
    debug_assert!(updates.len() + bytes.len() / CHUNK <= MAX_BATCH_HASH_UPDATES);
    updates.extend(bytes.chunks_exact(CHUNK).enumerate().map(|(i, chunk)| HashUpdate {
        at: at + i * CHUNK,
        hash: chunk_hash(chunk),
    }));
}

fn publish_hash_updates(mapped: &mut FxHashMap<usize, TileEntry>, updates: &[HashUpdate]) {
    for update in updates {
        let tile = update.at & !(TILE - 1);
        let Some(entry) = mapped.get_mut(&tile) else { continue };
        let page = update.at & !(PAGE - 1);
        let index = (page - tile) / PAGE;
        let chunk = (update.at - page) / CHUNK;
        if entry.holes & (1u16 << index) != 0 {
            continue;
        }
        if update.hash == ZERO_HASH && entry.sent[index].is_none() {
            continue;
        }
        entry.sent[index].get_or_insert_with(|| Box::new(ZERO_HASHES))[chunk] = update.hash;
    }
}

/// The byte ranges (offset, length) of `page` whose 64-byte chunks differ from `sent`
/// (updated to the page's chunks).
pub fn changed_chunks(page: &[u8], sent: &mut [u64; CHUNKS]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (k, c) in page.chunks_exact(CHUNK).enumerate() {
        let h = chunk_hash(c);
        if h == sent[k] {
            continue;
        }
        sent[k] = h;
        push_run(&mut out, k * CHUNK, CHUNK);
    }
    out
}

/// Plan a contiguous sub-allocation `(aligned_pos, chunk_len)` inside a ring buffer of `cap`
/// bytes starting at monotonic cursor `ring_write`, keeping each chunk `<= cap / 2` and
/// avoiding mid-copy wrap-around without wasting large tail gaps.
#[inline]
fn plan_ring_chunk(ring_write: u64, want: usize, cap: usize) -> (u64, usize) {
    let mut chunk = want.min(cap / 2);
    let mut aligned = (ring_write + (RING_ALIGN as u64 - 1)) & !(RING_ALIGN as u64 - 1);
    let off = (aligned as usize) % cap;
    if off + chunk > cap {
        let tail = cap - off;
        if tail >= PAGE {
            chunk = tail;
        } else {
            aligned += tail as u64;
        }
    }
    (aligned, chunk)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedDirtyCopy {
    ring_start: u64,
    staging_offset: usize,
    window_offset: usize,
    bytes: usize,
}

/// Bound one dirty copy to both its sparse window and its contiguous staging-ring allocation.
/// Keeping this arithmetic pure lets planning and recording use exactly the same split points.
fn prepare_dirty_copy(
    at: usize,
    remaining: usize,
    window_base: usize,
    window_size: usize,
    ring_write: u64,
    staging_cap: usize,
) -> PreparedDirtyCopy {
    debug_assert!(at >= window_base && at < window_base + window_size);
    let to_window_end = window_base + window_size - at;
    let (ring_start, bytes) = plan_ring_chunk(ring_write, remaining.min(to_window_end), staging_cap);
    PreparedDirtyCopy {
        ring_start,
        staging_offset: ring_start as usize % staging_cap,
        window_offset: at - window_base,
        bytes,
    }
}

// D3D12 objects are free-threaded; staging pointers and event handle are only used under STATE lock
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static NEXT_STATE_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Serialize mutation of upload slots, staging cursors and retry journals. Fence waits release this
/// gate while a shared generation lifetime keeps their exact native backing. Lock order is
/// `WAIT_GATES` → `UPLOAD_STATE_LIFETIME` (shared) → `UPLOAD_GATE` → `STATE`;
/// initialization takes `PLACEMENT` then the lifetime lock exclusively before `UPLOAD_GATE`.
static UPLOAD_GATE: Mutex<()> = Mutex::new(());

/// A waiter keeps its exact `State` generation and all of that generation's native backing alive
/// while `UPLOAD_GATE` is released. Other uploaders take shared ownership and can cross the wait;
/// setup takes exclusive ownership before `UPLOAD_GATE` and cannot replace `STATE` meanwhile.
static UPLOAD_STATE_LIFETIME: std::sync::RwLock<()> = std::sync::RwLock::new(());

type UploadGuard = std::sync::MutexGuard<'static, ()>;

#[derive(Clone, Copy, Default)]
struct UploadGateTimes {
    wait_us: u64,
    hold_us: u64,
    released_us: u64,
}

struct UploadGateTiming {
    enabled: bool,
    held_since: Option<std::time::Instant>,
    times: UploadGateTimes,
}

impl UploadGateTiming {
    fn acquire(enabled: bool) -> (Option<UploadGuard>, Self) {
        let waiting = enabled.then(std::time::Instant::now);
        let guard = UPLOAD_GATE.lock().unwrap();
        let acquired = enabled.then(std::time::Instant::now);
        let wait_us = waiting.zip(acquired).map_or(0, |(waiting, acquired)| acquired.duration_since(waiting).as_micros() as u64);
        (Some(guard), Self { enabled, held_since: acquired, times: UploadGateTimes { wait_us, ..Default::default() } })
    }

    fn with_released<T>(&mut self, guard: &mut Option<UploadGuard>, wait: impl FnOnce() -> T) -> T {
        let released = self.enabled.then(std::time::Instant::now);
        if let (Some(held_since), Some(released)) = (self.held_since.take(), released) {
            self.times.hold_us = self.times.hold_us.wrapping_add(released.duration_since(held_since).as_micros() as u64);
        }
        drop(guard.take());
        let result = wait();
        *guard = Some(UPLOAD_GATE.lock().unwrap());
        let acquired = self.enabled.then(std::time::Instant::now);
        if let (Some(released), Some(acquired)) = (released, acquired) {
            self.times.released_us = self.times.released_us.wrapping_add(acquired.duration_since(released).as_micros() as u64);
        }
        self.held_since = acquired;
        result
    }

    fn finish(mut self) -> UploadGateTimes {
        if let Some(held_since) = self.held_since.take() {
            self.times.hold_us = self.times.hold_us.wrapping_add(held_since.elapsed().as_micros() as u64);
        }
        self.times
    }
}

/// A placement may release `STATE` while the driver creates its reserved resource. Keep sparse
/// state replacement and another placement from crossing that exact-state transaction.
static PLACEMENT: Mutex<()> = Mutex::new(());

/// Serializes sparse preparation through wait enqueue for one title queue. Without this, a
/// concurrent submission could observe a reserved `waited` value and execute before the thread
/// which reserved it has called the native queue `Wait`.
static WAIT_GATES: Mutex<FxHashMap<usize, Arc<Mutex<()>>>> = Mutex::new(FxHashMap::with_hasher(BuildHasherDefault::new()));

/// The state's queue again, outside `STATE` (`queue`). `STATE` is held around D3D12 calls (an
/// upload's allocator `Reset` in `begin_slot`), and `late_release` asks for the queue from a
/// resource's last `Release`, which D3D12 can make inside a call of its own on another thread
/// (a list's `Reset`, with `XCL_DRED=1` on every start): the two then waited for each other.
static QUEUE: Mutex<Option<ID3D12CommandQueue>> = Mutex::new(None);

fn remove_queue_fence<F>(fences: &mut FxHashMap<usize, (F, u64)>, raw: usize) -> Option<(F, u64)> {
    fences.remove(&raw)
}

fn initial_quarantine_marks<F: Clone>(own: (F, u64), retiring: &[(F, u64)], blocked: bool) -> Option<Vec<(F, u64)>> {
    (!blocked).then(|| std::iter::once(own).chain(retiring.iter().cloned()).collect())
}

/// Remove raw queue-address state before a retired queue's final owner is released. Retired tile
/// batches own fence clones independently; the cached fence drops after unlocking.
pub(super) fn forget_queue(id: super::present::QueueId) {
    let gone = {
        let mut state = STATE.lock().unwrap();
        state.as_mut().and_then(|state| {
            state.waited.remove(&id.raw());
            remove_queue_fence(&mut state.queue_fences, id.raw())
        })
    };
    let gate = WAIT_GATES.lock().unwrap().remove(&id.raw());
    drop(gone);
    drop(gate);
}

/// (bytes of video memory in pool heaps, of those mapped) – `XCL_ACTIVITY`
pub fn pool_bytes() -> (usize, usize) {
    let g = lock_state();
    let Some(s) = g.as_ref() else { return (0, 0) };
    let pools = s.pools.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize * TILE;
    (pools, pools.saturating_sub(s.free.len() * TILE))
}

/// Active pool bytes, occupied tile bytes, retired pool bytes, and committed CPU page bytes for
/// `device`, from one state lock so a pool moving to the retired list cannot be counted twice.
pub(super) fn memory_bytes_for(device: usize) -> (usize, usize, usize, usize) {
    let g = lock_state();
    let Some(s) = g.as_ref() else { return (0, 0, 0, 0) };
    memory_byte_counts(
        device,
        s.dev.as_raw() as usize,
        s.pools.iter().filter(|p| p.is_some()).count(),
        s.free.len(),
        s.retired.len(),
        s.mapped.values().map(|e| e.holes),
    )
}

fn memory_byte_counts(
    device: usize,
    owner: usize,
    active_pools: usize,
    free_tiles: usize,
    retired_pools: usize,
    mapped_holes: impl Iterator<Item = u16>,
) -> (usize, usize, usize, usize) {
    if device != owner {
        return (0, 0, 0, 0);
    }
    let pool_bytes = (POOL_TILES as usize).saturating_mul(TILE);
    let pools = active_pools.saturating_mul(pool_bytes);
    let occupied = pools.saturating_sub(free_tiles.saturating_mul(TILE));
    let retired = retired_pools.saturating_mul(pool_bytes);
    let committed = mapped_holes.fold(0usize, |bytes, holes| {
        let pages = PAGES_PER_TILE.saturating_sub(holes.count_ones() as usize);
        bytes.saturating_add(pages.saturating_mul(PAGE))
    });
    (pools, occupied, retired, committed)
}
/// commits before the device existed: the pages committed, as disjoint page ranges (start →
/// end) merged with their neighbours, so a commit or decommit costs O(log n) and not a scan of
/// every page committed so far. Like `committed` / `decommitted` with a state, a range counts
/// every page it touches; `init` commits them once the windows exist.
static EARLY: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
/// Before the sparse device exists: pages the title protected other than writable. Commits clear
/// these ranges, and `init` replays what remains after the early commits.
static EARLY_NOT_RW: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

fn early_protection(early: &mut BTreeMap<usize, usize>, a: usize, n: usize, writable: bool) {
    let (start, end) = page_span(a, n);
    if writable {
        early_remove(early, start, end);
    } else {
        early_add(early, start, end);
    }
}

/// The pages `[a, a + n)` touches, as `[start, end)`.
fn page_span(a: usize, n: usize) -> (usize, usize) {
    (a & !(PAGE - 1), a.saturating_add(n).saturating_add(PAGE - 1) & !(PAGE - 1))
}

/// Add the pages `[a, b)` to `early`, merged with the ranges they overlap or touch.
fn early_add(early: &mut BTreeMap<usize, usize>, mut a: usize, mut b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..=a).next_back() {
        if e >= a {
            a = s;
            b = b.max(e);
            early.remove(&s);
        }
    }
    loop {
        let Some((s, e)) = early.range(a..=b).next().map(|(&s, &e)| (s, e)) else { break };
        b = b.max(e);
        early.remove(&s);
    }
    early.insert(a, b);
}

/// Take the pages `[a, b)` out of `early`; a range that overlaps it keeps the rest.
fn early_remove(early: &mut BTreeMap<usize, usize>, a: usize, b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..a).next_back() {
        if e > a {
            early.insert(s, a);
            if e > b {
                early.insert(b, e);
                return;
            }
        }
    }
    loop {
        let Some((s, e)) = early.range(a..b).next().map(|(&s, &e)| (s, e)) else { break };
        early.remove(&s);
        if e > b {
            early.insert(b, e);
            break;
        }
    }
}
/// generation counter incremented whenever the set of mapped tiles changes (for watch_ranges cache)
static MAPPED_GEN: AtomicU64 = AtomicU64::new(0);

/// Generation counter of `mapped_ranges()` (changes only when tiles are mapped or unmapped).
pub fn mapped_gen() -> u64 {
    MAPPED_GEN.load(Ordering::Acquire)
}

fn buffer_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

/// The sparse windows' buffers on `dev`: `(window_index, buffer_raw, gpu_va)` for each
/// sparse window in `gpu::uma` (empty: no sparse windows).
pub unsafe fn init(dev: &ID3D12Device) -> Vec<(usize, usize, u64)> {
    let Some(u) = crate::gpu::uma::get() else { return Vec::new() };
    let mut out = Vec::new();
    let mut windows = Vec::new();
    for k in 0..u.shader_windows() {
        if !u.is_sparse_window(k) {
            continue;
        }
        let (base, size) = u.window(k);
        let mut r: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateReservedResource(&buffer_desc(size as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut r) {
            crate::xlog_warn!("sparse window {k}: CreateReservedResource: {e}");
            return Vec::new();
        }
        let Some(r) = r else { return Vec::new() };
        out.push((k, r.clone().into_raw() as usize, r.GetGPUVirtualAddress()));
        windows.push((base, r));
    }
    if out.is_empty() {
        return out;
    }
    let made = (|| -> windows::core::Result<State> {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)?;
        let mut slots = Vec::with_capacity(UPLOAD_SLOTS);
        for _ in 0..UPLOAD_SLOTS {
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
            list.Close()?;
            let _ = list.SetName(windows::core::w!("xcl sparse upload list"));
            let _ = alloc.SetName(windows::core::w!("xcl sparse upload allocator"));
            slots.push(UploadSlot { alloc, list, fence_val: 0 });
        }
        let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None)?;
        rb_list.Close()?;
        let _ = rb_list.SetName(windows::core::w!("xcl sparse readback list"));
        let _ = rb_alloc.SetName(windows::core::w!("xcl sparse readback allocator"));
        let _ = queue.SetName(windows::core::w!("xcl sparse queue"));
        let generation = NEXT_STATE_GENERATION.fetch_add(1, Ordering::Relaxed);
        let fence_allocation = fence.as_raw() as usize as u64;
        let coordination = SparseCoordination::new(generation, fence_allocation)
            .expect("valid sparse state generation and fence identity");
        Ok(State {
            generation,
            mapping_epoch: 0,
            coordination,
            dev: dev.clone(),
            queue,
            fence,
            value: 0,
            event,
            rb_fence,
            rb_value: 0,
            windows,
            pools: Vec::new(),
            free: Vec::new(),
            retired: Vec::new(),
            mapped: FxHashMap::default(),
            pinned: FxHashMap::default(),
            parked: FxHashMap::default(),
            quarantined: VecDeque::new(),
            queue_fences: FxHashMap::default(),
            buffers: FxHashMap::default(),
            to_zero: Vec::new(),
            staging: None,
            ring_write: 0,
            ring_submitted: 0,
            ring_free: 0,
            inflight: VecDeque::new(),
            uploads_ordered: true,
            zero_buf: None,
            slots,
            next_slot: 0,
            readback: None,
            rb_alloc,
            rb_list,
            rb_last_fence: 0,
            waited: FxHashMap::default(),
            runs: None,
            scratch_runs: Vec::new(),
            retry_pages: Vec::new(),
        })
    })();
    match made {
        Ok(s) => {
            crate::xlog_info!("D3D12.X: {} sparse window(s) for the title's own memory (tiled buffers on video memory)", out.len());
            // a state this replaces (the windows set up again) goes after the lock: the release of
            // its last buffers reaches `forget_buffer`, which takes the lock (#344)
            // (the queue first: `queue()` never answers with one older than the state's)
            let _placement = PLACEMENT.lock().unwrap();
            let _upload_lifetime = UPLOAD_STATE_LIFETIME.write().unwrap();
            let _uploads = UPLOAD_GATE.lock().unwrap();
            let old_queue = QUEUE.lock().unwrap().replace(s.queue.clone());
            let mut state = STATE.lock().unwrap();
            let old = state.replace(s);
            let s = state.as_mut().unwrap();
            for (a, b) in std::mem::take(&mut *EARLY.lock().unwrap()) {
                s.commit_range(a, b - a);
            }
            for (a, b) in std::mem::take(&mut *EARLY_NOT_RW.lock().unwrap()) {
                mark_protection(&mut s.mapped, a, b - a, false);
            }
            drop(state);
            if old.as_ref().is_some_and(|state| !state.uploads_ordered) {
                // ExecuteCommandLists was accepted but no fence can prove when its allocator,
                // staging source, or destination windows become unused. Keep that generation
                // alive until process exit rather than release memory the GPU may still touch.
                crate::xlog_warn!("sparse: retaining an unordered upload generation after its queue Signal failed");
                std::mem::forget(old);
            } else {
                drop(old);
            }
            drop(old_queue);
            out
        }
        Err(e) => {
            crate::xlog_warn!("sparse windows: {e}");
            Vec::new()
        }
    }
}

impl State {
    #[inline]
    fn window_index(&self, a: usize) -> Option<usize> {
        self.windows.iter().position(|(b, _)| a >= *b && a < *b + crate::gpu::uma::WINDOW)
    }

    /// Capture every identity used by unlocked sparse planning while this State is authoritative.
    /// The returned value owns hashes and mapping metadata; it retains no title-memory borrow.
    unsafe fn upload_page_observation(&self, page: usize) -> Option<UploadPageObservation> {
        let tile = page & !(TILE - 1);
        let entry = self.mapped.get(&tile)?;
        let index = (page - tile) / PAGE;
        if index >= PAGES_PER_TILE { return None }
        let hole = entry.holes & (1u16 << index) != 0;
        let sent = entry.sent[index].as_deref().copied();
        let source = if hole {
            ZERO_HASHES
        } else {
            let mut bytes = [0u8; PAGE];
            if !crate::hle::dstorage::read_own(page, bytes.as_mut_ptr(), bytes.len()) {
                return None;
            }
            std::array::from_fn(|chunk| {
                let start = chunk * CHUNK;
                chunk_hash(&bytes[start..start + CHUNK])
            })
        };
        Some(UploadPageObservation { page, pool: entry.pool, hole, sent, source })
    }

    unsafe fn take_tile(&mut self) -> Option<((u32, u32), bool)> {
        self.drain_quarantined();
        if self.free.is_empty() && self.flush_parked() {
            self.drain_quarantined();
        }
        if let Some(t) = self.free.pop() {
            return Some((t, true));
        }
        // a pool retired a moment ago (its last tile came free) is taken back before a new
        // heap is made: freeing and allocating a tile in turn made a new pool each time,
        // up to 18 GB of retired heaps in 2 s (#913). Its tiles are reused as freed tiles are
        // (zeroed), so this is no less safe than reusing a tile of a pool in use.
        if let Some((heap, _, _)) = self.retired.pop() {
            let h = self.add_pool(heap);
            return Some(((h, 0), true));
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_TILES as u64 * TILE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| self.dev.CreateHeap(&desc, &mut heap)) {
            crate::xlog_warn!("sparse windows: no video memory for more tiles ({e})");
            return None;
        }
        let heap = heap?;
        let h = self.add_pool(heap);
        Some(((h, 0), false))
    }

    /// `heap` becomes a pool (in a given-back pool's index if there is one); its tiles but the
    /// first, which the caller takes, go to the free list.
    fn add_pool(&mut self, heap: ID3D12Heap) -> u32 {
        let h = match self.pools.iter().position(|p| p.is_none()) {
            Some(i) => {
                self.pools[i] = Some(heap);
                i as u32
            }
            None => {
                self.pools.push(Some(heap));
                self.pools.len() as u32 - 1
            }
        };
        self.free.extend((1..POOL_TILES).rev().map(|t| (h, t)));
        h
    }

    /// Pools whose every tile is free again go (their video memory – and as much of the
    /// system's commit: a retail game's homes of physical pages left 0.75 GB of pools for
    /// 0.29 GB of tiles in use). Call after their tiles are unmapped and signalled.
    fn retire_free_pools_with(&mut self, done: u64) {
        let now = std::time::Instant::now();
        self.retired.retain(|(_, v, at)| !(done >= *v && now.duration_since(*at).as_secs() >= 2));
        if self.free.len() < POOL_TILES as usize {
            return;
        }
        let mut count = vec![0u32; self.pools.len()];
        for &(h, _) in &self.free {
            count[h as usize] += 1;
        }
        // (a flag a pool: a title's world load frees hundreds of pools' worth of tiles at once)
        let gone: Vec<bool> = (0..count.len()).map(|h| count[h] == POOL_TILES && self.pools[h].is_some()).collect();
        if !gone.contains(&true) {
            return;
        }
        self.free.retain(|&(h, _)| !gone[h as usize]);
        for h in (0..gone.len()).filter(|&h| gone[h]) {
            if let Some(heap) = self.pools[h].take() {
                self.retired.push((heap, self.value, now));
            }
        }
    }

    fn retire_free_pools(&mut self) {
        self.retire_free_pools_with(unsafe { self.fence.GetCompletedValue() });
    }

    /// Map `tiles` (tile addresses) of `res` (starting at `res_base`) onto pool tiles,
    /// batched per pool heap with contiguous runs merged into single regions.
    unsafe fn map_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[(usize, (u32, u32))]) {
        if tiles.is_empty() {
            return;
        }
        let mut by_heap: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
        for &(a, (h, t)) in tiles {
            by_heap.entry(h).or_default().push((((a - res_base) / TILE) as u32, t));
        }
        for (h, mut pairs) in by_heap {
            pairs.sort_unstable_by_key(|&(x, _)| x);
            let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(pairs.len());
            let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(pairs.len());
            let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(pairs.len());
            let mut offsets: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut counts: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut i = 0;
            while i < pairs.len() {
                let (x0, t0) = pairs[i];
                let mut run = 1u32;
                while i + (run as usize) < pairs.len() {
                    let (nx, nt) = pairs[i + run as usize];
                    if nx == x0 + run && nt == t0 + run {
                        run += 1;
                    } else {
                        break;
                    }
                }
                coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
                regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
                flags.push(D3D12_TILE_RANGE_FLAG_NONE);
                offsets.push(t0);
                counts.push(run);
                i += run as usize;
            }
            self.queue.UpdateTileMappings(
                res,
                coords.len() as u32,
                Some(coords.as_ptr()),
                Some(regions.as_ptr()),
                self.pools[h as usize].as_ref().expect("a pool tile of a live pool"),
                flags.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }

    unsafe fn unmap_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[usize]) {
        if tiles.is_empty() {
            return;
        }
        let mut xs: Vec<u32> = tiles.iter().map(|&a| ((a - res_base) / TILE) as u32).collect();
        xs.sort_unstable();
        let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(xs.len());
        let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(xs.len());
        let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(xs.len());
        let mut counts: Vec<u32> = Vec::with_capacity(xs.len());
        let mut i = 0;
        while i < xs.len() {
            let x0 = xs[i];
            let mut run = 1u32;
            while i + (run as usize) < xs.len() && xs[i + run as usize] == x0 + run {
                run += 1;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
            regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
            flags.push(D3D12_TILE_RANGE_FLAG_NULL);
            counts.push(run);
            i += run as usize;
        }
        self.queue.UpdateTileMappings(
            res,
            coords.len() as u32,
            Some(coords.as_ptr()),
            Some(regions.as_ptr()),
            None,
            flags.len() as u32,
            Some(flags.as_ptr()),
            None,
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }

    /// tiles of `[a, a + n)` get video memory (mapped in their window); `initial_state` supplies
    /// the initial `(holes, untrusted protection)` masks for each newly mapped 64 KiB tile.
    unsafe fn ensure_mapped(&mut self, a: usize, n: usize, mut initial_state: impl FnMut(usize) -> (u16, u16)) {
        if n == 0 {
            return;
        }
        let first = a & !(TILE - 1);
        let end = a.saturating_add(n);
        let mut new: Vec<(usize, (u32, u32))> = Vec::new();
        let mut unparked = false;
        let mut t = first;
        while t < end {
            if !self.mapped.contains_key(&t) && self.window_index(t).is_some() {
                // still mapped there: no new mapping, what it held is cleared
                if let Some(tile) = self.parked.remove(&t) {
                    let (holes, not_rw) = initial_state(t);
                    let mut entry = TileEntry::with_holes(tile, holes);
                    entry.not_rw = not_rw;
                    self.mapped.insert(t, entry);
                    self.runs = None;
                    self.to_zero.push((t, TILE));
                    unparked = true;
                } else {
                    let Some((tile, reused)) = self.take_tile() else { break };
                    let (holes, not_rw) = initial_state(t);
                    let mut entry = TileEntry::with_holes(tile, holes);
                    entry.not_rw = not_rw;
                    self.mapped.insert(t, entry);
                    self.runs = None;
                    if reused {
                        self.to_zero.push((t, TILE));
                    }
                    new.push((t, tile));
                }
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        if new.is_empty() {
            if unparked {
                MAPPED_GEN.fetch_add(1, Ordering::Release);
            }
            return;
        }
        MAPPED_GEN.fetch_add(1, Ordering::Release);
        // per window
        let mut by_window: FxHashMap<usize, Vec<(usize, (u32, u32))>> = FxHashMap::default();
        for x in new {
            if let Some(wi) = self.window_index(x.0) {
                by_window.entry(wi).or_default().push(x);
            }
        }
        for (wi, tiles) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            self.map_tiles(&r, b, &tiles);
        }
        self.signal();
    }

    #[inline]
    unsafe fn signal(&mut self) {
        self.value += 1;
        let _ = super::real::native_queue_signal(&self.queue, &self.fence, self.value);
    }

    #[inline]
    unsafe fn wait_for(&self, target: u64) -> bool {
        self.wait_for_ms(target, 5000)
    }

    unsafe fn wait_for_ms(&self, target: u64, timeout_ms: u32) -> bool {
        if target == 0 || self.fence.GetCompletedValue() >= target {
            return true;
        }
        let _ = self.fence.SetEventOnCompletion(target, self.event);
        let wait_stall = crate::stall::span(crate::stall::StallKind::GpuWait);
        windows::Win32::System::Threading::WaitForSingleObject(self.event, timeout_ms);
        drop(wait_stall);
        if self.fence.GetCompletedValue() < target {
            crate::xlog_warn!("sparse: the queue has not reached fence value {target} after {timeout_ms} ms (at {})", self.fence.GetCompletedValue());
            return false;
        }
        true
    }

    #[inline]
    unsafe fn reclaim_ring(&mut self) {
        if self.inflight.is_empty() {
            return;
        }
        let completed = self.fence.GetCompletedValue();
        while let Some(&(fv, end_pos)) = self.inflight.front() {
            if fv <= completed {
                self.ring_free = end_pos;
                self.inflight.pop_front();
            } else {
                break;
            }
        }
    }

    /// Acquire the next upload command slot, or return a ticket to wait for without `STATE`.
    unsafe fn begin_slot(&mut self) -> BeginSlot {
        let next_slot = self.next_slot;
        let completed = self.fence.GetCompletedValue();
        let mut waiting = None;
        let mut ready = None;
        for offset in 0..self.slots.len() {
            let idx = (next_slot + offset) % self.slots.len();
            let fv = self.slots[idx].fence_val;
            if fv == 0 || fv <= completed {
                ready = Some(idx);
                break;
            }
            if waiting.is_none_or(|(_, earliest)| fv < earliest) {
                waiting = Some((idx, fv));
            }
        }
        let Some(idx) = ready else {
            let (idx, fv) = waiting.expect("sparse upload slots are not empty");
            return BeginSlot::Wait(self.wait_ticket(fv, WaitKind::Slot { index: idx, next_slot, fence: fv }));
        };
        let slot = &self.slots[idx];
        if slot.alloc.Reset().is_err() || slot.list.Reset(&slot.alloc, None).is_err() {
            return BeginSlot::RetryRequired;
        }
        self.next_slot = (idx + 1) % self.slots.len();
        BeginSlot::Ready(idx)
    }

    fn wait_ticket(&self, target: u64, kind: WaitKind) -> WaitTicket {
        WaitTicket {
            state_generation: self.generation,
            mapping_epoch: self.mapping_epoch,
            dev: self.dev.clone(),
            fence: self.fence.clone(),
            target,
            kind,
        }
    }

    fn wait_ticket_owns_upload(&self, ticket: &WaitTicket) -> bool {
        if !self.wait_ticket_generation(ticket) {
            return false;
        }
        match ticket.kind {
            WaitKind::Slot { index, next_slot, fence } => {
                self.next_slot == next_slot && self.slots.get(index).is_some_and(|slot| slot.fence_val == fence)
            }
            WaitKind::Ring { front, write, submitted, free } => {
                self.inflight.front().copied() == Some(front)
                    && self.ring_write == write && self.ring_submitted == submitted && self.ring_free == free
            }
        }
    }

    fn wait_ticket_generation(&self, ticket: &WaitTicket) -> bool {
        self.generation == ticket.state_generation
            && self.fence.as_raw() == ticket.fence.as_raw()
            && self.uploads_ordered
    }

    fn wait_ticket_valid(&self, ticket: &WaitTicket) -> bool {
        self.mapping_epoch == ticket.mapping_epoch && self.wait_ticket_owns_upload(ticket)
    }

    /// Close and submit upload slot `idx` asynchronously (NO CPU wait — callers order GPU work
    /// via `queue.Wait(&self.fence, self.value)`).
    unsafe fn submit_slot(&mut self, idx: usize) -> SubmitResult {
        if self.slots[idx].list.Close().is_err() {
            return SubmitResult::NotSubmitted;
        }
        let Ok(l) = self.slots[idx].list.cast::<ID3D12CommandList>() else {
            return SubmitResult::NotSubmitted;
        };
        super::title_textures::own_work(|| {
            super::real::native_queue_execute(&self.queue, &l);
        });
        let value = self.value + 1;
        if super::real::native_queue_signal(&self.queue, &self.fence, value).is_err() {
            self.uploads_ordered = false;
            self.slots[idx].fence_val = u64::MAX;
            return SubmitResult::Unordered;
        }
        self.value = value;
        self.slots[idx].fence_val = value;
        if self.ring_write > self.ring_submitted {
            self.inflight.push_back((value, self.ring_write));
            self.ring_submitted = self.ring_write;
        }
        SubmitResult::Submitted
    }

    unsafe fn staging_ptr(&mut self) -> Option<*mut u8> {
        if self.staging.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.staging = Some((r, p as *mut u8));
        }
        self.staging.as_ref().map(|&(_, p)| p)
    }

    unsafe fn ensure_zero_buf(&mut self) -> bool {
        if self.zero_buf.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(ZERO_BUF as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            if self
                .dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .is_err()
            {
                return false;
            }
            let Some(r) = r else { return false };
            let mut p = std::ptr::null_mut();
            if r.Map(0, None, Some(&mut p)).is_err() {
                return false;
            }
            std::ptr::write_bytes(p as *mut u8, 0, ZERO_BUF);
            r.Unmap(0, None);
            self.zero_buf = Some(r);
        }
        true
    }

    unsafe fn readback_buf(&mut self) -> Option<(ID3D12Resource, *mut u8)> {
        if self.readback.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(READBACK_STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.readback = Some((r, p as *mut u8));
        }
        self.readback.clone()
    }

    /// Clear `sent` hashes for every page overlapping `[a, a + n)`.
    fn clear_sent_range(&mut self, a: usize, n: usize) {
        let mut pg = a & !(PAGE - 1);
        let end = a + n;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    entry.sent[(p - t) / PAGE] = None;
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    fn publish_hashes(&mut self, updates: &[HashUpdate]) {
        publish_hash_updates(&mut self.mapped, updates);
    }

    /// Submit one exact dirty batch. Until submission succeeds its ring bytes and hash plan are
    /// provisional; a close/cast failure rolls the ring cursor back to this batch's start.
    unsafe fn submit_dirty_batch(
        &mut self,
        mut batch: PreparedDirtyBatch,
        reclaim: bool,
    ) -> Result<PreparedDirtyBatch, PreparationOutcome> {
        match self.submit_slot(batch.slot) {
            SubmitResult::Submitted => {
                self.publish_hashes(&batch.updates);
                batch.clear_after_submit();
                if reclaim {
                    self.reclaim_ring();
                }
                Ok(batch)
            }
            SubmitResult::NotSubmitted => {
                self.ring_write = batch.ring_start;
                Err(PreparationOutcome::RetryRequired)
            }
            SubmitResult::Unordered => Err(PreparationOutcome::ExecutedUntracked),
        }
    }

    /// Copy `ranges` (address, bytes) from sparse VRAM back into the CPU view and refresh
    /// `sent` hashes. Caller has already ordered `self.queue` after any GPU writes if needed.
    /// `gpu_written`: the ranges are bytes the GPU's commands wrote (`land_page`).
    unsafe fn readback_ranges(&mut self, ranges: &[(usize, usize)], gpu_written: bool) {
        let Some((rb, p)) = self.readback_buf() else { return };
        let mut pending: Vec<(usize, usize, usize)> = Vec::new(); // (va, rb_off, chunk)
        let mut used = 0usize;
        let mut open = false;
        let flush_and_copy = |s: &mut State, open: &mut bool, pending: &mut Vec<(usize, usize, usize)>, used: &mut usize, p: *mut u8| -> bool {
            if *open {
                *open = false;
                let Ok(()) = s.rb_list.Close() else { return false };
                let Ok(l) = s.rb_list.cast::<ID3D12CommandList>() else { return false };
                super::title_textures::own_work(|| {
                    super::real::native_queue_execute(&s.queue, &l);
                });
                s.signal();
                s.rb_last_fence = s.value;
                if !s.wait_for(s.value) {
                    pending.clear();
                    *used = 0;
                    return false;
                }
                s.reclaim_ring();
            }
            for &(dst, off, len) in pending.iter() {
                // copy only committed (non-hole) bytes
                let mut at = dst;
                let mut src_off = off;
                let end = dst + len;
                while at < end {
                    let pg = at & !(PAGE - 1);
                    let t = pg & !(TILE - 1);
                    let idx = (pg - t) / PAGE;
                    let step = (pg + PAGE).min(end) - at;
                    if let Some(entry) = s.mapped.get_mut(&t) {
                        if entry.holes & (1u16 << idx) == 0 {
                            let page = std::slice::from_raw_parts_mut(pg as *mut u8, PAGE);
                            let source = std::slice::from_raw_parts(p.add(src_off), step);
                            land_page(&mut entry.sent[idx], page, at - pg, source, gpu_written);
                        }
                    }
                    at += step;
                    src_off += step;
                }
            }
            pending.clear();
            *used = 0;
            true
        };
        for &(a, n) in ranges {
            if n == 0 {
                continue;
            }
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let tile = cur & !(TILE - 1);
                let Some(wi) = self.window_index(cur) else { break };
                let b = self.windows[wi].0;
                let to_tile_end = (tile + TILE) - cur;
                if !self.mapped.contains_key(&tile) {
                    done += (n - done).min(to_tile_end);
                    continue;
                }
                if used == READBACK_STAGING {
                    if !flush_and_copy(self, &mut open, &mut pending, &mut used, p) {
                        return;
                    }
                }
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
                if !open {
                    if !self.wait_for(self.rb_last_fence) || self.rb_alloc.Reset().is_err() || self.rb_list.Reset(&self.rb_alloc, None).is_err() {
                        return;
                    }
                    open = true;
                }
                self.rb_list.CopyBufferRegion(&rb, used as u64, &self.windows[wi].1, (cur - b) as u64, chunk as u64);
                pending.push((cur, used, chunk));
                used += chunk;
                done += chunk;
            }
        }
        if open {
            let _ = flush_and_copy(self, &mut open, &mut pending, &mut used, p);
        }
    }

    /// Asynchronously zero `runs` (address, bytes) in sparse VRAM using the pre-zeroed `zero_buf`
    /// (zero staging ring bytes consumed, zero CPU stall).
    unsafe fn upload_zeros(&mut self, runs: &[(usize, usize)]) -> UploadStep {
        if runs.is_empty() {
            return UploadStep::Done(PreparationOutcome::Ready);
        }
        if !self.uploads_ordered {
            return UploadStep::Done(PreparationOutcome::ExecutedUntracked);
        }
        if !self.ensure_zero_buf() {
            return UploadStep::Done(PreparationOutcome::RetryRequired);
        }
        let slot_idx = match self.begin_slot() {
            BeginSlot::Ready(index) => index,
            BeginSlot::Wait(ticket) => return UploadStep::Wait(ticket),
            BeginSlot::RetryRequired => return UploadStep::Done(PreparationOutcome::RetryRequired),
        };
        let mut last_wi: Option<usize> = None;
        let zero_buf = self.zero_buf.as_ref().unwrap();
        let list = &self.slots[slot_idx].list;
        for &(a, n) in runs {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(ZERO_BUF).min(to_win_end);
                list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, zero_buf, 0, chunk as u64);
                done += chunk;
            }
        }
        UploadStep::Done(self.submit_slot(slot_idx).into())
    }

    /// Asynchronously copy `runs` (address, bytes) from the CPU view into sparse VRAM tiles via
    /// the 64 MiB ring staging buffer (zero CPU stall on the common path).
    unsafe fn upload_dirty(&mut self, runs: &[(usize, usize)]) -> UploadStep {
        self.upload_dirty_with_timeout(runs, 5000)
    }

    unsafe fn upload_dirty_with_timeout(&mut self, runs: &[(usize, usize)], _wait_ms: u32) -> UploadStep {
        if runs.is_empty() {
            return UploadStep::Done(PreparationOutcome::Ready);
        }
        if !self.uploads_ordered {
            return UploadStep::Done(PreparationOutcome::ExecutedUntracked);
        }
        let Some(p) = self.staging_ptr() else {
            return UploadStep::Done(PreparationOutcome::RetryRequired);
        };
        let mut batch: Option<PreparedDirtyBatch> = None;
        let mut recycled_batch: Option<PreparedDirtyBatch> = None;
        let mut last_wi: Option<usize> = None;
        for &(a, n) in runs {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => {
                            if let Some(batch) = batch {
                                self.ring_write = batch.ring_start;
                            }
                            return UploadStep::Done(PreparationOutcome::RetryRequired);
                        }
                    },
                };
                let b = self.windows[wi].0;
                let copy = prepare_dirty_copy(
                    cur,
                    n - done,
                    b,
                    crate::gpu::uma::WINDOW,
                    self.ring_write,
                    STAGING,
                );
                if (copy.ring_start + copy.bytes as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                    self.reclaim_ring();
                    if (copy.ring_start + copy.bytes as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                        if let Some(ready) = batch.take() {
                            match self.submit_dirty_batch(ready, true) {
                                Ok(recycled) => recycled_batch = Some(recycled),
                                Err(outcome) => return UploadStep::Done(outcome),
                            }
                        }
                        while (copy.ring_start + copy.bytes as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                            let Some(&(fv, _)) = self.inflight.front() else {
                                crate::xlog_warn!("sparse: the staging ring is full without a retirement fence; the unsubmitted CPU writes will be retried");
                                return UploadStep::Done(PreparationOutcome::RetryRequired);
                            };
                            return UploadStep::Wait(self.wait_ticket(
                                fv,
                                WaitKind::Ring {
                                    front: *self.inflight.front().unwrap(),
                                    write: self.ring_write,
                                    submitted: self.ring_submitted,
                                    free: self.ring_free,
                                },
                            ));
                        }
                    }
                }
                if batch.as_ref().is_some_and(|batch| batch.hash_plan_would_overflow(copy.bytes)) {
                    match self.submit_dirty_batch(batch.take().unwrap(), true) {
                        Ok(recycled) => recycled_batch = Some(recycled),
                        Err(outcome) => return UploadStep::Done(outcome),
                    }
                }
                let slot_idx = match batch.as_ref() {
                    Some(batch) => batch.slot,
                    None => {
                        let idx = match self.begin_slot() {
                            BeginSlot::Ready(index) => index,
                            BeginSlot::Wait(ticket) => return UploadStep::Wait(ticket),
                            BeginSlot::RetryRequired => return UploadStep::Done(PreparationOutcome::RetryRequired),
                        };
                        batch = Some(match recycled_batch.take() {
                            Some(recycled) => recycled.reopen(idx, self.ring_write),
                            None => PreparedDirtyBatch::new(idx, self.ring_write),
                        });
                        idx
                    }
                };
                std::ptr::copy_nonoverlapping(cur as *const u8, p.add(copy.staging_offset), copy.bytes);
                batch.as_mut().unwrap().append_hashes(
                    cur,
                    std::slice::from_raw_parts(p.add(copy.staging_offset), copy.bytes),
                );
                let staging_res = &self.staging.as_ref().unwrap().0;
                self.slots[slot_idx].list.CopyBufferRegion(
                    &self.windows[wi].1,
                    copy.window_offset as u64,
                    staging_res,
                    copy.staging_offset as u64,
                    copy.bytes as u64,
                );
                self.ring_write = copy.ring_start + copy.bytes as u64;
                done += copy.bytes;
            }
        }
        if let Some(batch) = batch {
            if let Err(outcome) = self.submit_dirty_batch(batch, false) {
                return UploadStep::Done(outcome);
            }
        }
        UploadStep::Done(PreparationOutcome::Ready)
    }
}

/// Allocate a zeroed, write-watched block in the sparse VRAM windows (never straddling a
/// 4 GiB window boundary when `size <= 4 GiB`). Commits the CPU pages and maps VRAM tiles
/// (or queues them until the D3D12 device exists). Returns `(address, rounded_size)`.
pub unsafe fn alloc(size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc(size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Allocate a zeroed, write-watched block strictly inside sparse window `window` (e.g.
/// window 0 for 32-bit-addressed descriptor heaps). Returns `(address, rounded_size)`.
pub unsafe fn alloc_in(window: usize, size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc_in(window, size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Release a sparse allocation from `alloc` / `alloc_in` (false: not a sparse allocation).
pub unsafe fn free(p: usize) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    if !u.is_sparse(p) {
        return false;
    }
    // decommitted before the address goes back: once released, another thread may get it,
    // commit it and write there
    let Some((start, len)) = u.sparse_reservation(p).filter(|&(start, _)| start == p) else { return false };
    decommitted(start, len);
    VirtualFree(p as *mut c_void, len, MEM_DECOMMIT);
    u.sparse_release(p).is_some()
}

/// After `queue.ExecuteCommandLists`: copy GPU-written sparse ranges (e.g. `Write32/64`,
/// `FillMemoryWith32BitValueX`, `CopyMemoryToMemoryX`) back to their CPU view so CPU reads
/// and spin-waits see the GPU writes without clobbering `sent` hashes.
/// The queue tile mappings go out on, once there is one. Answered without `STATE` (`QUEUE`):
/// a caller may be inside a D3D12 call that a holder of `STATE` waits for.
pub fn queue() -> Option<ID3D12CommandQueue> {
    QUEUE.lock().unwrap().clone()
}

pub unsafe fn readback_after_execute(queue: &ID3D12CommandQueue, ranges: &[(usize, usize)]) {
    if ranges.is_empty() {
        return;
    }
    let measure = super::memory_report::enabled();
    let (mut uploads, timing) = UploadGateTiming::acquire(measure);
    readback_after_execute_locked(queue, ranges);
    let times = timing.finish();
    drop(uploads.take());
    if measure {
        note_readback_gate(times);
    }
}

unsafe fn readback_after_execute_locked(queue: &ID3D12CommandQueue, ranges: &[(usize, usize)]) {
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    let mut filtered: Vec<(usize, usize)> = ranges.iter().copied().filter(|&(a, n)| n > 0 && s.window_index(a).is_some()).collect();
    if filtered.is_empty() {
        return;
    }
    filtered.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(filtered.len());
    for (a, n) in filtered {
        match merged.last_mut() {
            Some(last) if a <= last.0.saturating_add(last.1) => {
                let end = (last.0 + last.1).max(a + n);
                last.1 = end - last.0;
            }
            _ => merged.push((a, n)),
        }
    }
    s.rb_value += 1;
    let v = s.rb_value;
    if super::real::native_queue_signal(queue, &s.rb_fence, v).is_ok() {
        let _ = super::real::native_queue_wait(&s.queue, &s.rb_fence, v);
        super::list_x::note_internal_wait(s.queue.as_raw() as usize, "sparse readback behind a title queue", &s.rb_fence, v);
    }
    // (the ranges are the lists' own GPU writes)
    s.readback_ranges(&merged, true);
}

/// Synchronize `[va, va + len)` in a sparse window for a CPU read (`ID3D12Resource::Map` with
/// a non-empty read range, or `Map(0, NULL)` on a sparse readback resource): flushes pending
/// tile clears and unsubmitted CPU-changed chunks first so CPU writes are preserved, then
/// copies the range back from VRAM into CPU memory.
pub unsafe fn readback_now(va: usize, len: usize) {
    if len == 0 {
        return;
    }
    let measure = super::memory_report::enabled();
    let _upload_lifetime = UPLOAD_STATE_LIFETIME.read().unwrap();
    let (mut uploads, mut timing) = UploadGateTiming::acquire(measure);
    readback_now_locked(va, len, &mut uploads, &mut timing);
    let times = timing.finish();
    drop(uploads.take());
    drop(_upload_lifetime);
    if measure {
        note_readback_gate(times);
    }
}

unsafe fn readback_now_locked(
    va: usize, len: usize, uploads: &mut Option<UploadGuard>, timing: &mut UploadGateTiming,
) {
    'prepare: loop {
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    if s.window_index(va).is_none() {
        return;
    }
    let zeroed = std::mem::take(&mut s.to_zero);
    if !zeroed.is_empty() {
        for &(a, n) in &zeroed {
            s.clear_sent_range(a, n);
        }
        match s.upload_zeros(&zeroed) {
            UploadStep::Done(PreparationOutcome::Ready) => {}
            UploadStep::Wait(ticket) => {
                // Publish the clear retry journal before another uploader can take the gate. It
                // may consume it while this thread waits; after reacquiring, restart from state.
                s.to_zero.extend(zeroed);
                drop(g);
                if !timing.with_released(uploads, || wait_upload_released(ticket)) { return }
                continue 'prepare;
            }
            UploadStep::Done(_) => {
                s.to_zero.extend(zeroed);
                return;
            }
        }
    }
    // Preserve any unsubmitted CPU writes on pages overlapping `[va, va + len)`
    let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
    let mut pg = va & !(PAGE - 1);
    let end = va + len;
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = (t + TILE).min(end);
        if let Some(entry) = s.mapped.get_mut(&t) {
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                if entry.holes & (1u16 << idx) == 0 {
                    let bytes = std::slice::from_raw_parts(p as *const u8, PAGE);
                    scan_page(bytes, p, &entry.sent[idx], &mut dirty_runs);
                }
                p += PAGE;
            }
        }
        pg = tile_end;
    }
    if !dirty_runs.is_empty() {
        match s.upload_dirty(&dirty_runs) {
            UploadStep::Done(PreparationOutcome::Ready) => {}
            UploadStep::Wait(ticket) => {
                // Dirty bytes were only observed, not removed from a journal. Another uploader
                // may send them while this thread waits; restart and rescan current hashes.
                drop(g);
                if !timing.with_released(uploads, || wait_upload_released(ticket)) { return }
                continue 'prepare;
            }
            UploadStep::Done(_) => return,
        }
    }
    s.readback_ranges(&[(va, len)], false);
    return;
    }
}

/// The title committed `[a, a + n)` in a sparse window. The pages of tiles mapped for it that
/// lie outside the range are not committed (holes); the range's pages are.
pub unsafe fn committed(a: usize, n: usize) {
    committed_with_protection(a, n, true);
}

/// `committed`, recording the protection established by the same successful commit without an
/// interval in which an affirmative writability query can observe the new page incorrectly.
pub unsafe fn committed_with_protection(a: usize, n: usize, writable: bool) {
    if n == 0 {
        return;
    }
    let mut g = lock_state();
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_add(&mut EARLY.lock().unwrap(), start, end);
        if writable {
            early_remove(&mut EARLY_NOT_RW.lock().unwrap(), start, end);
        } else {
            early_add(&mut EARLY_NOT_RW.lock().unwrap(), start, end);
        }
        return;
    };
    s.commit_range_with_protection(a, n, writable);
}

/// The title decommitted `[a, a + n)`: tiles whose pages are all uncommitted (whether in one
/// full-tile decommit or incrementally via sub-tile decommits) give their video memory back
/// (unless a placed buffer still pins the pool tile).
pub unsafe fn decommitted(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut g = lock_state();
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_remove(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    s.decommit_range(a, n);
}

impl State {
    /// `committed` with the state locked
    unsafe fn commit_range(&mut self, a: usize, n: usize) {
        self.commit_range_with_protection(a, n, true);
    }

    unsafe fn commit_range_with_protection(&mut self, a: usize, n: usize, writable: bool) {
        self.ensure_mapped(a, n, |_| (0xFFFF, 0));
        let mut pg = a & !(PAGE - 1);
        let end = a.saturating_add(n);
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = t.saturating_add(TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    let bit = 1u16 << ((p - t) / PAGE);
                    entry.holes &= !bit;
                    if writable {
                        entry.not_rw &= !bit;
                    } else {
                        entry.not_rw |= bit;
                    }
                    p += PAGE;
                }
            }
            if tile_end <= pg {
                break;
            }
            pg = tile_end;
        }
        self.mapping_epoch = self.mapping_epoch.wrapping_add(1);
    }

    /// `decommitted` with the state locked: the tiles that go are parked (`flush_parked` unmaps
    /// them and frees their pool tiles)
    unsafe fn decommit_range(&mut self, a: usize, n: usize) {
        let mut pools: Vec<(usize, (u32, u32))> = Vec::new();
        let mut t = a & !(TILE - 1);
        let end = a.saturating_add(n);
        while t < end {
            if let Some(e) = self.mapped.get(&t) {
                pools.push((t, e.pool));
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        let tiles = decommit_tiles(&mut self.mapped, &self.pinned, &mut Vec::new(), &mut self.to_zero, a, n);
        self.mapping_epoch = self.mapping_epoch.wrapping_add(1);
        if tiles.is_empty() {
            return;
        }
        self.runs = None;
        for &t in &tiles {
            if let Ok(i) = pools.binary_search_by_key(&t, |p| p.0) {
                self.parked.insert(t, pools[i].1);
            }
        }
        MAPPED_GEN.fetch_add(1, Ordering::Release);
        if self.parked.len() >= PARKED_MAX && self.flush_parked() {
            self.drain_quarantined();
            self.retire_free_pools();
        }
    }

    /// a pool tile no window maps, none is parked on and no buffer uses goes back to the free list
    fn release_if_unused(&mut self, tile: (u32, u32)) {
        if !self.pinned.contains_key(&tile)
            && !self.mapped.values().any(|e| e.pool == tile)
            && !self.parked.values().any(|&p| p == tile)
            && !self.quarantined.iter().any(|b| b.tiles.contains(&tile))
        {
            self.free.push(tile);
        }
    }

    unsafe fn quarantine_tiles_for(
        &mut self,
        tiles: Vec<(u32, u32)>,
        queues: &[ID3D12CommandQueue],
        retiring: &[(ID3D12Fence, u64)],
        blocked: bool,
    ) {
        if tiles.is_empty() {
            return;
        }
        let mut marks = initial_quarantine_marks((self.fence.clone(), self.value), retiring, blocked).unwrap_or_default();
        let mut ok = !blocked;
        let mut seen = Vec::with_capacity(queues.len());
        marks.reserve(queues.len());
        for q in queues.iter().filter(|_| ok) {
            let key = q.as_raw() as usize;
            if key == self.queue.as_raw() as usize || seen.contains(&key) {
                continue;
            }
            let mut qdev: Option<ID3D12Device> = None;
            if q.GetDevice(&mut qdev).is_err() || qdev.as_ref().is_none_or(|d| d.as_raw() != self.dev.as_raw()) {
                continue;
            }
            seen.push(key);
            let entry = match self.queue_fences.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => match self.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) {
                    Ok(fence) => entry.insert((fence, 0)),
                    Err(_) => {
                        ok = false;
                        break;
                    }
                },
            };
            let next = entry.1 + 1;
            if super::real::native_queue_signal(&q, &entry.0, next).is_ok() {
                entry.1 = next;
                marks.push((entry.0.clone(), next));
            } else {
                ok = false;
                break;
            }
        }
        self.quarantined.push_back(RetiredTiles { tiles, marks: ok.then_some(marks) });
    }

    unsafe fn drain_quarantined_with(&mut self, mut completed: impl FnMut(&ID3D12Fence) -> u64) -> bool {
        let mut freed = false;
        while let Some(front) = self.quarantined.front() {
            let ready = front.marks.as_ref().is_some_and(|marks| marks.iter().all(|(f, v)| completed(f) >= *v));
            if !ready {
                break;
            }
            let batch = self.quarantined.pop_front().unwrap();
            for tile in batch.tiles {
                if !self.pinned.contains_key(&tile) {
                    self.free.push(tile);
                    freed = true;
                }
            }
        }
        freed
    }

    unsafe fn drain_quarantined(&mut self) -> bool {
        self.drain_quarantined_with(|fence| fence.GetCompletedValue())
    }

    /// The parked tiles unmapped from their windows (one call per window) and their unpinned
    /// pool tiles quarantined behind `self.queue` and every known title queue's completion fence.
    /// Pools left all free are not retired here: callers do that (`retire_free_pools`) unless
    /// they take a tile right away.
    unsafe fn flush_parked(&mut self) -> bool {
        self.flush_parked_with(None)
    }

    unsafe fn flush_parked_with(&mut self, extra: Option<&ID3D12CommandQueue>) -> bool {
        if self.parked.is_empty() {
            return false;
        }
        let mut completion = super::present::queue_completion_snapshot();
        if let Some(queue) = extra.filter(|queue| {
            super::present::queue_accepts_internal_work(queue.as_raw() as usize)
                && !completion.active.iter().any(|active| active.as_raw() == queue.as_raw())
        }) {
            completion.active.push(queue.clone());
        }
        self.flush_parked_for(&completion.active, &completion.retiring, completion.blocked)
    }

    unsafe fn flush_parked_for(
        &mut self,
        queues: &[ID3D12CommandQueue],
        retiring: &[(ID3D12Fence, u64)],
        blocked: bool,
    ) -> bool {
        if self.parked.is_empty() {
            return false;
        }
        let mut by_window: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        let mut tiles: Vec<(u32, u32)> = Vec::new();
        for (t, tile) in std::mem::take(&mut self.parked) {
            if let Some(wi) = self.window_index(t) {
                by_window.entry(wi).or_default().push(t);
            }
            if !self.pinned.contains_key(&tile) {
                tiles.push(tile);
            }
        }
        for (wi, ts) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            self.unmap_tiles(&r, b, &ts);
        }
        self.mapping_epoch = self.mapping_epoch.wrapping_add(1);
        self.signal();
        self.quarantine_tiles_for(tiles, queues, retiring, blocked);
        true
    }
}

/// Whether `[p, p + n)` lies in a sparse window and every page is recorded as committed and
/// writable. False means the caller must ask Windows; this cache only supplies affirmative
/// answers, avoiding `VirtualQuery` walking a multi-gigabyte sparse reservation.
pub fn known_writable(p: usize, n: usize) -> bool {
    let Some(end) = p.checked_add(n) else { return false };
    if n == 0 || !crate::gpu::uma::get().is_some_and(|u| u.is_sparse(p) && u.is_sparse(end - 1)) {
        return false;
    }
    {
        let state = STATE.lock().unwrap();
        let Some(state) = state.as_ref() else { return false };
        if !recorded_committed(&state.mapped, p, end) {
            return false;
        }
    }
    resident_pages_writable(p, end)
}

/// Every page touched by `[p, end)` has video-memory backing, is committed, and has not been
/// given a non-writable protection. Check one 64 KiB tile at a time while holding `STATE`.
fn recorded_committed(mapped: &FxHashMap<usize, TileEntry>, p: usize, end: usize) -> bool {
    let mut tile = p & !(TILE - 1);
    while tile < end {
        let first = (p.max(tile) - tile) / PAGE;
        let tile_end = tile.saturating_add(TILE).min(end);
        let last = (tile_end - 1 - tile) / PAGE;
        let pages = last - first + 1;
        let mask = ((((1u32 << pages) - 1) << first) & u16::MAX as u32) as u16;
        if !mapped.get(&tile).is_some_and(|entry| (entry.holes | entry.not_rw) & mask == 0) {
            return false;
        }
        if tile_end <= tile {
            return false;
        }
        tile = tile_end;
    }
    true
}

/// Record a successful protection change in a sparse window. `writable` means a writable
/// protection without PAGE_GUARD. Before device creation, retain page ranges for `init` replay.
pub fn protection_changed(a: usize, n: usize, writable: bool) {
    let Some(end) = a.checked_add(n) else { return };
    let Some(uma) = crate::gpu::uma::get() else { return };
    let mut state = STATE.lock().unwrap();
    for k in 0..uma.shader_windows() {
        if !uma.is_sparse_window(k) {
            continue;
        }
        let (base, size) = uma.window(k);
        let (start, stop) = (a.max(base), end.min(base.saturating_add(size)));
        if start >= stop {
            continue;
        }
        match state.as_mut() {
            Some(state) => mark_protection(&mut state.mapped, start, stop - start, writable),
            None => early_protection(&mut EARLY_NOT_RW.lock().unwrap(), start, stop - start, writable),
        }
    }
}

fn mark_protection(mapped: &mut FxHashMap<usize, TileEntry>, a: usize, n: usize, writable: bool) {
    let end = a.saturating_add(n);
    let mut tile = a & !(TILE - 1);
    while tile < end {
        let first = (a.max(tile) - tile) / PAGE;
        let tile_end = tile.saturating_add(TILE).min(end);
        let last = (tile_end - 1 - tile) / PAGE;
        let pages = last - first + 1;
        let mask = ((((1u32 << pages) - 1) << first) & u16::MAX as u32) as u16;
        if let Some(entry) = mapped.get_mut(&tile) {
            if writable {
                entry.not_rw &= !mask;
            } else {
                entry.not_rw |= mask;
            }
        }
        if tile_end <= tile {
            break;
        }
        tile = tile_end;
    }
}

/// No resident page in `[p, end)` has a non-writable or guard protection. Nonresident pages are
/// judged by the protection record above; this query inspects metadata without touching pages.
fn resident_pages_writable(p: usize, end: usize) -> bool {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    const BATCH: usize = 64;
    const WRITABLE: usize = 0x04 | 0x08 | 0x40 | 0x80;
    let (first, last) = (p >> 12, (end - 1) >> 12);
    let mut entries: [PSAPI_WORKING_SET_EX_INFORMATION; BATCH] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let count = (last - page + 1).min(BATCH);
        for (i, entry) in entries[..count].iter_mut().enumerate() {
            *entry = unsafe { std::mem::zeroed() };
            entry.VirtualAddress = ((page + i) << 12) as *mut c_void;
        }
        let bytes = (count * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32;
        if unsafe {
            K32QueryWorkingSetEx(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                entries.as_mut_ptr() as *mut c_void,
                bytes,
            )
        } == 0
        {
            return false;
        }
        if entries[..count].iter().any(|entry| {
            let flags = unsafe { entry.VirtualAttributes.Flags };
            let protection = (flags >> 4) & 0x7FF;
            flags & 1 != 0 && (protection & WRITABLE == 0 || protection & PAGE_GUARD as usize != 0)
        }) {
            return false;
        }
        page += count;
    }
    true
}

/// The sparse windows' tiles with video memory as sorted (start, end) runs: where a CPU write
/// has to reach the GPU.
pub fn mapped_ranges() -> Vec<(usize, usize)> {
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return Vec::new() };
    if s.runs.is_none() {
        let mut tiles: Vec<usize> = s.mapped.keys().copied().collect();
        tiles.sort_unstable();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for t in tiles {
            match runs.last_mut() {
                Some(last) if last.1 == t => last.1 = t + TILE,
                _ => runs.push((t, t + TILE)),
            }
        }
        s.runs = Some(runs);
    }
    s.runs.clone().unwrap_or_default()
}

const LOOK_FIELDS: usize = 37;
const ACTIVE_US: usize = 0;
const REFRESH_WAIT_US: usize = 1;
const WATCH_US: usize = 2;
const PAGES: usize = 3;
const LOOKS: usize = 4;
const WATCHED: usize = 5;
const RANGES: usize = 6;
const QUEUE_WAIT_US: usize = 7;
const MAINTENANCE_US: usize = 8;
const SCAN_US: usize = 9;
const UPLOAD_US: usize = 10;
const RUNS: usize = 11;
const TRANSFER_BYTES: usize = 12;
const UPLOAD_WAIT_US: usize = 13;
const SUBMIT_GATE_WAIT_US: usize = 14;
const SUBMIT_GATE_HOLD_US: usize = 15;
const READBACK_GATE_WAIT_US: usize = 16;
const READBACK_GATE_HOLD_US: usize = 17;
const READBACK_GATE_RELEASED_US: usize = 18;
const TEXTURE_SELECT_WAIT_US: usize = 19;
const TEXTURE_SELECT_HOLD_US: usize = 20;
const TEXTURES_VISITED: usize = 21;
const TEXTURE_DIRTY_TESTS: usize = 22;
const TEXTURE_MATCHES: usize = 23;
const TEXTURES_RETAINED: usize = 24;
const VRAM_GATE_WAIT_US: usize = 25;
const VRAM_DEVICE_US: usize = 26;
const VRAM_UNMAP_POLL_US: usize = 27;
const VRAM_STATE_US: usize = 28;
const VRAM_QUEUE_WAIT_US: usize = 29;
const VRAM_RETIREMENT_US: usize = 30;
const VRAM_DEVICE_CACHE_HITS: usize = 31;
const VRAM_DEVICE_QUERIES: usize = 32;
const VRAM_CALLS: usize = 33;
const VRAM_MAPPING_WAITS: usize = 34;
const VRAM_RECLAIM_PASSES: usize = 35;
const VRAM_RETIREMENT_PASSES: usize = 36;

/// One complete `XCL_ACTIVITY` look. It is built without shared updates, then published once so
/// the reporting thread cannot split one look across adjacent intervals.
pub(super) struct LookSample([u64; LOOK_FIELDS]);

impl Default for LookSample {
    fn default() -> Self {
        Self([0; LOOK_FIELDS])
    }
}

impl LookSample {
    pub(super) fn add(&mut self, field: usize, value: u64) {
        self.0[field] = self.0[field].wrapping_add(value);
    }

    pub(super) fn add_time(&mut self, field: usize, value: std::time::Duration) {
        self.add(field, value.as_micros() as u64);
    }
}

/// Cumulative complete looks. Disabled title submission never locks or updates this mutex.
static LOOK: Mutex<[u64; LOOK_FIELDS]> = Mutex::new([0; LOOK_FIELDS]);

fn merge_look(total: &mut [u64; LOOK_FIELDS], sample: [u64; LOOK_FIELDS]) {
    for (total, value) in total.iter_mut().zip(sample) {
        *total = total.wrapping_add(value);
    }
}

fn add_gate_times(sample: &mut [u64; LOOK_FIELDS], wait: usize, hold: usize, released: Option<usize>, times: UploadGateTimes) {
    sample[wait] = sample[wait].wrapping_add(times.wait_us);
    sample[hold] = sample[hold].wrapping_add(times.hold_us);
    if let Some(released) = released {
        sample[released] = sample[released].wrapping_add(times.released_us);
    }
}

fn note_readback_gate(times: UploadGateTimes) {
    let mut total = LOOK.lock().unwrap();
    add_gate_times(
        &mut total, READBACK_GATE_WAIT_US, READBACK_GATE_HOLD_US, Some(READBACK_GATE_RELEASED_US), times,
    );
}

pub(super) fn note_watch(sample: &mut LookSample, t: std::time::Duration, pages: usize, watched: usize, ranges: usize) {
    sample.add_time(WATCH_US, t);
    sample.add(PAGES, pages as u64);
    sample.add(WATCHED, watched as u64);
    sample.add(RANGES, ranges as u64);
}

pub(super) fn note_texture_selection(
    sample: &mut LookSample,
    wait: std::time::Duration,
    hold: std::time::Duration,
    visited: usize,
    dirty_tests: usize,
    matches: usize,
    retained: usize,
) {
    sample.add_time(TEXTURE_SELECT_WAIT_US, wait);
    sample.add_time(TEXTURE_SELECT_HOLD_US, hold);
    sample.add(TEXTURES_VISITED, visited as u64);
    sample.add(TEXTURE_DIRTY_TESTS, dirty_tests as u64);
    sample.add(TEXTURE_MATCHES, matches as u64);
    sample.add(TEXTURES_RETAINED, retained as u64);
}

fn note_vram_before_execute(sample: &mut LookSample, timing: super::vram::BeforeExecuteTiming) {
    sample.add_time(VRAM_GATE_WAIT_US, timing.gate_wait);
    sample.add_time(VRAM_DEVICE_US, timing.device);
    sample.add_time(VRAM_UNMAP_POLL_US, timing.unmap_poll);
    sample.add_time(VRAM_STATE_US, timing.state);
    sample.add_time(VRAM_QUEUE_WAIT_US, timing.queue_wait);
    sample.add_time(VRAM_RETIREMENT_US, timing.retirement);
    sample.add(VRAM_DEVICE_CACHE_HITS, timing.device_cache_hits as u64);
    sample.add(VRAM_DEVICE_QUERIES, timing.device_queries as u64);
    sample.add(VRAM_CALLS, timing.calls as u64);
    sample.add(VRAM_MAPPING_WAITS, timing.mapping_waits as u64);
    sample.add(VRAM_RECLAIM_PASSES, timing.reclaim_passes as u64);
    sample.add(VRAM_RETIREMENT_PASSES, timing.retirement_passes as u64);
}

pub(super) fn note_look(mut sample: LookSample, active: std::time::Duration, refresh_wait: std::time::Duration) {
    if let Some(timing) = super::vram::take_before_execute_timing() {
        note_vram_before_execute(&mut sample, timing);
    }
    sample.add_time(ACTIVE_US, active);
    sample.add_time(REFRESH_WAIT_US, refresh_wait);
    sample.add(LOOKS, 1);
    let mut total = LOOK.lock().unwrap();
    merge_look(&mut total, sample.0);
}

pub fn look_counters() -> [u64; LOOK_FIELDS] {
    *LOOK.lock().unwrap()
}

pub fn look_delta(now: [u64; LOOK_FIELDS], before: [u64; LOOK_FIELDS]) -> [u64; LOOK_FIELDS] {
    std::array::from_fn(|i| now[i].wrapping_sub(before[i]))
}

/// Active foreground elapsed time and serialization wait before title submits in one interval.
pub fn look_line(d: [u64; LOOK_FIELDS], secs: u64) -> String {
    let [
        active, refresh_wait, watch, pages, looks, watched, ranges, queue_wait, maintenance, scan, upload,
        runs, transfer_bytes, upload_wait, submit_gate_wait, submit_gate_hold, readback_gate_wait,
        readback_gate_hold, readback_gate_released, texture_select_wait, texture_select_hold,
        textures_visited, texture_dirty_tests, texture_matches, textures_retained,
        vram_gate_wait, vram_device, vram_unmap_poll, vram_state, vram_queue_wait, vram_retirement,
        vram_device_cache_hits, vram_device_queries, vram_calls, vram_mapping_waits,
        vram_reclaim_passes, vram_retirement_passes,
    ] = d;
    let ms = |us: u64| us / 1000;
    let sparse = queue_wait.saturating_add(maintenance).saturating_add(scan).saturating_add(upload).saturating_add(upload_wait);
    let texture_select = texture_select_wait.saturating_add(texture_select_hold);
    let vram = vram_gate_wait
        .saturating_add(vram_device)
        .saturating_add(vram_unmap_poll)
        .saturating_add(vram_state)
        .saturating_add(vram_queue_wait)
        .saturating_add(vram_retirement);
    format!(
        "looks before submits ({secs} s): {looks} looks did {} ms active work ({:.1} % of the time) after {} ms waiting for the refresh turn: write-watch {} ms ({:.1} MB in {:.1} ranges/look, {pages} pages written); texture select wait/hold {}/{} ms ({textures_visited} visited, {texture_dirty_tests} dirty-tested, {texture_matches} matched, {textures_retained} retained); sparse queue wait/fence+state/scan/upload/ownership wait {}/{}/{}/{}/{} ms ({runs} runs, {:.1} MB zero/dirty bytes); VRAM gate/device/unmap poll/state/queue wait/retire {}/{}/{}/{}/{}/{} ms ({vram_calls} calls, {vram_device_cache_hits} device-cache hits, {vram_device_queries} GetDevice queries, {vram_mapping_waits} mapping waits, {vram_reclaim_passes} reclaim passes, {vram_retirement_passes} retirement passes); upload gate submit wait/hold {}/{} ms, readback wait/hold/released wait+reacquire {}/{}/{} ms; other active work {} ms",
        ms(active),
        active as f64 / (secs.max(1) * 10_000) as f64,
        ms(refresh_wait),
        ms(watch),
        watched as f64 / looks.max(1) as f64 / (1 << 20) as f64,
        ranges as f64 / looks.max(1) as f64,
        ms(texture_select_wait),
        ms(texture_select_hold),
        ms(queue_wait),
        ms(maintenance),
        ms(scan),
        ms(upload),
        ms(upload_wait),
        transfer_bytes as f64 / (1 << 20) as f64,
        ms(vram_gate_wait),
        ms(vram_device),
        ms(vram_unmap_poll),
        ms(vram_state),
        ms(vram_queue_wait),
        ms(vram_retirement),
        ms(submit_gate_wait),
        ms(submit_gate_hold),
        ms(readback_gate_wait),
        ms(readback_gate_hold),
        ms(readback_gate_released),
        ms(active.saturating_sub(watch.saturating_add(sparse).saturating_add(texture_select).saturating_add(vram)))
    )
}

fn poll_unique<F>(mut fences: Vec<F>, key: impl Fn(&F) -> usize, mut completed: impl FnMut(&F) -> u64) -> Vec<(F, u64)> {
    fences.sort_unstable_by_key(&key);
    fences.dedup_by(|a, b| key(a) == key(b));
    fences.into_iter().map(|fence| {
        let value = completed(&fence);
        (fence, value)
    }).collect()
}

fn observed_completion<F>(observed: &[(F, u64)], fence: &F, same: impl Fn(&F, &F) -> bool) -> u64 {
    observed.iter().find_map(|(known, value)| same(known, fence).then_some(*value)).unwrap_or(0)
}

fn reserve_queue_wait(waited: &mut FxHashMap<usize, u64>, queue: usize, value: u64) -> bool {
    if value == 0 || waited.get(&queue).copied().unwrap_or(0) >= value {
        return false;
    }
    waited.insert(queue, value);
    true
}

fn retry_journal(outcome: PreparationOutcome, mut deferred: Vec<usize>, runs: &[(usize, usize)]) -> Vec<usize> {
    if outcome != PreparationOutcome::Ready {
        for &(a, n) in runs {
            let mut pg = a & !(PAGE - 1);
            let end = a.saturating_add(n);
            while pg < end {
                deferred.push(pg);
                pg += PAGE;
            }
        }
    }
    deferred.sort_unstable();
    deferred.dedup();
    deferred
}

fn remember_metric_runs(seen: &mut BTreeMap<usize, usize>, runs: &[(usize, usize)]) {
    for &(a, n) in runs {
        early_add(seen, a, a.saturating_add(n));
    }
}

fn note_transfer_metrics(
    sample: Option<&mut LookSample>, seen: Option<&(BTreeMap<usize, usize>, BTreeMap<usize, usize>)>,
) {
    let (Some(sample), Some((zero, dirty))) = (sample, seen) else { return };
    sample.add(RUNS, (zero.len() + dirty.len()) as u64);
    let bytes = zero.iter().chain(dirty).fold(0usize, |sum, (&a, &b)| sum.saturating_add(b.saturating_sub(a)));
    sample.add(TRANSFER_BYTES, bytes as u64);
}

fn rollback_queue_wait(waited: &mut FxHashMap<usize, u64>, queue: usize, value: u64, previous: Option<u64>) {
    if waited.get(&queue) != Some(&value) {
        return;
    }
    match previous {
        Some(previous) => { waited.insert(queue, previous); }
        None => { waited.remove(&queue); }
    }
}

/// Wait with both `STATE` and `UPLOAD_GATE` released. The caller's shared generation lease keeps
/// every old-state allocator, list, staging byte and destination alive. Only an exact ticket may
/// retire ring storage; a crossing uploader otherwise makes the caller rebuild from current state.
unsafe fn wait_upload_released(ticket: WaitTicket) -> bool {
    loop {
        match crate::gpu::fence::wait(&ticket.fence, ticket.target, &ticket.dev, None, 5000) {
            Ok(()) => break,
            Err(crate::gpu::fence::FenceError::Timeout(_)) => {
                crate::xlog_warn!("sparse: upload retirement fence {} is still pending; STATE remains released", ticket.target);
            }
            Err(e) => {
                crate::xlog_warn!("sparse: upload retirement fence {} failed: {e}", ticket.target);
                return false;
            }
        }
    }
    let proceed = {
        let mut g = lock_state();
        match g.as_mut() {
            None => false,
            Some(s) if s.generation != ticket.state_generation || s.fence.as_raw() != ticket.fence.as_raw() => {
                // The caller's shared lifetime makes replacement impossible. Fail closed if a
                // future teardown path violates that protocol.
                false
            }
            Some(s) if !s.uploads_ordered => false,
            Some(s) => {
                // Another uploader can retire/reuse this slot or ring range after the waited
                // target completes. That invalidates exact ownership but not the completed fence;
                // loop on current state. Reclaim only if neither it nor a remap crossed this ticket.
                if s.wait_ticket_valid(&ticket) {
                    s.reclaim_ring();
                }
                true
            }
        }
    };
    drop(ticket);
    proceed
}

/// Snapshot strong fence owners under `STATE`, then query them without it. A batch or a higher
/// mark added while polling is checked against the same observation; a new fence has no
/// observation and therefore remains quarantined.
unsafe fn poll_retirement_fences() -> Vec<(ID3D12Fence, u64)> {
    let fences = {
        let g = lock_state();
        let Some(s) = g.as_ref() else { return Vec::new() };
        std::iter::once(s.fence.clone())
            .chain(s.quarantined.iter().flat_map(|batch| batch.marks.iter().flatten().map(|(fence, _)| fence.clone())))
            .collect()
    };
    poll_unique(fences, |fence| fence.as_raw() as usize, |fence| fence.GetCompletedValue())
}

/// Before the title's ExecuteCommandLists on `queue`: the pages the CPU wrote (`dirty`, 4 KiB
/// pages from the write-watch, sorted) and the reused tiles are copied asynchronously into video
/// memory, and `queue` GPU-waits for that and for new tile mappings.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue, dirty: &[usize], page: usize, mut look: Option<&mut LookSample>) -> PreparationOutcome {
    if look.is_some() {
        super::vram::begin_before_execute_timing();
    }
    let q = queue.as_raw() as usize;
    let t0 = look.is_some().then(std::time::Instant::now);
    let queue_gate = WAIT_GATES.lock().unwrap().entry(q).or_default().clone();
    let _queue_serial = queue_gate.lock().unwrap();
    let _upload_lifetime = UPLOAD_STATE_LIFETIME.read().unwrap();
    let (mut uploads, mut timing) = UploadGateTiming::acquire(look.is_some());
    let t_gated = t0.map(|_| std::time::Instant::now());
    if let (Some(sample), Some(t0), Some(t_gated)) = (look.as_deref_mut(), t0, t_gated) {
        sample.add_time(QUEUE_WAIT_US, t_gated.duration_since(t0));
    }
    let outcome = before_execute_locked(queue, dirty, page, q, t0, &mut look, &mut uploads, &mut timing);
    let times = timing.finish();
    drop(uploads.take());
    if let Some(sample) = look.as_deref_mut() {
        add_gate_times(&mut sample.0, SUBMIT_GATE_WAIT_US, SUBMIT_GATE_HOLD_US, None, times);
    }
    outcome
}

unsafe fn before_execute_locked(
    queue: &ID3D12CommandQueue,
    dirty: &[usize],
    page: usize,
    q: usize,
    t0: Option<std::time::Instant>,
    look: &mut Option<&mut LookSample>,
    uploads: &mut Option<UploadGuard>,
    timing: &mut UploadGateTiming,
) -> PreparationOutcome {
    let mut metric_runs = look.is_some().then(|| (BTreeMap::new(), BTreeMap::new()));
    'prepare: loop {
    let t_started = t0.map(|_| std::time::Instant::now());
    let observed = poll_retirement_fences();
    let mut g = lock_state();
    let Some(s) = g.as_mut() else {
        if let (Some(sample), Some(t_started)) = (look.as_deref_mut(), t_started) {
            sample.add_time(MAINTENANCE_US, t_started.elapsed());
        }
        note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
        return PreparationOutcome::Ready;
    };
    // Snapshot the exact-generation coordination owners while this State is authoritative. The
    // Arc-backed ledgers may later be used after STATE is released without borrowing replacement
    // state; this parity stage still leaves the legacy upload path authoritative.
    let coordination = s.coordination.clone();
    // the title's work sees what was decommitted unmapped, as it was
    let flushed = s.flush_parked_with(Some(queue));
    let drained = s.drain_quarantined_with(|fence| {
        observed_completion(&observed, fence, |known, current| known.as_raw() == current.as_raw())
    });
    if flushed || drained {
        let sparse_done = observed_completion(&observed, &s.fence, |known, current| known.as_raw() == current.as_raw());
        s.retire_free_pools_with(sparse_done);
    }
    let t_maintained = t0.map(|_| std::time::Instant::now());
    let zero_runs = std::mem::take(&mut s.to_zero);
    for &(a, n) in &zero_runs {
        s.clear_sent_range(a, n);
    }
    let mut dirty_runs = std::mem::take(&mut s.scratch_runs);
    dirty_runs.clear();
    let mut deferred = Vec::new();
    let retry_pages = std::mem::take(&mut s.retry_pages);
    let prior_retry = retry_pages.clone();
    if retry_pages.is_empty() {
        let mut i = 0;
        while i < dirty.len() {
            let a = dirty[i];
            let mut n = page;
            i += 1;
            while i < dirty.len() && dirty[i] == a + n {
                n += page;
                i += 1;
            }
            let mut pg = a & !(PAGE - 1);
            while pg < a + n {
                let t = pg & !(TILE - 1);
                if let Some(entry) = s.mapped.get_mut(&t) {
                    let idx = (pg - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        let bytes = std::slice::from_raw_parts(pg as *const u8, PAGE);
                        scan_page(bytes, pg, &entry.sent[idx], &mut dirty_runs);
                    }
                }
                pg += PAGE;
            }
        }
    } else {
        let mut pages = retry_pages;
        pages.extend(dirty.iter().flat_map(|&a| (a..a.saturating_add(page)).step_by(PAGE)));
        pages.sort_unstable();
        pages.dedup();
        for pg in pages {
            let t = pg & !(TILE - 1);
            let Some(entry) = s.mapped.get_mut(&t) else {
                if prior_retry.binary_search(&pg).is_ok() { deferred.push(pg) }
                continue;
            };
            let idx = (pg - t) / PAGE;
            if entry.holes & (1u16 << idx) != 0 {
                if prior_retry.binary_search(&pg).is_ok() { deferred.push(pg) }
                continue;
            }
            let bytes = std::slice::from_raw_parts(pg as *const u8, PAGE);
            scan_page(bytes, pg, &entry.sent[idx], &mut dirty_runs);
        }
    }
    let t_scanned = t0.map(|_| std::time::Instant::now());
    // Dual-journal exactly what the legacy attempt is about to consume. This sealed batch is a
    // parity record only: the established upload path remains authoritative and retires it as
    // obsolete after publishing its own retry state.
    let legacy_batch = coordination
        .seal_legacy(q as u64, &zero_runs, &prior_retry, &dirty_runs)
        .ok()
        .flatten();
    // Owned planning remains preparatory only. Capturing and revalidating source hashes both
    // require title-memory reads; doing either under STATE would add lock hold time and an extra
    // acquisition to the live uploader. Do not activate it until capture can be made genuinely
    // lock-free and measured on a Windows title workload.
    let state_generation = s.generation;
    let mapping_epoch = s.mapping_epoch;
    let mut legacy_parity = legacy_batch.map(|lease| LegacyParityUpload::Lease {
        lease, state_generation, mapping_epoch,
    });
    let mut g = g;
    let Some(s) = g.as_mut() else {
        note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
        return PreparationOutcome::Ready;
    };
    if let Some(parity) = legacy_parity.as_ref() {
        match parity.revalidate(s.generation, s.mapping_epoch, |page| s.upload_page_observation(page)) {
            // "Ready" accepts only the legacy attempt. The parity lease stays pending until that
            // authoritative path finishes; coordinated accept/publish remains a later activation.
            LegacyParityRevalidation::Ready => {}
            LegacyParityRevalidation::RestartForGeneration => {
                drop(g);
                let _ = legacy_parity.take().unwrap().retry();
                // The per-call budget is already consumed, so this rebuild cannot unlock again.
                continue 'prepare;
            }
            LegacyParityRevalidation::RetryCurrentGeneration => {
                s.to_zero.extend(zero_runs.iter().copied());
                s.retry_pages = retry_journal(PreparationOutcome::RetryRequired, deferred, &dirty_runs);
                drop(g);
                let _ = legacy_parity.take().unwrap().retry();
                // The per-call budget is already consumed, so this rebuild cannot unlock again.
                continue 'prepare;
            }
        }
    }
    // Submit tile clears before CPU writes so overlapping CopyBufferRegion destinations have no WAW hazard.
    let mut outcome = PreparationOutcome::Ready;
    let mut upload_wait = None;
    if !zero_runs.is_empty() {
        match s.upload_zeros(&zero_runs) {
            UploadStep::Done(done) => outcome = done,
            UploadStep::Wait(ticket) => {
                outcome = PreparationOutcome::RetryRequired;
                upload_wait = Some(ticket);
            }
        }
        if outcome != PreparationOutcome::Ready {
            s.to_zero.extend(zero_runs.iter().copied());
        }
    }
    if outcome == PreparationOutcome::Ready && !dirty_runs.is_empty() {
        match s.upload_dirty(&dirty_runs) {
            UploadStep::Done(done) => outcome = done,
            UploadStep::Wait(ticket) => {
                outcome = PreparationOutcome::RetryRequired;
                upload_wait = Some(ticket);
            }
        }
    }
    s.retry_pages = retry_journal(outcome, deferred, &dirty_runs);
    if let Some(parity) = legacy_parity {
        let _ = parity.obsolete();
    }
    if let Some((zero, dirty)) = metric_runs.as_mut() {
        remember_metric_runs(zero, &zero_runs);
        remember_metric_runs(dirty, &dirty_runs);
    }
    if let (Some(sample), Some(t_started), Some(t_maintained), Some(t_scanned)) =
        (look.as_deref_mut(), t_started, t_maintained, t_scanned)
    {
        let t_uploaded = std::time::Instant::now();
        sample.add_time(MAINTENANCE_US, t_maintained.duration_since(t_started));
        sample.add_time(SCAN_US, t_scanned.duration_since(t_maintained));
        sample.add_time(UPLOAD_US, t_uploaded.duration_since(t_scanned));
    }
    dirty_runs.clear();
    s.scratch_runs = dirty_runs;
    if let Some(ticket) = upload_wait {
        // `to_zero` and `retry_pages` were restored above before ownership is released. Another
        // queue may consume either journal; reacquisition is included in UPLOAD_WAIT_US and the
        // next iteration always rebuilds its plan from current state.
        drop(g);
        drop(observed);
        let wait_started = t0.map(|_| std::time::Instant::now());
        if !timing.with_released(uploads, || wait_upload_released(ticket)) {
            if let (Some(sample), Some(wait_started)) = (look.as_deref_mut(), wait_started) {
                sample.add_time(UPLOAD_WAIT_US, wait_started.elapsed());
            }
            note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
            return PreparationOutcome::RetryRequired;
        }
        if let (Some(sample), Some(wait_started)) = (look.as_deref_mut(), wait_started) {
            sample.add_time(UPLOAD_WAIT_US, wait_started.elapsed());
        }
        continue 'prepare;
    }
    if outcome != PreparationOutcome::Ready {
        drop(g);
        drop(observed);
        note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
        return outcome;
    }
    let value = s.value;
    let previous = s.waited.get(&q).copied();
    let wait = reserve_queue_wait(&mut s.waited, q, value).then(|| (s.fence.clone(), value, previous));
    drop(g);
    drop(observed);
    if let Some((fence, value, previous)) = wait {
        if super::real::native_queue_wait(queue, &fence, value).is_err() {
            let mut g = lock_state();
            if let Some(s) = g.as_mut().filter(|s| s.fence.as_raw() == fence.as_raw() && s.waited.get(&q) == Some(&value)) {
                rollback_queue_wait(&mut s.waited, q, value, previous);
            }
            note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
            return PreparationOutcome::RetryRequired;
        }
        super::list_x::note_internal_wait(q, "sparse uploads and tile mappings", &fence, value);
    }
    note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
    return PreparationOutcome::Ready;
    }
}

/// A buffer the title places at `va` in a sparse window (`bytes` long, `desc` the PC
/// description): a reserved resource over the window's tiles there, starting at the 64 KiB
/// below `va` (returned: its distance from there).
pub unsafe fn place_buffer(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<(crate::util::HRESULT, usize)> {
    let _placement = PLACEMENT.lock().unwrap();
    place_buffer_in(
        &STATE,
        &FORGOTTEN,
        va,
        desc,
        state,
        |d| {
            let hr = super::real::dev_create_reserved_resource(this, d as *const _ as usize, state as usize, 0, riid, ppv);
            (hr >= 0 && !ppv.is_null() && !(*ppv).is_null()).then_some((hr, *ppv))
        },
        |raw| {
            if !ppv.is_null() && *ppv == raw {
                *ppv = std::ptr::null_mut();
            }
            drop(ID3D12Resource::from_raw(raw));
        },
    )
}

unsafe fn place_buffer_in(
    state_lock: &Mutex<Option<State>>,
    forgotten: &Mutex<Vec<usize>>,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    create: impl FnOnce(&D3D12_RESOURCE_DESC) -> Option<(crate::util::HRESULT, *mut c_void)>,
    on_abort: impl FnOnce(*mut c_void),
) -> Option<(crate::util::HRESULT, usize)> {
    let start = va & !(TILE - 1);
    let delta = va - start;
    let bytes = desc.Width as usize;
    if bytes == 0 {
        return None;
    }
    let end = va.checked_add(bytes)?.checked_sub(1)?;
    let generation = {
        let g = lock_state_in(state_lock, forgotten);
        let s = g.as_ref()?;
        s.window_index(start)?;
        s.window_index(end)?;
        s.fence.clone()
    };
    let mut d = *desc;
    d.Width = (delta + bytes) as u64;
    d.Alignment = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Reserved buffers only accept NONE or ALLOW_UNORDERED_ACCESS (strip console flags and ALLOW_CROSS_ADAPTER).
    d.Flags &= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if state & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32 | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32) != 0 {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    // Driver creation may call back into a release path which needs `STATE`. Revalidate the
    // strongly owned fence generation before publishing the resulting resource.
    let (hr, raw) = create(&d)?;
    let mut g = lock_state_in(state_lock, forgotten);
    let valid = g.as_ref().is_some_and(|s| {
        s.fence.as_raw() == generation.as_raw() && s.window_index(start).is_some() && s.window_index(end).is_some()
    });
    let Some(res) = valid.then(|| ID3D12Resource::from_raw_borrowed(&raw).cloned()).flatten() else {
        drop(g);
        drop(generation);
        on_abort(raw);
        return None;
    };
    let s = g.as_mut().unwrap();
    // its tiles: the window's video memory at the same addresses; probe actual OS commitment
    // so uncommitted pages (e.g. [start, va) or reserved tails) stay marked as holes
    s.ensure_mapped(start, delta + bytes, |t| {
        let holes = query_tile_holes(t);
        (holes, !holes)
    });
    s.mapping_epoch = s.mapping_epoch.wrapping_add(1);
    let tiles: Vec<(usize, (u32, u32))> = (start..va + bytes).step_by(TILE).filter_map(|t| s.mapped.get(&t).map(|e| (t, e.pool))).collect();
    s.map_tiles(&res, start, &tiles);
    if s.buffers.contains_key(&(raw as usize)) {
        forget_in(s, raw as usize);
    }
    for &(_, tile) in &tiles {
        *s.pinned.entry(tile).or_default() += 1;
    }
    s.buffers.insert(raw as usize, tiles.iter().map(|x| x.1).collect());
    s.signal();
    drop(g);
    drop(res);
    drop(generation);
    Some((hr, delta))
}

/// Placed buffers released while `STATE` was held, let go by the next `lock_state`. Held only
/// to push or take, never across a call out.
static FORGOTTEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// `STATE`, with the buffers released while it was held let go first.
fn lock_state() -> std::sync::MutexGuard<'static, Option<State>> {
    lock_state_in(&STATE, &FORGOTTEN)
}

fn lock_state_in<'a>(state: &'a Mutex<Option<State>>, forgotten: &Mutex<Vec<usize>>) -> std::sync::MutexGuard<'a, Option<State>> {
    let mut g = state.lock().unwrap();
    forget_released_in(&mut g, forgotten);
    g
}

fn forget_released_in(g: &mut Option<State>, forgotten: &Mutex<Vec<usize>>) {
    let gone = std::mem::take(&mut *forgotten.lock().unwrap());
    let Some(s) = g.as_mut() else { return };
    for res in gone {
        forget_in(s, res);
    }
}

/// A placed buffer is gone: its tiles are not pinned by it any more. This runs from the
/// resource's last Release, which D3D12 can make while it holds a lock of its own (in a title's
/// command list Reset), and `before_execute` calls D3D12 with `STATE` held: waiting for `STATE`
/// here deadlocked the two threads. So it does not wait: with `STATE` held, the holder's next
/// `lock_state` lets the buffer go.
pub fn forget_buffer(res: usize) {
    forget_buffer_in(&STATE, &FORGOTTEN, res);
}

fn forget_buffer_in(state: &Mutex<Option<State>>, forgotten: &Mutex<Vec<usize>>, res: usize) {
    forgotten.lock().unwrap().push(res);
    match state.try_lock() {
        Ok(mut g) => forget_released_in(&mut g, forgotten),
        Err(std::sync::TryLockError::WouldBlock) => {}
        Err(e) => panic!("sparse state: {e}"),
    }
}

fn forget_in(s: &mut State, res: usize) {
    let Some(tiles) = s.buffers.remove(&res) else { return };
    for tile in tiles {
        if let Some(c) = s.pinned.get_mut(&tile) {
            *c -= 1;
            if *c == 0 {
                s.pinned.remove(&tile);
                s.release_if_unused(tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_look_line_splits_foreground_costs() {
        let line = look_line([
            830_000,
            120_000,
            610_000,
            4_800,
            240,
            240 * 1_641 << 20,
            240 * 69,
            10_000,
            30_000,
            40_000,
            40_000,
            90,
            96 << 20,
            20_000,
            5_000,
            105_000,
            3_000,
            70_000,
            12_000,
            7_000,
            13_000,
            24_000,
            8_000,
            1_200,
            1_190,
            1_000,
            2_000,
            3_000,
            4_000,
            5_000,
            6_000,
            237,
            3,
            240,
            8,
            2,
            1,
        ], 2);
        assert!(line.contains("240 looks did 830 ms active work (41.5 % of the time) after 120 ms waiting"), "{line}");
        assert!(line.contains("write-watch 610 ms (1641.0 MB in 69.0 ranges/look, 4800 pages written)"), "{line}");
        assert!(line.contains("texture select wait/hold 7/13 ms (24000 visited, 8000 dirty-tested, 1200 matched, 1190 retained)"), "{line}");
        assert!(line.contains("sparse queue wait/fence+state/scan/upload/ownership wait 10/30/40/40/20 ms (90 runs, 96.0 MB zero/dirty bytes)"), "{line}");
        assert!(line.contains("VRAM gate/device/unmap poll/state/queue wait/retire 1/2/3/4/5/6 ms (240 calls, 237 device-cache hits, 3 GetDevice queries, 8 mapping waits, 2 reclaim passes, 1 retirement passes)"), "{line}");
        assert!(line.contains("upload gate submit wait/hold 5/105 ms, readback wait/hold/released wait+reacquire 3/70/12 ms; other active work 39 ms"), "{line}");
        let idle = look_line([0; LOOK_FIELDS], 2);
        assert!(idle.contains("0 looks did 0 ms active work (0.0 % of the time) after 0 ms waiting"), "{idle}");
        assert!(idle.contains("0.0 MB in 0.0 ranges/look"), "{idle}");
    }

    #[test]
    fn activity_look_counter_delta_survives_wrap() {
        assert_eq!(look_delta([0; LOOK_FIELDS], [u64::MAX; LOOK_FIELDS]), [1; LOOK_FIELDS]);
    }

    #[test]
    fn texture_selection_metrics_keep_wait_hold_and_counts_distinct() {
        let mut sample = LookSample::default();
        note_texture_selection(
            &mut sample,
            std::time::Duration::from_micros(7),
            std::time::Duration::from_micros(13),
            101,
            79,
            11,
            10,
        );
        assert_eq!(sample.0[TEXTURE_SELECT_WAIT_US..=TEXTURES_RETAINED], [7, 13, 101, 79, 11, 10]);
    }

    #[test]
    fn vram_metrics_keep_phases_and_counts_distinct() {
        let mut sample = LookSample::default();
        note_vram_before_execute(&mut sample, super::super::vram::BeforeExecuteTiming {
            gate_wait: std::time::Duration::from_micros(1),
            device: std::time::Duration::from_micros(2),
            unmap_poll: std::time::Duration::from_micros(3),
            state: std::time::Duration::from_micros(4),
            queue_wait: std::time::Duration::from_micros(5),
            retirement: std::time::Duration::from_micros(6),
            device_cache_hits: 7,
            device_queries: 8,
            calls: 9,
            mapping_waits: 10,
            reclaim_passes: 11,
            retirement_passes: 12,
        });
        assert_eq!(sample.0[VRAM_GATE_WAIT_US..=VRAM_RETIREMENT_PASSES], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let line = look_line(sample.0, 2);
        assert!(line.contains("9 calls, 7 device-cache hits, 8 GetDevice queries"), "{line}");
    }

    #[test]
    fn upload_gate_times_accumulate_into_distinct_submit_and_readback_fields() {
        let mut sample = [0; LOOK_FIELDS];
        add_gate_times(
            &mut sample,
            SUBMIT_GATE_WAIT_US,
            SUBMIT_GATE_HOLD_US,
            None,
            UploadGateTimes { wait_us: 11, hold_us: 22, released_us: 33 },
        );
        add_gate_times(
            &mut sample,
            READBACK_GATE_WAIT_US,
            READBACK_GATE_HOLD_US,
            Some(READBACK_GATE_RELEASED_US),
            UploadGateTimes { wait_us: 44, hold_us: 55, released_us: 66 },
        );
        assert_eq!(sample[SUBMIT_GATE_WAIT_US], 11);
        assert_eq!(sample[SUBMIT_GATE_HOLD_US], 22);
        assert_eq!(sample[READBACK_GATE_WAIT_US], 44);
        assert_eq!(sample[READBACK_GATE_HOLD_US], 55);
        assert_eq!(sample[READBACK_GATE_RELEASED_US], 66);
    }

    #[test]
    fn retried_upload_ranges_are_counted_once() {
        let mut zero = BTreeMap::new();
        let mut dirty = BTreeMap::new();
        remember_metric_runs(&mut zero, &[(0x1000, 0x1000)]);
        remember_metric_runs(&mut zero, &[(0x1000, 0x1000)]);
        remember_metric_runs(&mut dirty, &[(0x4000, 0x1000)]);
        remember_metric_runs(&mut dirty, &[(0x4000, 0x2000)]);
        let mut sample = LookSample::default();
        note_transfer_metrics(Some(&mut sample), Some(&(zero, dirty)));
        assert_eq!(sample.0[RUNS], 2);
        assert_eq!(sample.0[TRANSFER_BYTES], 0x3000);
    }

    #[test]
    fn activity_look_publishes_one_complete_sample() {
        let mut total = [10; LOOK_FIELDS];
        let sample = std::array::from_fn(|i| i as u64);
        merge_look(&mut total, sample);
        assert_eq!(total, std::array::from_fn(|i| 10 + i as u64));
        merge_look(&mut total, std::array::from_fn(|i| u64::MAX - 9 - i as u64));
        assert_eq!(total, [0; LOOK_FIELDS]);
    }

    #[test]
    fn writable_record_requires_every_touched_page_and_tracks_protection() {
        let mut mapped = FxHashMap::default();
        let (t0, t1) = (0x10_0000usize, 0x11_0000usize);
        mapped.insert(t0, TileEntry::with_holes((0, 1), 0));
        mapped.insert(t1, TileEntry::with_holes((0, 2), 1 << 2));

        assert!(recorded_committed(&mapped, t0, t0 + TILE));
        assert!(recorded_committed(&mapped, t0 + 0x800, t1 + 2 * PAGE));
        assert!(!recorded_committed(&mapped, t1 + 2 * PAGE, t1 + 2 * PAGE + 1));
        assert!(!recorded_committed(&mapped, t1 + TILE - 8, t1 + TILE + 8));

        mark_protection(&mut mapped, t0 + 5 * PAGE + 16, 8, false);
        assert!(!recorded_committed(&mapped, t0 + 5 * PAGE, t0 + 5 * PAGE + 1));
        assert!(recorded_committed(&mapped, t0 + 6 * PAGE, t0 + TILE));
        mark_protection(&mut mapped, t0 + 5 * PAGE, PAGE, true);
        assert!(recorded_committed(&mapped, t0, t0 + TILE));
    }

    #[test]
    fn early_protection_ranges_merge_split_and_commit_clears_them() {
        let base = 0x20_0000usize;
        let mut protected = BTreeMap::new();
        early_protection(&mut protected, base + PAGE + 16, 8, false);
        early_protection(&mut protected, base + 3 * PAGE, 3 * PAGE, false);
        early_protection(&mut protected, base + 5 * PAGE, PAGE, true);
        assert_eq!(protected.into_iter().collect::<Vec<_>>(), [(base + PAGE, base + 2 * PAGE), (base + 3 * PAGE, base + 5 * PAGE)]);

        let mut protected = BTreeMap::from([(base, base + 4 * PAGE)]);
        early_remove(&mut protected, base + PAGE, base + 3 * PAGE);
        assert_eq!(protected.into_iter().collect::<Vec<_>>(), [(base, base + PAGE), (base + 3 * PAGE, base + 4 * PAGE)]);
    }

    #[test]
    fn working_set_check_rejects_resident_read_only_pages_without_touching_cold_pages() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_READONLY};
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert_ne!(base, 0);
            assert!(!VirtualAlloc(base as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            assert!(!VirtualAlloc((base + PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READONLY).is_null());
            assert!(!VirtualAlloc((base + 2 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::write_volatile(base as *mut u8, 1);
            assert_eq!(std::ptr::read_volatile((base + PAGE) as *const u8), 0);
            assert!(resident_pages_writable(base, base + PAGE));
            assert!(!resident_pages_writable(base + PAGE, base + 2 * PAGE));
            assert!(resident_pages_writable(base + 2 * PAGE, base + 3 * PAGE));

            let mut cold: windows_sys::Win32::System::ProcessStatus::PSAPI_WORKING_SET_EX_INFORMATION = std::mem::zeroed();
            cold.VirtualAddress = (base + 2 * PAGE) as *mut c_void;
            assert_ne!(
                windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx(
                    windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    &mut cold as *mut _ as *mut c_void,
                    std::mem::size_of_val(&cold) as u32,
                ),
                0
            );
            assert_eq!(cold.VirtualAttributes.Flags & 1, 0, "the metadata query paged memory in");
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn memory_counts_committed_pages_and_separates_devices_and_retired_pools() {
        let holes = [0xFFFE, 0xFFF0].into_iter(); // one committed page, then four
        let (pools, occupied, retired, committed) = memory_byte_counts(7, 7, 1, 60, 1, holes);
        assert_eq!(pools, POOL_TILES as usize * TILE);
        assert_eq!(occupied, 4 * TILE, "occupied video tiles are independent of CPU page commitment");
        assert_eq!(retired, POOL_TILES as usize * TILE, "the retired pool is not also active");
        assert_eq!(committed, 5 * PAGE, "holes in a 64 KiB tile do not consume CPU commit");
        assert_eq!(memory_byte_counts(8, 7, 1, 60, 1, [0].into_iter()), (0, 0, 0, 0));
    }

    /// `queue()` answers while the state's lock is held: a resource's last `Release` asks for
    /// it (`late_release::drain_known`) from inside a D3D12 call that the lock's holder can be
    /// waiting for (a list's `Reset` against an upload's allocator `Reset`, with `XCL_DRED=1`).
    #[test]
    fn the_queue_is_answered_while_the_state_is_locked() {
        let held = STATE.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let asked = std::thread::spawn(move || {
            let _ = queue();
            let _ = tx.send(());
        });
        let answered = rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok();
        drop(held);
        asked.join().unwrap();
        assert!(answered, "queue() waited for the sparse state's lock");
    }

    #[test]
    fn exact_retirement_forgets_cache_but_inflight_batch_keeps_its_fence() {
        use std::sync::Arc;
        let old = Arc::new(7u32);
        let mut fences = FxHashMap::default();
        fences.insert(0xA, (old.clone(), 3));
        let batch_mark = fences[&0xA].0.clone();

        assert!(fences.contains_key(&0xA), "marker failure must leave the cache pinned");
        let gone = remove_queue_fence(&mut fences, 0xA).expect("exact retirement callback");
        assert!(fences.is_empty() && Arc::ptr_eq(&gone.0, &batch_mark));
        drop(gone);
        drop(old);
        assert_eq!(Arc::strong_count(&batch_mark), 1, "cache removal invalidated an in-flight batch mark");

        let replacement = Arc::new(9u32);
        fences.insert(0xA, (replacement.clone(), 1));
        assert!(!Arc::ptr_eq(&batch_mark, &fences[&0xA].0), "reused raw address kept the retired generation's fence");
    }

    #[test]
    fn retirement_fence_poll_is_unlocked_and_new_requirements_fail_closed() {
        use std::sync::{atomic::{AtomicU64, AtomicUsize, Ordering}, Arc};
        let first = Arc::new(AtomicU64::new(1));
        let late = Arc::new(AtomicU64::new(9));
        let state = Mutex::new(vec![first.clone(), first.clone()]);
        let polls = AtomicUsize::new(0);

        let snapshot = state.lock().unwrap().clone();
        let observed = poll_unique(snapshot, |fence| Arc::as_ptr(fence) as usize, |fence| {
            assert!(state.try_lock().is_ok(), "completion polling held sparse state");
            state.lock().unwrap().push(late.clone());
            polls.fetch_add(1, Ordering::Relaxed);
            fence.load(Ordering::Relaxed)
        });

        assert_eq!(polls.load(Ordering::Relaxed), 1, "the same fence is queried once");
        assert_eq!(observed_completion(&observed, &first, Arc::ptr_eq), 1);
        assert!(observed_completion(&observed, &first, Arc::ptr_eq) < 2, "a higher mark added during polling remains held");
        assert_eq!(observed_completion(&observed, &late, Arc::ptr_eq), 0, "a newly added fence remains held");
    }

    #[test]
    fn title_queue_wait_reservation_is_monotonic_and_per_queue() {
        let mut waited = FxHashMap::default();
        assert!(!reserve_queue_wait(&mut waited, 1, 0));
        assert!(reserve_queue_wait(&mut waited, 1, 4));
        assert!(!reserve_queue_wait(&mut waited, 1, 4));
        assert!(!reserve_queue_wait(&mut waited, 1, 3));
        assert!(reserve_queue_wait(&mut waited, 2, 3));
        assert!(reserve_queue_wait(&mut waited, 1, 5));
        assert_eq!(waited, FxHashMap::from_iter([(1, 5), (2, 3)]));
    }

    #[test]
    fn failed_write_watch_pages_survive_until_a_ready_retry() {
        let runs = [(0x1040, PAGE), (0x2000, CHUNK)];
        assert_eq!(retry_journal(PreparationOutcome::RetryRequired, vec![], &runs), [0x1000, 0x2000]);
        assert!(retry_journal(PreparationOutcome::Ready, vec![], &runs).is_empty());
        assert_eq!(retry_journal(PreparationOutcome::ExecutedUntracked, vec![0x3000], &runs), [0x1000, 0x2000, 0x3000]);
    }

    #[test]
    fn late_claim_completion_cannot_change_a_newer_owner() {
        let mut ledger = WorkClaimLedger::new(11);
        ledger.add_dirty_pages([0x8000]);
        let old = ledger.claim(1).unwrap();
        assert_eq!(ledger.commit(old.token), WorkClaimResolution::Committed);

        ledger.add_retry_pages([0x8000]);
        let newer = ledger.claim(1).unwrap();
        assert_eq!(ledger.restore(old.token), WorkClaimResolution::AlreadyFinished);
        assert_eq!(ledger.owners.get(&0x8000), Some(&newer.token.serial));
        assert_eq!(ledger.commit(newer.token), WorkClaimResolution::Committed);
        assert!(ledger.available.is_empty());
    }

    #[test]
    fn foreground_queue_cannot_pass_or_steal_an_unreceipted_claim() {
        let mut frontier = ClaimFrontier::new(23);
        let planner = frontier.claim(100).unwrap();
        assert_eq!(
            frontier.observe(23, planner.epoch, 200),
            Ok(ClaimFrontierAction::TakeOver(planner)),
            "another queue mistook pending planner work for an empty frontier",
        );

        let helper = frontier.take_over(planner, 200).unwrap();
        assert_eq!(frontier.accept(planner), Err(ClaimFrontierError::StaleToken));
        assert_eq!(frontier.receipt(planner, 7), Err(ClaimFrontierError::StaleToken));
        assert_eq!(frontier.consume_obsolete(planner), Err(ClaimFrontierError::StaleToken));
        assert_eq!(frontier.take_over(planner, 300), Err(ClaimFrontierError::StaleToken));
        assert_eq!(frontier.observe(23, helper.epoch, 200), Ok(ClaimFrontierAction::Progress(helper)));

        frontier.accept(helper).unwrap();
        assert_eq!(frontier.take_over(helper, 100), Err(ClaimFrontierError::WrongState));
        assert_eq!(frontier.consume_obsolete(helper), Err(ClaimFrontierError::WrongState));
        assert_eq!(
            frontier.observe(23, helper.epoch, 100),
            Ok(ClaimFrontierAction::WaitForReceipt(helper)),
            "accepted GPU work was exposed for retry",
        );
        frontier.receipt(helper, 7).unwrap();
        assert_eq!(
            frontier.observe(23, helper.epoch, 100),
            Ok(ClaimFrontierAction::Ready { through: helper.epoch, required_fence: Some(7) }),
        );
    }

    #[test]
    fn receipts_and_obsolete_holes_advance_only_a_contiguous_frontier() {
        let mut frontier = ClaimFrontier::new(29);
        let first = frontier.claim(1).unwrap();
        let second = frontier.claim(2).unwrap();
        let hole = frontier.claim(3).unwrap();
        frontier.accept(first).unwrap();
        frontier.accept(second).unwrap();
        frontier.receipt(second, 11).unwrap();
        frontier.consume_obsolete(hole).unwrap();

        assert_eq!(frontier.retired_through, 0, "out-of-order terminal work crossed the first claim");
        assert_eq!(
            frontier.observe(29, hole.epoch, 9),
            Ok(ClaimFrontierAction::WaitForReceipt(first)),
        );
        frontier.receipt(first, 8).unwrap();
        assert_eq!(frontier.retired_through, hole.epoch, "an obsolete hole left the frontier stuck");
        assert_eq!(
            frontier.observe(29, hole.epoch, 9),
            Ok(ClaimFrontierAction::Ready { through: hole.epoch, required_fence: Some(11) }),
        );
        assert_eq!(
            frontier.observe(29, first.epoch, 9),
            Ok(ClaimFrontierAction::Ready { through: first.epoch, required_fence: Some(8) }),
            "an older observer unnecessarily inherited a newer fence",
        );
    }

    #[test]
    fn frontier_rejects_wrong_generations_unknown_epochs_and_invalid_receipts() {
        let mut frontier = ClaimFrontier::new(31);
        let token = frontier.claim(4).unwrap();
        assert_eq!(frontier.observe(30, token.epoch, 4), Err(ClaimFrontierError::StaleGeneration));
        assert_eq!(frontier.observe(31, token.epoch + 1, 4), Err(ClaimFrontierError::UnknownEpoch));
        assert_eq!(frontier.receipt(token, 1), Err(ClaimFrontierError::WrongState));
        frontier.accept(token).unwrap();
        assert_eq!(frontier.receipt(token, 0), Err(ClaimFrontierError::Fence));
        assert_eq!(frontier.observe(31, 0, 4), Ok(ClaimFrontierAction::Ready { through: 0, required_fence: None }));

        let stale_generation = ClaimFrontierToken { state_generation: 32, ..token };
        assert_eq!(frontier.receipt(stale_generation, 1), Err(ClaimFrontierError::StaleGeneration));
    }

    #[test]
    fn frontier_epoch_and_lease_overflow_fail_without_partial_publication() {
        let mut epoch = ClaimFrontier::new(37);
        epoch.next_epoch = u64::MAX;
        assert_eq!(epoch.claim(1), Err(ClaimFrontierError::Exhausted));
        assert!(epoch.records.is_empty());

        let mut lease = ClaimFrontier::new(41);
        lease.next_lease = u64::MAX;
        assert_eq!(lease.claim(1), Err(ClaimFrontierError::Exhausted));
        assert_eq!(lease.next_epoch, 0);
        assert!(lease.records.is_empty());

        let mut takeover = ClaimFrontier::new(43);
        let token = takeover.claim(1).unwrap();
        takeover.next_lease = u64::MAX;
        assert_eq!(takeover.take_over(token, 2), Err(ClaimFrontierError::Exhausted));
        assert_eq!(takeover.observe(43, token.epoch, 1), Ok(ClaimFrontierAction::Progress(token)));
    }

    #[test]
    fn work_claim_ledger_merges_orders_and_restores_exact_out_of_order_work() {
        let mut ledger = WorkClaimLedger::new(7);
        ledger.add_zero_ranges(&[(0x3000, 2 * PAGE)]);
        ledger.add_retry_pages([0x3001, 0x1000, 0x1000]);
        ledger.add_dirty_pages([0x4000, 0x6001, 0x6000]);

        let first = ledger.claim(2).unwrap();
        assert_eq!(first.pages.iter().map(|(page, reasons)| (*page, reasons.0)).collect::<Vec<_>>(), [
            (0x1000, WorkReasons::RETRY.0),
            (0x3000, WorkReasons::ZERO.0 | WorkReasons::RETRY.0),
        ]);
        let second = ledger.claim(8).unwrap();
        assert_eq!(second.pages.iter().map(|(page, reasons)| (*page, reasons.0)).collect::<Vec<_>>(), [
            (0x4000, WorkReasons::ZERO.0 | WorkReasons::DIRTY.0),
            (0x6000, WorkReasons::DIRTY.0),
        ]);

        ledger.add_dirty_pages([0x1000]);
        assert!(ledger.claim(1).is_none(), "new work escaped an older owner of the same page");
        assert_eq!(ledger.cancel(second.token), WorkClaimResolution::Restored);
        let out_of_order = ledger.claim(1).unwrap();
        assert_eq!(out_of_order.pages[0].0, 0x4000);
        assert_eq!(ledger.commit(first.token), WorkClaimResolution::Committed);
        let later = ledger.claim(1).unwrap();
        assert_eq!(later.pages, [(0x1000, WorkReasons::DIRTY)]);
        assert_eq!(ledger.commit(later.token), WorkClaimResolution::Committed);
        assert_eq!(ledger.cancel(out_of_order.token), WorkClaimResolution::Restored);
        let restored = ledger.claim(8).unwrap();
        assert_eq!(restored.pages.iter().map(|(page, _)| *page).collect::<Vec<_>>(), [0x4000, 0x6000]);
    }

    #[test]
    fn replaced_sparse_batch_handles_reject_stale_additions() {
        let batches = SparseBatches::new(17);
        let stale_clone = batches.clone();
        let current = batches.replace_generation(18).unwrap();
        assert!(!batches.add_dirty_pages([0xC000]));
        assert!(!stale_clone.add_retry_pages([0xD000]));
        assert!(matches!(batches.seal(1, 1), Err(SparseBatchError::StaleGeneration)));
        assert!(matches!(stale_clone.replace_generation(19), Err(SparseBatchError::StaleGeneration)));
        assert!(current.add_dirty_pages([0xB000]));
        assert_eq!(current.seal(1, 1).unwrap().unwrap().pages()[0].0, 0xB000);
    }

    #[test]
    fn sparse_batch_takeover_revokes_without_restoring_or_duplicating_work() {
        let batches = SparseBatches::new(47);
        assert!(batches.add_dirty_pages([0x1_1000]));
        let old = batches.seal(1, 1).unwrap().unwrap();
        let through = batches.through().unwrap();
        assert_eq!(old.pages()[0].0, 0x1_1000);

        let replacement = old.take_over(2).unwrap();
        assert!(matches!(
            old.obsolete(),
            Err(SparseBatchError::Frontier(ClaimFrontierError::StaleToken)),
        ));
        assert_eq!(replacement.pages()[0].0, 0x1_1000);
        replacement.accept().unwrap().receipt(5).unwrap();
        assert_eq!(
            batches.observe(through, 9),
            Ok(ClaimFrontierAction::Ready { through, required_fence: Some(5) }),
            "the revoked owner restored or duplicated its work",
        );
        assert!(batches.seal(3, 1).unwrap().is_none());
    }

    #[test]
    fn dropped_pending_batch_remains_visible_for_foreground_takeover() {
        let batches = SparseBatches::new(53);
        assert!(batches.add_dirty_pages([0x2_1000]));
        let abandoned = batches.seal(10, 1).unwrap().unwrap();
        let token = abandoned.token;
        let through = batches.through().unwrap();
        drop(abandoned);
        assert_eq!(batches.observe(through, 20), Ok(ClaimFrontierAction::TakeOver(token)));

        let foreground = batches.take_over(token, 20).unwrap();
        foreground.accept().unwrap().receipt(7).unwrap();
        assert_eq!(
            batches.observe(through, 20),
            Ok(ClaimFrontierAction::Ready { through, required_fence: Some(7) }),
        );
    }

    #[test]
    fn sparse_batch_producers_are_collected_before_locking() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let batches = SparseBatches::new(57);
        let unlocked = AtomicBool::new(false);
        let ledger = batches.ledger.clone();
        let pages = std::iter::once_with(|| { unlocked.store(ledger.try_lock().is_ok(), Ordering::Relaxed); 0x2_2001 });
        assert!(batches.add_dirty_pages(pages));
        assert!(unlocked.load(Ordering::Relaxed), "a producer iterator ran under the batch lock");
        assert_eq!(batches.seal(1, 1).unwrap().unwrap().pages()[0].0, 0x2_2000);
    }

    #[test]
    fn sparse_batch_ready_requires_new_reasons_to_be_sealed() {
        let batches = SparseBatches::new(59);
        assert!(batches.add_dirty_pages([0x3_1000]));
        let first = batches.seal(1, 1).unwrap().unwrap();
        let first_epoch = batches.through().unwrap();
        assert!(batches.add_dirty_pages([0x3_1001]));
        assert_eq!(batches.through(), Err(SparseBatchError::Unsealed));
        assert_eq!(batches.observe(first_epoch, 2), Err(SparseBatchError::Unsealed));

        first.accept().unwrap().receipt(8).unwrap();
        assert_eq!(batches.through(), Err(SparseBatchError::Unsealed));
        let successor = batches.seal(2, 1).unwrap().unwrap();
        let through = batches.through().unwrap();
        assert!(through > first_epoch);
        successor.obsolete().unwrap();
        assert_eq!(
            batches.observe(through, 2),
            Ok(ClaimFrontierAction::Ready { through, required_fence: Some(8) }),
        );
    }

    #[test]
    fn sparse_batch_receipts_are_contiguous_and_acceptance_is_fail_closed() {
        let batches = SparseBatches::new(61);
        assert!(batches.add_dirty_pages([0x4_1000, 0x4_2000]));
        let first = batches.seal(1, 1).unwrap().unwrap();
        let second = batches.seal(2, 1).unwrap().unwrap();
        let through = batches.through().unwrap();
        let first_token = first.token;
        let first = first.accept().unwrap();
        let second = second.accept().unwrap();
        second.receipt(11).unwrap();
        assert_eq!(batches.observe(through, 3), Ok(ClaimFrontierAction::WaitForReceipt(first_token)));
        first.receipt(8).unwrap();
        assert_eq!(
            batches.observe(through, 3),
            Ok(ClaimFrontierAction::Ready { through, required_fence: Some(11) }),
        );

        assert!(batches.add_dirty_pages([0x4_3000]));
        let accepted = batches.seal(4, 1).unwrap().unwrap().accept().unwrap();
        let accepted_token = accepted.token.unwrap().0;
        assert!(matches!(
            batches.take_over(accepted_token, 5),
            Err(SparseBatchError::Frontier(ClaimFrontierError::WrongState)),
        ));
        drop(accepted);
        assert_eq!(batches.through(), Err(SparseBatchError::Poisoned));

        let current = batches.replace_generation(62).unwrap();
        assert_eq!(batches.through(), Err(SparseBatchError::StaleGeneration));
        assert!(matches!(current.replace_generation(61), Err(SparseBatchError::StaleGeneration)));
        assert!(matches!(current.take_over(accepted_token, 6), Err(SparseBatchError::StaleGeneration)));
        assert!(current.add_dirty_pages([0x5_1000]));
        let stale = current.seal(6, 1).unwrap().unwrap();
        let replacement = current.replace_generation(63).unwrap();
        assert_eq!(stale.obsolete(), Err(SparseBatchError::StaleGeneration));
        assert_eq!(replacement.through(), Ok(0));
        assert!(matches!(SparseBatches::new(u64::MAX).replace_generation(0), Err(SparseBatchError::StaleGeneration)));
    }

    #[test]
    fn live_sparse_coordination_owns_one_exact_generation() {
        let coordination = SparseCoordination::new(101, 501).unwrap();
        {
            let batches = coordination.batches.ledger.lock().unwrap();
            assert_eq!(batches.work.state_generation, 101);
            assert_eq!(batches.fence, Some(SparseFenceIdentity { state_generation: 101, allocation: 501 }));
        }
        assert_eq!(coordination.waits.fence(), SparseFenceIdentity { state_generation: 101, allocation: 501 });
        let reservation = coordination.reservations.reserve(0, RING_ALIGN).unwrap();
        assert_eq!(reservation.token.unwrap().state_generation, 101);
        assert!(SparseCoordination::new(0, 501).is_none());
        assert!(SparseCoordination::new(101, 0).is_none());
    }

    #[test]
    fn legacy_sparse_attempt_is_mirrored_and_retired_as_one_exact_batch() {
        let coordination = SparseCoordination::new(103, 503).unwrap();
        let snapshot = coordination.clone();
        assert!(Arc::ptr_eq(&coordination.batches.ledger, &snapshot.batches.ledger));

        let batch = snapshot.seal_legacy(
            0xA,
            &[(0x3000, 2 * PAGE)],
            &[0x3001, 0x1000],
            &[(0x4000 + CHUNK, CHUNK)],
        ).unwrap().unwrap();
        assert_eq!(batch.token.state_generation, 103);
        assert_eq!(
            batch.pages().iter().map(|(page, reasons)| (*page, reasons.0)).collect::<Vec<_>>(),
            [
                (0x1000, WorkReasons::RETRY.0),
                (0x3000, WorkReasons::ZERO.0 | WorkReasons::RETRY.0),
                (0x4000, WorkReasons::ZERO.0 | WorkReasons::DIRTY.0),
            ],
        );
        let through = snapshot.batches.through().unwrap();
        batch.obsolete().unwrap();
        assert_eq!(
            coordination.batches.observe(through, 0xB),
            Ok(ClaimFrontierAction::Ready { through, required_fence: None }),
            "the parity batch remained visible after the legacy path retired it",
        );
        assert!(coordination.batches.seal(0xB, usize::MAX).unwrap().is_none());
    }

    fn wait_ready(fence: SparseFenceIdentity, through: u64, value: Option<u64>) -> SparseReadyReceipt {
        SparseReadyReceipt { state_generation: fence.state_generation, fence, through, required_fence: value }
    }

    fn wait_reservation(decision: ForegroundWaitDecision) -> ForegroundWaitReservation {
        match decision { ForegroundWaitDecision::Reserve(reservation) => reservation, _ => panic!("wait not reserved") }
    }

    #[test]
    fn foreground_wait_pairs_out_of_order_frontier_with_exact_fence() {
        let batches = SparseBatches::with_fence(101, 501).unwrap();
        assert!(batches.add_dirty_pages([0x6_1000, 0x6_2000]));
        let first = batches.seal(1, 1).unwrap().unwrap();
        let first_token = first.token;
        let second = batches.seal(2, 1).unwrap().unwrap();
        let through = batches.through().unwrap();
        let first = first.accept().unwrap();
        second.accept().unwrap().receipt(11).unwrap();
        let waits = ForegroundWaits::new(101, 501).unwrap();
        let queue = waits.register_queue(0xA).unwrap();
        assert_eq!(waits.register_queue(0xA), Ok(queue));
        assert_eq!(batches.ready_receipt(through, 3), Err(ForegroundWaitError::NotReady));
        assert_eq!(batches.observe(through, 3), Ok(ClaimFrontierAction::WaitForReceipt(first_token)));
        first.receipt(8).unwrap();

        let ready = batches.ready_receipt(through, 3).unwrap();
        let wrong = SparseReadyReceipt { fence: SparseFenceIdentity { allocation: 502, ..ready.fence }, ..ready };
        assert!(matches!(waits.reserve(queue, wrong), Err(ForegroundWaitError::Fence)));
        let reservation = wait_reservation(waits.reserve(queue, ready).unwrap());
        assert_eq!(reservation.arm().unwrap().publish(), Ok((through, 11)));
        assert!(matches!(waits.reserve(queue, ready), Ok(ForegroundWaitDecision::Installed { through: t, value: 11 }) if t == through));
        assert!(matches!(waits.reserve(queue, wait_ready(waits.fence(), through + 1, None)),
            Ok(ForegroundWaitDecision::NoWait { through: t }) if t == through + 1));

        let other = SparseBatches::with_fence(101, 502).unwrap();
        let other_waits = ForegroundWaits::new(101, 502).unwrap();
        let other_queue = other_waits.register_queue(0xE).unwrap();
        assert!(matches!(other.ready_receipt(through, 3), Err(ForegroundWaitError::Batch(_))));
        assert!(matches!(other_waits.reserve(other_queue, ready), Err(ForegroundWaitError::Fence)));

        assert!(matches!(batches.replace_generation_with_fence(102, 501), Err(SparseBatchError::Fence)));
        let stale = batches.clone();
        let batches = batches.replace_generation_with_fence(102, 503).unwrap();
        assert_eq!(batches.ledger.lock().unwrap().fence.unwrap().allocation, 503);
        assert_eq!(stale.ready_receipt(through, 3), Err(ForegroundWaitError::Batch(SparseBatchError::StaleGeneration)))
    }

    #[test]
    fn foreground_wait_rollbacks_are_exact_and_queue_reuse_is_stale() {
        let waits = ForegroundWaits::new(103, 601).unwrap();
        let fence = waits.fence();
        let queue = waits.register_queue(0xB).unwrap();
        let ready = wait_ready(fence, 1, Some(5));
        let abandoned = wait_reservation(waits.reserve(queue, ready).unwrap());
        let stale = abandoned.token.unwrap();
        assert!(matches!(waits.reserve(queue, ready), Err(ForegroundWaitError::Pending(token)) if token == stale));
        assert!(matches!(waits.reserve(queue, wait_ready(fence, 0, None)), Err(ForegroundWaitError::Pending(_))));
        drop(abandoned);

        let accepted = wait_reservation(waits.reserve(queue, wait_ready(fence, 2, Some(6))).unwrap()).arm().unwrap();
        let accepted_token = accepted.token.unwrap();
        accepted.rollback().unwrap();
        let newer = wait_reservation(waits.reserve(queue, wait_ready(fence, 3, Some(7))).unwrap());
        assert_eq!(waits.0.lock().unwrap().transition(accepted_token, ForegroundWaitState::Accepted, None),
            Err(ForegroundWaitError::StaleToken));
        assert!(matches!(waits.reserve(queue, ready), Err(ForegroundWaitError::Pending(token)) if token == newer.token.unwrap()));
        drop(newer);
        waits.retire_queue(queue).unwrap();
        let replacement = waits.register_queue(0xB).unwrap();
        assert!(replacement.generation > queue.generation);
        assert!(matches!(waits.reserve(queue, ready), Err(ForegroundWaitError::Queue)));
        assert!(matches!(waits.reserve(replacement, ready), Ok(ForegroundWaitDecision::Reserve(_))));
    }

    #[test]
    fn foreground_wait_unpublished_acceptance_fails_closed() {
        let waits = ForegroundWaits::new(107, 701).unwrap();
        let queue = waits.register_queue(0xC).unwrap();
        let ready = wait_ready(waits.fence(), 1, Some(9));
        let accepted = wait_reservation(waits.reserve(queue, ready).unwrap()).arm().unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _accepted = accepted;
            panic!("after native wait acceptance");
        }));
        assert!(result.is_err());
        assert!(matches!(waits.reserve(queue, ready), Err(ForegroundWaitError::Uncertain)));
        assert!(matches!(waits.reserve(queue, wait_ready(waits.fence(), 0, None)), Err(ForegroundWaitError::Uncertain)));
    }

    #[test]
    fn foreground_wait_generation_replacement_rejects_old_guards_and_fences() {
        let waits = ForegroundWaits::new(109, 801).unwrap();
        let queue = waits.register_queue(0xD).unwrap();
        let old_fence = waits.fence();
        let old_ready = wait_ready(old_fence, 1, Some(3));
        let old = wait_reservation(waits.reserve(queue, old_ready).unwrap());
        waits.replace_generation(109, 110, 802).unwrap();
        drop(old);
        assert!(matches!(waits.reserve(queue, old_ready), Err(ForegroundWaitError::StaleGeneration)));
        assert_eq!(waits.replace_generation(110, 109, 803), Err(ForegroundWaitError::StaleGeneration));
        assert_eq!(waits.replace_generation(110, 111, 801), Err(ForegroundWaitError::Fence));
        let current = wait_ready(waits.fence(), 1, Some(4));
        assert!(matches!(waits.reserve(queue, current), Ok(ForegroundWaitDecision::Reserve(_))));
    }

    fn upload_observation(page: usize, pool: (u32, u32), source_word: u64) -> UploadPageObservation {
        let mut source = ZERO_HASHES;
        source[1] = source_word;
        source[2] = source_word;
        UploadPageObservation { page, pool, hole: false, sent: Some(ZERO_HASHES), source }
    }

    fn coordinated_upload_fixture(
        owner: u64,
    ) -> (SparseBatches, CoordinatedSparseUploadPlan, UploadPageObservation, ClaimFrontierToken) {
        let batches = SparseBatches::new(71);
        assert!(batches.add_dirty_pages([0x2_1000]));
        let lease = batches.seal(owner, 1).unwrap().unwrap();
        let token = lease.token;
        let observation = upload_observation(0x2_1000, (4, 9), 0xA5);
        let plan = CoordinatedSparseUploadPlan::build(lease, 71, 7, |_| Some(observation.clone())).unwrap();
        (batches, plan, observation, token)
    }

    fn commit_coordinated(
        plan: CoordinatedSparseUploadPlan, observation: &UploadPageObservation,
        reservations: &UploadReservations, slot: usize, fence: u64,
    ) -> Result<UploadSubmissionReceipt, CoordinatedSparseUploadError<(), (), ()>> {
        plan.commit_reserved(
            71, 7, |_| Some(observation.clone()), reservations, slot, 64,
            |_, _| Ok(vec![0; 64]), |_, _| Ok(()), |_, _| {}, || Ok(fence),
        )
    }

    #[test]
    fn shared_upload_page_builder_preserves_intent_order_and_dirty_ranges() {
        let claimed = [
            (0x1_1000, WorkReasons(WorkReasons::ZERO.0 | WorkReasons::DIRTY.0)),
            (0x1_2000, WorkReasons::RETRY),
        ];
        let first = upload_observation(0x1_1000, (2, 3), 0x91);
        let second = upload_observation(0x1_2000, (2, 3), ZERO_HASH);
        let pages = build_upload_pages(&claimed, |page| match page {
            0x1_1000 => Some(first.clone()),
            0x1_2000 => Some(second.clone()),
            _ => None,
        }).unwrap();

        assert_eq!(pages.iter().map(|page| page.page).collect::<Vec<_>>(), [0x1_1000, 0x1_2000]);
        assert!(pages[0].zero && pages[0].dirty);
        assert_eq!(pages[0].dirty_ranges, [(0x1_1000 + CHUNK, 2 * CHUNK)]);
        assert!(!pages[1].zero && pages[1].dirty);
        assert!(pages[1].dirty_ranges.is_empty(), "retry intent was lost when no chunks changed");
        assert!(build_upload_pages(&claimed, |page| {
            (page == 0x1_1000).then(|| first.clone())
        }).is_none());
    }

    #[test]
    fn coordinated_plan_revalidates_every_sparse_page_identity() {
        let (_, plan, observation, _) = coordinated_upload_fixture(1);
        assert_eq!(plan.validate(72, 7, |_| Some(observation.clone())), Err(UploadPlanInvalid::StateGeneration));
        assert_eq!(plan.validate(71, 8, |_| Some(observation.clone())), Err(UploadPlanInvalid::MappingEpoch));
        assert_eq!(plan.validate(71, 7, |_| None), Err(UploadPlanInvalid::MissingPage));

        let mut remapped = observation.clone();
        remapped.page += TILE;
        assert_eq!(plan.validate(71, 7, |_| Some(remapped.clone())), Err(UploadPlanInvalid::Mapping));
        let mut repooled = observation.clone();
        repooled.pool.1 += 1;
        assert_eq!(plan.validate(71, 7, |_| Some(repooled.clone())), Err(UploadPlanInvalid::Mapping));
        let mut hole = observation.clone();
        hole.hole = true;
        assert_eq!(plan.validate(71, 7, |_| Some(hole.clone())), Err(UploadPlanInvalid::Mapping));
        let mut resent = observation.clone();
        resent.sent.as_mut().unwrap()[3] ^= 1;
        assert_eq!(plan.validate(71, 7, |_| Some(resent.clone())), Err(UploadPlanInvalid::SentHashes));
        let mut rewritten = observation.clone();
        rewritten.source[3] ^= 1;
        assert_eq!(plan.validate(71, 7, |_| Some(rewritten.clone())), Err(UploadPlanInvalid::SourceHashes));
    }

    #[test]
    fn live_parity_snapshot_revalidates_concurrent_changes_and_retries() {
        let batches = SparseBatches::new(73);
        assert!(batches.add_dirty_pages([0x2_3000]));
        let lease = batches.seal(3, 1).unwrap().unwrap();
        let token = lease.token;
        let through = batches.through().unwrap();
        let observed = upload_observation(0x2_3000, (6, 11), 0xB7);
        let mut captures = 0;
        let snapshot = UploadPlanningSnapshot::capture(73, 9, lease.pages(), |_| {
            captures += 1;
            Some(observed.clone())
        }).unwrap();
        assert_eq!(captures, 1);

        // Construction consumes only the owned snapshot; no live State observation is available
        // while this work runs outside its lock.
        let plan = CoordinatedSparseUploadPlan::from_snapshot(lease, snapshot).unwrap();
        assert_eq!(plan.pages[0].source, observed.source);
        assert_eq!(plan.validate(74, 9, |_| Some(observed.clone())), Err(UploadPlanInvalid::StateGeneration));
        assert_eq!(plan.validate(73, 10, |_| Some(observed.clone())), Err(UploadPlanInvalid::MappingEpoch));
        let mut rewritten = observed.clone();
        rewritten.source[5] ^= 1;
        assert_eq!(plan.validate(73, 9, |_| Some(rewritten.clone())), Err(UploadPlanInvalid::SourceHashes));
        assert_eq!(plan.validate(73, 9, |_| Some(observed.clone())), Ok(()));

        let parity = LegacyParityUpload::Planned(plan);
        assert_eq!(
            parity.revalidate(74, 9, |_| Some(observed.clone())),
            LegacyParityRevalidation::RestartForGeneration,
        );
        assert_eq!(
            parity.revalidate(73, 10, |_| Some(observed.clone())),
            LegacyParityRevalidation::RetryCurrentGeneration,
        );
        assert_eq!(
            parity.revalidate(73, 9, |_| Some(rewritten.clone())),
            LegacyParityRevalidation::RetryCurrentGeneration,
        );
        assert_eq!(
            parity.revalidate(73, 9, |_| Some(observed.clone())),
            LegacyParityRevalidation::Ready,
        );
        assert_eq!(
            batches.observe(through, 4),
            Ok(ClaimFrontierAction::TakeOver(token)),
            "validation must not accept or publish the behavior-neutral parity lease",
        );
        parity.retry().unwrap();
        let restored = batches.seal(4, 1).unwrap().expect(
            "a stale unlocked plan did not restore its exact work for retry",
        );
        assert_eq!(restored.pages(), &[(0x2_3000, WorkReasons::DIRTY)]);
        let fallback = LegacyParityUpload::Lease {
            lease: restored,
            state_generation: 73,
            mapping_epoch: 9,
        };
        assert_eq!(
            fallback.revalidate(74, 9, |_| None),
            LegacyParityRevalidation::RestartForGeneration,
        );
        assert_eq!(
            fallback.revalidate(73, 10, |_| None),
            LegacyParityRevalidation::RetryCurrentGeneration,
        );
        fallback.retry().unwrap();
        let exact = batches.seal(5, 1).unwrap().unwrap();
        assert_eq!(exact.pages(), &[(0x2_3000, WorkReasons::DIRTY)]);
        exact.obsolete().unwrap();
    }

    #[test]
    fn coordinated_preaccept_unwind_cancels_reservation_and_leaves_batch_visible() {
        for panic_at in ["copy", "prepare"] {
            let (batches, plan, observation, token) = coordinated_upload_fixture(1);
            let reservations = UploadReservations::new(71, 1, 256, 64);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = plan.commit_reserved(
                    71, 7, |_| Some(observation.clone()), &reservations, 0, 64,
                    |_, _| -> Result<Vec<u8>, ()> {
                        if panic_at == "copy" { panic!("copy panicked") }
                        Ok(vec![0; 64])
                    },
                    |_, _| -> Result<(), ()> {
                        if panic_at == "prepare" { panic!("prepare panicked") }
                        Ok(())
                    },
                    |_, _| {}, || Ok::<_, ()>(9),
                );
            }));
            assert!(result.is_err());
            assert_eq!(batches.observe(1, 9), Ok(ClaimFrontierAction::TakeOver(token)));
            let ledger = reservations.0.lock().unwrap();
            assert!(!ledger.poisoned && ledger.records.is_empty());
            assert!(ledger.slots.iter().all(Option::is_none));
        }
    }

    #[test]
    fn coordinated_commit_orders_callbacks_and_owns_copied_staging() {
        let (_, plan, observation, _) = coordinated_upload_fixture(1);
        let reservations = UploadReservations::new(71, 1, 256, 64);
        let calls = std::cell::RefCell::new(Vec::new());
        let receipt = plan.commit_reserved(
            71, 7, |_| Some(observation.clone()), &reservations, 0, 4,
            |range, pages| {
                calls.borrow_mut().push("copy");
                assert_eq!(range, UploadRingRange { start: 0, end: 4 });
                assert_eq!(pages[0].page, observation.page);
                Ok::<_, ()>(vec![1, 2, 3, 4])
            },
            |staging, _| {
                calls.borrow_mut().push("prepare");
                assert_eq!(&*staging.bytes, &[1, 2, 3, 4]);
                Ok::<_, ()>(())
            },
            |staging, _| {
                calls.borrow_mut().push("execute");
                assert_eq!(&*staging.bytes, &[1, 2, 3, 4]);
            },
            || { calls.borrow_mut().push("signal"); Ok::<_, ()>(9) },
        ).unwrap();
        assert_eq!(&*calls.borrow(), &["copy", "prepare", "execute", "signal"]);
        assert_eq!(reservations.complete(receipt), Ok(4));
    }

    #[test]
    fn coordinated_plan_preaccept_failures_stay_visible_and_takeover_revokes() {
        for failure in ["validate", "reservation generation", "reserve", "copy", "size", "prepare"] {
            let (batches, plan, observation, token) = coordinated_upload_fixture(1);
            let reservations = UploadReservations::new(if failure == "reservation generation" { 72 } else { 71 }, 1, 256, 64);
            let occupied = (failure == "reserve").then(|| reservations.reserve(0, 64).unwrap());
            let result = plan.commit_reserved(
                if failure == "validate" { 72 } else { 71 }, 7, |_| Some(observation.clone()),
                &reservations, 0, 64,
                |_, _| if failure == "copy" { Err("copy") } else { Ok(vec![0; if failure == "size" { 63 } else { 64 }]) },
                |_, _| if failure == "prepare" { Err("prepare") } else { Ok(()) },
                |_, _| {}, || Ok::<_, &'static str>(9),
            );
            let expected = match failure {
                "validate" => matches!(result, Err(CoordinatedSparseUploadError::Invalid(UploadPlanInvalid::StateGeneration))),
                "reservation generation" => matches!(result, Err(CoordinatedSparseUploadError::Reserve(UploadReservationError::StaleGeneration))),
                "reserve" => matches!(result, Err(CoordinatedSparseUploadError::Reserve(UploadReservationError::SlotBusy))),
                "copy" => matches!(result, Err(CoordinatedSparseUploadError::Copy("copy"))),
                "size" => matches!(result, Err(CoordinatedSparseUploadError::StagingSize)),
                _ => matches!(result, Err(CoordinatedSparseUploadError::Prepare("prepare"))),
            };
            assert!(expected, "wrong {failure} boundary result");
            assert_eq!(batches.observe(1, 9), Ok(ClaimFrontierAction::TakeOver(token)));
            assert!(!reservations.0.lock().unwrap().poisoned);
            drop(occupied);
        }

        let (batches, stale, observation, token) = coordinated_upload_fixture(1);
        let replacement = batches.take_over(token, 2).unwrap();
        let result = commit_coordinated(stale, &observation, &UploadReservations::new(71, 1, 256, 64), 0, 9);
        assert!(matches!(result, Err(CoordinatedSparseUploadError::Accept(
            SparseBatchError::Frontier(ClaimFrontierError::StaleToken)
        ))));
        assert_eq!(replacement.pages()[0].0, 0x2_1000);
    }

    #[test]
    fn coordinated_submit_then_panic_poisons_both_generations() {
        let (batches, plan, observation, _) = coordinated_upload_fixture(1);
        let reservations = UploadReservations::new(71, 1, 256, 64);
        let submitted = std::cell::Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.commit_reserved(
                71, 7, |_| Some(observation.clone()), &reservations, 0, 64,
                |_, _| Ok::<_, ()>(vec![0; 64]), |_, _| Ok::<_, ()>(()),
                |_, _| { submitted.set(true); panic!("execute panicked after submit") },
                || Ok::<_, ()>(9),
            );
        }));
        assert!(result.is_err() && submitted.get());
        assert!(reservations.0.lock().unwrap().poisoned);
        assert_eq!(batches.through(), Err(SparseBatchError::Poisoned));

        let (batches, plan, observation, _) = coordinated_upload_fixture(1);
        let reservations = UploadReservations::new(71, 1, 256, 64);
        let result = plan.commit_reserved(
            71, 7, |_| Some(observation.clone()), &reservations, 0, 64,
            |_, _| Ok::<_, ()>(vec![0; 64]), |_, _| Ok::<_, ()>(()), |_, _| {}, || Err::<u64, _>("signal"),
        );
        assert!(matches!(result, Err(CoordinatedSparseUploadError::Signal("signal"))));
        assert!(reservations.0.lock().unwrap().poisoned);
        assert_eq!(batches.through(), Err(SparseBatchError::Poisoned));

        let (batches, plan, observation, _) = coordinated_upload_fixture(1);
        let reservations = UploadReservations::new(71, 1, 256, 64);
        assert!(matches!(commit_coordinated(plan, &observation, &reservations, 0, 0),
            Err(CoordinatedSparseUploadError::PublishReservation(UploadReservationError::State))));
        assert!(reservations.0.lock().unwrap().poisoned);
        assert_eq!(batches.through(), Err(SparseBatchError::Poisoned));
    }

    #[test]
    fn coordinated_frontier_publication_failure_quarantines_submitted_reservation() {
        let (batches, plan, _, _) = coordinated_upload_fixture(1);
        let lease = plan.lease.unwrap();
        let batch = lease.accept().unwrap();
        let reservations = UploadReservations::new(71, 1, 256, 64);
        let reservation = reservations.reserve(0, 64).unwrap();
        let mut accepted = AcceptedCoordinatedUpload::new(reservation, batch);
        let receipt = accepted.reservation.take().unwrap().publish_after_accept(13).unwrap();
        accepted.batch_ledger.lock().unwrap().bindings.remove(&accepted.batch_token.epoch);
        assert_eq!(accepted.batch.take().unwrap().receipt(13), Err(SparseBatchError::Corrupt));
        drop(accepted);
        let ledger = reservations.0.lock().unwrap();
        assert!(ledger.poisoned);
        assert_eq!(ledger.records[&receipt.token.serial].state, UploadReservationState::Poisoned);
        drop(ledger);
        let ledger = batches.ledger.lock().unwrap();
        assert!(ledger.poisoned && ledger.work.claims.is_empty() && ledger.work.owners.is_empty());
        assert!(ledger.bindings.is_empty());
    }

    #[test]
    fn a_prior_poison_does_not_leave_later_accepted_work_owned() {
        let batches = SparseBatches::new(71);
        assert!(batches.add_dirty_pages([0x3_1000, 0x3_2000]));
        let first = batches.seal(1, 1).unwrap().unwrap().accept().unwrap();
        let second = batches.seal(2, 1).unwrap().unwrap().accept().unwrap();
        let reservations = UploadReservations::new(71, 2, 256, 64);
        let first = AcceptedCoordinatedUpload::new(reservations.reserve(0, 64).unwrap(), first);
        let second = AcceptedCoordinatedUpload::new(reservations.reserve(1, 64).unwrap(), second);
        drop(first);
        assert!(matches!(second.publish(17), Err(CoordinatedPublishError::Batch(SparseBatchError::Poisoned))));
        let ledger = batches.ledger.lock().unwrap();
        assert!(ledger.poisoned && ledger.work.claims.is_empty() && ledger.work.owners.is_empty());
        assert!(ledger.bindings.is_empty());
        drop(ledger);
        let reservations = reservations.0.lock().unwrap();
        assert!(reservations.poisoned && reservations.records.values().all(|r| r.state == UploadReservationState::Poisoned));
    }

    #[test]
    fn coordinated_receipts_publish_exact_out_of_order_fence_prefix() {
        let batches = SparseBatches::new(71);
        assert!(batches.add_dirty_pages([0x2_1000, 0x2_2000]));
        let observation = |page| upload_observation(page, (4, 9), 0xA5);
        let first_lease = batches.seal(1, 1).unwrap().unwrap();
        let first_token = first_lease.token;
        let second_lease = batches.seal(2, 1).unwrap().unwrap();
        let through = batches.through().unwrap();
        let first = CoordinatedSparseUploadPlan::build(first_lease, 71, 7, |page| Some(observation(page))).unwrap();
        let second = CoordinatedSparseUploadPlan::build(second_lease, 71, 7, |page| Some(observation(page))).unwrap();
        let reservations = UploadReservations::new(71, 2, 256, 64);
        let second_receipt = commit_coordinated(second, &observation(0x2_2000), &reservations, 1, 11).unwrap();
        assert_eq!(batches.observe(through, 3), Ok(ClaimFrontierAction::TakeOver(first_token)));
        let first_receipt = commit_coordinated(first, &observation(0x2_1000), &reservations, 0, 8).unwrap();
        assert_eq!(batches.observe(through, 3), Ok(ClaimFrontierAction::Ready { through, required_fence: Some(11) }));
        assert_eq!(reservations.complete(first_receipt), Ok(0));
        assert_eq!(reservations.complete(second_receipt), Ok(128));
    }

    #[test]
    fn upload_reservations_reclaim_only_a_contiguous_retired_prefix() {
        let reservations = UploadReservations::new(41, 2, 256, 64);
        let first = reservations.reserve(0, 80).unwrap();
        let second = reservations.reserve(1, 80).unwrap();
        assert_eq!(first.range(), UploadRingRange { start: 0, end: 80 });
        assert_eq!(second.range(), UploadRingRange { start: 128, end: 208 });
        let first = first.submit(5).unwrap();
        let second = second.submit(6).unwrap();

        assert_eq!(reservations.complete(second), Ok(0), "later retirement crossed an older owner");
        assert!(matches!(reservations.reserve(1, 64), Err(UploadReservationError::Full)));
        assert_eq!(reservations.complete(first), Ok(208));

        let wrapped = reservations.reserve(1, 64).unwrap();
        assert_eq!(wrapped.range(), UploadRingRange { start: 256, end: 320 });
        assert_eq!(wrapped.submit(7).and_then(|receipt| reservations.complete(receipt)), Ok(320));
    }

    #[test]
    fn an_empty_ring_discards_tail_padding_before_a_full_capacity_reservation() {
        let reservations = UploadReservations::new(42, 1, 256, 64);
        drop(reservations.reserve(0, 64).unwrap());
        {
            let ledger = reservations.0.lock().unwrap();
            assert_eq!((ledger.cursor, ledger.reclaimed), (64, 64));
        }

        let full = reservations.reserve(0, 256).unwrap();
        assert_eq!(full.range(), UploadRingRange { start: 256, end: 512 });
        let ledger = reservations.0.lock().unwrap();
        assert_eq!((ledger.cursor, ledger.reclaimed), (512, 256));
    }

    #[test]
    fn failed_empty_ring_reservation_does_not_publish_boundary_normalization() {
        let mut ledger = UploadReservationLedger::new(42, 1, 256, 64);
        let token = ledger.reserve(42, 0, 64).unwrap().0;
        assert_eq!(ledger.cancel(token), Ok(64));
        ledger.next_serial = u64::MAX;

        assert!(matches!(ledger.reserve(42, 0, 256), Err(UploadReservationError::Exhausted)));
        assert_eq!((ledger.cursor, ledger.reclaimed), (64, 64));
        assert_eq!(ledger.next_serial, u64::MAX);
        assert!(ledger.order.is_empty());
        assert!(ledger.records.is_empty());
        assert!(ledger.slots[0].is_none());
    }

    #[test]
    fn cancelling_an_older_reservation_never_rewinds_newer_ring_work() {
        let reservations = UploadReservations::new(43, 3, 512, 64);
        let first = reservations.reserve(0, 64).unwrap();
        let middle = reservations.reserve(1, 64).unwrap();
        let third = reservations.reserve(2, 64).unwrap();
        let middle_token = middle.token.unwrap();
        drop(middle);
        assert_eq!(reservations.0.lock().unwrap().reclaimed, 0);

        let later = reservations.reserve(1, 64).unwrap();
        assert!(later.range().start >= third.range().end, "a cancellation rewound across newer work");
        drop(first);
        assert_eq!(reservations.0.lock().unwrap().reclaimed, 128);
        assert_eq!(reservations.0.lock().unwrap().cancel(middle_token), Err(UploadReservationError::Missing));

        let third = third.submit(8).unwrap();
        assert_eq!(reservations.complete(UploadSubmissionReceipt { fence: 9, ..third }), Err(UploadReservationError::Fence));
        assert_eq!(reservations.complete(third), Ok(192));
        drop(later);
        assert_eq!(reservations.0.lock().unwrap().reclaimed, 256);
    }

    #[test]
    fn dropped_reservation_cancels_exactly_it_and_signal_failure_poisons_reuse() {
        let reservations = UploadReservations::new(47, 2, 256, 64);
        let abandoned = reservations.reserve(0, 64).unwrap();
        let stale = abandoned.token.unwrap();
        drop(abandoned);
        assert_eq!(reservations.0.lock().unwrap().reclaimed, 64);
        assert_eq!(reservations.0.lock().unwrap().cancel(stale), Err(UploadReservationError::Missing));

        let poisoned = reservations.reserve(0, 64).unwrap();
        poisoned.poison().unwrap();
        let ledger = reservations.0.lock().unwrap();
        assert!(ledger.poisoned);
        assert_eq!(ledger.slots[0], Some(2), "poison released an allocator with untracked GPU work");
        drop(ledger);
        assert!(matches!(reservations.reserve(1, 64), Err(UploadReservationError::Poisoned)));
    }

    #[test]
    fn upload_reservation_bounds_generation_and_monotonic_overflow_fail_closed() {
        let mut ledger = UploadReservationLedger::new(53, 1, 256, 64);
        assert!(matches!(ledger.reserve(52, 0, 64), Err(UploadReservationError::StaleGeneration)));
        assert!(matches!(ledger.reserve(53, 1, 0), Err(UploadReservationError::Slot)));
        assert!(matches!(ledger.reserve(53, 0, 0), Err(UploadReservationError::Empty)));
        assert!(matches!(ledger.reserve(53, 0, 257), Err(UploadReservationError::TooLarge)));

        ledger.cursor = u64::MAX - 31;
        assert!(matches!(ledger.reserve(53, 0, 64), Err(UploadReservationError::Exhausted)));
        ledger.cursor = 0;
        ledger.next_serial = u64::MAX;
        assert!(matches!(ledger.reserve(53, 0, 64), Err(UploadReservationError::Exhausted)));
        assert!(ledger.records.is_empty());
        assert!(ledger.order.is_empty());
        assert!(ledger.slots[0].is_none());
    }

    #[test]
    fn a_failed_title_queue_wait_rolls_back_only_its_reservation() {
        let mut waited = FxHashMap::from_iter([(1, 4), (2, 7)]);
        assert!(reserve_queue_wait(&mut waited, 1, 5));
        rollback_queue_wait(&mut waited, 1, 5, Some(4));
        assert_eq!(waited, FxHashMap::from_iter([(1, 4), (2, 7)]));
        rollback_queue_wait(&mut waited, 1, 9, None);
        assert_eq!(waited.get(&1), Some(&4), "a stale failure changed a newer reservation");
    }

    #[test]
    fn quarantine_uses_installed_retirement_marks_and_fails_closed_without_one() {
        use std::sync::Arc;
        let own = Arc::new(1u32);
        let retired = Arc::new(2u32);
        let marks = initial_quarantine_marks((own.clone(), 3), &[(retired.clone(), 4)], false).unwrap();
        assert_eq!(marks.iter().map(|(_, value)| *value).collect::<Vec<_>>(), [3, 4]);
        drop(retired);
        assert_eq!(Arc::strong_count(&marks[1].0), 1, "batch did not own the retirement marker");
        assert!(initial_quarantine_marks((own, 3), &[(marks[1].0.clone(), 4)], true).is_none(), "Preparing/Failed reused tiles without a completion source");
    }

    /// A block freed on one thread while another allocates: its address goes back to the
    /// windows only once it is decommitted (released first, the decommit came after another
    /// thread had the same address again: that thread's writes faulted).
    #[test]
    fn a_freed_block_is_decommitted_before_its_address_is_reused() {
        if crate::gpu::uma::get().is_none_or(|u| !u.has_sparse()) {
            return;
        }
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| unsafe {
                    for i in 0..2000u64 {
                        let Some((p, n)) = alloc(TILE, TILE) else { continue };
                        std::ptr::write_volatile(p as *mut u64, i);
                        std::ptr::write_volatile((p + n - 8) as *mut u64, i);
                        assert_eq!(std::ptr::read_volatile(p as *const u64), i);
                        assert!(free(p));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }

    /// `f` on another thread while this one holds the sparse state (as `before_execute` does
    /// around its D3D12 calls, which can wait for that thread): did `f` finish without waiting
    /// for the state? The thread runs before the state is taken, so its start is not timed.
    fn returns_while_the_state_is_held(f: impl FnOnce() + Send + 'static) -> bool {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            let _ = ready_tx.send(());
            let _ = go_rx.recv();
            f();
            let _ = tx.send(());
        });
        ready_rx.recv().unwrap();
        let held = STATE.lock().unwrap();
        go_tx.send(()).unwrap();
        let done = rx.recv_timeout(std::time::Duration::from_secs(5));
        drop(held);
        let joined = t.join();
        assert!(joined.is_ok(), "the thread panicked: {joined:?}");
        done.is_ok()
    }

    /// A placed buffer released while another thread holds the sparse state: the release does
    /// not wait for the state, and the next `lock_state` lets the buffer go.
    #[test]
    fn a_buffer_released_while_the_state_is_held_does_not_wait_for_it() {
        const RES: usize = 0x7ff0_dead_0000;
        assert!(returns_while_the_state_is_held(|| forget_buffer(RES)), "forget_buffer waited for the sparse state's lock");
        drop(lock_state());
        assert!(!FORGOTTEN.lock().unwrap().contains(&RES));
    }


    /// Freeing the last tile in use of a pool and taking a tile again takes the retired pool
    /// back, not a new heap each time (#913: 288 pools of 64 MB, 18 GB asked of a 12 GB card).
    #[test]
    fn a_pool_retired_a_moment_ago_is_taken_back() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");
            let mut s = State {
                generation: 1,
                mapping_epoch: 0,
                coordination: SparseCoordination::new(1, 1).unwrap(),
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                windows: Vec::new(),
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                parked: FxHashMap::default(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                uploads_ordered: true,
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: None,
                scratch_runs: Vec::new(),
                retry_pages: Vec::new(),
            };

            let (first, reused) = s.take_tile().expect("a tile from a new pool");
            assert!(!reused, "a new pool's tile needs no clearing");
            s.free.push(first);
            s.retire_free_pools();
            assert_eq!(s.retired.len(), 1, "an all-free pool is retired");
            for round in 0..8 {
                let (tile, reused) = s.take_tile().expect("a tile");
                assert!(reused, "round {round}: a tile of a retired pool is cleared before use");
                s.free.push(tile);
                s.retire_free_pools();
                assert_eq!(s.retired.len(), 1, "round {round}: the retired pool was taken back, not a new one made");
                assert_eq!(s.pools.iter().filter(|p| p.is_some()).count(), 0, "round {round}");
            }
        }
    }

    /// A title frees its tiles in runs spread over all its pools: the pools of the freed runs go
    /// back. With pools of 64 MiB none of these did (every pool kept the tiles still in use).
    #[test]
    fn pools_of_freed_runs_go_back() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            // 192 MiB of tiles, taken in order as a title's first commits take them
            const TILES: usize = 3072;
            let tiles: Vec<(u32, u32)> = (0..TILES).map(|_| s.take_tile().expect("a tile").0).collect();
            let made = s.pools.iter().filter(|p| p.is_some()).count();
            assert_eq!(made * POOL_TILES as usize, TILES, "the tiles fill whole pools");
            // of every 64 MiB the title keeps the first 4 MiB and frees the rest
            let (mut kept, mut freed) = (0, 0);
            for (i, &t) in tiles.iter().enumerate() {
                if i % 1024 < 64 {
                    kept += 1;
                } else {
                    s.free.push(t);
                    freed += 1;
                }
            }
            s.retire_free_pools();
            let left = s.pools.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize;
            assert_eq!(s.free.len() + s.retired.len() * POOL_TILES as usize, freed, "every freed tile is free or its pool retired");
            assert!(left <= kept * 2, "{left} tiles of video memory stay in pools for {kept} tiles in use ({} MiB for {} MiB)", left * TILE >> 20, kept * TILE >> 20);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn aligned_tile_addresses_spread_across_hash_buckets() {
        use std::hash::BuildHasher;

        let hasher = BuildHasherDefault::<FxHasher>::default();
        let mut used = [false; 256];
        for tile in 0..used.len() {
            let address = (1usize << 32) + tile * TILE;
            let bucket = hasher.hash_one(address) as usize & (used.len() - 1);
            used[bucket] = true;
        }
        let occupied = used.iter().filter(|&&bucket| bucket).count();
        assert!(occupied >= used.len() / 2, "256 aligned addresses used {occupied} buckets");
    }

    /// Only the 64-byte chunks the CPU changed are sent again – the rest of the page may hold
    /// what the GPU wrote.
    #[test]
    fn only_changed_chunks_are_sent() {
        let mut page = vec![0u8; PAGE];
        let mut sent = ZERO_HASHES;
        assert!(changed_chunks(&page, &mut sent).is_empty(), "a zero page is what the GPU has");
        page[10] = 1;
        page[200] = 2;
        page[255] = 3;
        page[4000] = 4;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(0, 64), (192, 64), (3968, 64)]);
        assert!(changed_chunks(&page, &mut sent).is_empty(), "nothing changed since");
        page[256] = 5; // next to the chunk at 192: one run
        page[200] = 9;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(192, 128)]);
    }

    #[test]
    fn clean_chunks_take_gpu_copy_back_bytes() {
        let mut page = vec![0x11u8; PAGE];
        let mut sent = Box::new(ZERO_HASHES);
        changed_chunks(&page, &mut sent);
        let mut slot = Some(sent);
        let gpu = vec![0x22u8; 2 * CHUNK];

        land_page(&mut slot, &mut page, CHUNK, &gpu, false);

        assert!(page[..CHUNK].iter().all(|&byte| byte == 0x11));
        assert!(page[CHUNK..3 * CHUNK].iter().all(|&byte| byte == 0x22));
        assert!(changed_chunks(&page, slot.as_deref_mut().unwrap()).is_empty(), "landed bytes were left dirty");
    }

    #[test]
    fn dirty_chunks_survive_gpu_copy_back_and_upload_next() {
        let mut page = vec![0x11u8; PAGE];
        let mut sent = Box::new(ZERO_HASHES);
        changed_chunks(&page, &mut sent);
        let old_dirty_hash = sent[1];
        let mut slot = Some(sent);
        page[CHUNK..2 * CHUNK].fill(0x33);
        let gpu = vec![0x22u8; 2 * CHUNK];

        land_page(&mut slot, &mut page, CHUNK, &gpu, false);

        assert!(page[CHUNK..2 * CHUNK].iter().all(|&byte| byte == 0x33), "copy-back overwrote newer CPU bytes");
        assert_eq!(slot.as_ref().unwrap()[1], old_dirty_hash, "dirty chunk lost its last-sent hash");
        assert!(page[2 * CHUNK..3 * CHUNK].iter().all(|&byte| byte == 0x22), "clean neighbour did not land");
        assert_eq!(changed_chunks(&page, slot.as_deref_mut().unwrap()), vec![(CHUNK, CHUNK)]);
    }

    /// A wait's cells and the write after it share a chunk (#3093, the test title's Wait32 / Wait64
    /// at +0x300 / +0x308 and Write32 at +0x310): the CPU wrote the cells, so the chunk changed,
    /// and the GPU's own write still lands; the chunk keeps its old hash and uploads next.
    #[test]
    fn gpu_written_bytes_land_in_a_chunk_the_cpu_changed() {
        let mut page = vec![0u8; PAGE];
        let mut sent = Box::new(ZERO_HASHES);
        changed_chunks(&page, &mut sent);
        let old_hash = sent[0x300 / CHUNK];
        let mut slot = Some(sent);
        page[0x300..0x304].copy_from_slice(&0xDEAD_00AAu32.to_le_bytes());
        page[0x308..0x310].copy_from_slice(&0x0000_0003_0000_0000u64.to_le_bytes());

        land_page(&mut slot, &mut page, 0x310, &0x55AA_55AAu32.to_le_bytes(), true);

        assert_eq!(page[0x310..0x314], 0x55AA_55AAu32.to_le_bytes(), "the GPU's write was dropped");
        assert_eq!(page[0x300..0x304], 0xDEAD_00AAu32.to_le_bytes(), "the CPU's cells were overwritten");
        assert_eq!(page[0x308..0x310], 0x0000_0003_0000_0000u64.to_le_bytes());
        assert_eq!(slot.as_ref().unwrap()[0x300 / CHUNK], old_hash, "the changed chunk lost its last-sent hash");
        assert_eq!(changed_chunks(&page, slot.as_deref_mut().unwrap()), vec![(0x300, CHUNK)]);
    }

    /// `scan_page` leaves published hashes unchanged, skips allocation for zero pages, and
    /// coalesces adjacent dirty chunks.
    #[test]
    fn scan_page_plans_without_publishing_and_coalesces() {
        let mut page = vec![0u8; PAGE];
        let slot: Option<Box<[u64; CHUNKS]>> = None;
        let mut out = Vec::new();
        scan_page(&page, 0x1_0000, &slot, &mut out);
        assert!(slot.is_none(), "all-zero page must not allocate hash box");
        assert!(out.is_empty());

        page[PAGE - 64] = 0xAA;
        scan_page(&page, 0x1_0000, &slot, &mut out);
        assert!(slot.is_none(), "planning must not publish a last-sent hash");
        assert_eq!(out, vec![(0x1_0FC0, 64)]);

        // Adjacent chunk at the start of the next page coalesces into a single run.
        let mut next = vec![0u8; PAGE];
        next[0] = 0xBB;
        let next_slot: Option<Box<[u64; CHUNKS]>> = None;
        scan_page(&next, 0x1_1000, &next_slot, &mut out);
        assert_eq!(out, vec![(0x1_0FC0, 128)]);
    }

    #[test]
    fn hash_plan_is_published_only_when_applied() {
        let base = 0x2_0000;
        let mut mapped = FxHashMap::default();
        mapped.insert(base, TileEntry::with_holes((0, 0), 0));
        let mut prefix = Vec::new();
        append_hash_updates(base + CHUNK, &[1; CHUNK], &mut prefix);
        let mut suffix = Vec::new();
        append_hash_updates(base + PAGE + 2 * CHUNK, &[2; CHUNK], &mut suffix);
        assert!(mapped[&base].sent[0].is_none());
        assert!(mapped[&base].sent[1].is_none());
        publish_hash_updates(&mut mapped, &prefix);
        assert_eq!(mapped[&base].sent[0].as_ref().unwrap()[1], chunk_hash(&[1; CHUNK]));
        assert!(mapped[&base].sent[1].is_none(), "an unacknowledged suffix was published");
        publish_hash_updates(&mut mapped, &suffix);
        assert_eq!(mapped[&base].sent[1].as_ref().unwrap()[2], chunk_hash(&[2; CHUNK]));
    }

    #[test]
    fn cpu_write_after_planning_is_dirty_after_receipt_publication() {
        let base = 0x3_0000;
        let mut mapped = FxHashMap::default();
        mapped.insert(base, TileEntry::with_holes((0, 0), 0));
        let mut page = vec![0u8; PAGE];
        page[0] = 1;
        let staged = page[..CHUNK].to_vec();
        let mut planned = Vec::new();
        append_hash_updates(base, &staged, &mut planned);
        page[0] = 2;
        publish_hash_updates(&mut mapped, &planned);

        let mut runs = Vec::new();
        scan_page(&page, base, &mapped[&base].sent[0], &mut runs);
        assert_eq!(runs, [(base, CHUNK)]);
    }

    #[test]
    fn batch_hash_plan_is_bounded_by_staging() {
        assert_eq!(MAX_BATCH_HASH_UPDATES, STAGING / CHUNK);
        assert_eq!(MAX_BATCH_HASH_UPDATES * std::mem::size_of::<HashUpdate>(), STAGING / 4);
        assert!(!batch_hash_plan_would_overflow(MAX_BATCH_HASH_UPDATES - 1, CHUNK));
        assert!(batch_hash_plan_would_overflow(MAX_BATCH_HASH_UPDATES, CHUNK));
        assert!(batch_hash_plan_would_overflow(MAX_BATCH_HASH_UPDATES / 2 + 1, STAGING / 2));
    }

    #[test]
    fn prepared_dirty_batch_keeps_exact_hash_plan_and_rollback_point() {
        let mut batch = PreparedDirtyBatch::new(3, 0x12_3400);
        let bytes = [0xA5; 2 * CHUNK];
        batch.append_hashes(0x20_0000, &bytes);
        assert_eq!(batch.slot, 3);
        assert_eq!(batch.ring_start, 0x12_3400);
        assert_eq!(batch.updates.len(), 2);
        assert_eq!(batch.updates[0], HashUpdate { at: 0x20_0000, hash: chunk_hash(&bytes[..CHUNK]) });
        assert_eq!(batch.updates[1], HashUpdate { at: 0x20_0000 + CHUNK, hash: chunk_hash(&bytes[CHUNK..]) });
        assert!(!batch.hash_plan_would_overflow(CHUNK));

        let storage = batch.updates.as_ptr();
        let capacity = batch.updates.capacity();
        for index in 0..3 {
            batch.clear_after_submit();
            batch = batch.reopen(4 + index, 0x20_0000 + index as u64 * PAGE as u64);
            batch.append_hashes(0x30_0000, &bytes);
            assert_eq!(batch.updates.as_ptr(), storage, "batch {index} replaced its hash-plan allocation");
            assert_eq!(batch.updates.capacity(), capacity);
        }
    }

    #[test]
    fn prepared_dirty_copy_splits_at_windows_and_ring_wrap() {
        let window = 4 * PAGE;
        let base = 0x10_0000;
        let first = prepare_dirty_copy(base + window - CHUNK, 2 * CHUNK, base, window, 0, 8 * PAGE);
        assert_eq!(first, PreparedDirtyCopy {
            ring_start: 0,
            staging_offset: 0,
            window_offset: window - CHUNK,
            bytes: CHUNK,
        });
        let second = prepare_dirty_copy(base + window, CHUNK, base + window, window, CHUNK as u64, 8 * PAGE);
        assert_eq!(second.window_offset, 0);
        assert_eq!(second.bytes, CHUNK);

        let cap = 8 * PAGE;
        let wrapped = prepare_dirty_copy(base, PAGE, base, window, (cap - 512) as u64, cap);
        assert_eq!(wrapped.ring_start, cap as u64);
        assert_eq!(wrapped.staging_offset, 0);
        assert_eq!(wrapped.bytes, PAGE);
    }

    /// Ring-buffer planner keeps every chunk contiguous, 64-byte aligned, <= cap/2, and splits
    /// cleanly across the end of the ring buffer without wrapping mid-copy.
    #[test]
    fn ring_planner_handles_wrap_and_large_chunks() {
        let cap = 64 << 20;
        // Normal aligned chunk
        let (a, n) = plan_ring_chunk(0, 4096, cap);
        assert_eq!((a, n), (0, 4096));
        // Unaligned cursor aligns up to 64
        let (a, n) = plan_ring_chunk(17, 256, cap);
        assert_eq!((a, n), (64, 256));
        // Oversized chunk capped at cap / 2
        let (a, n) = plan_ring_chunk(0, cap * 2, cap);
        assert_eq!((a, n), (0, cap / 2));
        // Near end of ring with >= 4 KiB left: shrinks to fill tail
        let near = (cap - 8192) as u64;
        let (a, n) = plan_ring_chunk(near, 16384, cap);
        assert_eq!((a, n), (near, 8192));
        assert_eq!((a as usize % cap) + n, cap);
        // Tiny tail (< 4 KiB): wraps cleanly to 0 of next cycle
        let tiny = (cap - 512) as u64;
        let (a, n) = plan_ring_chunk(tiny, 4096, cap);
        assert_eq!((a, n), (cap as u64, 4096));
        assert_eq!(a as usize % cap, 0);
    }

    /// `query_tile_holes` marks reserved, free, and `PAGE_NOACCESS` pages as holes while clearing
    /// bits only for committed, accessible 4 KiB pages.
    #[test]
    fn query_tile_holes_detects_reserved_noaccess_and_committed_pages() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            // SAFETY: Reserving 2 tiles (128 KiB) with VirtualAlloc returns a 64 KiB-aligned VA.
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);

            // Initially every page in both reserved tiles is a hole.
            assert_eq!(query_tile_holes(base), 0xFFFF);
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 4..8 as PAGE_READWRITE and page 9 as PAGE_NOACCESS.
            let p4 = VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p4.is_null());
            let p9 = VirtualAlloc((base + 9 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_NOACCESS);
            assert!(!p9.is_null());

            // Only bits 4..8 are cleared; reserved [0..4), PAGE_NOACCESS [9], and reserved [10..16) remain holes.
            assert_eq!(query_tile_holes(base), 0xFFFF & !(0b1111u16 << 4));
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 0..4 and confirm bits 0..4 clear too.
            let p0 = VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p0.is_null());
            assert_eq!(query_tile_holes(base), 0xFF00);

            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
            // Released (MEM_FREE) memory reports all holes.
            assert_eq!(query_tile_holes(base), 0xFFFF);
        }
    }

    /// Mapping tiles for a placed buffer over partially-committed/reserved memory must preserve
    /// uncommitted hole bits so `readback_ranges` does not access-violate,
    /// and later commits clear those hole bits so subsequent writes scan cleanly.
    #[test]
    fn ensure_mapped_with_query_holes_skips_uncommitted_pages_on_readback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");

            // Reserve 2 tiles; commit only [base + 0x4000, base + 0x8000) to simulate a
            // non-64KiB-aligned placed buffer with uncommitted prefix [base, base + 0x4000) and
            // uncommitted second tile [base + TILE, base + 2*TILE).
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());

            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                generation: 1,
                mapping_epoch: 0,
                coordination: SparseCoordination::new(1, 1).unwrap(),
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                uploads_ordered: true,
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: None,
                scratch_runs: Vec::new(),
                retry_pages: Vec::new(),
            };

            // Mirrors `place_buffer`: map both tiles probing real OS page commitment.
            s.ensure_mapped(base, 2 * TILE, |t| {
                let holes = query_tile_holes(t);
                (holes, !holes)
            });
            assert_eq!(s.mapped[&base].holes, 0xFF0F, "uncommitted prefix and suffix pages must stay holes");
            assert_eq!(s.mapped[&(base + TILE)].holes, 0xFFFF, "fully reserved tile must have all 16 holes set");

            // Reading back the full 2-tile span must not fault on uncommitted pages.
            s.readback_ranges(&[(base, 2 * TILE)], false);

            // Now incrementally commit [base, base + 4*PAGE) and clear its hole bits (as `committed` does).
            assert!(!VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let entry = s.mapped.get_mut(&base).unwrap();
            for i in 0..4 {
                entry.holes &= !(1u16 << i);
            }
            assert_eq!(entry.holes, 0xFF00);

            // Writing to the newly committed page is detected by `scan_page`.
            *(base as *mut u8) = 0x5A;
            let mut dirty = Vec::new();
            let page0 = std::slice::from_raw_parts(base as *const u8, PAGE);
            scan_page(page0, base, &entry.sent[0], &mut dirty);
            assert_eq!(dirty, vec![(base, CHUNK)]);

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A tile decommitted and committed again before the next submit gets its pool tile back
    /// with no new tile mapping (cleared instead). One that is not is unmapped and freed by
    /// `flush_parked` (before a submit), or when the free list runs out; a placed buffer's
    /// release does not free it while it is parked.
    #[test]
    fn a_tile_decommitted_and_committed_again_keeps_its_video_memory() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                generation: 1,
                mapping_epoch: 0,
                coordination: SparseCoordination::new(1, 1).unwrap(),
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                uploads_ordered: true,
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: None,
                scratch_runs: Vec::new(),
                retry_pages: Vec::new(),
            };

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            let v = s.value;
            assert!(v > 0, "a new tile is mapped");
            mark_protection(&mut s.mapped, base + PAGE, PAGE, false);
            assert_eq!(s.mapped[&base].not_rw, 1 << 1);

            // decommitted: parked, still mapped, nothing sent to the GPU
            s.decommit_range(base, TILE);
            assert!(!s.mapped.contains_key(&base));
            assert_eq!(s.parked.get(&base), Some(&tile));
            assert!(!s.free.contains(&tile));
            assert_eq!(s.value, v, "no tile mapping changed");
            s.release_if_unused(tile);
            assert!(!s.free.contains(&tile), "a parked tile is not free");

            // committed again: the same tile, cleared, no new mapping
            s.to_zero.clear();
            let gen = mapped_gen();
            s.commit_range(base, TILE);
            assert_eq!(s.mapped[&base].pool, tile);
            assert_eq!(s.mapped[&base].holes, 0);
            assert_eq!(s.mapped[&base].not_rw, 0, "recommitted pages use their commit protection");
            assert!(s.parked.is_empty());
            assert_eq!(s.to_zero, vec![(base, TILE)]);
            assert_eq!(s.value, v, "no tile mapping changed");
            assert!(mapped_gen() > gen, "readers of `mapped` see it changed");
            mark_protection(&mut s.mapped, base + 2 * PAGE, PAGE, false);
            s.decommit_range(base + 2 * PAGE, PAGE);
            s.commit_range(base + 2 * PAGE, PAGE);
            assert_eq!(s.mapped[&base].not_rw, 0, "a partially recommitted page is writable again");
            s.decommit_range(base + 3 * PAGE, PAGE);
            s.commit_range_with_protection(base + 3 * PAGE, PAGE, false);
            assert_eq!(s.mapped[&base].not_rw, 1 << 3, "commit and protection publish as one record update");
            s.commit_range(base + 3 * PAGE, PAGE);

            // decommitted and flushed (as before a submit): unmapped and free once unmap fence completes
            s.decommit_range(base, TILE);
            assert!(s.flush_parked());
            assert!(s.parked.is_empty());
            assert!(s.value > v, "the unmapping is sent");
            assert!(!s.flush_parked(), "nothing left to flush");
            s.wait_for(s.value);
            assert!(s.drain_quarantined());
            assert!(s.free.contains(&tile));

            // no free tile left: the parked one is flushed and reused once its unmap fence finishes
            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());
            s.wait_for(s.value);
            s.commit_range(base + TILE, TILE);
            assert!(s.parked.is_empty());
            assert_eq!(s.mapped[&(base + TILE)].pool, tile);
            assert!(!s.mapped.contains_key(&base));

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    struct SignalOnDrop(ID3D12Fence, u64);

    impl Drop for SignalOnDrop {
        fn drop(&mut self) {
            unsafe {
                let _ = self.0.Signal(self.1);
            }
        }
    }

    unsafe fn test_state_with_reserved(tiles: usize) -> (usize, State) {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::MEM_RESERVE;

        let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
            D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
        }
        let dev = dev.expect("D3D12 device");
        let base = VirtualAlloc(std::ptr::null(), tiles * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
        assert!(base != 0 && base % TILE == 0);
        let mut reserved: Option<ID3D12Resource> = None;
        dev.CreateReservedResource(&buffer_desc((tiles * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
            .expect("CreateReservedResource");
        let reserved = reserved.expect("reserved buffer");
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
        let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
        let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
        let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
        let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
        rb_list.Close().expect("rb_list close");

        let s = State {
            generation: 1,
            mapping_epoch: 0,
            coordination: SparseCoordination::new(1, 1).unwrap(),
            dev,
            queue,
            fence,
            value: 0,
            event,
            rb_fence,
            rb_value: 0,
            windows: vec![(base, reserved)],
            pools: Vec::new(),
            free: Vec::new(),
            retired: Vec::new(),
            mapped: FxHashMap::default(),
            pinned: FxHashMap::default(),
            parked: FxHashMap::default(),
            quarantined: VecDeque::new(),
            queue_fences: FxHashMap::default(),
            buffers: FxHashMap::default(),
            to_zero: Vec::new(),
            staging: None,
            ring_write: 0,
            ring_submitted: 0,
            ring_free: 0,
            inflight: VecDeque::new(),
            uploads_ordered: true,
            zero_buf: None,
            slots: Vec::new(),
            next_slot: 0,
            readback: None,
            rb_alloc,
            rb_list,
            rb_last_fence: 0,
            waited: FxHashMap::default(),
            runs: None,
            scratch_runs: Vec::new(),
            retry_pages: Vec::new(),
        };
        (base, s)
    }

    unsafe fn add_upload_slot(s: &mut State, fence_val: u64, close: bool) {
        let alloc = s.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("upload allocator");
        let list: ID3D12GraphicsCommandList = s.dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .expect("upload list");
        if close {
            list.Close().expect("upload list close");
        }
        s.slots.push(UploadSlot { alloc, list, fence_val });
    }

    #[test]
    fn begin_slot_skips_a_busy_head_for_a_ready_slot() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            add_upload_slot(&mut s, 9, true);
            add_upload_slot(&mut s, 0, true);

            assert!(matches!(s.begin_slot(), BeginSlot::Ready(1)));
            assert_eq!(s.next_slot, 0);
            s.slots[1].list.Close().expect("selected list close");

            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn begin_slot_waits_for_the_earliest_busy_slot_without_advancing() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            add_upload_slot(&mut s, 9, true);
            add_upload_slot(&mut s, 7, true);
            add_upload_slot(&mut s, 8, true);
            let ticket = match s.begin_slot() {
                BeginSlot::Wait(ticket) => ticket,
                _ => panic!("all busy slots did not return a retirement ticket"),
            };

            assert_eq!(s.next_slot, 0);
            assert_eq!(ticket.target, 7);
            assert!(matches!(ticket.kind, WaitKind::Slot { index: 1, next_slot: 0, fence: 7 }));
            assert!(s.wait_ticket_owns_upload(&ticket));
            s.slots[1].fence_val = 10;
            assert!(!s.wait_ticket_owns_upload(&ticket), "a changed selected slot accepted its old ticket");

            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn begin_slot_reset_failure_does_not_open_or_advance_a_slot() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            add_upload_slot(&mut s, 0, false);

            assert!(matches!(s.begin_slot(), BeginSlot::RetryRequired));
            assert_eq!(s.next_slot, 0);

            s.slots[0].list.Close().expect("failed selected list close");
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn upload_wait_releases_and_reacquires_its_ownership_gate() {
        let (acquired_tx, acquired_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (mut guard, mut timing) = UploadGateTiming::acquire(false);
        let worker = std::thread::spawn(move || {
            let _crossing_uploader = UPLOAD_GATE.lock().unwrap();
            acquired_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        let released = timing.with_released(&mut guard, || {
            let acquired = acquired_rx.recv_timeout(std::time::Duration::from_secs(1)).is_ok();
            release_tx.send(()).unwrap();
            worker.join().unwrap();
            acquired
        });
        assert!(released, "another uploader could not cross the fence wait");
        assert!(UPLOAD_GATE.try_lock().is_err(), "the process upload gate was not reacquired");
        let times = timing.finish();
        assert_eq!((times.wait_us, times.hold_us, times.released_us), (0, 0, 0), "disabled timing read the clock");
        drop(guard);
    }

    #[test]
    fn replacement_waits_for_every_upload_generation_reader() {
        let _placement = PLACEMENT.lock().unwrap();
        let lower_target = UPLOAD_STATE_LIFETIME.read().unwrap();
        let higher_target = UPLOAD_STATE_LIFETIME.read().unwrap();
        let (acquired_tx, acquired_rx) = std::sync::mpsc::channel();
        let replacement = std::thread::spawn(move || {
            let _exclusive = UPLOAD_STATE_LIFETIME.write().unwrap();
            acquired_tx.send(()).unwrap();
        });

        assert!(acquired_rx.recv_timeout(std::time::Duration::from_millis(50)).is_err());
        drop(lower_target);
        assert!(acquired_rx.recv_timeout(std::time::Duration::from_millis(50)).is_err(),
            "replacement passed the lower target while a higher upload was still in flight");
        drop(higher_target);
        acquired_rx.recv_timeout(std::time::Duration::from_secs(1)).expect("replacement stayed blocked after every upload reader retired");
        replacement.join().unwrap();
    }

    #[test]
    fn a_stalled_upload_ring_yields_without_publishing_hashes() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE};

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            assert_eq!(VirtualAlloc(base as *mut c_void, TILE, MEM_COMMIT, PAGE_READWRITE) as usize, base);
            std::ptr::write_bytes(base as *mut u8, 0x5a, PAGE);
            s.commit_range(base, TILE);
            s.wait_for(s.value);
            let page = std::slice::from_raw_parts(base as *const u8, PAGE);
            let mut sent = Box::new(ZERO_HASHES);
            changed_chunks(page, &mut sent);
            s.mapped.get_mut(&base).unwrap().sent[0] = Some(sent);
            let before = *s.mapped[&base].sent[0].as_ref().unwrap().clone();

            s.ring_write = STAGING as u64;
            s.ring_submitted = STAGING as u64;
            s.ring_free = 0;
            s.inflight.push_back((s.value + 1, STAGING as u64));
            let ticket = match s.upload_dirty_with_timeout(&[(base + CHUNK, 2 * CHUNK)], 20) {
                UploadStep::Wait(ticket) => ticket,
                UploadStep::Done(_) => panic!("a full ring did not return its retirement ticket"),
            };
            assert_eq!(ticket.target, s.value + 1);
            assert!(s.wait_ticket_valid(&ticket));
            s.mapping_epoch += 1;
            assert!(!s.wait_ticket_valid(&ticket), "a remapped destination accepted an old upload ticket");
            assert!(s.wait_ticket_generation(&ticket), "a remap invalidated the live state generation");
            let after = s.mapped[&base].sent[0].as_ref().unwrap();
            assert_eq!(after.as_ref(), &before, "a failed upload changed the published hash state");

            let alloc = s.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = s.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            list.Close().unwrap();
            s.slots.push(UploadSlot { alloc, list, fence_val: s.value + 1 });
            s.next_slot = 0;
            let before_slot = s.next_slot;
            let ticket = match s.begin_slot() {
                BeginSlot::Wait(ticket) => ticket,
                _ => panic!("a busy slot did not return its retirement ticket"),
            };
            assert_eq!(s.next_slot, before_slot, "a busy slot advanced before its fence completed");
            assert!(matches!(ticket.kind, WaitKind::Slot { index: 0, next_slot: 0, .. }));
            assert!(s.wait_ticket_owns_upload(&ticket));
            s.slots[0].fence_val += 1;
            assert!(!s.wait_ticket_owns_upload(&ticket), "a reused slot accepted its predecessor ticket");
            assert!(s.wait_ticket_generation(&ticket), "safe slot reuse invalidated the live generation");
            s.generation += 1;
            assert!(!s.wait_ticket_owns_upload(&ticket), "a replacement state accepted an old upload ticket");
            s.uploads_ordered = false;
            assert!(matches!(
                s.upload_dirty_with_timeout(&[(base + CHUNK, CHUNK)], 20),
                UploadStep::Done(PreparationOutcome::ExecutedUntracked)
            ));

            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn placement_create_is_unlocked_drains_releases_and_rejects_a_new_generation() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE};

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            assert_eq!(VirtualAlloc(base as *mut c_void, 2 * TILE, MEM_COMMIT, PAGE_READWRITE) as usize, base);
            s.commit_range(base, 2 * TILE);
            s.wait_for(s.value);
            let old_tile = s.mapped[&base].pool;
            let new_tile = s.mapped[&(base + TILE)].pool;

            let mut placed: Option<ID3D12Resource> = None;
            s.dev.CreateReservedResource(&buffer_desc(TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut placed)
                .expect("placed resource");
            let placed = placed.unwrap();
            let ptr = placed.as_raw() as usize;
            s.buffers.insert(ptr, vec![old_tile]);
            s.pinned.insert(old_tile, 1);

            let state = Mutex::new(Some(s));
            let forgotten = Mutex::new(Vec::new());
            let desc = buffer_desc(TILE as u64);
            let returned = placed.clone().into_raw();
            let mut unlocked = false;
            let result = place_buffer_in(
                &state,
                &forgotten,
                base + TILE,
                &desc,
                0,
                |_| {
                    unlocked = state.try_lock().is_ok();
                    forget_buffer_in(&state, &forgotten, ptr);
                    forgotten.lock().unwrap().push(ptr);
                    Some((0, returned))
                },
                |_| panic!("same-generation placement aborted"),
            );
            assert_eq!(result, Some((0, 0)));
            assert!(unlocked, "resource creation held sparse STATE");
            {
                let g = lock_state_in(&state, &forgotten);
                let s = g.as_ref().unwrap();
                assert_eq!(s.buffers.get(&ptr), Some(&vec![new_tile]));
                assert!(!s.pinned.contains_key(&old_tile));
                assert_eq!(s.pinned.get(&new_tile), Some(&1));
            }

            let original_fence = state.lock().unwrap().as_ref().unwrap().fence.clone();
            let replacement = state.lock().unwrap().as_ref().unwrap().dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let returned2 = placed.clone().into_raw();
            let mut aborted = false;
            let result = place_buffer_in(
                &state,
                &forgotten,
                base + TILE,
                &desc,
                0,
                |_| {
                    state.lock().unwrap().as_mut().unwrap().fence = replacement;
                    Some((0, returned2))
                },
                |raw| {
                    aborted = true;
                    drop(ID3D12Resource::from_raw(raw));
                },
            );
            assert!(result.is_none() && aborted, "a replacement state accepted the old generation's resource");
            state.lock().unwrap().as_mut().unwrap().fence = original_fence;

            drop(ID3D12Resource::from_raw(returned));
            let s = state.lock().unwrap().take().unwrap();
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// `flush_parked` must keep unmapped pool tiles out of `free` (and `take_tile` must not reuse
    /// them) while the sparse queue's unmap fence is still in flight on the GPU.
    #[test]
    fn flush_parked_does_not_free_or_reuse_tiles_while_unmap_is_in_flight() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);

            let gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate fence");
            let _unblock = SignalOnDrop(gate.clone(), 1);
            s.queue.Wait(&gate, 1).expect("gate sparse queue");

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());
            assert!(!s.free.contains(&tile), "unmapped tile must remain quarantined until unmap fence completes");

            // While the unmap is still blocked on `gate`, committing another tile must not steal `tile`.
            s.commit_range(base + TILE, TILE);
            assert_ne!(s.mapped[&(base + TILE)].pool, tile, "in-flight unmapped tile must not be reused before GPU completion");

            gate.Signal(1).expect("release sparse queue");
            s.wait_for(s.value);
            assert!(s.drain_quarantined(), "completed unmap batch should drain after fence completion");
            assert!(s.free.contains(&tile), "unmapped tile must return to free list once unmap fence completes");
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// `flush_parked` must keep unmapped pool tiles quarantined until every active title queue has
    /// finished its prior work, even after `s.queue` itself finishes unmapping.
    #[test]
    fn flush_parked_quarantines_tiles_until_title_queues_finish() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            let title_q: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);

            let title_gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("title gate fence");
            let _unblock = SignalOnDrop(title_gate.clone(), 1);
            title_q.Wait(&title_gate, 1).expect("gate title queue");

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked_for(std::slice::from_ref(&title_q), &[], false));

            // Even after `s.queue` finishes `unmap_tiles`, `title_q` is still gated behind `title_gate`.
            s.wait_for(s.value);
            assert!(!s.drain_quarantined(), "tile must remain quarantined while title queue has work in flight");
            assert!(!s.free.contains(&tile));

            s.commit_range(base + TILE, TILE);
            assert_ne!(s.mapped[&(base + TILE)].pool, tile, "must not reuse tile still in use by title queue");

            title_gate.Signal(1).expect("release title queue");
            let start = std::time::Instant::now();
            while !s.drain_quarantined() && start.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            assert!(s.free.contains(&tile), "tile must return to free list after title queue completes");
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// Sub-tile decommits (two 32 KiB halves, sixteen 4 KiB pages, or decommitting the last
    /// remaining committed page) must unmap the tile and return unpinned pool tiles to `free`
    /// once `holes` reaches `0xFFFF`, pruning stale `to_zero` entries for unmapped tiles.
    #[test]
    fn decommit_sub_tile_reclaims_pool_tile_when_all_pages_become_holes() {
        let mut mapped: FxHashMap<usize, TileEntry> = FxHashMap::default();
        let mut pinned: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut free: Vec<(u32, u32)> = Vec::new();
        let mut to_zero: Vec<(usize, usize)> = Vec::new();

        // 1. Two 32 KiB decommits across a full 64 KiB tile.
        let t0 = 0x10_0000usize;
        mapped.insert(t0, TileEntry::with_holes((0, 10), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0, TILE / 2).is_empty());
        assert_eq!(mapped[&t0].holes, 0x00FF);
        assert_eq!(to_zero.len(), 8);
        assert!(free.is_empty());

        let unmapped = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0 + TILE / 2, TILE / 2);
        assert_eq!(unmapped, vec![t0]);
        assert!(!mapped.contains_key(&t0));
        assert_eq!(free, vec![(0, 10)]);
        assert!(to_zero.is_empty(), "stale to_zero entries for the unmapped tile must be pruned");

        // 2. Sixteen individual 4 KiB decommits in reverse order.
        let t1 = 0x20_0000usize;
        free.clear();
        mapped.insert(t1, TileEntry::with_holes((0, 11), 0));
        for i in (1..PAGES_PER_TILE).rev() {
            let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1 + i * PAGE, PAGE);
            assert!(u.is_empty(), "tile still has committed pages after step {i}");
        }
        assert!(free.is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1, PAGE);
        assert_eq!(u, vec![t1]);
        assert!(!mapped.contains_key(&t1));
        assert_eq!(free, vec![(0, 11)]);
        assert!(to_zero.is_empty());

        // 3. Pinned tile decommitted in two halves: unmapped from `mapped`, NOT returned to `free`.
        let t2 = 0x30_0000usize;
        free.clear();
        mapped.insert(t2, TileEntry::with_holes((0, 12), 0));
        pinned.insert((0, 12), 1);
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2, TILE / 2).is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2 + TILE / 2, TILE / 2);
        assert_eq!(u, vec![t2]);
        assert!(!mapped.contains_key(&t2));
        assert!(free.is_empty(), "pinned pool tile must not return to free list");
        assert!(to_zero.is_empty());

        // 4. Single-committed-page tile decommit reclaims tile without dropping another tile's to_zero.
        let t3 = 0x40_0000usize;
        let t4 = 0x50_0000usize;
        mapped.insert(t3, TileEntry::with_holes((0, 13), 0xFFFF & !(1u16 << 2)));
        mapped.insert(t4, TileEntry::with_holes((0, 14), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t4, PAGE).is_empty());
        assert_eq!(to_zero, vec![(t4, PAGE)]);
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t3 + 2 * PAGE, PAGE);
        assert_eq!(u, vec![t3]);
        assert_eq!(free, vec![(0, 13)]);
        assert_eq!(to_zero, vec![(t4, PAGE)], "unrelated mapped tile's to_zero must remain");
    }

    /// The pages in `early`, checking that its ranges are page-aligned, not empty, disjoint and
    /// merged (no two touch).
    fn early_pages(early: &BTreeMap<usize, usize>) -> std::collections::BTreeSet<usize> {
        let mut last_end = None;
        let mut pages = std::collections::BTreeSet::new();
        for (&a, &b) in early {
            assert!(a % PAGE == 0 && b % PAGE == 0 && a < b, "range {a:#x}..{b:#x}");
            assert!(last_end.is_none_or(|e| e < a), "range {a:#x}..{b:#x} touches the one before");
            last_end = Some(b);
            pages.extend((a..b).step_by(PAGE));
        }
        pages
    }

    /// xorshift64*: the same sequence on every run
    fn next(x: &mut u64) -> u64 {
        *x ^= *x >> 12;
        *x ^= *x << 25;
        *x ^= *x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Before the device exists, commits and decommits leave the pages a plain page set would
    /// hold, with the live path's rule (a range counts every page it touches): random ranges,
    /// unaligned too, over 300 pages.
    #[test]
    fn early_commits_are_the_pages_a_page_set_holds() {
        let base = 0x4_2000_0000usize;
        for seed in 1..=8u64 {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let (mut early, mut model) = (BTreeMap::new(), std::collections::BTreeSet::new());
            for _ in 0..4000 {
                let a = base + (next(&mut x) % (300 * PAGE as u64)) as usize;
                let n = match next(&mut x) % 4 {
                    0 => PAGE,
                    1 => 1 + (next(&mut x) % 64) as usize,
                    _ => 1 + (next(&mut x) % (24 * PAGE as u64)) as usize,
                };
                let (start, end) = page_span(a, n);
                let pages = (start..end).step_by(PAGE);
                if next(&mut x) % 3 == 0 {
                    early_remove(&mut early, start, end);
                    pages.for_each(|p| {
                        model.remove(&p);
                    });
                } else {
                    early_add(&mut early, start, end);
                    model.extend(pages);
                }
                assert_eq!(early_pages(&early), model, "seed {seed}");
            }
        }
    }

    /// The same pages as the list this replaced, for what that list handled exactly: one
    /// entry per committed page (`XMemMapPhysicalPagesScatter`, `home_again`) and page-aligned
    /// decommits of any length.
    #[test]
    fn early_commits_match_the_old_list_for_page_entries() {
        let base = 0x4_2000_0000usize;
        let mut x = 0x1234_5678_9ABC_DEF1u64;
        let (mut early, mut old): (BTreeMap<usize, usize>, Vec<(usize, usize)>) = (BTreeMap::new(), Vec::new());
        for _ in 0..20000 {
            let a = base + (next(&mut x) % 512) as usize * PAGE;
            if next(&mut x) % 2 == 0 {
                old.push((a, PAGE));
                early_add(&mut early, a, a + PAGE);
            } else {
                let n = (1 + next(&mut x) % 8) as usize * PAGE;
                old.retain(|&(y, m)| y + m <= a || y >= a + n);
                early_remove(&mut early, a, a + n);
            }
        }
        let old_pages: std::collections::BTreeSet<usize> = old.iter().map(|&(y, _)| y).collect();
        assert_eq!(early_pages(&early), old_pages);
    }

    /// A range committed in one call keeps its other pages when one page of it is decommitted
    /// (`commit_homes` commits several pages, `move_home_away` gives one back). The list this
    /// replaced dropped the whole entry, so `init` left those still-committed pages as holes.
    #[test]
    fn early_decommit_of_one_page_keeps_the_rest() {
        let a = 0x4_2000_0000usize;
        let mut early = BTreeMap::new();
        early_add(&mut early, a, a + 4 * TILE);
        early_remove(&mut early, a + TILE, a + 2 * TILE);
        assert_eq!(early.into_iter().collect::<Vec<_>>(), vec![(a, a + TILE), (a + 2 * TILE, a + 4 * TILE)]);
    }

    /// Time for n 64 KiB pages committed one at a time and decommitted one at a time in
    /// reverse, before the device exists (`cargo test --release -p xcl --lib --
    /// --ignored --exact hle::d3d12x::sparse::tests::early_commit_timing --nocapture`,
    /// `XCL_EARLY_PAGES=16384,32768`). Run it alone: a test that made the device keeps a state.
    #[test]
    #[ignore]
    fn early_commit_timing() {
        if STATE.lock().unwrap().is_some() {
            return eprintln!("early_commit_timing: a sparse state exists – run it alone");
        }
        let sizes: Vec<usize> = std::env::var("XCL_EARLY_PAGES").ok().map_or(vec![4096, 8192, 16384, 32768], |v| v.split(',').map(|s| s.trim().parse().unwrap()).collect());
        let base = 0x4_2000_0000usize;
        for n in sizes {
            let t = std::time::Instant::now();
            for i in 0..n {
                unsafe { committed(base + i * TILE, TILE) };
            }
            let commit = t.elapsed();
            let t = std::time::Instant::now();
            for i in (0..n).rev() {
                unsafe { decommitted(base + i * TILE, TILE) };
            }
            let decommit = t.elapsed();
            println!("early: {n} pages: commit {:.1} ms, decommit {:.1} ms", commit.as_secs_f64() * 1e3, decommit.as_secs_f64() * 1e3);
        }
    }
}
