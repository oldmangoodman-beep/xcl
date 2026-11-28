//! PSOs for a split compute entry. The first piece is the real PSO returned
//! to the caller. Other pieces and the lazily built whole form belong to it.
use super::*;
use super::super::used_journal::Journal;
use crate::gpu::{dxil::kernel::DispatchChain, recomp::pso};
use std::collections::HashSet;
use std::sync::{Arc, LazyLock, OnceLock, Weak};

static PROGRAMS: AtomicU64 = AtomicU64::new(0);
static PIECES: AtomicU64 = AtomicU64::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// slabs dispatched (`pso::dispatch_slabs`)
static SLABS: AtomicU64 = AtomicU64::new(0);
/// scratch alive now, the most there was, and the most a dispatch would have needed at once (bytes)
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static WANTED: AtomicU64 = AtomicU64::new(0);

struct IndirectUsed {
    now: Journal,
}

static INDIRECT_USED: OnceLock<IndirectUsed> = OnceLock::new();
static EARLIER_INDIRECT_USED: LazyLock<HashSet<u64>> = LazyLock::new(|| {
    if cfg!(test) {
        HashSet::new()
    } else {
        parse_used(&std::fs::read_to_string(indirect_used_path()).unwrap_or_default())
    }
});

fn parse_used(text: &str) -> HashSet<u64> {
    text.lines()
        .filter_map(|line| u64::from_str_radix(line.trim(), 16).ok())
        .filter(|&record| record != 0)
        .collect()
}

fn indirect_used_path() -> std::path::PathBuf {
    crate::gpu::recomp::dxc::cache_dir().join("used-indirect-split.txt")
}

fn indirect_used() -> &'static IndirectUsed {
    INDIRECT_USED.get_or_init(|| {
        if cfg!(test) {
            return IndirectUsed { now: Journal::memory(HashSet::new()) };
        }
        let path = indirect_used_path();
        let now = Journal::file(path, EARLIER_INDIRECT_USED.clone());
        IndirectUsed { now }
    })
}

fn note_indirect_use(chain: &Chain) {
    if !chain.indirect_noted.swap(true, Ordering::Relaxed) {
        indirect_used().now.note(chain.builder.record);
    }
}

pub(crate) fn earlier_indirect_used() -> HashSet<u64> {
    EARLIER_INDIRECT_USED.clone()
}

pub(crate) fn shutdown_indirect_used(timeout: std::time::Duration) {
    if let Some(used) = INDIRECT_USED.get() {
        let _ = used.now.shutdown_bounded(timeout);
    }
}

pub(crate) fn flush_indirect_used(timeout: std::time::Duration) {
    if let Some(used) = INDIRECT_USED.get() {
        let _ = used.now.flush_bounded(timeout);
    }
}

pub(crate) fn log_counts() {
    let mib = |a: &AtomicU64| a.load(Ordering::Relaxed) >> 20;
    crate::xlog_info!("compute split: {} programs split, {} pieces built, {} chained dispatches issued in {} slabs; scratch {} MiB (most {}, largest dispatch {} in one slab)",
        PROGRAMS.load(Ordering::Relaxed), PIECES.load(Ordering::Relaxed), DISPATCHES.load(Ordering::Relaxed), SLABS.load(Ordering::Relaxed),
        mib(&LIVE), mib(&PEAK), mib(&WANTED));
}

/// The scratch a dispatch uses at once, `XCL_SPLIT_SCRATCH_MB` (256): more groups than fit run in
/// slabs. A retail title's split programs need 190 KB for each group (median), 3 GB for 960x540
/// pixels in groups of 8x4.
fn scratch_budget() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_SCRATCH_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(256u64).min(4096) << 20)
}

struct PipelineBuilder {
    device: ID3D12Device,
    create: PsoFn,
    signature: ID3D12RootSignature,
    samplers: Vec<pso::StaticSampler>,
    flags: D3D12_PIPELINE_STATE_FLAGS,
    node: u32,
    record: u64,
    mode: ComputeBuild,
}

impl PipelineBuilder {
    unsafe fn pipeline(&self, dxil: &[u8], iid: &GUID) -> Result<ID3D12PipelineState, HRESULT> {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { NodeMask: self.node, Flags: self.flags, ..Default::default() };
        let mut raw = std::ptr::null_mut();
        let (hr, _, _) = compute_container(self.device.as_raw(), self.create, &desc, self.signature.as_raw(),
            dxil, &self.samplers, self.record, iid, &mut raw, self.mode);
        if hr >= 0 && !raw.is_null() { Ok(ID3D12PipelineState::from_raw(raw)) } else { Err(hr) }
    }
}

/// One exact chain's whole-form outcome. The owning `Chain` fixes device, DXIL, root signature,
/// flags and node; releasing/recreating the pipeline gets a fresh outcome.
pub(super) struct WholeBuild<T> {
    result: OnceLock<Result<T, HRESULT>>,
    retry_at: Mutex<Option<std::time::Instant>>,
}

impl<T: Clone> WholeBuild<T> {
    pub(super) fn new() -> Self {
        Self { result: OnceLock::new(), retry_at: Mutex::new(None) }
    }

    pub(super) fn get_or_build(&self, now: std::time::Instant, build: impl FnOnce() -> Result<T, HRESULT>) -> Result<T, HRESULT> {
        if let Some(result) = self.result.get() {
            return result.clone();
        }
        let mut retry_at = self.retry_at.lock().unwrap();
        if let Some(result) = self.result.get() {
            return result.clone();
        }
        if retry_at.is_some_and(|at| now < at) {
            return Err(crate::util::E_OUTOFMEMORY);
        }
        match build() {
            Err(hr) if hr == crate::util::E_OUTOFMEMORY => {
                *retry_at = Some(now + WHOLE_RETRY);
                Err(hr)
            }
            result => {
                let _ = self.result.set(result.clone());
                result
            }
        }
    }

    fn ready(&self) -> Option<Result<T, HRESULT>> {
        self.result.get().cloned()
    }
}

/// A transient whole-form OOM is retried after memory has had time to drain.
const WHOLE_RETRY: std::time::Duration = std::time::Duration::from_secs(10);

struct Chain {
    builder: Arc<PipelineBuilder>,
    tail: Vec<ID3D12PipelineState>,
    words: u32,
    whole_code: Vec<u8>,
    whole: WholeBuild<ID3D12PipelineState>,
    plan: Weak<DispatchChain>,
    indirect: IndirectBuild,
    indirect_noted: AtomicBool,
}

struct IndirectChain {
    signature: ID3D12RootSignature,
    whole: ID3D12PipelineState,
    pieces: Vec<ID3D12PipelineState>,
    _charge: IndirectCharge,
}

struct IndirectCharge(usize);

impl Drop for IndirectCharge {
    fn drop(&mut self) {
        INDIRECT_VARIANT_SOURCE.fetch_sub(self.0, Ordering::AcqRel);
    }
}

struct IndirectBuild {
    requested: AtomicBool,
    result: OnceLock<Result<Arc<IndirectChain>, HRESULT>>,
}

impl IndirectBuild {
    fn new() -> Self {
        Self { requested: AtomicBool::new(false), result: OnceLock::new() }
    }
}

impl Chain {
    unsafe fn whole(&self) -> Result<ID3D12PipelineState, String> {
        self.whole
            .get_or_build(std::time::Instant::now(), || self.builder.pipeline(&self.whole_code, &pc_iid(Kind::PipelineState)))
            .map_err(|hr| if hr == crate::util::E_OUTOFMEMORY { "whole compute pipeline: out of memory".into() } else { format!("whole compute pipeline ({hr:#x})") })
    }
}

static CHAINS: LazyLock<Mutex<HashMap<usize, Arc<Chain>>>> = LazyLock::new(Default::default);
/// Shared for the process lifetime so chains with a common piece also share its driver blob.
static SIGNATURES: LazyLock<Mutex<HashMap<(usize, bool, Vec<u32>), ID3D12RootSignature>>> = LazyLock::new(Default::default);

fn sampler_key(samplers: &[pso::StaticSampler]) -> Vec<u32> {
    samplers.iter().flat_map(|(words, comparison)| words.iter().copied().chain([*comparison as u32])).collect()
}

const INDIRECT_PLAN_BYTES: usize = 64 << 20;
const INDIRECT_VARIANT_BYTES: usize = 64 << 20;
static INDIRECT_VARIANT_SOURCE: AtomicUsize = AtomicUsize::new(0);

fn reserve_indirect_source(used: &AtomicUsize, bytes: usize, limit: usize) -> bool {
    used.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        current.checked_add(bytes).filter(|&next| next <= limit)
    }).is_ok()
}

#[derive(Default)]
struct PlanCache {
    bytes: usize,
    plans: std::collections::VecDeque<(usize, Arc<DispatchChain>)>,
}

impl PlanCache {
    fn insert(&mut self, plan: Arc<DispatchChain>, limit: usize) {
        let key = Arc::as_ptr(&plan) as usize;
        if let Some(at) = self.plans.iter().position(|p| p.0 == key) {
            self.plans.remove(at);
        } else {
            self.bytes = self.bytes.saturating_add(plan.pieces.iter().map(Vec::capacity).sum::<usize>());
        }
        self.plans.push_back((key, plan));
        while self.bytes > limit {
            let Some((_, old)) = self.plans.pop_front() else { break };
            self.bytes = self.bytes.saturating_sub(old.pieces.iter().map(Vec::capacity).sum::<usize>());
        }
    }
}

fn retain_indirect_plan(plan: Arc<DispatchChain>) {
    static PLANS: LazyLock<Mutex<PlanCache>> = LazyLock::new(Default::default);
    PLANS.lock().unwrap().insert(plan, INDIRECT_PLAN_BYTES);
}

fn build_indirect(chain: &Arc<Chain>) -> Result<Arc<IndirectChain>, HRESULT> {
    let plan = chain.plan.upgrade().ok_or(E_INVALIDARG)?;
    let source_bytes = plan.pieces.iter().map(Vec::len).sum::<usize>().saturating_add(chain.whole_code.len());
    if !reserve_indirect_source(&INDIRECT_VARIANT_SOURCE, source_bytes, INDIRECT_VARIANT_BYTES) {
        return Err(E_OUTOFMEMORY);
    }
    let charge = IndirectCharge(source_bytes);
    let signature = unsafe { signature_for(&chain.builder.device, &chain.builder.samplers, true) }.ok_or(E_INVALIDARG)?;
    let builder = Arc::new(PipelineBuilder {
        device: chain.builder.device.clone(),
        create: chain.builder.create,
        signature: signature.clone(),
        samplers: chain.builder.samplers.clone(),
        flags: chain.builder.flags,
        node: chain.builder.node,
        record: chain.builder.record,
        mode: ComputeBuild::IndirectChain,
    });
    // These variants are speculative: the ready direct whole form remains correct. Keep their
    // cold driver compiles on this one low-impact lane instead of fanning out across PSO workers.
    let mut pieces = Vec::with_capacity(plan.pieces.len());
    for code in &plan.pieces {
        pieces.push(unsafe { builder.pipeline(code, &pc_iid(Kind::PipelineState))? });
    }
    let whole = unsafe { builder.pipeline(&chain.whole_code, &pc_iid(Kind::PipelineState))? };
    Ok(Arc::new(IndirectChain { signature, whole, pieces, _charge: charge }))
}

fn indirect_blobs_present(chain: &Chain, plan: &DispatchChain) -> bool {
    plan.pieces.iter().map(Vec::as_slice).chain([chain.whole_code.as_slice()]).all(|dxil| {
        super::driver_cache::path_for_chain_root_node(
            dxil,
            chain.builder.flags.0,
            super::driver_cache::ChainRoot::IndirectCbv,
            &chain.builder.samplers,
            chain.builder.node,
        )
        .and_then(|path| std::fs::metadata(path).ok())
        .is_some_and(|metadata| metadata.len() != 0)
    })
}

fn request_indirect(chain: &Arc<Chain>) {
    if chain.indirect.requested.swap(true, Ordering::AcqRel) {
        return;
    }
    static BUILDS: LazyLock<Option<std::sync::mpsc::Sender<Weak<Chain>>>> = LazyLock::new(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Weak<Chain>>();
        std::thread::Builder::new().name("xcl-pso-chain-indirect".into()).stack_size(64 << 20).spawn(move || {
            while let Ok(weak) = rx.recv() {
                let Some(chain) = weak.upgrade() else { continue };
                let result = build_indirect(&chain);
                let _ = chain.indirect.result.set(result);
            }
        }).ok().map(|_| tx)
    });
    if BUILDS.as_ref().is_none_or(|queue| queue.send(Arc::downgrade(chain)).is_err()) {
        chain.indirect.requested.store(false, Ordering::Release);
    }
}

unsafe fn signature_for(dev: &ID3D12Device, samplers: &[pso::StaticSampler], indirect: bool) -> Option<ID3D12RootSignature> {
    let key = (dev.as_raw() as usize, indirect, sampler_key(samplers));
    if let Some(signature) = SIGNATURES.lock().unwrap().get(&key).cloned() {
        return Some(signature);
    }
    let created = match if indirect {
        pso::dispatch_chain_indirect_root_signature(dev, samplers)
    } else {
        pso::dispatch_chain_root_signature(dev, samplers)
    } {
        Ok(signature) => signature,
        Err(e) => {
            note_translation_error(format!("dispatch-chain root signature: {e}"));
            return None;
        }
    };
    Some(SIGNATURES.lock().unwrap().entry(key).or_insert(created).clone())
}

pub(super) unsafe fn build(this: *mut c_void, create: PsoFn, desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
    iid: *const GUID, result: *mut *mut c_void, record: u64, whole: &[u8], samplers: &[pso::StaticSampler],
    plan: Arc<DispatchChain>) -> Option<usize> {
    if result.is_null() || iid.is_null() || plan.pieces.len() < 2 { return None; }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    let signature = signature_for(dev, samplers, false)?;
    let rs = signature.as_raw() as usize;
    let builder = Arc::new(PipelineBuilder { device: dev.clone(), create, signature, samplers: samplers.to_vec(),
        flags: desc.Flags, node: desc.NodeMask, record, mode: ComputeBuild::DirectChain });
    let jobs = (0..plan.pieces.len()).map(|i| {
        let plan = plan.clone();
        let builder = builder.clone();
        let iid = if i == 0 { *iid } else { pc_iid(Kind::PipelineState) };
        Box::new(move || builder.pipeline(&plan.pieces[i], &iid).map_or(0, |p| p.into_raw() as usize)) as super::super::pso_async::Job
    }).collect();
    // Take ownership of every success, including when a different piece failed.
    let built: Vec<_> = super::super::pso_async::build_pieces(jobs).into_iter()
        .map(|p| (p != 0).then(|| ID3D12PipelineState::from_raw(p as *mut c_void))).collect();
    let mut pieces = built.into_iter().collect::<Option<Vec<_>>>()?.into_iter();
    let first = pieces.next()?;
    let chain = Arc::new(Chain { builder, tail: pieces.collect(), words: plan.dwords_per_group,
        whole_code: whole.to_vec(), whole: WholeBuild::new(), plan: Arc::downgrade(&plan),
        indirect: IndirectBuild::new(), indirect_noted: AtomicBool::new(false) });
    retain_indirect_plan(plan.clone());
    // Initialize the journal on this preparation worker, never on the first recording call.
    let _ = indirect_used();
    if EARLIER_INDIRECT_USED.contains(&record) && indirect_blobs_present(&chain, &plan) {
        request_indirect(&chain);
    }
    PROGRAMS.fetch_add(1, Ordering::Relaxed);
    PIECES.fetch_add(plan.pieces.len() as u64, Ordering::Relaxed);
    *result = first.into_raw();
    CHAINS.lock().unwrap().insert(*result as usize, chain);
    Some(rs)
}

pub(super) fn forget(pso: usize) {
    let gone = CHAINS.lock().unwrap().remove(&pso);
    drop(gone); // releasing a tail PSO also goes through forget_pso
}

pub(in crate::hle::d3d12x) unsafe fn whole(pso: usize) -> Result<Option<ID3D12PipelineState>, String> {
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    chain.map(|c| c.whole()).transpose()
}

struct Scratch {
    buffer: ID3D12Resource,
    bytes: u64,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        LIVE.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

struct Recording {
    scratch: Arc<Scratch>,
    /// Keep the title-visible direct first piece alive while this recording may still execute.
    /// Indirect execution uses its separate CBV-compatible duplicates.
    first: ID3D12PipelineState,
    chain: Arc<Chain>,
    _indirect_commands: Option<Arc<CommandStorage>>,
}

static RECORDINGS: LazyLock<Mutex<HashMap<usize, Vec<Arc<Recording>>>>> = LazyLock::new(Default::default);
/// list → the allocator it records on and the scratch its chains share: they are dispatched one
/// after another, with a barrier between them (the recordings keep it alive)
static SHARED: LazyLock<Mutex<HashMap<usize, (usize, std::sync::Weak<Scratch>)>>> = LazyLock::new(Default::default);
/// One recording list reuses indirect commands at offset zero. Buffer state transitions are
/// resource-wide, so storage is deliberately never shared across lists.
static COMMANDS: LazyLock<Mutex<HashMap<usize, (usize, Weak<CommandStorage>)>>> = LazyLock::new(Default::default);

struct CommandStorage {
    buffer: ID3D12Resource,
    bytes: u64,
}

const COMMAND_STORAGE_BYTES: u64 = 64 << 10;

fn command_storage_bytes(bytes: u64) -> u64 {
    bytes.max(COMMAND_STORAGE_BYTES).next_multiple_of(COMMAND_STORAGE_BYTES)
}

unsafe fn commands_for(
    dev: &ID3D12Device, list: usize, allocator: usize, bytes: u64,
) -> Option<Arc<CommandStorage>> {
    {
        let commands = COMMANDS.lock().unwrap();
        if let Some((_, storage)) = commands.get(&list).filter(|entry| entry.0 == allocator) {
            if let Some(storage) = storage.upgrade().filter(|storage| storage.bytes >= bytes) {
                return Some(storage);
            }
        }
    }
    let size = command_storage_bytes(bytes);
    let storage = Arc::new(CommandStorage { buffer: pso::chain_commands_storage(dev, size)?, bytes: size });
    COMMANDS.lock().unwrap().insert(list, (allocator, Arc::downgrade(&storage)));
    Some(storage)
}

/// `list` begins another recording, which the GPU may run beside the last one.
pub(in crate::hle::d3d12x) fn list_reset(list: usize) {
    SHARED.lock().unwrap().remove(&list);
    COMMANDS.lock().unwrap().remove(&list);
}

/// `to` records right after `from` on the same allocator: its chains go on in `from`'s scratch.
pub(in crate::hle::d3d12x) fn list_follows(from: usize, to: usize) {
    let mut shared = SHARED.lock().unwrap();
    if let Some(e) = shared.get(&from).cloned() {
        shared.insert(to, e);
    }
}

/// The scratch of all recordings alive together, `XCL_SPLIT_SCRATCH_TOTAL_MB` (384). A recording
/// keeps its scratch until its allocator is reset, and a title that records ahead of the GPU has
/// many: with a bound on one dispatch alone a retail title's world took 1.2 GB and went on growing.
fn scratch_total() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_SCRATCH_TOTAL_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(384u64) << 20)
}

/// A recording gets at least this much (or one group's worth) however much the others hold.
const LEAST: u64 = 2 << 20;

/// How much scratch a dispatch of `needed` bytes (`group` for one group) runs with at once: the
/// list's own buffer (`had` bytes) when that is enough or no more may be made, else a new one of
/// the bytes given (second: make one).
fn scratch_bytes(needed: u64, group: u64, budget: u64, had: Option<u64>, live: u64, total: u64) -> (u64, bool) {
    let want = needed.min(budget).max(group);
    let room = total.saturating_sub(live);
    match had {
        Some(h) if h >= want || (room < want && h >= group) => (h, false),
        _ => (want.min(room.max(LEAST)).max(group), true),
    }
}

/// A scratch for a chain of `needed` bytes recorded on `list` with `allocator`, and whether a
/// chain before it on the list used it.
unsafe fn scratch_for(dev: &ID3D12Device, list: usize, allocator: usize, needed: u64, group: u64) -> Option<(Arc<Scratch>, bool)> {
    let had = SHARED.lock().unwrap().get(&list).filter(|s| s.0 == allocator).and_then(|s| s.1.upgrade());
    let (bytes, make) = scratch_bytes(needed, group, scratch_budget(), had.as_ref().map(|s| s.bytes), LIVE.load(Ordering::Relaxed), scratch_total());
    if !make {
        return had.map(|s| (s, true));
    }
    let s = Arc::new(Scratch { buffer: scratch_buffer(dev, bytes)?, bytes });
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
    SHARED.lock().unwrap().insert(list, (allocator, Arc::downgrade(&s)));
    Some((s, false))
}

pub(in crate::hle::d3d12x) fn allocator_reset(allocator: usize) {
    let recordings = RECORDINGS.lock().unwrap().remove(&allocator);
    drop(recordings);
}

unsafe fn scratch_buffer(dev: &ID3D12Device, bytes: u64) -> Option<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..Default::default()
    };
    let mut resource = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() },
        D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut resource).ok()?;
    resource
}

fn remapped_grid(mut grid: [u32; 3], remap: Option<(u32, u32)>) -> Option<[u32; 3]> {
    if grid.into_iter().any(|n| n > 65_535) {
        return None;
    }
    if let Some((ratio, dim)) = remap {
        let count = grid.get_mut(dim as usize)?;
        let scaled = count.checked_mul(ratio)?;
        // D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION (an SDK macro, not in windows-rs).
        if scaled > 65_535 {
            return None;
        }
        *count = scaled;
    }
    Some(grid)
}

/// Called with the original six ABI views bound. The extended signature is
/// common to every piece and the whole fallback. Choose before recording any
/// part of the dispatch, including when scratch allocation fails.
pub(in crate::hle::d3d12x) unsafe fn dispatch(
    list: &ID3D12GraphicsCommandList, original: usize, pso: usize, allocator: usize, grid: [u32; 3],
) -> Result<(), String> {
    if grid.into_iter().any(|n| n > 65_535) {
        return Err("dispatch group count exceeds 65535 on one axis".into());
    }
    let [x, y, z] = grid;
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    let Some(chain) = chain else {
        match remapped_grid(grid, super::pso_dispatch_remap_exact(pso)) {
            Some([x, y, z]) => list.Dispatch(x, y, z),
            None => {
                let whole = super::remap_fallback::whole_exact(pso)?;
                super::bind_pipeline_exact(list.as_raw(), original, whole.as_raw() as usize);
                list.Dispatch(x, y, z);
            }
        }
        return Ok(())
    };
    if grid.contains(&0) { return Ok(()); }
    // A title Reset can switch to a different PC allocator while this
    // recording is still submitted on the old one. Retain by backing.
    let allocator = if super::super::alloc_pool::enabled() {
        super::super::alloc_pool::recording_backing(list.as_raw() as usize).unwrap_or(0)
    } else { allocator };
    let group = u64::from(chain.words) * 4;
    let needed = grid.into_iter().fold(group, |s, n| s.saturating_mul(n as u64));
    WANTED.fetch_max(needed, Ordering::Relaxed);
    let scratch = (allocator != 0 && list.GetType() != D3D12_COMMAND_LIST_TYPE_BUNDLE)
        .then(|| scratch_for(&chain.builder.device, list.as_raw() as usize, allocator, needed, group)).flatten();
    let most = scratch.as_ref().map_or(1, |(s, _)| (s.bytes / group).max(1));
    let slabs = pso::dispatch_slabs(grid, most);
    let Some((scratch, used)) = scratch else {
        let whole = chain.whole()?;
        super::bind_pipeline_exact(list.as_raw(), original, whole.as_raw() as usize);
        list.Dispatch(x, y, z);
        return Ok(());
    };
    let first = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).ok_or("first compute piece")?.clone();
    let held = Arc::new(Recording { scratch, first, chain, _indirect_commands: None });
    RECORDINGS.lock().unwrap().entry(allocator).or_default().push(held.clone());
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    SLABS.fetch_add(slabs.len() as u64, Ordering::Relaxed);
    pso::record_dispatch_chain(
        list, grid, most, &held.scratch.buffer, used,
        std::iter::once(&held.first).chain(&held.chain.tail),
        |piece| super::bind_pipeline_exact(list.as_raw(), original, piece.as_raw() as usize),
    );
    Ok(())
}

fn indirect_rounds() -> u32 {
    static ROUNDS: OnceLock<u32> = OnceLock::new();
    *ROUNDS.get_or_init(|| std::env::var("XCL_SPLIT_INDIRECT_ROUNDS").ok().and_then(|v| v.trim().parse().ok())
        .filter(|&n| n > 0).unwrap_or(16u32).min(1024))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IndirectReadiness {
    Run,
    Request,
    Fallback,
    Fail(HRESULT),
}

fn indirect_readiness(
    whole: Option<Result<(), HRESULT>>,
    indirect: Option<Result<(), HRESULT>>,
) -> IndirectReadiness {
    match (whole, indirect) {
        (_, Some(Ok(()))) => IndirectReadiness::Run,
        (Some(Err(hr)), _) => IndirectReadiness::Fail(hr),
        (Some(Ok(())), None) => IndirectReadiness::Request,
        _ => IndirectReadiness::Fallback,
    }
}

/// Run one Dispatch-only indirect command through an existing split chain. A complete indirect
/// set is independent of the direct whole-form build. Otherwise preserve the synchronous fallback;
/// the slicer writes either bounded piece commands or the whole-form command, never a partial chain.
pub(in crate::hle::d3d12x) unsafe fn execute_indirect(
    list: &ID3D12GraphicsCommandList, original: usize, pso: usize, allocator: usize, args: u64,
    bind: impl FnOnce(usize) -> Result<(), String>,
) -> Result<bool, String> {
    let Some(chain) = CHAINS.lock().unwrap().get(&pso).cloned() else { return Ok(false) };
    note_indirect_use(&chain);
    let whole = chain.whole.ready().map(|result| result.map(|_| ()));
    let indirect = chain.indirect.result.get().map(|result| match result {
        Ok(_) => Ok(()),
        Err(hr) => Err(*hr),
    });
    match indirect_readiness(whole, indirect) {
        IndirectReadiness::Run => {}
        IndirectReadiness::Request => {
            request_indirect(&chain);
            return Ok(false);
        }
        IndirectReadiness::Fallback => return Ok(false),
        IndirectReadiness::Fail(hr) => return Err(format!("whole compute pipeline ({hr:#x})")),
    };
    let Some(Ok(indirect)) = chain.indirect.result.get() else { unreachable!() };
    let indirect = indirect.clone();
    let allocator = if super::super::alloc_pool::enabled() {
        super::super::alloc_pool::recording_backing(list.as_raw() as usize).unwrap_or(0)
    } else { allocator };
    if allocator == 0 || list.GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE { return Ok(false); }
    let group = u64::from(chain.words) * 4;
    let Some((scratch, _)) = scratch_for(&chain.builder.device, list.as_raw() as usize, allocator, scratch_budget(), group) else {
        return Ok(false);
    };
    let signature = pso::dispatch_command_signature(&chain.builder.device)?;
    let rounds = indirect_rounds();
    let command_bytes = pso::chain_tail_offset(rounds) + 12;
    let commands = commands_for(
        &chain.builder.device, list.as_raw() as usize, allocator, command_bytes,
    ).ok_or("no command buffer for split ExecuteIndirect")?;
    let first = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).ok_or("first compute piece")?.clone();
    pso::record_indirect_slicer(
        list, args, rounds, (scratch.bytes / group).max(1), &commands.buffer,
        |slicer| super::bind_pipeline_exact(list.as_raw(), original, slicer.as_raw() as usize),
    )?;
    let bound = bind(indirect.signature.as_raw() as usize);
    if bound.is_ok() {
        super::bind_pipeline_exact(list.as_raw(), original, indirect.whole.as_raw() as usize);
        list.ExecuteIndirect(&signature, 1, &commands.buffer,
            pso::chain_tail_offset(rounds), None::<&ID3D12Resource>, 0);
        pso::record_dispatch_chain_indirect(
            list, &signature, &commands.buffer, rounds, &scratch.buffer, indirect.pieces.iter(),
            |piece| super::bind_pipeline_exact(list.as_raw(), original, piece.as_raw() as usize),
        );
    }
    pso::restore_indirect_commands(list, &commands.buffer);
    RECORDINGS.lock().unwrap().entry(allocator).or_default().push(Arc::new(Recording {
        scratch, first, chain, _indirect_commands: Some(commands),
    }));
    bound?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indirect_used_records_parse_unique_nonzero_hex() {
        assert_eq!(parse_used("1\n0000000000000002\ninvalid\n0\n1\n"), [1, 2].into());
    }

    #[test]
    fn ready_indirect_set_does_not_wait_for_direct_whole() {
        assert_eq!(indirect_readiness(None, Some(Ok(()))), IndirectReadiness::Run);
        assert_eq!(
            indirect_readiness(Some(Err(E_INVALIDARG)), Some(Ok(()))),
            IndirectReadiness::Run,
        );
        assert_eq!(indirect_readiness(Some(Ok(())), None), IndirectReadiness::Request);
        assert_eq!(indirect_readiness(None, None), IndirectReadiness::Fallback);
        assert_eq!(
            indirect_readiness(Some(Ok(())), Some(Err(E_INVALIDARG))),
            IndirectReadiness::Fallback,
        );
        assert_eq!(
            indirect_readiness(Some(Err(E_INVALIDARG)), Some(Err(E_OUTOFMEMORY))),
            IndirectReadiness::Fail(E_INVALIDARG),
        );
    }

    const MB: u64 = 1 << 20;

    #[test]
    fn indirect_command_storage_covers_the_fixed_recording_at_64k_granularity() {
        assert_eq!(command_storage_bytes(12), 64 << 10);
        assert_eq!(command_storage_bytes((64 << 10) + 1), 128 << 10);
        assert!(command_storage_bytes(pso::chain_tail_offset(1024) + 12)
            >= pso::chain_tail_offset(1024) + 12);
    }

    #[test]
    fn retained_indirect_plans_are_deduplicated_and_bounded() {
        let plan = |byte: u8| Arc::new(DispatchChain {
            pieces: vec![vec![byte; 40], vec![byte; 40]],
            dwords_per_group: 4,
        });
        let (first, second) = (plan(1), plan(2));
        let first_weak = Arc::downgrade(&first);
        let mut cache = PlanCache::default();
        cache.insert(first.clone(), 100);
        cache.insert(first, 100);
        assert_eq!((cache.plans.len(), cache.bytes), (1, 80));
        cache.insert(second, 100);
        assert_eq!((cache.plans.len(), cache.bytes), (1, 80));
        assert!(first_weak.upgrade().is_none(), "the oldest plan was released at the cap");
    }

    #[test]
    fn indirect_variant_source_admission_never_exceeds_its_cap() {
        let used = AtomicUsize::new(0);
        assert!(reserve_indirect_source(&used, 60, 100));
        assert!(!reserve_indirect_source(&used, 41, 100));
        assert_eq!(used.load(Ordering::Relaxed), 60);
        assert!(reserve_indirect_source(&used, 40, 100));
        assert_eq!(used.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn signature_identity_includes_sampler_words_order_and_comparison() {
        let a = ([1, 2, 3, 4], false);
        let b = ([5, 6, 7, 8], true);
        assert_eq!(sampler_key(&[a, b]), sampler_key(&[a, b]));
        assert_ne!(sampler_key(&[a, b]), sampler_key(&[b, a]));
        assert_ne!(sampler_key(&[a]), sampler_key(&[([1, 2, 3, 4], true)]));
    }

    #[test]
    fn remapped_grid_scales_only_its_dimension_and_refuses_overflow() {
        assert_eq!(remapped_grid([3, 5, 7], Some((4, 1))), Some([3, 20, 7]));
        assert_eq!(remapped_grid([3, 5, 7], None), Some([3, 5, 7]));
        assert_eq!(remapped_grid([65_535, 1, 1], None), Some([65_535, 1, 1]));
        assert_eq!(remapped_grid([65_536, 1, 1], None), None);
        assert_eq!(remapped_grid([1, 65_536, 1], None), None);
        assert_eq!(remapped_grid([1, 1, 65_536], None), None);
        assert_eq!(remapped_grid([32_768, 1, 1], Some((2, 0))), None);
        assert_eq!(remapped_grid([u32::MAX, 1, 1], Some((2, 0))), None);
        assert_eq!(remapped_grid([1, 1, 1], Some((2, 3))), None);
    }

    #[test]
    fn released_pipeline_does_not_lend_its_remap_to_address_reuse() {
        let pso = 0x7E57_6400usize;
        super::super::note_pso_dispatch_remap(pso as *mut c_void, Some((4, 0)));
        assert_eq!(super::super::pso_dispatch_remap(pso), Some((4, 0)));
        super::super::forget_pso_for_test(pso);
        assert_eq!(super::super::pso_dispatch_remap(pso), None);
    }

    #[test]
    fn a_list_recorded_after_another_goes_on_in_its_scratch() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
        use windows::Win32::Graphics::Dxgi::*;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            let warp = unsafe { CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) };
            if warp.and_then(|w| unsafe { D3D12CreateDevice(&w, D3D_FEATURE_LEVEL_11_0, &mut dev) }).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
        }
        let dev = dev.unwrap();
        let (a, b, c, alloc) = (0x7E57_5C01usize, 0x7E57_5C02usize, 0x7E57_5C03usize, 0x7E57_5CA0usize);
        let (first, used) = unsafe { super::scratch_for(&dev, a, alloc, MB, 4096) }.unwrap();
        assert!(!used);
        super::list_follows(a, b);
        let (second, used) = unsafe { super::scratch_for(&dev, b, alloc, MB, 4096) }.unwrap();
        assert!(used && std::sync::Arc::ptr_eq(&first, &second), "the list after it made a scratch of its own");
        let (third, _) = unsafe { super::scratch_for(&dev, c, alloc, MB, 4096) }.unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, &third));
        for list in [a, b, c] {
            super::list_reset(list);
        }
    }

    #[test]
    fn a_chains_scratch_is_bounded_by_what_all_recordings_hold() {
        let group = 187 << 10;
        // the first of a recording: as much as it needs, up to the budget of one dispatch
        assert_eq!(scratch_bytes(15 * MB, group, 64 * MB, None, 0, 384 * MB), (15 * MB, true));
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 0, 384 * MB), (64 * MB, true));
        // the others hold nearly all: what is left, and at least LEAST
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 364 * MB, 384 * MB), (20 * MB, true));
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 1200 * MB, 384 * MB), (LEAST, true));
        // the list's own buffer: kept when big enough, or when no more may be made
        assert_eq!(scratch_bytes(8 * MB, group, 64 * MB, Some(16 * MB), 100 * MB, 384 * MB), (16 * MB, false));
        assert_eq!(scratch_bytes(60 * MB, group, 64 * MB, Some(16 * MB), 380 * MB, 384 * MB), (16 * MB, false));
        // … but made again, larger, while there is room
        assert_eq!(scratch_bytes(60 * MB, group, 64 * MB, Some(16 * MB), 100 * MB, 384 * MB), (60 * MB, true));
        // one group always fits, whatever the bounds say
        assert_eq!(scratch_bytes(9 * MB, 9 * MB, 4 * MB, None, 400 * MB, 384 * MB), (9 * MB, true));
        assert_eq!(scratch_bytes(9 * MB, 9 * MB, 4 * MB, Some(MB), 400 * MB, 384 * MB), (9 * MB, true));
    }

    #[test]
    fn a_non_oom_whole_failure_is_final_only_for_that_chain() {
        let at = std::time::Instant::now();
        let whole: WholeBuild<u32> = WholeBuild::new();
        let mut calls = 0;
        assert_eq!(whole.get_or_build(at, || { calls += 1; Err(crate::util::E_FAIL) }), Err(crate::util::E_FAIL));
        assert_eq!(whole.get_or_build(at + WHOLE_RETRY, || { calls += 1; Ok(7) }), Err(crate::util::E_FAIL));
        assert_eq!(calls, 1, "the same immutable chain does not repeat a deterministic driver failure");

        let rebuilt: WholeBuild<u32> = WholeBuild::new();
        assert_eq!(rebuilt.get_or_build(at, || Ok(7)), Ok(7), "a new chain after policy/cache changes still builds");
    }

    #[test]
    fn a_whole_oom_is_throttled_then_retried() {
        let at = std::time::Instant::now();
        let whole: WholeBuild<u32> = WholeBuild::new();
        let mut calls = 0;
        assert_eq!(whole.get_or_build(at, || { calls += 1; Err(crate::util::E_OUTOFMEMORY) }), Err(crate::util::E_OUTOFMEMORY));
        assert_eq!(whole.get_or_build(at + WHOLE_RETRY / 2, || { calls += 1; Ok(7) }), Err(crate::util::E_OUTOFMEMORY));
        assert_eq!(calls, 1, "calls during the cooldown do not enter the driver");
        assert_eq!(whole.get_or_build(at + WHOLE_RETRY, || { calls += 1; Ok(7) }), Ok(7));
        assert_eq!(whole.get_or_build(at + WHOLE_RETRY * 2, || { calls += 1; Ok(8) }), Ok(7));
        assert_eq!(calls, 2, "the successful retry is retained");
    }

    #[test]
    fn failed_binding_keeps_indirect_commands_until_allocator_reset() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        unsafe extern "system" fn never(_: *mut c_void, _: *const c_void, _: *const GUID, _: *mut *mut c_void) -> HRESULT { E_INVALIDARG }
        unsafe {
            let mut dev = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let dev = dev.unwrap();
            let Ok(dxil) = crate::gpu::recomp::dxc::compile("[numthreads(1,1,1)] void main() {}", "main", "cs_6_0", &[]) else {
                return crate::gpu::recomp::runner::skip_test("no DXC");
            };
            let rs = pso::dispatch_chain_root_signature(&dev, &[]).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: std::mem::transmute_copy(&rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() }, ..Default::default() };
            let first: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).unwrap();
            let whole = WholeBuild::new();
            whole.get_or_build(std::time::Instant::now(), || Ok(first.clone())).unwrap();
            let builder = Arc::new(PipelineBuilder { device: dev.clone(), create: never, signature: rs.clone(), samplers: Vec::new(),
                flags: D3D12_PIPELINE_STATE_FLAG_NONE, node: 0, record: 0, mode: ComputeBuild::DirectChain });
            let key = first.as_raw() as usize;
            let plan = Arc::new(DispatchChain { pieces: vec![dxil.clone(), dxil.clone()], dwords_per_group: 4 });
            assert!(reserve_indirect_source(&INDIRECT_VARIANT_SOURCE, 1, INDIRECT_VARIANT_BYTES));
            let indirect = Arc::new(IndirectChain { signature: rs.clone(), whole: first.clone(),
                pieces: vec![first.clone(), first.clone()], _charge: IndirectCharge(1) });
            let indirect_build = IndirectBuild::new();
            indirect_build.requested.store(true, Ordering::Relaxed);
            indirect_build.result.set(Ok(indirect)).unwrap_or_else(|_| unreachable!());
            CHAINS.lock().unwrap().insert(key, Arc::new(Chain { builder, tail: vec![first.clone()], words: 4,
                whole_code: Vec::new(), whole, plan: Arc::downgrade(&plan), indirect: indirect_build,
                indirect_noted: AtomicBool::new(false) }));
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let (raw_list, allocator) = (list.as_raw() as usize, alloc.as_raw() as usize);
            super::super::super::alloc_pool::created_on(raw_list, allocator, ty.0);
            let args = pso::chain_commands_buffer(&dev, 1).unwrap();
            assert_eq!(execute_indirect(&list, key, key, allocator, args.GetGPUVirtualAddress(), |_| Err("bind failed".into())), Err("bind failed".into()));
            let recorded = RECORDINGS.lock().unwrap().get(&allocator).unwrap()[0]._indirect_commands.clone().unwrap();
            let reused = commands_for(&dev, raw_list, allocator, pso::chain_tail_offset(1) + 12).unwrap();
            assert!(Arc::ptr_eq(&recorded, &reused));
            list_reset(raw_list);
            let next_recording = commands_for(&dev, raw_list, allocator, pso::chain_tail_offset(1) + 12).unwrap();
            assert!(!Arc::ptr_eq(&recorded, &next_recording));
            list.Close().unwrap();
            allocator_reset(allocator);
            list_reset(raw_list);
            forget(key);
            super::super::super::alloc_pool::forget_list(raw_list);
        }
    }

    #[test]
    fn remap_compile_oom_retries_but_deterministic_failure_is_final() {
        let at = std::time::Instant::now();
        let oom = super::super::remap_fallback::compile_error_hr("DXC: E_OUTOFMEMORY (0x8007000E)");
        let whole: WholeBuild<u32> = WholeBuild::new();
        assert_eq!(whole.get_or_build(at, || Err(oom)), Err(crate::util::E_OUTOFMEMORY));
        assert_eq!(whole.get_or_build(at + WHOLE_RETRY, || Ok(7)), Ok(7));

        let compile = super::super::remap_fallback::compile_error_hr("dxc: invalid operation");
        let deterministic: WholeBuild<u32> = WholeBuild::new();
        assert_eq!(deterministic.get_or_build(at, || Err(compile)), Err(crate::util::E_INVALIDARG));
        assert_eq!(deterministic.get_or_build(at + WHOLE_RETRY, || Ok(8)), Err(crate::util::E_INVALIDARG));
    }
}
