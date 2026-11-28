//! Fence #12 of the newer console layout (a retail game): the address of the fence's completed value,
//! which the title reads and sleeps on (`monitorx` / `mwaitx`). On the console the GPU writes
//! that memory; a PC fence's value is only behind GetCompletedValue, so each fence asked gets a
//! cell of its own that a thread keeps equal to it: every Signal the title queues arms an event.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::ID3D12Fence;
use windows_sys::Win32::System::Threading as th;

/// a cache line of its own: the title's monitors wake on writes to the line
#[repr(C, align(64))]
struct Cell(AtomicU64);

/// Heap-isolated cell plus per-fence refresh lock (`true` = live, `false` = retired).
/// Keeping `Box<Cell>` off the lock's cache line avoids false-sharing `monitorx`/`mwaitx` wakes.
struct CellMirror {
    cell: Box<Cell>,
    refresh_lock: Mutex<bool>,
}

#[derive(Default)]
struct Fence {
    cell: Option<Arc<CellMirror>>,
    /// the highest value a queue was told to signal
    signalled: u64,
    /// the highest value a queue was told to wait for
    waited: u64,
}

#[derive(Default)]
struct State {
    map: HashMap<usize, Fence>,
    cells: Vec<(usize, Arc<CellMirror>)>,
}

/// (fences are never freed in production: XCL holds a reference to each)
struct FenceTable {
    /// Keeps test-owned raw fence pointers alive across diagnostic COM calls.
    summary_lifetime: Mutex<()>,
    inner: Mutex<State>,
}

impl FenceTable {
    fn new() -> Self {
        Self { summary_lifetime: Mutex::new(()), inner: Mutex::new(State::default()) }
    }

    fn refresh_one(fence: usize, mirror: &CellMirror) {
        let live = mirror.refresh_lock.lock().unwrap_or_else(|e| e.into_inner());
        if !*live {
            return;
        }
        if let Some(v) = unsafe { borrowed(fence, |f| f.GetCompletedValue()) } {
            if super::title_textures::complete_fence_downloads(fence, v) {
                mirror.cell.0.store(v, Ordering::Release);
            }
        }
    }

    fn refresh(&self) {
        let cells = self.inner.lock().unwrap_or_else(|e| e.into_inner()).cells.clone();
        for (p, mirror) in cells {
            Self::refresh_one(p, &mirror);
        }
    }

    fn note_queue_signal(&self, fence: usize, value: u64) -> bool {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let f = g.map.entry(fence).or_default();
        f.signalled = f.signalled.max(value);
        f.cell.is_some()
    }

    fn note_queue_wait(&self, fence: usize, value: u64) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let f = g.map.entry(fence).or_default();
        f.waited = f.waited.max(value);
    }

    fn note_cpu_signal(&self, fence: usize) {
        let mirror = self.inner.lock().unwrap_or_else(|e| e.into_inner()).map.get(&fence).and_then(|f| f.cell.clone());
        if let Some(mirror) = mirror {
            Self::refresh_one(fence, &mirror);
        }
    }

    fn pending_summary(&self) -> String {
        // Test teardown keeps this order: summary lifetime -> table -> per-cell refresh lock.
        let _lifetime = self.summary_lifetime.lock().unwrap_or_else(|e| e.into_inner());
        let snap: Vec<(usize, u64, u64, bool)> = {
            let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.map.iter().map(|(&p, f)| (p, f.signalled, f.waited, f.cell.is_some())).collect()
        };
        let mut parts = Vec::new();
        for (p, signalled, waited, has_cell) in snap {
            let done = unsafe { borrowed(p, |f| f.GetCompletedValue()) }.unwrap_or(0);
            if done < signalled || done < waited {
                parts.push(format!("{p:#x}(done={done:#x}/sig={signalled:#x}/wait={waited:#x},cell={has_cell})"));
            }
        }
        if parts.is_empty() { "none".into() } else { parts.join(", ") }
    }

    fn completed_value_address_with(&self, this: *mut c_void, arm_fn: impl FnOnce(usize, u64)) -> *const u64 {
        let key = this as usize;
        let (mirror, pending) = {
            let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(m) = g.map.get(&key).and_then(|f| f.cell.as_ref()) {
                return m.cell.0.as_ptr();
            }
            let m = Arc::new(CellMirror { cell: Box::new(Cell(AtomicU64::new(0))), refresh_lock: Mutex::new(true) });
            let pending = {
                let f = g.map.entry(key).or_default();
                f.cell = Some(Arc::clone(&m));
                f.signalled
            };
            g.cells.push((key, Arc::clone(&m)));
            (m, pending)
        };
        if pending > 0 {
            arm_fn(key, pending);
        }
        Self::refresh_one(key, &mirror);
        mirror.cell.0.as_ptr()
    }

    #[cfg(test)]
    fn forget_fence(&self, fence: usize) {
        let _lifetime = self.summary_lifetime.lock().unwrap_or_else(|e| e.into_inner());
        let removed = {
            let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.cells.retain(|(p, _)| *p != fence);
            g.map.remove(&fence).and_then(|f| f.cell)
        };
        if let Some(mirror) = removed {
            *mirror.refresh_lock.lock().unwrap_or_else(|e| e.into_inner()) = false;
        }
    }
}

fn table() -> &'static FenceTable {
    static T: OnceLock<FenceTable> = OnceLock::new();
    T.get_or_init(FenceTable::new)
}

/// the event a completion sets (auto-reset); the first use starts the thread that follows it
fn event() -> usize {
    static E: OnceLock<usize> = OnceLock::new();
    *E.get_or_init(|| unsafe {
        let e = th::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) as usize;
        let spawned = std::thread::Builder::new().name("xcl-fences".into()).spawn(move || loop {
            th::WaitForSingleObject(e as _, th::INFINITE);
            super::title_textures::poll_fence_downloads();
            table().refresh();
        });
        if let Err(err) = spawned {
            crate::xlog_error!("D3D12.X: no thread for the fences' completed values: {err}");
        }
        e
    })
}

pub(super) fn wake_event() -> HANDLE {
    HANDLE(event() as *mut c_void)
}

unsafe fn borrowed<R>(fence: usize, f: impl FnOnce(&ID3D12Fence) -> R) -> Option<R> {
    let p = fence as *mut c_void;
    ID3D12Fence::from_raw_borrowed(&p).map(f)
}

unsafe fn arm(fence: usize, value: u64) {
    if let Some(Err(e)) = borrowed(fence, |f| f.SetEventOnCompletion(value, HANDLE(event() as *mut c_void))) {
        crate::xlog_warn!("D3D12.X: fence {fence:#x}: SetEventOnCompletion({value:#x}): {e}");
    }
}

/// A queue was told to signal `fence` (one of the title's) with `value`.
pub(super) unsafe fn note_queue_signal(fence: usize, value: u64) {
    if table().note_queue_signal(fence, value) {
        arm(fence, value);
    }
}

/// A queue was told to wait for `fence` (one of the title's) to reach `value`.
pub(super) fn note_queue_wait(fence: usize, value: u64) {
    table().note_queue_wait(fence, value);
}

/// The CPU signalled `fence`.
pub(super) fn note_cpu_signal(fence: usize) {
    table().note_cpu_signal(fence);
}

/// Diagnostic summary of title fences whose completed value lags their signalled or waited value.
pub(super) fn pending_summary() -> String {
    table().pending_summary()
}

/// fence #12: the address of its completed value
pub(super) unsafe extern "system" fn completed_value_address(this: *mut c_void) -> *const u64 {
    table().completed_value_address_with(this, |f, v| arm(f, v))
}

/// Remove a synthetic test fence from `FENCES` before it is dropped.
#[cfg(test)]
pub(super) fn forget_fence_for_test(fence: usize) {
    table().forget_fence(fence);
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::*;

    /// The cell follows the fence: CPU signals at once, queue signals when the GPU is there –
    /// also one queued before the address was asked for.
    #[test]
    fn the_cell_follows_the_fence() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(5, D3D12_FENCE_FLAG_NONE).unwrap();
            let p = fence.as_raw();
            let wait = |cell: *const u64, v: u64| {
                let t = std::time::Instant::now();
                while (*(cell as *const AtomicU64)).load(Ordering::Acquire) != v {
                    assert!(t.elapsed().as_secs() < 5, "cell never became {v}");
                    std::thread::yield_now();
                }
            };
            queue.Signal(&fence, 6).unwrap();
            note_queue_signal(p as usize, 6);
            let cell = completed_value_address(p);
            assert_eq!(cell, completed_value_address(p));
            wait(cell, 6);
            queue.Signal(&fence, 7).unwrap();
            note_queue_signal(p as usize, 7);
            wait(cell, 7);
            fence.Signal(9).unwrap();
            note_cpu_signal(p as usize);
            assert_eq!(*cell, 9);
            std::mem::forget(fence); // the map keeps its address
        }
    }

    struct Gate {
        entered: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    }

    #[repr(C)]
    struct MockFence {
        vtable: *const usize,
        val: AtomicU64,
        calls: std::sync::atomic::AtomicUsize,
        gate: Mutex<Option<Gate>>,
    }
    unsafe impl Send for MockFence {}
    unsafe impl Sync for MockFence {}

    impl MockFence {
        fn new(val: u64) -> Box<Self> {
            use std::mem::{offset_of, size_of};
            unsafe extern "system" fn get_val(this: *mut c_void) -> u64 {
                let f = &*this.cast::<MockFence>();
                f.calls.fetch_add(1, Ordering::AcqRel);
                let v = f.val.load(Ordering::Acquire);
                if let Some(g) = f.gate.lock().unwrap().take() {
                    g.entered.wait();
                    g.release.wait();
                }
                v
            }
            extern "system" fn unexpected() { std::process::abort(); }
            const SLOT: usize = size_of::<usize>();
            const LEN: usize = size_of::<ID3D12Fence_Vtbl>() / SLOT;
            static VT: OnceLock<[usize; LEN]> = OnceLock::new();
            let vt = VT.get_or_init(|| {
                let mut t = [unexpected as usize; LEN];
                t[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = get_val as usize;
                t
            });
            Box::new(Self { vtable: vt.as_ptr(), val: AtomicU64::new(val), calls: Default::default(), gate: Mutex::new(None) })
        }
        fn as_ptr(&self) -> *mut c_void { std::ptr::from_ref(self) as *mut c_void }
        fn arm_gate(&self) -> (Arc<std::sync::Barrier>, Arc<std::sync::Barrier>) {
            let (e, r) = (Arc::new(std::sync::Barrier::new(2)), Arc::new(std::sync::Barrier::new(2)));
            *self.gate.lock().unwrap() = Some(Gate { entered: e.clone(), release: r.clone() });
            (e, r)
        }
    }

    #[test]
    fn refresh_and_pending_summary_do_not_block_concurrent_queue_and_cpu_signals() {
        let table = Arc::new(FenceTable::new());
        let (a, b) = (MockFence::new(1), MockFence::new(2));
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let _cell_a = table.completed_value_address_with(pa, |_, _| {});
        let cell_b = table.completed_value_address_with(pb, |_, _| {});
        assert!(table.note_queue_signal(pa as usize, 10));

        // 1. refresh() must not hold global lock across GetCompletedValue().
        let (entered, release) = a.arm_gate();
        let t1 = Arc::clone(&table);
        let h1 = std::thread::spawn(move || t1.refresh());
        entered.wait();
        b.val.store(4, Ordering::Release);
        assert!(table.note_queue_signal(pb as usize, 4));
        table.note_queue_wait(pb as usize, 4);
        table.note_cpu_signal(pb as usize);
        assert_eq!(unsafe { (*(cell_b as *const AtomicU64)).load(Ordering::Acquire) }, 4);
        release.wait();
        h1.join().unwrap();

        // 2. pending_summary() must not hold global lock across GetCompletedValue().
        let (entered, release) = a.arm_gate();
        let t2 = Arc::clone(&table);
        let h2 = std::thread::spawn(move || t2.pending_summary());
        entered.wait();
        b.val.store(7, Ordering::Release);
        assert!(table.note_queue_signal(pb as usize, 7));
        table.note_queue_wait(pb as usize, 7);
        table.note_cpu_signal(pb as usize);
        assert_eq!(unsafe { (*(cell_b as *const AtomicU64)).load(Ordering::Acquire) }, 7);
        release.wait();
        let summary = h2.join().unwrap();
        assert!(summary.contains("done=0x1/sig=0xa/"), "unexpected summary: {summary}");
    }

    #[test]
    fn pending_summary_keeps_non_cell_fence_live_until_forget_returns() {
        let table = Arc::new(FenceTable::new());
        let fence = MockFence::new(1);
        let p = fence.as_ptr() as usize;
        assert!(!table.note_queue_signal(p, 2), "the fence must not have a cell");
        let (entered, release) = fence.arm_gate();
        let summary_table = Arc::clone(&table);
        let summary = std::thread::spawn(move || summary_table.pending_summary());
        entered.wait();

        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let forget_table = Arc::clone(&table);
        let forget = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            forget_table.forget_fence(p);
            done_tx.send(()).unwrap();
        });
        started_rx.recv().unwrap();
        assert!(matches!(table.summary_lifetime.try_lock(), Err(std::sync::TryLockError::WouldBlock)));
        assert!(matches!(done_rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)), "forget returned during the raw COM query");

        release.wait();
        assert!(summary.join().unwrap().contains("done=0x1/sig=0x2/"));
        forget.join().unwrap();
        done_rx.recv().unwrap();
        drop(fence);
    }

    #[test]
    fn note_cpu_signal_and_refresh_avoid_unrelated_and_non_cell_fences() {
        let table = FenceTable::new();
        let (a, b) = (MockFence::new(3), MockFence::new(5));
        let non_cells: Vec<Box<MockFence>> = (0..64).map(|i| MockFence::new(i)).collect();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let _ = table.completed_value_address_with(pa, |_, _| {});
        let _ = table.completed_value_address_with(pb, |_, _| {});
        for f in &non_cells {
            assert!(!table.note_queue_signal(f.as_ptr() as usize, 10));
            table.note_queue_wait(f.as_ptr() as usize, 10);
        }
        a.calls.store(0, Ordering::Release);
        b.calls.store(0, Ordering::Release);
        table.note_cpu_signal(pb as usize);
        assert_eq!((a.calls.load(Ordering::Acquire), b.calls.load(Ordering::Acquire)), (0, 1));
        assert!(non_cells.iter().all(|f| f.calls.load(Ordering::Acquire) == 0));

        a.calls.store(0, Ordering::Release);
        b.calls.store(0, Ordering::Release);
        table.refresh();
        assert_eq!((a.calls.load(Ordering::Acquire), b.calls.load(Ordering::Acquire)), (1, 1));
        assert!(non_cells.iter().all(|f| f.calls.load(Ordering::Acquire) == 0));

        table.forget_fence(pa as usize);
        a.calls.store(0, Ordering::Release);
        b.calls.store(0, Ordering::Release);
        table.refresh();
        assert_eq!((a.calls.load(Ordering::Acquire), b.calls.load(Ordering::Acquire)), (0, 1));
    }

    #[test]
    fn per_cell_lock_prevents_stale_refresh_overwrite() {
        let table = Arc::new(FenceTable::new());
        let a = MockFence::new(10);
        let pa = a.as_ptr();
        let cell = table.completed_value_address_with(pa, |_, _| {});
        a.calls.store(0, Ordering::Release);

        let (entered, release) = a.arm_gate();
        let t1 = Arc::clone(&table);
        let h1 = std::thread::spawn(move || t1.refresh());
        entered.wait();
        a.val.store(11, Ordering::Release);
        let t2 = Arc::clone(&table);
        let pa_addr = pa as usize;
        let h2 = std::thread::spawn(move || t2.note_cpu_signal(pa_addr));
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(a.calls.load(Ordering::Acquire), 1, "note_cpu_signal must serialize behind in-flight refresh_lock");
        release.wait();
        h1.join().unwrap();
        h2.join().unwrap();
        assert_eq!(a.calls.load(Ordering::Acquire), 2);
        assert_eq!(unsafe { (*(cell as *const AtomicU64)).load(Ordering::Acquire) }, 11);
    }
}
