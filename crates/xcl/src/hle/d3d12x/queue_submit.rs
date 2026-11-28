//! Per-command-queue ordering for title and XCL submissions.
//!
//! XCL sometimes issues setup work from inside a title queue call. The lease is therefore
//! re-entrant on one thread, but excludes another thread until the whole ordered operation ends.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::thread::ThreadId;

#[derive(Default)]
struct State {
    owner: Option<ThreadId>,
    depth: usize,
    #[cfg(test)]
    waiters: usize,
}

#[derive(Default)]
struct Lease {
    state: Mutex<State>,
    changed: Condvar,
}

impl Lease {
    fn enter(self: &Arc<Self>) -> Guard {
        let me = std::thread::current().id();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let _stall = state.owner.is_some_and(|owner| owner != me)
            .then(|| crate::stall::span(crate::stall::StallKind::LockContention))
            .flatten();
        while state.owner.is_some_and(|owner| owner != me) {
            #[cfg(test)]
            {
                state.waiters += 1;
            }
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
            #[cfg(test)]
            {
                state.waiters -= 1;
            }
        }
        state.owner = Some(me);
        state.depth += 1;
        drop(state);
        Guard(Arc::clone(self))
    }
}

struct Guard(Arc<Lease>);

impl Drop for Guard {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        debug_assert_eq!(state.owner, Some(std::thread::current().id()));
        debug_assert!(state.depth > 0);
        state.depth -= 1;
        if state.depth == 0 {
            state.owner = None;
            self.0.changed.notify_one();
        }
    }
}

fn leases() -> &'static RwLock<HashMap<usize, Arc<Lease>>> {
    static LEASES: OnceLock<RwLock<HashMap<usize, Arc<Lease>>>> = OnceLock::new();
    LEASES.get_or_init(|| RwLock::new(HashMap::new()))
}

fn lease(queue: usize) -> Arc<Lease> {
    if let Some(lease) = leases().read().unwrap_or_else(|e| e.into_inner()).get(&queue).cloned() {
        return lease;
    }
    let mut leases = leases().write().unwrap_or_else(|e| e.into_inner());
    Arc::clone(leases.entry(queue).or_default())
}

/// Start a new raw-address generation before the newly created queue is returned to its caller.
pub(super) fn created(queue: usize) {
    if queue != 0 {
        leases().write().unwrap_or_else(|e| e.into_inner()).insert(queue, Arc::new(Lease::default()));
    }
}

/// Run one ordered queue operation. Calls made recursively by XCL setup work retain the same turn.
pub(super) fn with<R>(queue: usize, f: impl FnOnce() -> R) -> R {
    if queue == 0 {
        return f();
    }
    let _guard = lease(queue).enter();
    f()
}

/// Forward Release and discard the raw-address lease after a final native release. Pointer
/// identity prevents a newly created generation at the same address from being removed.
pub(super) fn release(queue: usize, f: impl FnOnce() -> u32) -> u32 {
    let old = leases().read().unwrap_or_else(|e| e.into_inner()).get(&queue).cloned();
    let n = f();
    if n == 0 {
        let mut leases = leases().write().unwrap_or_else(|e| e.into_inner());
        if old.as_ref().is_some_and(|old| leases.get(&queue).is_some_and(|current| Arc::ptr_eq(old, current))) {
            leases.remove(&queue);
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::sync::mpsc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::{ID3D12CommandQueue, ID3D12Fence};

    static NATIVE_SIGNAL: AtomicU64 = AtomicU64::new(0);
    static WRAPPER_SIGNAL: AtomicU64 = AtomicU64::new(0);

    unsafe extern "system" fn fake_native_signal(_: *mut c_void, _: *mut c_void, value: u64) -> i32 {
        NATIVE_SIGNAL.store(value, Ordering::Relaxed);
        0
    }

    unsafe extern "system" fn fake_shifted_wrapper(_: *mut c_void, _: *mut c_void, value: u64) -> i32 {
        WRAPPER_SIGNAL.store(value, Ordering::Relaxed);
        0
    }

    #[test]
    fn same_queue_is_reentrant_but_excludes_another_thread() {
        let queue = 0x5151_0001;
        let (entered_tx, entered_rx) = mpsc::channel();
        let (leave_tx, leave_rx) = mpsc::channel();
        let first = std::thread::spawn(move || {
            with(queue, || {
                with(queue, || entered_tx.send(()).unwrap());
                leave_rx.recv().unwrap();
            });
        });
        entered_rx.recv().unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (second_tx, second_rx) = mpsc::channel();
        let second = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            with(queue, || second_tx.send(()).unwrap())
        });
        ready_rx.recv().unwrap();
        let start = std::time::Instant::now();
        while old_waiter_count(queue) == 0 && start.elapsed() < Duration::from_secs(1) {
            std::thread::yield_now();
        }
        assert_eq!(old_waiter_count(queue), 1, "the second thread reached the held lease");
        assert!(second_rx.try_recv().is_err());
        leave_tx.send(()).unwrap();
        second_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        first.join().unwrap();
        second.join().unwrap();
        release(queue, || 0);
    }

    #[test]
    fn different_queues_do_not_share_a_turn() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (leave_tx, leave_rx) = mpsc::channel();
        let first = std::thread::spawn(move || with(0x5151_0002, || {
            entered_tx.send(()).unwrap();
            leave_rx.recv().unwrap();
        }));
        entered_rx.recv().unwrap();
        with(0x5151_0003, || {});
        leave_tx.send(()).unwrap();
        first.join().unwrap();
        release(0x5151_0002, || 0);
        release(0x5151_0003, || 0);
    }

    #[test]
    fn an_old_final_release_does_not_remove_a_reused_address() {
        let queue = 0x5151_0004;
        created(queue);
        let old = lease(queue);
        let n = release(queue, || {
            created(queue);
            0
        });
        assert_eq!(n, 0);
        assert!(!Arc::ptr_eq(&old, &lease(queue)));
        release(queue, || 0);
    }

    #[test]
    fn internal_signal_reenters_the_lease_and_uses_the_native_slot() {
        NATIVE_SIGNAL.store(0, Ordering::Relaxed);
        WRAPPER_SIGNAL.store(0, Ordering::Relaxed);
        let mut original = vec![0usize; 16];
        original[14] = fake_native_signal as usize;
        let mut extended = vec![0usize; 17];
        extended[0] = original.as_ptr() as usize;
        extended[15] = fake_shifted_wrapper as usize; // ext[14], as a shifted title layout sees it
        let ext = unsafe { extended.as_mut_ptr().add(1) };
        struct Registered(usize);
        impl Drop for Registered {
            fn drop(&mut self) { super::super::real::register_test_queue_vtable(self.0, false); }
        }
        super::super::real::register_test_queue_vtable(ext as usize, true);
        let _registered = Registered(ext as usize);
        let mut queue_object = [ext as usize];
        let mut fence_object = [0usize];
        let queue_raw = queue_object.as_mut_ptr() as *mut c_void;
        let fence_raw = fence_object.as_mut_ptr() as *mut c_void;
        let queue = unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue_raw).unwrap() };
        let fence = unsafe { ID3D12Fence::from_raw_borrowed(&fence_raw).unwrap() };

        with(queue_raw as usize, || unsafe {
            super::super::real::native_queue_signal(queue, fence, 0x5151).unwrap();
        });
        assert_eq!(NATIVE_SIGNAL.load(Ordering::Relaxed), 0x5151);
        assert_eq!(WRAPPER_SIGNAL.load(Ordering::Relaxed), 0, "the adopted title wrapper must be bypassed");
        release(queue_raw as usize, || 0);
    }

    fn old_waiter_count(queue: usize) -> usize {
        lease(queue).state.lock().unwrap_or_else(|e| e.into_inner()).waiters
    }
}
