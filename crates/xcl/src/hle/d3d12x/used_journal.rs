//! Ordered background append journal for `used-pipelines.txt`.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const COALESCE: Duration = Duration::from_millis(25);
const FULL_BATCH: usize = 64;

trait Sink: Send {
    fn append(&mut self, bytes: &[u8]) -> bool;
}

struct FileSink {
    path: PathBuf,
    file: Option<std::fs::File>,
}

impl Sink for FileSink {
    fn append(&mut self, bytes: &[u8]) -> bool {
        if self.file.is_none() {
            if let Some(dir) = self.path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            self.file = std::fs::OpenOptions::new().create(true).append(true).open(&self.path).ok();
        }
        let ok = self.file.as_mut().is_some_and(|file| file.write_all(bytes).and_then(|_| file.flush()).is_ok());
        if !ok {
            self.file = None;
        }
        ok
    }
}

#[derive(Clone)]
struct Batch {
    end_seq: u64,
    keys: Arc<[u64]>,
}

struct State {
    seen: HashSet<u64>,
    pending: Vec<(u64, u64)>,
    in_flight: Option<Batch>,
    fallback_path: Option<PathBuf>,
    fallback_writing: bool,
    next_seq: u64,
    completed_seq: u64,
    flush_now: bool,
    closed: bool,
    stopped: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    completed: Condvar,
}

/// A single writer owns the append handle. Producers only touch `Shared::state`; the batch being
/// written stays there until it succeeds, so a later producer can never hide or overtake it.
pub(super) struct Journal {
    shared: Arc<Shared>,
}

impl Journal {
    pub(super) fn file(path: PathBuf, seen: HashSet<u64>) -> Self {
        Self::with_sink(seen, Box::new(FileSink { path: path.clone(), file: None }), COALESCE, Some(path))
    }

    pub(super) fn memory(seen: HashSet<u64>) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                seen,
                pending: Vec::new(),
                in_flight: None,
                fallback_path: None,
                fallback_writing: false,
                next_seq: 0,
                completed_seq: 0,
                flush_now: false,
                closed: false,
                stopped: true,
            }),
            wake: Condvar::new(),
            completed: Condvar::new(),
        });
        Self { shared }
    }

    fn with_sink(seen: HashSet<u64>, sink: Box<dyn Sink>, coalesce: Duration, fallback_path: Option<PathBuf>) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                seen,
                pending: Vec::new(),
                in_flight: None,
                fallback_path,
                fallback_writing: false,
                next_seq: 0,
                completed_seq: 0,
                flush_now: false,
                closed: false,
                stopped: false,
            }),
            wake: Condvar::new(),
            completed: Condvar::new(),
        });
        let worker = shared.clone();
        if std::thread::Builder::new().name("xcl-pso-used".into()).spawn(move || writer(worker, sink, coalesce)).is_err() {
            shared.state.lock().unwrap().stopped = true;
        }
        Self { shared }
    }

    #[cfg(test)]
    fn stopped_file(path: PathBuf, seen: HashSet<u64>) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                seen,
                pending: Vec::new(),
                in_flight: None,
                fallback_path: Some(path),
                fallback_writing: false,
                next_seq: 0,
                completed_seq: 0,
                flush_now: false,
                closed: false,
                stopped: true,
            }),
            wake: Condvar::new(),
            completed: Condvar::new(),
        });
        Self { shared }
    }

    pub(super) fn note(&self, key: u64) {
        if key == 0 {
            return;
        }
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed || !state.seen.insert(key) {
            return;
        }
        // An in-memory journal still deduplicates. A file journal retains keys for the bounded
        // shutdown fallback when its worker could not start.
        if state.stopped && state.fallback_path.is_none() {
            return;
        }
        state.next_seq += 1;
        let seq = state.next_seq;
        state.pending.push((seq, key));
        if state.pending.len() == 1 || state.pending.len() == FULL_BATCH {
            self.shared.wake.notify_one();
        }
    }

    #[cfg(test)]
    pub(super) fn contains(&self, key: u64) -> bool { self.shared.state.lock().unwrap_or_else(|e| e.into_inner()).seen.contains(&key) }

    pub(super) fn flush_bounded(&self, timeout: Duration) -> bool { self.wait_bounded(timeout, false) }

    pub(super) fn shutdown_bounded(&self, timeout: Duration) -> bool { self.wait_bounded(timeout, true) }

    fn wait_bounded(&self, timeout: Duration, close: bool) -> bool {
        let deadline = Instant::now() + timeout;
        let mut state = loop {
            match self.shared.state.try_lock() {
                Ok(state) => break state,
                Err(std::sync::TryLockError::Poisoned(e)) => break e.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => {
                    if Instant::now() >= deadline {
                        return false;
                    }
                    std::thread::yield_now();
                }
            }
        };
        if close {
            state.closed = true;
        }
        let target = state.next_seq;
        state.flush_now = true;
        self.shared.wake.notify_one();
        while (state.completed_seq < target || close && !state.stopped) && !state.stopped {
            let Some(left) = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) else { break };
            let (next, waited) = self.shared.completed.wait_timeout(state, left).unwrap_or_else(|e| e.into_inner());
            state = next;
            if waited.timed_out() {
                break;
            }
        }
        let complete = state.completed_seq >= target && (!close || state.stopped);
        drop(state);
        complete || self.append_after_worker_stopped()
    }

    /// A failed spawn (or a worker panic) has dropped its sink, so this is the only writer. Keep
    /// the bind path memory-only and persist its ordered backlog at an explicit flush boundary.
    fn append_after_worker_stopped(&self) -> bool {
        let (path, end_seq, interrupted_append, keys) = {
            let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if !state.stopped || state.fallback_writing {
                return false;
            }
            let Some(path) = state.fallback_path.clone() else { return false };
            let mut entries = Vec::new();
            let interrupted_append = state.in_flight.is_some();
            if let Some(batch) = &state.in_flight {
                entries.extend(batch.keys.iter().copied().map(|key| (batch.end_seq, key)));
            }
            entries.extend(state.pending.iter().copied());
            let Some(end_seq) = entries.last().map(|entry| entry.0) else { return true };
            state.fallback_writing = true;
            (path, end_seq, interrupted_append, entries.into_iter().map(|(_, key)| key).collect::<Vec<_>>())
        };

        // A stopped worker may have left a partial hexadecimal line. The marker makes that
        // fragment invalid before the complete records are retried without rejecting old files
        // that used shorter hexadecimal keys.
        let mut bytes = String::new();
        if interrupted_append {
            bytes.push_str("!\n");
        }
        bytes.reserve(keys.len() * 17);
        for key in keys {
            let _ = writeln!(bytes, "{key:016x}");
        }
        let written = (FileSink { path, file: None }).append(bytes.as_bytes());

        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.fallback_writing = false;
        if written && state.stopped {
            state.in_flight = state.in_flight.take().filter(|batch| batch.end_seq > end_seq);
            state.pending.retain(|(seq, _)| *seq > end_seq);
            state.completed_seq = state.completed_seq.max(end_seq);
            self.shared.completed.notify_all();
        }
        written && state.completed_seq >= end_seq
    }
}

impl Drop for Journal {
    fn drop(&mut self) {
        let _ = self.shutdown_bounded(Duration::from_millis(100));
    }
}

struct Stopped {
    shared: Arc<Shared>,
    sink: Option<Box<dyn Sink>>,
}

impl Drop for Stopped {
    fn drop(&mut self) {
        // Publish `stopped` only after the worker's append handle is gone. The fallback can then
        // open the path without ever overlapping the worker, including during panic unwinding.
        drop(self.sink.take());
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.stopped = true;
        self.shared.completed.notify_all();
    }
}

fn writer(shared: Arc<Shared>, sink: Box<dyn Sink>, coalesce: Duration) {
    let mut stopped = Stopped { shared: shared.clone(), sink: Some(sink) };
    let mut bytes = String::new();
    let mut retrying = false;
    loop {
        let batch = {
            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.in_flight.is_none() && state.pending.is_empty() && !state.closed {
                state = shared.wake.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            if state.in_flight.is_none() && !state.pending.is_empty() && !coalesce.is_zero() && !state.flush_now && !state.closed {
                let until = Instant::now() + coalesce;
                while state.pending.len() < FULL_BATCH && !state.flush_now && !state.closed {
                    let Some(left) = until.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) else { break };
                    let (next, waited) = shared.wake.wait_timeout(state, left).unwrap_or_else(|e| e.into_inner());
                    state = next;
                    if waited.timed_out() {
                        break;
                    }
                }
            }
            state.flush_now = false;
            if state.in_flight.is_none() && !state.pending.is_empty() {
                let pending = std::mem::take(&mut state.pending);
                let end_seq = pending.last().unwrap().0;
                state.in_flight = Some(Batch { end_seq, keys: pending.into_iter().map(|(_, key)| key).collect() });
            }
            let Some(batch) = state.in_flight.clone() else {
                if state.closed { break } else { continue };
            };
            batch
        };

        bytes.clear();
        if retrying {
            // `write_all` may have appended an arbitrary prefix before reporting failure. Make
            // that fragment invalid so replay cannot rank a torn hexadecimal prefix as a key.
            bytes.push_str("!\n");
        }
        bytes.reserve(batch.keys.len() * 17);
        for key in batch.keys.iter() {
            let _ = writeln!(bytes, "{key:016x}");
        }
        let written = stopped.sink.as_mut().unwrap().append(bytes.as_bytes());
        retrying = !written;
        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if written {
            if state.in_flight.as_ref().is_some_and(|current| current.end_seq == batch.end_seq) {
                state.in_flight = None;
                state.completed_seq = state.completed_seq.max(batch.end_seq);
                shared.completed.notify_all();
            }
        } else {
            let (next, _) = shared.wake.wait_timeout(state, coalesce.max(Duration::from_millis(1))).unwrap_or_else(|e| e.into_inner());
            drop(next);
        }
    }
    drop(stopped);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    struct Blocking {
        first: bool,
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        output: Arc<Mutex<Vec<u8>>>,
    }

    impl Sink for Blocking {
        fn append(&mut self, bytes: &[u8]) -> bool {
            if self.first {
                self.first = false;
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
            }
            self.output.lock().unwrap().extend_from_slice(bytes);
            true
        }
    }

    #[test]
    fn flush_waits_for_the_shared_in_flight_batch_and_later_producers() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let output = Arc::new(Mutex::new(Vec::new()));
        let journal = Arc::new(Journal::with_sink(
            HashSet::new(),
            Box::new(Blocking { first: true, entered: entered_tx, release: release_rx, output: output.clone() }),
            Duration::ZERO,
            None,
        ));
        journal.note(1);
        entered_rx.recv_timeout(Duration::from_secs(2)).expect("writer did not take the first batch");
        journal.note(2);
        let waiter = journal.clone();
        let (flushed_tx, flushed_rx) = mpsc::channel();
        std::thread::spawn(move || flushed_tx.send(waiter.flush_bounded(Duration::from_secs(2))).unwrap());
        assert!(matches!(flushed_rx.recv_timeout(Duration::from_millis(20)), Err(mpsc::RecvTimeoutError::Timeout)));
        release_tx.send(()).unwrap();
        assert!(flushed_rx.recv_timeout(Duration::from_secs(2)).unwrap());
        assert_eq!(String::from_utf8(output.lock().unwrap().clone()).unwrap(), "0000000000000001\n0000000000000002\n");
        assert!(journal.shutdown_bounded(Duration::from_secs(2)));
    }

    struct FailOnce {
        failed: bool,
        output: Arc<Mutex<Vec<u8>>>,
    }

    impl Sink for FailOnce {
        fn append(&mut self, bytes: &[u8]) -> bool {
            if !self.failed {
                self.failed = true;
                self.output.lock().unwrap().extend_from_slice(&bytes[..8.min(bytes.len())]);
                return false;
            }
            self.output.lock().unwrap().extend_from_slice(bytes);
            true
        }
    }

    #[test]
    fn failed_batch_is_retried_before_later_keys() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let journal = Journal::with_sink(
            HashSet::new(),
            Box::new(FailOnce { failed: false, output: output.clone() }),
            Duration::from_millis(1),
            None,
        );
        journal.note(0xdead_beef_0000_0003);
        journal.note(0xdead_beef_0000_0004);
        assert!(journal.flush_bounded(Duration::from_secs(2)));
        journal.note(0xdead_beef_0000_0005);
        assert!(journal.flush_bounded(Duration::from_secs(2)));
        let text = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        let keys: Vec<u64> = text.lines().filter_map(|line| u64::from_str_radix(line, 16).ok()).filter(|&key| key != 0).collect();
        assert_eq!(keys, [0xdead_beef_0000_0003, 0xdead_beef_0000_0004, 0xdead_beef_0000_0005]);
        assert!(journal.shutdown_bounded(Duration::from_secs(2)));
    }

    #[test]
    fn shutdown_is_bounded_when_the_only_writer_is_blocked() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let journal = Journal::with_sink(
            HashSet::new(),
            Box::new(Blocking { first: true, entered: entered_tx, release: release_rx, output: Arc::new(Mutex::new(Vec::new())) }),
            Duration::ZERO,
            None,
        );
        journal.note(6);
        entered_rx.recv_timeout(Duration::from_secs(2)).expect("writer did not block");
        let start = Instant::now();
        assert!(!journal.shutdown_bounded(Duration::from_millis(20)));
        assert!(start.elapsed() < Duration::from_millis(500), "bounded shutdown waited {:?}", start.elapsed());
        release_tx.send(()).unwrap();
        assert!(journal.shutdown_bounded(Duration::from_secs(2)));
    }

    #[test]
    fn worker_start_failure_defers_ordered_unique_writes_until_shutdown() {
        let path = std::env::temp_dir().join(format!("xcl-used-no-worker-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let journal = Journal::stopped_file(path.clone(), [7].into());
        journal.note(8);
        journal.note(7);
        journal.note(9);
        journal.note(8);
        assert!(!path.exists(), "the producer performed file I/O");
        assert!(journal.shutdown_bounded(Duration::from_secs(2)));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "0000000000000008\n0000000000000009\n");
        let _ = std::fs::remove_file(path);
    }
}
