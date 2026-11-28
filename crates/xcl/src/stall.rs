//! Opt-in attribution of time spent in coarse runtime operations.
//!
//! A disabled span is one relaxed atomic load. In particular, it does not read the clock, allocate,
//! lock, or update a counter. High-frequency command-recording spans have a separate detail switch,
//! so enabling coarse attribution does not turn them on.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub const HISTOGRAM_BUCKETS: usize = 64;
pub const SLOW_SPAN_CAPACITY: usize = 128;
pub const SLOW_SPAN_NS: u64 = 50_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum StallKind {
    ShaderTranslation,
    DxcQueue,
    DxcCompile,
    PipelineQueueAdmission,
    PipelineQueueSpeculative,
    PipelineQueueDemand,
    PipelineWait,
    PipelineRestoreDemand,
    PipelineRestoreWait,
    PipelineRestoreReconcile,
    DriverCompileGate,
    DriverCompile,
    SubmitPreparation,
    MemorySynchronization,
    Residency,
    CacheIo,
    LockContention,
    GpuWait,
    Present,
    DirectStorage,
    DetailDraw,
    DetailDispatch,
    DetailBind,
    DetailViewDescriptor,
    DetailBarrier,
    DetailCopy,
    DetailListLifecycle,
}

impl StallKind {
    pub const ALL: [Self; 27] = [
        Self::ShaderTranslation,
        Self::DxcQueue,
        Self::DxcCompile,
        Self::PipelineQueueAdmission,
        Self::PipelineQueueSpeculative,
        Self::PipelineQueueDemand,
        Self::PipelineWait,
        Self::PipelineRestoreDemand,
        Self::PipelineRestoreWait,
        Self::PipelineRestoreReconcile,
        Self::DriverCompileGate,
        Self::DriverCompile,
        Self::SubmitPreparation,
        Self::MemorySynchronization,
        Self::Residency,
        Self::CacheIo,
        Self::LockContention,
        Self::GpuWait,
        Self::Present,
        Self::DirectStorage,
        Self::DetailDraw,
        Self::DetailDispatch,
        Self::DetailBind,
        Self::DetailViewDescriptor,
        Self::DetailBarrier,
        Self::DetailCopy,
        Self::DetailListLifecycle,
    ];

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AggregateSnapshot {
    pub kind: StallKind,
    pub count: u64,
    pub total_ns: u64,
    pub max_ns: u64,
    /// Duration counts in base-2 nanosecond buckets. Bucket zero also contains zero-duration spans.
    pub histogram: [u64; HISTOGRAM_BUCKETS],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlowSpan {
    pub sequence: u64,
    pub kind: StallKind,
    pub started_ns: u64,
    pub duration_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    pub aggregates: Vec<AggregateSnapshot>,
    pub slow_spans: Vec<SlowSpan>,
}

struct Aggregate {
    count: AtomicU64,
    total_ns: AtomicU64,
    max_ns: AtomicU64,
    histogram: [AtomicU64; HISTOGRAM_BUCKETS],
}

impl Aggregate {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
            max_ns: AtomicU64::new(0),
            histogram: [const { AtomicU64::new(0) }; HISTOGRAM_BUCKETS],
        }
    }

    fn record(&self, duration_ns: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(duration_ns, Ordering::Relaxed);
        self.max_ns.fetch_max(duration_ns, Ordering::Relaxed);
        self.histogram[histogram_bucket(duration_ns)].fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self, kind: StallKind) -> AggregateSnapshot {
        AggregateSnapshot {
            kind,
            count: self.count.load(Ordering::Relaxed),
            total_ns: self.total_ns.load(Ordering::Relaxed),
            max_ns: self.max_ns.load(Ordering::Relaxed),
            histogram: std::array::from_fn(|i| self.histogram[i].load(Ordering::Relaxed)),
        }
    }
}

const fn histogram_bucket(duration_ns: u64) -> usize {
    if duration_ns == 0 { 0 } else { (u64::BITS - 1 - duration_ns.leading_zeros()) as usize }
}

struct SlowSpans {
    next_sequence: u64,
    spans: VecDeque<SlowSpan>,
}

struct Recorder {
    enabled: AtomicBool,
    detail_enabled: AtomicBool,
    /// First title thread observed submitting command lists; zero until then.
    submitter: AtomicU32,
    aggregates: [Aggregate; StallKind::ALL.len()],
    slow_spans: Mutex<SlowSpans>,
}

trait Clock: Sync {
    fn now_ns(&self) -> u64;
}

impl Recorder {
    const fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            detail_enabled: AtomicBool::new(false),
            submitter: AtomicU32::new(0),
            aggregates: [const { Aggregate::new() }; StallKind::ALL.len()],
            slow_spans: Mutex::new(SlowSpans { next_sequence: 0, spans: VecDeque::new() }),
        }
    }

    fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    fn enable_detail(&self) {
        self.enable();
        self.detail_enabled.store(true, Ordering::Relaxed);
    }

    fn note_submitter(&self, thread: u32) {
        let _ = self.submitter.compare_exchange(0, thread, Ordering::Relaxed, Ordering::Relaxed);
    }

    #[inline]
    fn start_with<'a>(&'a self, kind: StallKind, clock: &'a dyn Clock) -> Option<Span<'a>> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }
        Some(Span { recorder: self, kind, started_ns: clock.now_ns(), clock, finished: false })
    }

    #[inline]
    fn start_detail_with<'a>(&'a self, kind: StallKind, clock: &'a dyn Clock, thread: u32) -> Option<Span<'a>> {
        if !self.detail_enabled.load(Ordering::Relaxed) || self.submitter.load(Ordering::Relaxed) != thread {
            return None;
        }
        Some(Span { recorder: self, kind, started_ns: clock.now_ns(), clock, finished: false })
    }

    #[inline]
    fn defer_with(&self, clock: &dyn Clock) -> Option<DeferredSpan> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }
        Some(DeferredSpan { started_ns: clock.now_ns() })
    }

    fn record(&self, kind: StallKind, started_ns: u64, duration_ns: u64) {
        self.aggregates[kind.index()].record(duration_ns);
        if duration_ns < SLOW_SPAN_NS {
            return;
        }
        let mut slow = self.slow_spans.lock().unwrap_or_else(|e| e.into_inner());
        let span = SlowSpan {
            sequence: slow.next_sequence,
            kind,
            started_ns,
            duration_ns,
        };
        slow.next_sequence = slow.next_sequence.wrapping_add(1);
        if slow.spans.len() == SLOW_SPAN_CAPACITY {
            slow.spans.pop_front();
        }
        slow.spans.push_back(span);
    }

    fn snapshot(&self) -> Snapshot {
        let aggregates = StallKind::ALL.iter().map(|&kind| self.aggregates[kind.index()].snapshot(kind)).collect();
        let slow_spans = self.slow_spans.lock().unwrap_or_else(|e| e.into_inner()).spans.iter().copied().collect();
        Snapshot { aggregates, slow_spans }
    }
}

pub struct Span<'a> {
    recorder: &'a Recorder,
    kind: StallKind,
    started_ns: u64,
    clock: &'a dyn Clock,
    finished: bool,
}

/// A duration whose kind is chosen when it finishes. Dropping it records nothing, so work refused
/// or cancelled before it starts does not look like completed queue latency.
pub struct DeferredSpan {
    started_ns: u64,
}

impl DeferredSpan {
    fn stop_with(self, clock: &dyn Clock) -> DeferredDuration {
        DeferredDuration {
            started_ns: self.started_ns,
            duration_ns: clock.now_ns().saturating_sub(self.started_ns),
        }
    }

    pub fn stop(self) -> DeferredDuration {
        self.stop_with(&CLOCK)
    }

    pub fn finish(self, kind: StallKind) {
        self.stop().finish(kind);
    }
}

/// A stopped deferred duration. Dropping it records nothing.
pub struct DeferredDuration {
    started_ns: u64,
    duration_ns: u64,
}

impl DeferredDuration {
    fn finish_with(self, recorder: &Recorder, kind: StallKind) {
        recorder.record(kind, self.started_ns, self.duration_ns);
    }

    pub fn finish(self, kind: StallKind) {
        self.finish_with(&RECORDER, kind);
    }
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.finished = true;
            self.recorder.record(self.kind, self.started_ns, self.clock.now_ns().saturating_sub(self.started_ns));
        }
    }
}

static RECORDER: Recorder = Recorder::new();
static CLOCK: SystemClock = SystemClock;

pub fn enable() {
    RECORDER.enable();
}

pub fn enable_detail() {
    RECORDER.enable_detail();
}

/// Mark the first title thread that submits command lists as the thread detailed spans describe.
pub fn note_submitter() {
    RECORDER.note_submitter(current_thread());
}

#[inline]
pub fn span(kind: StallKind) -> Option<Span<'static>> {
    RECORDER.start_with(kind, &CLOCK)
}

/// Begin an opt-in duration which records only when [`DeferredSpan::finish`] is called.
#[inline]
pub fn deferred_span() -> Option<DeferredSpan> {
    RECORDER.defer_with(&CLOCK)
}

/// A high-frequency command-recording span, enabled only by `XCL_STALL_TRACE_DETAIL`.
#[inline]
pub fn detail_span(kind: StallKind) -> Option<Span<'static>> {
    if !RECORDER.detail_enabled.load(Ordering::Relaxed) {
        return None;
    }
    RECORDER.start_detail_with(kind, &CLOCK, current_thread())
}

/// Like [`detail_span`], without even consulting the detail switch when `on` is false.
#[inline]
pub fn detail_span_if(on: bool, kind: StallKind) -> Option<Span<'static>> {
    on.then(|| detail_span(kind)).flatten()
}

pub fn snapshot() -> Option<Snapshot> {
    RECORDER.enabled.load(Ordering::Relaxed).then(|| RECORDER.snapshot())
}

struct SystemClock;

fn current_thread() -> u32 {
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

impl Clock for SystemClock {
    fn now_ns(&self) -> u64 {
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        ORIGIN.get_or_init(Instant::now).elapsed().as_nanos().min(u64::MAX as u128) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize};

    struct TestClock {
        now: AtomicU64,
        calls: AtomicUsize,
    }

    impl TestClock {
        fn new(now: u64) -> Self {
            Self { now: AtomicU64::new(now), calls: AtomicUsize::new(0) }
        }
    }

    impl Clock for TestClock {
        fn now_ns(&self) -> u64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.now.load(Ordering::Relaxed)
        }
    }

    #[test]
    fn disabled_span_does_not_read_the_clock_or_record() {
        let recorder = Recorder::new();
        let clock = TestClock::new(10);
        assert!(recorder.start_with(StallKind::DxcCompile, &clock).is_none());
        assert!(recorder.defer_with(&clock).is_none());
        assert_eq!(clock.calls.load(Ordering::Relaxed), 0);
        assert_eq!(recorder.snapshot().aggregates[StallKind::DxcCompile.index()].count, 0);
    }

    #[test]
    fn deferred_spans_split_queue_bands_and_abandoned_work_records_nothing() {
        let recorder = Recorder::new();
        recorder.enable();
        let clock = TestClock::new(10);

        let speculative = recorder.defer_with(&clock).unwrap();
        clock.now.store(30, Ordering::Relaxed);
        let speculative = speculative.stop_with(&clock);
        let demand = recorder.defer_with(&clock).unwrap();
        clock.now.store(75, Ordering::Relaxed);
        speculative.finish_with(&recorder, StallKind::PipelineQueueSpeculative);
        demand.stop_with(&clock).finish_with(&recorder, StallKind::PipelineQueueDemand);
        let abandoned_previous = recorder.defer_with(&clock).unwrap();
        clock.now.store(100, Ordering::Relaxed);
        let abandoned_previous = abandoned_previous.stop_with(&clock);
        let abandoned_active = recorder.defer_with(&clock).unwrap();
        clock.now.store(1_000, Ordering::Relaxed);
        drop((abandoned_previous, abandoned_active));

        let snapshot = recorder.snapshot();
        let speculative = &snapshot.aggregates[StallKind::PipelineQueueSpeculative.index()];
        let demand = &snapshot.aggregates[StallKind::PipelineQueueDemand.index()];
        assert_eq!((speculative.count, speculative.total_ns), (1, 20));
        assert_eq!((demand.count, demand.total_ns), (1, 45));
        assert_eq!(clock.calls.load(Ordering::Relaxed), 7, "abandoning deferred work must not read the clock again");
    }

    #[test]
    fn coarse_tracing_does_not_enable_detail_spans() {
        let recorder = Recorder::new();
        recorder.enable();
        let clock = TestClock::new(10);
        recorder.note_submitter(7);
        assert!(recorder.start_detail_with(StallKind::DetailDraw, &clock, 7).is_none());
        assert_eq!(clock.calls.load(Ordering::Relaxed), 0);

        recorder.enable_detail();
        assert!(recorder.start_detail_with(StallKind::DetailDraw, &clock, 8).is_none());
        let span = recorder.start_detail_with(StallKind::DetailDraw, &clock, 7).unwrap();
        clock.now.store(42, Ordering::Relaxed);
        drop(span);
        let aggregate = &recorder.snapshot().aggregates[StallKind::DetailDraw.index()];
        assert_eq!((aggregate.count, aggregate.total_ns), (1, 32));
    }

    #[test]
    fn descriptor_detail_is_attributed_only_to_the_submitter() {
        let recorder = Recorder::new();
        recorder.enable_detail();
        recorder.note_submitter(7);
        let clock = TestClock::new(100);
        assert!(recorder.start_detail_with(StallKind::DetailViewDescriptor, &clock, 8).is_none());
        let span = recorder.start_detail_with(StallKind::DetailViewDescriptor, &clock, 7).unwrap();
        clock.now.store(145, Ordering::Relaxed);
        drop(span);
        let aggregate = &recorder.snapshot().aggregates[StallKind::DetailViewDescriptor.index()];
        assert_eq!((aggregate.count, aggregate.total_ns), (1, 45));
    }

    #[test]
    fn a_span_updates_its_aggregate_and_log2_bucket() {
        let recorder = Recorder::new();
        recorder.enable();
        let clock = TestClock::new(10);
        let span = recorder.start_with(StallKind::PipelineWait, &clock).unwrap();
        clock.now.store(10 + 1024, Ordering::Relaxed);
        drop(span);

        let aggregate = &recorder.snapshot().aggregates[StallKind::PipelineWait.index()];
        assert_eq!((aggregate.count, aggregate.total_ns, aggregate.max_ns), (1, 1024, 1024));
        assert_eq!(aggregate.histogram[10], 1);
        assert_eq!(aggregate.histogram.iter().sum::<u64>(), 1);
        assert_eq!(clock.calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn only_slow_spans_enter_the_bounded_ring() {
        let recorder = Recorder::new();
        recorder.enable();
        let clock = TestClock::new(0);
        for i in 0..SLOW_SPAN_CAPACITY + 3 {
            clock.now.store(i as u64, Ordering::Relaxed);
            let span = recorder.start_with(StallKind::GpuWait, &clock).unwrap();
            clock.now.store(i as u64 + SLOW_SPAN_NS, Ordering::Relaxed);
            drop(span);
        }
        clock.now.store(1_000_000_000, Ordering::Relaxed);
        let fast = recorder.start_with(StallKind::Present, &clock).unwrap();
        clock.now.store(1_000_000_001, Ordering::Relaxed);
        drop(fast);

        let snapshot = recorder.snapshot();
        assert_eq!(snapshot.slow_spans.len(), SLOW_SPAN_CAPACITY);
        assert_eq!(snapshot.slow_spans[0].sequence, 3);
        assert!(snapshot.slow_spans.iter().all(|span| span.kind == StallKind::GpuWait));
        assert_eq!(snapshot.aggregates[StallKind::Present.index()].count, 1);
    }
}
