//! `report.json`: what the title imported, which stubs it actually called, which console
//! paths were not mapped, which game files were redirected – the to-do list for the
//! next iteration.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::resolver::{self, Source};

pub const STALL_SUMMARY_VERSION: u32 = 1;
pub const PREPARATION_SUMMARY_VERSION: u32 = 3;

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct StubCall {
    pub dll: String,
    pub function: String,
    pub calls: u64,
    pub first_caller: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct MissingImport {
    pub importer: String,
    pub dll: String,
    pub function: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ModuleInfo {
    pub name: String,
    pub base: String,
    pub size: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct VfsInfo {
    pub unmapped_paths: Vec<String>,
    pub redirected_writes: Vec<String>,
    pub whiteouts: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct StallAggregate {
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub count: u64,
    #[serde(default)]
    pub total_ns: u64,
    #[serde(default)]
    pub max_ns: u64,
    /// Index N counts durations from 2^N through 2^(N+1)-1 nanoseconds; bucket zero
    /// also contains zero-duration spans.
    #[serde(default)]
    pub histogram_log2_ns: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct StallEvent {
    #[serde(default)]
    pub sequence: u64,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub started_ns: u64,
    #[serde(default)]
    pub duration_ns: u64,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct StallSummary {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub aggregates: Vec<StallAggregate>,
    #[serde(default)]
    pub slow_events: Vec<StallEvent>,
}

fn stall_kind_name(kind: crate::stall::StallKind) -> &'static str {
    use crate::stall::StallKind;
    match kind {
        StallKind::ShaderTranslation => "shader_translation",
        StallKind::DxcQueue => "dxc_queue",
        StallKind::DxcCompile => "dxc_compile",
        StallKind::PipelineQueueAdmission => "pipeline_queue_admission",
        StallKind::PipelineQueueSpeculative => "pipeline_queue_speculative",
        StallKind::PipelineQueueDemand => "pipeline_queue_demand",
        StallKind::PipelineWait => "pipeline_wait",
        StallKind::PipelineRestoreDemand => "pipeline_restore_demand",
        StallKind::PipelineRestoreWait => "pipeline_restore_wait",
        StallKind::PipelineRestoreReconcile => "pipeline_restore_reconcile",
        StallKind::DriverCompileGate => "driver_compile_gate",
        StallKind::DriverCompile => "driver_compile",
        StallKind::SubmitPreparation => "submit_preparation",
        StallKind::MemorySynchronization => "memory_synchronization",
        StallKind::Residency => "residency",
        StallKind::CacheIo => "cache_io",
        StallKind::LockContention => "lock_contention",
        StallKind::GpuWait => "gpu_wait",
        StallKind::Present => "present",
        StallKind::DirectStorage => "direct_storage",
        StallKind::DetailDraw => "detail_draw",
        StallKind::DetailDispatch => "detail_dispatch",
        StallKind::DetailBind => "detail_bind",
        StallKind::DetailViewDescriptor => "detail_view_descriptor",
        StallKind::DetailBarrier => "detail_barrier",
        StallKind::DetailCopy => "detail_copy",
        StallKind::DetailListLifecycle => "detail_list_lifecycle",
    }
}

impl From<crate::stall::Snapshot> for StallSummary {
    fn from(snapshot: crate::stall::Snapshot) -> Self {
        let slow_skip = snapshot.slow_spans.len().saturating_sub(crate::stall::SLOW_SPAN_CAPACITY);
        Self {
            version: STALL_SUMMARY_VERSION,
            aggregates: snapshot
                .aggregates
                .into_iter()
                .map(|a| StallAggregate {
                    kind: stall_kind_name(a.kind).into(),
                    count: a.count,
                    total_ns: a.total_ns,
                    max_ns: a.max_ns,
                    histogram_log2_ns: a.histogram.into(),
                })
                .collect(),
            slow_events: snapshot
                .slow_spans
                .into_iter()
                .skip(slow_skip)
                .map(|s| StallEvent {
                    sequence: s.sequence,
                    kind: stall_kind_name(s.kind).into(),
                    started_ns: s.started_ns,
                    duration_ns: s.duration_ns,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct PreparationSummary {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub compute_build_coalescing: crate::hle::d3d12x::pso_async::ComputeCoalescingStats,
    #[serde(default)]
    pub ranked_preloads: crate::hle::d3d12x::pso_async::RankedPreloadStats,
    #[serde(default)]
    pub allocator_submits: crate::hle::d3d12x::AllocatorSubmitStats,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Report {
    pub xcl_version: String,
    pub title: String,
    pub exe: String,
    pub crash: Option<String>,
    pub exit_code: Option<i64>,
    /// what the PC has of what a run needs (DXC, GPU features), checked by `xcl run` first
    #[serde(default)]
    pub environment: Vec<crate::preflight::Check>,
    /// counts of static imports by how they were resolved
    pub imports_by_source: BTreeMap<String, usize>,
    /// imports that ended up as auto-stubs (static analysis)
    pub stubbed_imports: Vec<MissingImport>,
    /// stubs the title actually called at run time (sorted by call count)
    pub called_stubs: Vec<StubCall>,
    pub modules: Vec<ModuleInfo>,
    pub vfs: VfsInfo,
    /// shader formats seen at pipeline creation (dxil / dxbc / container-unknown / rdna2-raw …)
    #[serde(default)]
    pub shaders: crate::gpu::shader::ShaderStats,
    /// the DXC that compiled them: folder and versions of dxcompiler.dll and dxil.dll (empty:
    /// none was loaded)
    #[serde(default)]
    pub dxc: crate::gpu::recomp::dxc::Toolchain,
    /// Versioned preparation-plane counters. Optional so reports from older XCL builds load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation: Option<PreparationSummary>,
    /// D3D12.X backend: adapter, interface layouts, unknown IIDs, fallbacks, presented frames
    #[serde(default)]
    pub d3d12: crate::hle::d3d12x::real::D3d12Stats,
    /// XAudio2: engines, mastering/source voices, XMA voices played as silence, dropped console flags
    #[serde(default)]
    pub audio: crate::hle::xaudio2::AudioStats,
    /// XGameRuntime through the GDK thunks: which classes the title asked for and who answered,
    /// XCL slots it called that XCL does not implement
    #[serde(default)]
    pub xgameruntime: crate::hle::xgameruntime::api::ApiStats,
    /// DirectStorage: BCPack requests unpacked and refused (no decoder) by mode, and bytes filled under
    /// `XCL_DSTORAGE_BCPACK_FILL`
    #[serde(default)]
    pub dstorage: crate::hle::dstorage::DStorageStats,
    /// Coarse runtime timings when `XCL_STALL_TRACE` was enabled. The events contain only
    /// categories and monotonic times, never title paths, shader contents, or pipeline keys.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stalls: Option<StallSummary>,
    pub notes: Vec<String>,
}

static CRASH: Mutex<Option<String>> = Mutex::new(None);
static WRITE_LOCK: Mutex<()> = Mutex::new(());

fn source_label(s: &Source) -> String {
    match s {
        Source::Hle(m) => format!("hle:{m}"),
        Source::HleOther(m) => format!("hle-cross:{m}"),
        Source::System(d) => format!("system:{d}"),
        Source::GameDll(d) => format!("game-dll:{d}"),
        Source::Stub => "stub".into(),
    }
}

pub fn build() -> Report {
    let boot = crate::state::boot();
    let records = resolver::records();
    let mut by_source: BTreeMap<String, usize> = BTreeMap::new();
    let mut stubbed = Vec::new();
    for r in &records {
        let label = match &r.source {
            Source::System(_) => "system".to_string(),
            other => source_label(other),
        };
        *by_source.entry(label).or_default() += 1;
        if r.source == Source::Stub {
            stubbed.push(MissingImport { importer: r.importer.clone(), dll: r.dll.clone(), function: r.function.clone() });
        }
    }
    let mut called: Vec<StubCall> = crate::stubs::all_stubs()
        .into_iter()
        .filter(|s| s.calls.load(Ordering::Relaxed) > 0)
        .map(|s| StubCall {
            dll: s.dll.clone(),
            function: s.name.clone(),
            calls: s.calls.load(Ordering::Relaxed),
            first_caller: crate::modules::describe_address(s.first_caller.load(Ordering::Relaxed)),
        })
        .collect();
    called.sort_by(|a, b| b.calls.cmp(&a.calls));
    let modules = crate::modules::mapped_modules()
        .iter()
        .map(|m| ModuleInfo { name: m.name.clone(), base: format!("{:#x}", m.base), size: format!("{:#x}", m.size) })
        .collect();
    let vfs = crate::vfs::get()
        .map(|v| VfsInfo { unmapped_paths: v.unmapped_paths(), redirected_writes: v.redirected_writes(), whiteouts: v.whiteouts() })
        .unwrap_or_default();
    Report {
        xcl_version: env!("CARGO_PKG_VERSION").to_string(),
        title: boot.map(|b| b.title.clone()).unwrap_or_default(),
        exe: boot.map(|b| b.exe_path.to_string_lossy().into_owned()).unwrap_or_default(),
        crash: CRASH.lock().ok().and_then(|g| g.clone()),
        exit_code: None,
        environment: Vec::new(),
        imports_by_source: by_source,
        stubbed_imports: stubbed,
        called_stubs: called,
        modules,
        vfs,
        shaders: crate::gpu::shader::stats(),
        dxc: crate::gpu::recomp::dxc::toolchain(),
        preparation: Some(PreparationSummary {
            version: PREPARATION_SUMMARY_VERSION,
            compute_build_coalescing: crate::hle::d3d12x::pso_async::compute_coalescing_stats(),
            ranked_preloads: crate::hle::d3d12x::pso_async::ranked_preload_stats(),
            allocator_submits: crate::hle::d3d12x::allocator_submit_stats(),
        }),
        d3d12: crate::hle::d3d12x::real::stats(),
        audio: crate::hle::xaudio2::stats(),
        xgameruntime: crate::hle::xgameruntime::api::stats(),
        dstorage: crate::hle::dstorage::stats(),
        stalls: crate::stall::snapshot().map(Into::into),
        notes: vec![
            "D3D12.X methods missing on PC show up in called_stubs as '<Interface>::#slot (not on PC)'.".into(),
            "Console-format shaders are replaced (fallback_pipelines) until the RDNA2 recompiler is finished.".into(),
            "Direct Nt* calls on files do not go through the VFS.".into(),
        ],
    }
}

/// `m.try_lock()`, retried for `wait`: never an endless wait – the holder may be this very
/// thread, crashed inside `build()`.
fn lock_within<T>(m: &Mutex<T>, wait: Duration) -> Option<MutexGuard<'_, T>> {
    let end = Instant::now() + wait;
    loop {
        match m.try_lock() {
            Ok(g) => return Some(g),
            Err(TryLockError::WouldBlock) if Instant::now() < end => std::thread::sleep(Duration::from_millis(10)),
            Err(_) => return None,
        }
    }
}

/// Write `report.json` (host side). `crash` is remembered for later writes.
pub fn write_final(crash: Option<String>) {
    // the process ends: a cursor held in the XCL window is let go
    crate::hle::kbm::release();
    // `XCL_DIRECT_DXIL_STATS=1`: the direct DXIL translator's live fallback summary
    crate::gpu::dxil::coverage::dump_fallbacks();
    // a crash is kept even while another thread writes the report, and is written once that
    // thread is done – the process ends right after (#106); other calls give way at once
    let wait = if crash.is_some() { Duration::from_secs(2) } else { Duration::ZERO };
    if let Some(c) = crash {
        if let Some(mut g) = lock_within(&CRASH, wait) {
            g.get_or_insert(c);
        }
    }
    let Some(_g) = lock_within(&WRITE_LOCK, wait) else { return };
    let Some(boot) = crate::state::boot() else { return };
    let r = build();
    if let Ok(s) = serde_json::to_string_pretty(&r) {
        let _ = std::fs::write(boot.report_path(), s);
    }
}

pub fn load(path: &std::path::Path) -> Option<Report> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_reports_without_a_stall_summary_still_deserialize() {
        let old = serde_json::to_value(Report::default()).unwrap();
        assert!(old.get("stalls").is_none(), "an absent summary is not serialized");
        assert!(old.get("preparation").is_none(), "an absent preparation summary is not serialized");
        let report: Report = serde_json::from_value(old).unwrap();
        assert!(report.stalls.is_none());
        assert!(report.preparation.is_none());
    }

    #[test]
    fn preparation_summary_has_a_versioned_stable_json_shape() {
        let summary = PreparationSummary {
            version: PREPARATION_SUMMARY_VERSION,
            compute_build_coalescing: crate::hle::d3d12x::pso_async::ComputeCoalescingStats {
                keyed_submissions: 12,
                coalescing_hits: 5,
                unique_keyed_builds: 7,
                current_in_flight: 3,
                peak_in_flight: 6,
            },
            ranked_preloads: crate::hle::d3d12x::pso_async::RankedPreloadStats {
                submissions: 19,
                started: 13,
                promoted: 4,
            },
            allocator_submits: crate::hle::d3d12x::AllocatorSubmitStats {
                outer_execute_calls: 11,
                chunks: 14,
                multi_chunk_calls: 2,
                forced_wait_boundaries: 3,
                lists_marked: 37,
                backing_mark_writes: 23,
                signal_attempts: 14,
                signal_failures: 1,
                state_lock_wait_ns: 101,
                state_lock_hold_ns: 202,
            },
        };
        let json = serde_json::to_value(summary).unwrap();
        assert_eq!(json["version"], PREPARATION_SUMMARY_VERSION);
        assert_eq!(json["compute_build_coalescing"]["keyed_submissions"], 12);
        assert_eq!(json["compute_build_coalescing"]["coalescing_hits"], 5);
        assert_eq!(json["compute_build_coalescing"]["unique_keyed_builds"], 7);
        assert_eq!(json["compute_build_coalescing"]["current_in_flight"], 3);
        assert_eq!(json["compute_build_coalescing"]["peak_in_flight"], 6);
        assert_eq!(json["ranked_preloads"]["submissions"], 19);
        assert_eq!(json["ranked_preloads"]["started"], 13);
        assert_eq!(json["ranked_preloads"]["promoted"], 4);
        assert_eq!(json["allocator_submits"]["outer_execute_calls"], 11);
        assert_eq!(json["allocator_submits"]["state_lock_hold_ns"], 202);

        let mut version_two = json;
        version_two["version"] = 2.into();
        version_two.as_object_mut().unwrap().remove("allocator_submits");
        let loaded: PreparationSummary = serde_json::from_value(version_two.clone()).unwrap();
        assert_eq!(loaded.allocator_submits, Default::default());

        let mut version_one = version_two;
        version_one["version"] = 1.into();
        version_one.as_object_mut().unwrap().remove("ranked_preloads");
        let loaded: PreparationSummary = serde_json::from_value(version_one).unwrap();
        assert_eq!(loaded.ranked_preloads, Default::default());
    }

    #[test]
    fn a_stall_snapshot_has_a_versioned_stable_json_shape() {
        let mut histogram = [0; crate::stall::HISTOGRAM_BUCKETS];
        histogram[10] = 2;
        let summary = StallSummary::from(crate::stall::Snapshot {
            aggregates: vec![crate::stall::AggregateSnapshot {
                kind: crate::stall::StallKind::DxcCompile,
                count: 2,
                total_ns: 2500,
                max_ns: 1476,
                histogram,
            }],
            slow_spans: vec![crate::stall::SlowSpan {
                sequence: 7,
                kind: crate::stall::StallKind::GpuWait,
                started_ns: 100,
                duration_ns: 50_000_000,
            }],
        });
        let json = serde_json::to_value(summary).unwrap();

        assert_eq!(json["version"], STALL_SUMMARY_VERSION);
        assert_eq!(json["aggregates"][0]["kind"], "dxc_compile");
        assert_eq!(json["aggregates"][0]["count"], 2);
        assert_eq!(json["aggregates"][0]["histogram_log2_ns"][10], 2);
        assert_eq!(json["aggregates"][0]["histogram_log2_ns"].as_array().unwrap().len(), crate::stall::HISTOGRAM_BUCKETS);
        assert_eq!(json["slow_events"][0]["kind"], "gpu_wait");
        assert_eq!(json["slow_events"][0]["sequence"], 7);
        assert_eq!(json["slow_events"][0]["duration_ns"], 50_000_000);
    }

    #[test]
    fn pipeline_stall_names_are_stable() {
        assert_eq!(stall_kind_name(crate::stall::StallKind::PipelineQueueAdmission), "pipeline_queue_admission");
        assert_eq!(stall_kind_name(crate::stall::StallKind::PipelineQueueSpeculative), "pipeline_queue_speculative");
        assert_eq!(stall_kind_name(crate::stall::StallKind::PipelineQueueDemand), "pipeline_queue_demand");
        assert_eq!(stall_kind_name(crate::stall::StallKind::PipelineRestoreDemand), "pipeline_restore_demand");
        assert_eq!(stall_kind_name(crate::stall::StallKind::PipelineRestoreWait), "pipeline_restore_wait");
        assert_eq!(stall_kind_name(crate::stall::StallKind::PipelineRestoreReconcile), "pipeline_restore_reconcile");
    }

    #[test]
    fn missing_fields_in_a_stall_summary_take_safe_defaults() {
        let summary: StallSummary = serde_json::from_str(r#"{"version":1}"#).unwrap();
        assert_eq!(summary, StallSummary { version: 1, aggregates: Vec::new(), slow_events: Vec::new() });
    }

    #[test]
    fn a_stall_summary_keeps_only_the_newest_bounded_events() {
        let slow_spans = (0..crate::stall::SLOW_SPAN_CAPACITY + 2)
            .map(|sequence| crate::stall::SlowSpan {
                sequence: sequence as u64,
                kind: crate::stall::StallKind::GpuWait,
                started_ns: sequence as u64,
                duration_ns: crate::stall::SLOW_SPAN_NS,
            })
            .collect();
        let summary = StallSummary::from(crate::stall::Snapshot { aggregates: Vec::new(), slow_spans });
        assert_eq!(summary.slow_events.len(), crate::stall::SLOW_SPAN_CAPACITY);
        assert_eq!(summary.slow_events[0].sequence, 2);
    }

    #[test]
    fn a_crash_is_kept_while_another_thread_writes_the_report() {
        let (held, wait_held) = std::sync::mpsc::channel();
        let writer = std::thread::spawn(move || {
            let _g = WRITE_LOCK.lock().unwrap();
            held.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(100));
        });
        wait_held.recv().unwrap();
        // no boot record in unit tests: nothing is written to disk
        write_final(Some("boom".into()));
        writer.join().unwrap();
        assert_eq!(CRASH.lock().unwrap().as_deref(), Some("boom"));
        *CRASH.lock().unwrap() = None;
    }
}
