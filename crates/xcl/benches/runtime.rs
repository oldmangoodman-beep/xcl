mod support;

use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;
use support::{BatchSize, BenchmarkId, Criterion, Throughput};
use xcl::config::Config;
use xcl::manifest;
use xcl::util;
use xcl::vfs::Vfs;

const MANIFEST: &str = r#"<Game configVersion="1">
  <Identity Name="Synthetic.Benchmark" Publisher="CN=Benchmark" Version="1.0.0.0" />
  <PersistentLocalStorage><SizeMB>4096</SizeMB></PersistentLocalStorage>
  <VirtualMachine>
    <XboxAnacondaTitleMemory>Advanced</XboxAnacondaTitleMemory>
    <XboxLockhartTitleMemory>Standard</XboxLockhartTitleMemory>
  </VirtualMachine>
  <TitleId>12345678</TitleId>
  <Resources>
    <Resource Language="en-US"/><Resource Language="fr-FR"/>
    <Resource Language="de-DE"/><Resource Language="ja-JP"/>
  </Resources>
</Game>"#;

const CONFIG: &str = r#"
device_type = "anaconda"
gamertag = "Player"
user_id = 281474976710655
title_memory = "auto"
pls_mb = 0
uma_mb = 0
log_level = "info"
graphics = "d3d12"
d3d12_adapter = "auto"
present = "auto"
raytracing = "auto"
window_width = 1280
window_height = 720
keyboard_mouse = true
mouse_sensitivity = 1.0

[mounts]
"GAME:" = "{game}"
"G:" = "{game}"
"S:" = "{data}\\pls"

[aliases]
"api-ms-win-*" = "@system"
"kernelx.dll" = "kernelx.dll"

[stub_returns]
"example.dll!Example" = "S_OK"
"#;

const ALLOCATION_JOBS: usize = 128;

struct AllocationJob {
    owner: Arc<[u8; 64]>,
    label: String,
    payload: Vec<u8>,
}

fn allocation_job_batch(seed: usize) -> usize {
    let owner = Arc::new([seed as u8; 64]);
    let mut jobs = Vec::with_capacity(ALLOCATION_JOBS);
    for index in 0..ALLOCATION_JOBS {
        let mut label = String::with_capacity(48);
        label.push_str("pipeline-job-");
        label.extend(std::iter::repeat_n((b'a' + (index % 26) as u8) as char, 16 + index % 16));
        jobs.push(Box::new(AllocationJob {
            owner: owner.clone(),
            label,
            payload: vec![(seed ^ index) as u8; 96 + index % 64],
        }));
    }
    let checksum = jobs.iter().fold(0usize, |checksum, job| {
        checksum ^ job.owner[0] as usize ^ job.label.len() ^ job.payload.len()
    });
    black_box(&jobs);
    checksum
}

#[repr(align(64))]
struct AllocationResult(AtomicUsize);

struct AllocationWorkers {
    start: Arc<Barrier>,
    done: Arc<Barrier>,
    stop: Arc<AtomicBool>,
    epoch: Arc<AtomicUsize>,
    results: Arc<Vec<AllocationResult>>,
    threads: Vec<JoinHandle<()>>,
}

impl AllocationWorkers {
    fn new(count: usize) -> Self {
        let start = Arc::new(Barrier::new(count + 1));
        let done = Arc::new(Barrier::new(count + 1));
        let stop = Arc::new(AtomicBool::new(false));
        let epoch = Arc::new(AtomicUsize::new(0));
        let results: Arc<Vec<AllocationResult>> =
            Arc::new((0..count).map(|_| AllocationResult(AtomicUsize::new(0))).collect());
        let threads = (0..count).map(|worker| {
            let start = start.clone();
            let done = done.clone();
            let stop = stop.clone();
            let epoch = epoch.clone();
            let results = results.clone();
            std::thread::Builder::new().name(format!("alloc-bench-{worker}")).spawn(move || loop {
                start.wait();
                if stop.load(Ordering::Acquire) { break }
                let checksum = allocation_job_batch(epoch.load(Ordering::Acquire).wrapping_add(worker));
                results[worker].0.store(checksum, Ordering::Release);
                done.wait();
            }).expect("allocator benchmark worker")
        }).collect();
        Self { start, done, stop, epoch, results, threads }
    }

    fn run(&self) -> usize {
        self.epoch.fetch_add(1, Ordering::Release);
        self.start.wait();
        self.done.wait();
        self.results.iter().fold(0, |checksum, result| checksum ^ result.0.load(Ordering::Acquire))
    }
}

impl Drop for AllocationWorkers {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.start.wait();
        for thread in self.threads.drain(..) {
            thread.join().expect("allocator benchmark worker panicked");
        }
    }
}

fn allocator_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime/allocator");
    group.throughput(Throughput::Elements(ALLOCATION_JOBS as u64));
    group.bench_function("job_mix_single", |b| {
        let mut seed = 0usize;
        b.iter(|| {
            seed = seed.wrapping_add(1);
            allocation_job_batch(black_box(seed))
        });
    });

    let workers = std::thread::available_parallelism().map_or(4, |count| count.get())
        .saturating_sub(1).clamp(2, 8);
    let pool = AllocationWorkers::new(workers);
    group.throughput(Throughput::Elements((ALLOCATION_JOBS * workers) as u64));
    group.bench_function(&format!("job_mix_parallel_{workers}"), |b| b.iter(|| pool.run()));
    group.finish();
}

#[cfg(feature = "texture-selection-bench")]
fn texture_selection_benches(c: &mut Criterion) {
    use xcl::hle::d3d12x::title_textures::{TextureSelectionBench, TextureSelectionScenario as Scenario};

    let mut group = c.benchmark_group("runtime/texture_selection");
    for entries in [1_024usize, 4_096, 16_384] {
        for (name, scenario) in [
            ("empty_gate", Scenario::Empty),
            ("pending_sparse", Scenario::SparsePending),
            ("pending_all", Scenario::AllPending),
            ("gpu_written_all", Scenario::AllGpuWritten),
            ("dirty_1", Scenario::OneDirty),
            ("dirty_sparse_64", Scenario::SparseDirty),
            ("dirty_dense_4096", Scenario::DenseDirty),
            ("mixed_aliases", Scenario::MixedAliases),
        ] {
            let fixture = TextureSelectionBench::new(entries, scenario);
            let operations = if matches!(scenario, Scenario::Empty) { 1 } else { entries };
            group.throughput(Throughput::Elements(operations as u64));
            group.bench_function(&format!("{name}_{entries}"), |b| {
                b.iter(|| black_box(fixture.scan()));
            });
        }
    }

    const ENTRIES: usize = 4_096;
    const BATCH: usize = 64;
    let workers = std::thread::available_parallelism().map_or(4, |count| count.get())
        .saturating_sub(1).clamp(2, 8);
    group.throughput(Throughput::Elements((ENTRIES * BATCH * workers) as u64));
    for shared in [true, false] {
        let pool = TextureSelectionWorkers::new(ENTRIES, workers, BATCH, shared);
        let name = if shared { "mixed_shared_lock" } else { "mixed_distinct_locks" };
        group.bench_function(&format!("{name}_{workers}"), |b| b.iter(|| pool.run()));
    }
    group.finish();
}

#[cfg(feature = "texture-selection-bench")]
struct TextureSelectionWorkers {
    start: Arc<Barrier>,
    done: Arc<Barrier>,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

#[cfg(feature = "texture-selection-bench")]
impl TextureSelectionWorkers {
    fn new(entries: usize, workers: usize, batch: usize, shared: bool) -> Self {
        use xcl::hle::d3d12x::title_textures::{TextureSelectionBench, TextureSelectionScenario};

        let start = Arc::new(Barrier::new(workers + 1));
        let done = Arc::new(Barrier::new(workers + 1));
        let stop = Arc::new(AtomicBool::new(false));
        let common = shared.then(|| Arc::new(TextureSelectionBench::new(entries, TextureSelectionScenario::MixedAliases)));
        let threads = (0..workers).map(|worker| {
            let fixture = common.clone().unwrap_or_else(|| Arc::new(TextureSelectionBench::new(entries, TextureSelectionScenario::MixedAliases)));
            let (start, done, stop) = (start.clone(), done.clone(), stop.clone());
            std::thread::Builder::new().name(format!("texture-select-bench-{worker}")).spawn(move || loop {
                start.wait();
                if stop.load(Ordering::Acquire) { break }
                let mut checksum = 0usize;
                for _ in 0..batch {
                    let (matches, ids) = fixture.scan();
                    checksum = checksum.wrapping_add(matches ^ ids);
                }
                black_box(checksum);
                done.wait();
            }).expect("texture selection benchmark worker")
        }).collect();
        Self { start, done, stop, threads }
    }

    fn run(&self) {
        self.start.wait();
        self.done.wait();
    }
}

#[cfg(feature = "texture-selection-bench")]
impl Drop for TextureSelectionWorkers {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.start.wait();
        for thread in self.threads.drain(..) {
            thread.join().expect("texture selection benchmark worker panicked");
        }
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
fn pipeline_resolution_benches(c: &mut Criterion) {
    let fixture = Arc::new(xcl::hle::d3d12x::pso_async::PipelineResolutionBench::new());
    let mut group = c.benchmark_group("runtime/pipeline_resolution");
    group.throughput(Throughput::Elements(1));
    group.bench_function("retained_ready", |b| {
        b.iter(|| black_box(&fixture).retained_ready());
    });
    group.bench_function("retained_ready_recipe_tracked", |b| {
        b.iter(|| black_box(&fixture).retained_ready_recipe_tracked());
    });
    // Bind-time registry lookup + classification as a comparison ceiling, not another retained
    // emission path.
    group.bench_function("registry_ready", |b| {
        b.iter(|| black_box(&fixture).registry_ready());
    });
    group.bench_function("initial_completed", |b| {
        b.iter(|| black_box(&fixture).initial_completed());
    });
    group.bench_function("initial_completed_dispatch", |b| {
        b.iter(|| black_box(&fixture).initial_completed_dispatch());
    });
    group.bench_function("initial_unpromoted_completed_probe", |b| {
        b.iter(|| black_box(&fixture).initial_unpromoted_shared_probe());
    });

    let workers = std::thread::available_parallelism().map_or(4, |count| count.get())
        .saturating_sub(1).clamp(2, 8);
    group.throughput(Throughput::Elements((PIPELINE_PROBE_BATCH * workers) as u64));
    for (name, mode) in [
        ("initial_unpromoted_shared", PipelineProbeMode::UnpromotedShared),
        ("initial_unpromoted_distinct", PipelineProbeMode::UnpromotedDistinct),
        ("initial_promoted_control", PipelineProbeMode::Promoted),
        ("retained_ready", PipelineProbeMode::RetainedReady),
        ("retained_ready_recipe_tracked", PipelineProbeMode::RetainedReadyRecipeTracked),
    ] {
        let pool = PipelineProbeWorkers::new(fixture.clone(), workers, mode);
        group.bench_function(&format!("{name}_parallel_{workers}"), |b| b.iter(|| pool.run()));
    }
    group.finish();
}

#[cfg(feature = "pipeline-resolution-bench")]
const PIPELINE_PROBE_BATCH: usize = 4_096;

#[cfg(feature = "pipeline-resolution-bench")]
#[derive(Clone, Copy)]
enum PipelineProbeMode {
    UnpromotedShared,
    UnpromotedDistinct,
    Promoted,
    RetainedReady,
    RetainedReadyRecipeTracked,
}

#[cfg(feature = "pipeline-resolution-bench")]
struct PipelineProbeWorkers {
    start: Arc<Barrier>,
    done: Arc<Barrier>,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

#[cfg(feature = "pipeline-resolution-bench")]
impl PipelineProbeWorkers {
    fn new(
        fixture: Arc<xcl::hle::d3d12x::pso_async::PipelineResolutionBench>,
        workers: usize,
        mode: PipelineProbeMode,
    ) -> Self {
        let start = Arc::new(Barrier::new(workers + 1));
        let done = Arc::new(Barrier::new(workers + 1));
        let stop = Arc::new(AtomicBool::new(false));
        let threads = (0..workers)
            .map(|worker| {
                let fixture = fixture.clone();
                let start = start.clone();
                let done = done.clone();
                let stop = stop.clone();
                std::thread::Builder::new()
                    .name(format!("pipeline-probe-bench-{worker}"))
                    .spawn(move || loop {
                        start.wait();
                        if stop.load(Ordering::Acquire) {
                            break;
                        }
                        let mut pending = 0usize;
                        for _ in 0..PIPELINE_PROBE_BATCH {
                            pending += match mode {
                                PipelineProbeMode::UnpromotedShared => fixture.initial_unpromoted_shared_probe(),
                                PipelineProbeMode::UnpromotedDistinct => fixture.initial_unpromoted_distinct_probe(worker),
                                PipelineProbeMode::Promoted => fixture.initial_promoted_probe(),
                                PipelineProbeMode::RetainedReady => fixture.retained_ready() != 0,
                                PipelineProbeMode::RetainedReadyRecipeTracked => fixture.retained_ready_recipe_tracked() != 0,
                            } as usize;
                        }
                        black_box(pending);
                        done.wait();
                    })
                    .expect("pipeline probe benchmark worker")
            })
            .collect();
        Self { start, done, stop, threads }
    }

    fn run(&self) {
        self.start.wait();
        self.done.wait();
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
impl Drop for PipelineProbeWorkers {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.start.wait();
        for thread in self.threads.drain(..) {
            thread.join().expect("pipeline probe benchmark worker panicked");
        }
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
const EMISSION_BATCH: usize = 1_024;

#[cfg(feature = "pipeline-resolution-bench")]
struct EmissionWorkers {
    start: Arc<Barrier>,
    done: Arc<Barrier>,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

#[cfg(feature = "pipeline-resolution-bench")]
impl EmissionWorkers {
    fn new(
        fixture: Arc<xcl::hle::d3d12x::list_x::EmissionBookkeepingBench>,
        workers: usize,
    ) -> Self {
        let start = Arc::new(Barrier::new(workers + 1));
        let done = Arc::new(Barrier::new(workers + 1));
        let stop = Arc::new(AtomicBool::new(false));
        let threads = (0..workers)
            .map(|worker| {
                let fixture = fixture.clone();
                let start = start.clone();
                let done = done.clone();
                let stop = stop.clone();
                std::thread::Builder::new()
                    .name(format!("emission-bench-{worker}"))
                    .spawn(move || loop {
                        start.wait();
                        if stop.load(Ordering::Acquire) {
                            break;
                        }
                        for _ in 0..EMISSION_BATCH {
                            fixture.record_worker(worker);
                        }
                        done.wait();
                    })
                    .expect("emission benchmark worker")
            })
            .collect();
        Self { start, done, stop, threads }
    }

    fn run(&self) {
        self.start.wait();
        self.done.wait();
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
impl Drop for EmissionWorkers {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.start.wait();
        for thread in self.threads.drain(..) {
            thread.join().expect("emission benchmark worker panicked");
        }
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
struct BindWorkers {
    start: Arc<Barrier>,
    done: Arc<Barrier>,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

#[cfg(feature = "pipeline-resolution-bench")]
impl BindWorkers {
    fn new(fixture: Arc<xcl::hle::d3d12x::bind::BindStateBench>, workers: usize) -> Self {
        let start = Arc::new(Barrier::new(workers + 1));
        let done = Arc::new(Barrier::new(workers + 1));
        let stop = Arc::new(AtomicBool::new(false));
        let threads = (0..workers)
            .map(|worker| {
                let fixture = fixture.clone();
                let start = start.clone();
                let done = done.clone();
                let stop = stop.clone();
                std::thread::Builder::new()
                    .name(format!("bind-bench-{worker}"))
                    .spawn(move || loop {
                        start.wait();
                        if stop.load(Ordering::Acquire) {
                            break;
                        }
                        let mut checksum = 0usize;
                        for _ in 0..EMISSION_BATCH {
                            checksum = checksum.wrapping_add(fixture.snapshot_worker(worker));
                        }
                        black_box(checksum);
                        done.wait();
                    })
                    .expect("bind-state benchmark worker")
            })
            .collect();
        Self { start, done, stop, threads }
    }

    fn run(&self) {
        self.start.wait();
        self.done.wait();
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
impl Drop for BindWorkers {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.start.wait();
        for thread in self.threads.drain(..) {
            thread.join().expect("bind-state benchmark worker panicked");
        }
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
fn emission_bookkeeping_benches(c: &mut Criterion) {
    let workers = std::thread::available_parallelism().map_or(4, |count| count.get())
        .saturating_sub(1).clamp(2, 8);
    let fixture = Arc::new(xcl::hle::d3d12x::list_x::EmissionBookkeepingBench::new(workers));

    let mut group = c.benchmark_group("runtime/emission_bookkeeping/record_existing");
    group.throughput(Throughput::Elements(1));
    for (slot, depth) in [1usize, 16, 64, 256].into_iter().enumerate() {
        group.bench_with_input(BenchmarkId::from_parameter(depth), &slot, |b, &slot| {
            b.iter(|| black_box(&fixture).record_at_depth(black_box(slot)));
        });
    }
    group.finish();

    let pool = EmissionWorkers::new(fixture.clone(), workers);
    let mut group = c.benchmark_group("runtime/emission_bookkeeping/contention");
    group.throughput(Throughput::Elements((workers * EMISSION_BATCH) as u64));
    group.bench_function(&format!("distinct_lists_serial_{workers}"), |b| {
        b.iter(|| black_box(&fixture).record_workers_serial(workers, EMISSION_BATCH));
    });
    group.bench_function(&format!("distinct_lists_parallel_{workers}"), |b| {
        b.iter(|| pool.run());
    });
    group.finish();
}

#[cfg(feature = "pipeline-resolution-bench")]
fn bind_state_benches(c: &mut Criterion) {
    let workers = std::thread::available_parallelism().map_or(4, |count| count.get())
        .saturating_sub(1).clamp(2, 8);
    let fixture = Arc::new(xcl::hle::d3d12x::bind::BindStateBench::new(workers));
    let pool = BindWorkers::new(fixture.clone(), workers);
    // This is one retained-pipeline snapshot, not a whole draw or dispatch; those take STATES
    // several more times while preparing their root arguments and ABI memory.
    let mut group = c.benchmark_group("runtime/emission_bookkeeping/warm_bind_pipeline_snapshot");
    group.throughput(Throughput::Elements((workers * EMISSION_BATCH) as u64));
    group.bench_function(&format!("distinct_lists_serial_{workers}"), |b| {
        b.iter(|| black_box(&fixture).snapshot_workers_serial(workers, EMISSION_BATCH));
    });
    group.bench_function(&format!("distinct_lists_parallel_{workers}"), |b| {
        b.iter(|| pool.run());
    });
    group.finish();
}

fn utility_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime/util");
    for (name, pattern, value) in [
        ("glob_exact", "kernelx.dll", "kernelx.dll"),
        (
            "glob_prefix",
            "api-ms-win-*",
            "api-ms-win-core-file-l2-1-0.dll",
        ),
        ("glob_bounded", "xgame*.dll", "xgameruntime.dll"),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| util::glob_match(black_box(pattern), black_box(value)))
        });
    }
    for value in [
        "kernelx",
        "D:\\Game\\D3D12_XS.DLL",
        "api-ms-win-core-file-l2-1-0",
    ] {
        group.bench_with_input(
            BenchmarkId::new("dll_key", value.len()),
            value,
            |b, value| {
                b.iter(|| util::dll_key(black_box(value)));
            },
        );
    }
    for value in ["S_OK", "E_NOTIMPL", "0x80004005", "42"] {
        group.bench_with_input(
            BenchmarkId::new("parse_return", value),
            value,
            |b, value| {
                b.iter(|| util::parse_ret(black_box(value)));
            },
        );
    }
    for value in [
        "short",
        "D:\\Synthetic Game\\Content\\localization\\en-US\\strings.bin",
    ] {
        group.throughput(Throughput::Bytes(value.len() as u64));
        group.bench_with_input(BenchmarkId::new("utf16", value.len()), value, |b, value| {
            b.iter(|| util::wide(black_box(value)));
        });
    }
    group.finish();
}

fn manifest_and_config_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime/config");
    group.throughput(Throughput::Bytes(MANIFEST.len() as u64));
    group.bench_function("manifest_parse", |b| {
        b.iter(|| manifest::parse(black_box(MANIFEST)))
    });

    group.throughput(Throughput::Bytes(CONFIG.len() as u64));
    group.bench_function("toml_parse", |b| {
        b.iter(|| toml::from_str::<Config>(black_box(CONFIG)).unwrap());
    });

    let parsed_manifest = manifest::parse(MANIFEST);
    group.throughput(Throughput::Elements(1));
    group.bench_function("resolve_defaults", |b| {
        b.iter_batched(
            Config::default,
            |mut config| black_box(config.resolve(black_box(Some(&parsed_manifest)))),
            BatchSize::SmallInput,
        );
    });

    let mut mounts = BTreeMap::new();
    mounts.insert("GAME:".to_owned(), "{game}".to_owned());
    mounts.insert("S:".to_owned(), "{data}\\pls".to_owned());
    let mut resolved = Config::default();
    resolved.mounts = mounts;
    resolved.resolve(Some(&parsed_manifest));
    group.throughput(Throughput::Elements(1));
    group.bench_function("resolved_memory_queries", |b| {
        b.iter(|| {
            black_box(resolved.title_memory_budget_mb());
            black_box(resolved.pls_drive());
            black_box(resolved.raytracing_tier_override());
        });
    });
    group.finish();
}

fn vfs_benches(c: &mut Criterion) {
    let vfs = Vfs::new(
        Path::new(r"C:\bench\game"),
        Path::new(r"C:\bench\data"),
        vec![
            ("GAME:".to_owned(), r"C:\bench\game".to_owned()),
            ("G:".to_owned(), r"C:\bench\game".to_owned()),
            ("S:".to_owned(), r"C:\bench\data\pls".to_owned()),
        ],
    );
    let full = |path: &str| Some(path.replace('/', "\\"));
    let cases = [
        ("mounted_game", r"GAME:\data\textures.bin"),
        ("mounted_drive", r"G:\data\textures.bin"),
        ("embedded_mount", r"C:\cwd\GAME:\data\textures.bin"),
        ("direct_game", r"C:\bench\game\data\textures.bin"),
        ("host_path", r"C:\Windows\System32\kernel32.dll"),
        ("unc", r"\\server\share\file.bin"),
    ];

    let mut group = c.benchmark_group("runtime/vfs");
    for (name, path) in cases {
        group.throughput(Throughput::Bytes(path.len() as u64));
        group.bench_with_input(BenchmarkId::new("classify", name), path, |b, path| {
            b.iter(|| vfs.classify_with(black_box(path), black_box(full)));
        });
    }
    group.throughput(Throughput::Bytes(
        cases.iter().map(|(_, path)| path.len() as u64).sum(),
    ));
    group.bench_function("classify_mixed_batch", |b| {
        b.iter(|| {
            for (_, path) in black_box(&cases) {
                black_box(vfs.classify_with(path, full));
            }
        });
    });
    for rel in ["", "data", r"data\textures\terrain.bin"] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("overlay_path", rel.len()),
            rel,
            |b, rel| {
                b.iter(|| vfs.overlay_path(black_box(rel)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("original_path", rel.len()),
            rel,
            |b, rel| {
                b.iter(|| vfs.original_path(black_box(rel)));
            },
        );
    }
    group.finish();
}

fn main() {
    support::run(&[
        utility_benches,
        manifest_and_config_benches,
        vfs_benches,
        allocator_benches,
        #[cfg(feature = "texture-selection-bench")]
        texture_selection_benches,
        #[cfg(feature = "pipeline-resolution-bench")]
        pipeline_resolution_benches,
        #[cfg(feature = "pipeline-resolution-bench")]
        emission_bookkeeping_benches,
        #[cfg(feature = "pipeline-resolution-bench")]
        bind_state_benches,
    ]);
}
