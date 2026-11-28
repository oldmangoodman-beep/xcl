#![allow(dead_code)]

use std::fmt::Display;
use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct Config {
    test: bool,
    json: bool,
    filter: Option<String>,
    identity: BenchmarkIdentity,
}

pub struct Criterion {
    config: Config,
    ran: usize,
}

pub struct BenchmarkGroup<'a> {
    criterion: &'a mut Criterion,
    name: String,
    throughput: Option<Throughput>,
    sample_size: usize,
    measurement_time: Duration,
}

pub struct Bencher {
    name: String,
    test: bool,
    json: bool,
    throughput: Option<Throughput>,
    sample_size: usize,
    measurement_time: Duration,
    identity: BenchmarkIdentity,
}

#[derive(Clone, Copy)]
pub enum Throughput {
    Bytes(u64),
    Elements(u64),
}

pub enum BatchSize {
    SmallInput,
}

pub struct BenchmarkId(String);

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

const BENCHMARK_JSON_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
struct BenchmarkIdentity {
    environment: serde_json::Value,
    hardware: serde_json::Value,
}

fn nonempty(value: Option<String>) -> Option<String> {
    value.map(|value| value.trim().to_owned()).filter(|value| !value.is_empty())
}

fn cpu_identity() -> Option<String> {
    if cfg!(windows) {
        return nonempty(std::env::var("PROCESSOR_IDENTIFIER").ok());
    }
    if cfg!(target_os = "linux") {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        return cpuinfo.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            matches!(key.trim(), "model name" | "Hardware")
                .then(|| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        });
    }
    if cfg!(target_os = "macos") {
        let output = Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]).output().ok()?;
        return output.status.success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .filter(|value| !value.is_empty());
    }
    None
}

fn benchmark_identity_with(
    get: impl Fn(&str) -> Option<String>,
    build_profile: &str,
    build_toolchain: &str,
    build_opt_level: &str,
    build_debug_assertions: &str,
    default_cpu: impl FnOnce() -> Option<String>,
) -> BenchmarkIdentity {
    let field = |name: &str, fallback: &str| nonempty(get(name)).unwrap_or_else(|| fallback.to_owned());
    let profile = field("XCL_BENCH_PROFILE", build_profile);
    let toolchain = field("XCL_BENCH_TOOLCHAIN", build_toolchain);
    let opt_level = field("XCL_BENCH_OPT_LEVEL", build_opt_level);
    let debug_assertions = field("XCL_BENCH_DEBUG_ASSERTIONS", build_debug_assertions);
    let cpu = nonempty(get("XCL_BENCH_CPU")).or_else(|| nonempty(default_cpu()));
    BenchmarkIdentity {
        environment: serde_json::json!({
            "profile": profile,
            "toolchain": toolchain,
            "opt_level": opt_level,
            "debug_assertions": debug_assertions,
            "os": field("XCL_BENCH_OS", std::env::consts::OS),
            "arch": field("XCL_BENCH_ARCH", std::env::consts::ARCH),
        }),
        hardware: serde_json::json!({ "cpu": cpu }),
    }
}

fn benchmark_identity() -> BenchmarkIdentity {
    benchmark_identity_with(
        |name| std::env::var(name).ok(),
        env!("XCL_BUILD_PROFILE"),
        env!("XCL_BUILD_RUSTC"),
        env!("XCL_BUILD_OPT_LEVEL"),
        env!("XCL_BUILD_DEBUG_ASSERTIONS"),
        cpu_identity,
    )
}

fn measurement_json(
    name: &str,
    sample_size: usize,
    sample_iterations: u64,
    median: f64,
    mean: f64,
    minimum: f64,
    throughput: Option<Throughput>,
    identity: &BenchmarkIdentity,
) -> serde_json::Value {
    let throughput = throughput.map(|throughput| {
        let (kind, amount) = match throughput {
            Throughput::Bytes(amount) => ("bytes", amount),
            Throughput::Elements(amount) => ("elements", amount),
        };
        serde_json::json!({
            "kind": kind,
            "per_iteration": amount,
            "per_second": amount as f64 / median.max(1e-12),
        })
    });
    serde_json::json!({
        "schema": "xcl-benchmark",
        "version": BENCHMARK_JSON_VERSION,
        "benchmark": name,
        "sample_size": sample_size,
        "sample_iterations": sample_iterations,
        "median_ns_per_iter": median * 1e9,
        "mean_ns_per_iter": mean * 1e9,
        "min_ns_per_iter": minimum * 1e9,
        "throughput": throughput,
        "environment": identity.environment,
        "hardware": identity.hardware,
    })
}

fn test_helpers() {
    assert_eq!(median(&[1.0, 3.0]), 2.0);
    assert_eq!(median(&[1.0, 4.0, 9.0]), 4.0);
    let identity = benchmark_identity_with(
        |name| match name {
            "XCL_BENCH_PROFILE" => Some("ci".into()),
            "XCL_BENCH_TOOLCHAIN" => Some("rustc-test".into()),
            "XCL_BENCH_OPT_LEVEL" => Some("3".into()),
            "XCL_BENCH_DEBUG_ASSERTIONS" => Some("false".into()),
            "XCL_BENCH_OS" => Some("test-os".into()),
            "XCL_BENCH_ARCH" => Some("test-arch".into()),
            "XCL_BENCH_CPU" => Some("test-cpu".into()),
            _ => None,
        },
        "ignored-profile",
        "ignored-toolchain",
        "ignored-opt-level",
        "ignored-debug-assertions",
        || Some("ignored-cpu".into()),
    );
    let json = measurement_json(
        "group/case", 30, 17, 2e-9, 3e-9, 1e-9, Some(Throughput::Bytes(8)), &identity,
    );
    assert_eq!(json["schema"], "xcl-benchmark");
    assert_eq!(json["version"], BENCHMARK_JSON_VERSION);
    assert_eq!(json["benchmark"], "group/case");
    assert_eq!(json["sample_size"], 30);
    assert_eq!(json["sample_iterations"], 17);
    assert_eq!(json["median_ns_per_iter"], 2.0);
    assert_eq!(json["throughput"]["kind"], "bytes");
    assert_eq!(json["throughput"]["per_iteration"], 8);
    assert_eq!(json["environment"]["profile"], "ci");
    assert_eq!(json["environment"]["toolchain"], "rustc-test");
    assert_eq!(json["environment"]["opt_level"], "3");
    assert_eq!(json["environment"]["debug_assertions"], "false");
    assert_eq!(json["environment"]["os"], "test-os");
    assert_eq!(json["environment"]["arch"], "test-arch");
    assert_eq!(json["hardware"]["cpu"], "test-cpu");
    let second = measurement_json(
        "group/other", 30, 17, 4e-9, 5e-9, 3e-9, None, &identity,
    );
    assert_eq!(second["environment"], json["environment"]);
    assert_eq!(second["hardware"], json["hardware"]);

    let defaults = benchmark_identity_with(
        |_| None,
        "bench-profile",
        "build-time-toolchain",
        "s",
        "true",
        || None,
    );
    assert_eq!(defaults.environment["profile"], "bench-profile");
    assert_eq!(defaults.environment["toolchain"], "build-time-toolchain");
    assert_eq!(defaults.environment["opt_level"], "s");
    assert_eq!(defaults.environment["debug_assertions"], "true");
    assert!(defaults.hardware["cpu"].is_null());
}

impl BenchmarkId {
    pub fn new(name: impl Display, parameter: impl Display) -> Self {
        Self(format!("{name}/{parameter}"))
    }

    pub fn from_parameter(parameter: impl Display) -> Self {
        Self(parameter.to_string())
    }
}

impl Display for BenchmarkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Criterion {
    fn from_args() -> Self {
        let mut test = false;
        let mut json = false;
        let mut filter = None;
        for argument in std::env::args().skip(1) {
            if argument == "--test" {
                test = true;
            } else if argument == "--json" {
                json = true;
            } else if !argument.starts_with('-') && filter.is_none() {
                filter = Some(argument);
            }
        }
        Self { config: Config { test, json, filter, identity: benchmark_identity() }, ran: 0 }
    }

    pub fn benchmark_group<'a>(&'a mut self, name: &str) -> BenchmarkGroup<'a> {
        BenchmarkGroup {
            criterion: self,
            name: name.to_owned(),
            throughput: None,
            sample_size: 30,
            measurement_time: Duration::from_secs(2),
        }
    }

    pub fn bench_function(&mut self, name: &str, routine: impl FnMut(&mut Bencher)) {
        self.run_one(name.to_owned(), None, 30, Duration::from_secs(2), routine);
    }

    fn run_one(
        &mut self,
        name: String,
        throughput: Option<Throughput>,
        sample_size: usize,
        measurement_time: Duration,
        mut routine: impl FnMut(&mut Bencher),
    ) {
        if self.config.filter.as_ref().is_some_and(|filter| !name.contains(filter)) {
            return;
        }
        self.ran += 1;
        routine(&mut Bencher {
            name,
            test: self.config.test,
            json: self.config.json,
            throughput,
            sample_size: sample_size.max(3),
            measurement_time,
            identity: self.config.identity.clone(),
        });
    }
}

impl BenchmarkGroup<'_> {
    pub fn throughput(&mut self, throughput: Throughput) {
        self.throughput = Some(throughput);
    }

    pub fn sample_size(&mut self, sample_size: usize) {
        self.sample_size = sample_size;
    }

    pub fn measurement_time(&mut self, measurement_time: Duration) {
        self.measurement_time = measurement_time;
    }

    pub fn bench_function(&mut self, name: &str, routine: impl FnMut(&mut Bencher)) {
        let name = format!("{}/{}", self.name, name);
        self.criterion.run_one(
            name, self.throughput, self.sample_size, self.measurement_time, routine,
        );
    }

    pub fn bench_with_input<T: ?Sized>(
        &mut self,
        id: BenchmarkId,
        input: &T,
        mut routine: impl FnMut(&mut Bencher, &T),
    ) {
        let name = format!("{}/{}", self.name, id);
        self.criterion.run_one(
            name,
            self.throughput,
            self.sample_size,
            self.measurement_time,
            |bencher| routine(bencher, input),
        );
    }

    pub fn finish(self) {}
}

impl Bencher {
    pub fn iter<R>(&mut self, mut routine: impl FnMut() -> R) {
        if self.test {
            let _ = black_box(routine());
            println!("test {} ... ok", self.name);
            return;
        }
        self.measure_iterations(|iterations| {
            let started = Instant::now();
            for _ in 0..iterations {
                let _ = black_box(routine());
            }
            started.elapsed()
        });
    }

    pub fn iter_batched<I, R>(
        &mut self,
        mut setup: impl FnMut() -> I,
        mut routine: impl FnMut(I) -> R,
        _batch_size: BatchSize,
    ) {
        if self.test {
            let input = setup();
            let _ = black_box(routine(input));
            println!("test {} ... ok", self.name);
            return;
        }
        self.measure_iterations(|iterations| {
            let mut elapsed = Duration::ZERO;
            let mut remaining = iterations;
            while remaining != 0 {
                let batch = remaining.min(1_024) as usize;
                let inputs: Vec<_> = (0..batch).map(|_| setup()).collect();
                let started = Instant::now();
                for input in inputs {
                    let _ = black_box(routine(input));
                }
                elapsed += started.elapsed();
                remaining -= batch as u64;
            }
            elapsed
        });
    }

    fn measure_iterations(&self, mut run: impl FnMut(u64) -> Duration) {
        let warmup = self.measurement_time.div_f64(10.0).min(Duration::from_millis(100));
        let mut iterations = 1u64;
        let elapsed = loop {
            let elapsed = run(iterations);
            if elapsed >= warmup || iterations >= 1 << 40 {
                break elapsed;
            }
            let scale = (warmup.as_secs_f64() / elapsed.as_secs_f64().max(1e-9)).ceil() as u64;
            iterations = iterations.saturating_mul(scale.clamp(2, 10)).min(1 << 40);
        };

        let seconds_per_iteration = elapsed.as_secs_f64() / iterations as f64;
        let target_sample = self.measurement_time.as_secs_f64() / self.sample_size as f64;
        let sample_iterations = (target_sample / seconds_per_iteration.max(1e-12))
            .ceil()
            .clamp(1.0, (1u64 << 40) as f64) as u64;
        let mut samples = Vec::with_capacity(self.sample_size);
        for _ in 0..self.sample_size {
            samples.push(run(sample_iterations).as_secs_f64() / sample_iterations as f64);
        }
        samples.sort_unstable_by(f64::total_cmp);
        let median = median(&samples);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let minimum = samples[0];
        if self.json {
            println!(
                "{}",
                measurement_json(
                    &self.name,
                    samples.len(),
                    sample_iterations,
                    median,
                    mean,
                    minimum,
                    self.throughput,
                    &self.identity,
                )
            );
            return;
        }
        print!(
            "bench {}: median {:.1} ns/iter, mean {:.1}, min {:.1}",
            self.name,
            median * 1e9,
            mean * 1e9,
            minimum * 1e9,
        );
        if let Some(throughput) = self.throughput {
            let (amount, unit) = match throughput {
                Throughput::Bytes(amount) => (amount, "B/s"),
                Throughput::Elements(amount) => (amount, "elem/s"),
            };
            print!(", {:.1} {unit}", amount as f64 / median.max(1e-12));
        }
        println!();
    }
}

pub fn run(groups: &[fn(&mut Criterion)]) {
    let mut criterion = Criterion::from_args();
    if criterion.config.test {
        test_helpers();
    }
    for group in groups {
        group(&mut criterion);
    }
    if criterion.ran == 0 {
        eprintln!("no benchmark matched the requested filter");
        std::process::exit(2);
    }
}
