//! Compare two version-1 `xcl-benchmark` NDJSON streams against an explicit limit.

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Deserialize)]
struct Record {
    schema: String,
    version: u32,
    benchmark: String,
    sample_size: usize,
    sample_iterations: u64,
    median_ns_per_iter: f64,
    mean_ns_per_iter: f64,
    min_ns_per_iter: f64,
    #[serde(default)]
    environment: Option<Value>,
    #[serde(default)]
    hardware: Option<Value>,
}

#[derive(Clone, Debug, PartialEq)]
struct Metadata {
    environment: Option<Value>,
    hardware: Option<Value>,
}

#[derive(Debug)]
struct Results {
    records: BTreeMap<String, Record>,
    metadata: Metadata,
}

#[derive(Debug, PartialEq)]
struct Comparison {
    benchmark: String,
    baseline_ns: f64,
    candidate_ns: f64,
    change_percent: f64,
    regression: bool,
}

fn valid_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn parse_records(reader: impl BufRead, label: &str) -> Result<Results, String> {
    let mut records = BTreeMap::new();
    let mut metadata = None;
    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|e| format!("cannot read {label} line {line_number}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(&line)
            .map_err(|e| format!("invalid {label} line {line_number}: {e}"))?;
        if record.schema != "xcl-benchmark" || record.version != 1 {
            return Err(format!("unsupported {label} schema/version on line {line_number}"));
        }
        if record.benchmark.trim().is_empty()
            || record.sample_size == 0
            || record.sample_iterations == 0
            || !record.median_ns_per_iter.is_finite()
            || record.median_ns_per_iter <= 0.0
            || !valid_nonnegative(record.mean_ns_per_iter)
            || !valid_nonnegative(record.min_ns_per_iter)
            || record.environment.as_ref().is_some_and(|value| !value.is_object())
            || record.hardware.as_ref().is_some_and(|value| !value.is_object())
        {
            return Err(format!("malformed {label} measurement on line {line_number}"));
        }
        let record_metadata = Metadata {
            environment: record.environment.clone(),
            hardware: record.hardware.clone(),
        };
        if let Some(expected) = &metadata {
            if expected != &record_metadata {
                return Err(format!("{label} metadata changes between benchmark records"));
            }
        } else {
            metadata = Some(record_metadata);
        }
        let name = record.benchmark.clone();
        if records.insert(name.clone(), record).is_some() {
            return Err(format!("duplicate {label} benchmark: {name}"));
        }
    }
    if records.is_empty() {
        return Err(format!("{label} contains no benchmark records"));
    }
    Ok(Results {
        records,
        metadata: metadata.expect("a non-empty result has metadata"),
    })
}

fn read_records(path: &str, label: &str) -> Result<Results, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {label} {path}: {e}"))?;
    parse_records(BufReader::new(file), label)
}

fn compare(baseline: &Results, candidate: &Results, limit: f64) -> Result<Vec<Comparison>, String> {
    if !limit.is_finite() || limit < 0.0 {
        return Err("maximum regression percent must be a finite nonnegative number".into());
    }
    if baseline.metadata != candidate.metadata {
        return Err("environment or hardware metadata differs; results are incomparable".into());
    }
    let baseline_names: Vec<_> = baseline.records.keys().collect();
    let candidate_names: Vec<_> = candidate.records.keys().collect();
    if baseline_names != candidate_names {
        return Err("benchmark sets differ; results are incomparable".into());
    }
    Ok(baseline.records.iter().map(|(name, before)| {
        let after = &candidate.records[name];
        let change_percent = (after.median_ns_per_iter / before.median_ns_per_iter - 1.0) * 100.0;
        // Classify in the measurement domain. Computing the percentage first can turn an exact
        // boundary such as 100 -> 105 at 5% into 5.000000000000004 and reject it.
        let maximum_ns = before.median_ns_per_iter * (1.0 + limit / 100.0);
        Comparison {
            benchmark: name.clone(),
            baseline_ns: before.median_ns_per_iter,
            candidate_ns: after.median_ns_per_iter,
            change_percent,
            regression: after.median_ns_per_iter > maximum_ns,
        }
    }).collect())
}

fn usage() -> &'static str {
    "usage: xcl-bench-compare BASELINE.ndjson CANDIDATE.ndjson --max-regression-percent PERCENT"
}

fn run() -> Result<bool, String> {
    let mut paths = Vec::new();
    let mut limit = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max-regression-percent" => {
                let value = args.next().ok_or_else(|| usage().to_string())?;
                limit = Some(value.parse::<f64>()
                    .map_err(|_| "maximum regression percent is not a number".to_string())?);
            }
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(false);
            }
            _ if arg.starts_with('-') => return Err(format!("unknown option: {arg}\n{}", usage())),
            _ => paths.push(arg),
        }
    }
    if paths.len() != 2 || limit.is_none() {
        return Err(usage().into());
    }
    let limit = limit.unwrap();
    let baseline = read_records(&paths[0], "baseline")?;
    let candidate = read_records(&paths[1], "candidate")?;
    let comparisons = compare(&baseline, &candidate, limit)?;
    let mut failed = false;
    for comparison in comparisons {
        let status = if comparison.regression { "REGRESSION" } else { "PASS" };
        println!(
            "{status} {}: baseline {:.3} ns, candidate {:.3} ns, change {:+.2}%, limit +{:.2}%",
            comparison.benchmark,
            comparison.baseline_ns,
            comparison.candidate_ns,
            comparison.change_percent,
            limit,
        );
        failed |= comparison.regression;
    }
    Ok(failed)
}

fn main() {
    match run() {
        Ok(true) => std::process::exit(1),
        Ok(false) => {}
        Err(error) => {
            eprintln!("xcl-bench-compare: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn input(records: &[(&str, f64)], metadata: Option<(&str, &str)>) -> Results {
        let lines = records.iter().map(|(name, median)| {
            let mut value = serde_json::json!({
                "schema": "xcl-benchmark",
                "version": 1,
                "benchmark": name,
                "sample_size": 30,
                "sample_iterations": 10,
                "median_ns_per_iter": median,
                "mean_ns_per_iter": median,
                "min_ns_per_iter": median,
                "throughput": null,
            });
            if let Some((environment, hardware)) = metadata {
                value["environment"] = serde_json::json!({ "profile": environment });
                value["hardware"] = serde_json::json!({ "cpu": hardware });
            }
            value.to_string()
        }).collect::<Vec<_>>().join("\n");
        parse_records(Cursor::new(lines), "test").unwrap()
    }

    #[test]
    fn explicit_limit_classifies_observed_median_changes() {
        let baseline = input(&[("a", 100.0), ("b", 200.0)], Some(("release", "cpu-a")));
        let candidate = input(&[("a", 104.0), ("b", 222.0)], Some(("release", "cpu-a")));
        let compared = compare(&baseline, &candidate, 5.0).unwrap();
        assert_eq!(compared[0].benchmark, "a");
        assert!(!compared[0].regression);
        assert!((compared[0].change_percent - 4.0).abs() < 1e-9);
        assert_eq!(compared[1].benchmark, "b");
        assert!(compared[1].regression);
        assert!((compared[1].change_percent - 11.0).abs() < 1e-9);
    }

    #[test]
    fn exact_regression_limit_is_inclusive() {
        let baseline = input(&[("at-limit", 100.0), ("over-limit", 100.0)], None);
        let candidate = input(&[("at-limit", 105.0), ("over-limit", 105.000_001)], None);
        let compared = compare(&baseline, &candidate, 5.0).unwrap();
        assert!(!compared[0].regression, "an exact 5% change is within a 5% limit");
        assert!(compared[1].regression, "a change beyond the limit is rejected");
    }

    #[test]
    fn metadata_and_benchmark_sets_must_match() {
        let baseline = input(&[("a", 100.0)], Some(("release", "cpu-a")));
        let other_hardware = input(&[("a", 100.0)], Some(("release", "cpu-b")));
        assert!(compare(&baseline, &other_hardware, 5.0).unwrap_err().contains("metadata differs"));
        let other_benchmark = input(&[("b", 100.0)], Some(("release", "cpu-a")));
        assert!(compare(&baseline, &other_benchmark, 5.0).unwrap_err().contains("benchmark sets differ"));
    }

    #[test]
    fn malformed_and_duplicate_records_are_rejected() {
        let duplicate = input(&[("a", 100.0)], None).records["a"].benchmark.clone();
        let line = serde_json::json!({
            "schema": "xcl-benchmark", "version": 1, "benchmark": duplicate,
            "sample_size": 1, "sample_iterations": 1,
            "median_ns_per_iter": 1.0, "mean_ns_per_iter": 1.0, "min_ns_per_iter": 1.0
        }).to_string();
        assert!(parse_records(Cursor::new(format!("{line}\n{line}")), "test")
            .unwrap_err().contains("duplicate"));
        let unsupported = line.replace("\"version\":1", "\"version\":2");
        assert!(parse_records(Cursor::new(unsupported), "test").unwrap_err().contains("unsupported"));
        let bad_metadata = serde_json::json!({
            "schema": "xcl-benchmark", "version": 1, "benchmark": "a",
            "sample_size": 1, "sample_iterations": 1,
            "median_ns_per_iter": 1.0, "mean_ns_per_iter": 1.0, "min_ns_per_iter": 1.0,
            "environment": "not an identity object"
        }).to_string();
        assert!(parse_records(Cursor::new(bad_metadata), "test").unwrap_err().contains("malformed"));
    }
}
