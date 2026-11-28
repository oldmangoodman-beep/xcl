//! A staged set of native graphics and compute recipes, shared by startup and offline preparation.
use super::{graphics_summary, progress_count, recomp, worker_command, LaunchPreparation, Pipelines};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex};
use std::time::Duration;

pub(crate) struct Corpus {
    directory: PathBuf,
    expected: [usize; 2],
    graphics: Option<(PathBuf, usize)>,
    compute: Vec<(u64, PathBuf)>,
    windows: u32,
    identity: Option<[u8; 32]>,
}

fn identity(path: &Path) -> Result<Option<[u8; 32]>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("Could not read pipeline corpus identity: {e}")),
    };
    let text = text.trim();
    if text.len() != 64 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Invalid pipeline corpus identity".into());
    }
    let mut out = [0; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2*i..2*i+2], 16).map_err(|_| "Invalid pipeline corpus identity")?;
    }
    Ok(Some(out))
}

fn metadata(root: &Path) -> Result<Option<(PathBuf, [usize; 2], u32, Option<[u8; 32]>)>, String> {
    let dir = root.join(".xcl").join("pso-corpus");
    if !dir.try_exists().map_err(|e| e.to_string())? { return Ok(None); }
    let marker = std::fs::read_to_string(dir.join("complete")).map_err(|e| format!("Incomplete pipeline corpus: {e}"))?;
    let expected = marker.split_whitespace().map(str::parse::<usize>).collect::<Result<Vec<_>, _>>()
        .map_err(|_| "Invalid pipeline corpus counts")?;
    let expected: [usize; 2] = expected.try_into().map_err(|_| "Expected graphics and compute counts in the corpus marker")?;
    let encoded = std::fs::read_to_string(dir.join("windows")).map_err(|e| e.to_string())?;
    let encoded = encoded.trim().parse::<u8>().map_err(|_| "Invalid corpus memory window count")?;
    let count = (encoded & 127) as usize;
    let heap = encoded & 128 != 0;
    if count == 0 || !recomp::valid_window_count(count, heap) || heap && count <= 1 {
        return Err(format!("Invalid corpus memory window configuration {encoded}"));
    }
    let windows = count as u32 | if heap { recomp::HEAP_WINDOWS } else { 0 };
    let identity = identity(&dir.join("identity"))?;
    Ok(Some((dir, expected, windows, identity)))
}

pub(crate) fn summary(root: &Path) -> Result<Option<Corpus>, String> {
    Ok(metadata(root)?.map(|(directory, expected, windows, identity)| Corpus {
        directory,
        expected,
        graphics: None,
        compute: Vec::new(),
        windows,
        identity,
    }))
}

pub(crate) fn discover(root: &Path) -> Result<Option<Corpus>, String> {
    let Some((dir, expected, windows, identity)) = metadata(root)? else { return Ok(None) };
    let path = dir.join("graphics.bin");
    let graphics = if path.try_exists().map_err(|e| e.to_string())? {
        if expected[0] == 0 { return Err("Unexpected graphics job in compute-only corpus".into()); }
        // The worker validates and decodes the job. Require its independent native-success
        // count to match the marker before handoff, without loading the job twice.
        Some((path, expected[0]))
    } else { None };
    let mut compute = Vec::new();
    let folder = dir.join("compute");
    if folder.try_exists().map_err(|e| e.to_string())? {
        for entry in std::fs::read_dir(&folder).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            let name = entry.file_name();
            let key = name.to_str().and_then(|s| s.strip_suffix(".dxbc"))
                .filter(|s| s.len() == 16).and_then(|s| u64::from_str_radix(s, 16).ok());
            let key = key.ok_or_else(|| format!("Invalid compute entry {}", path.display()))?;
            if !entry.file_type().map_err(|e| e.to_string())?.is_file() {
                return Err(format!("Expected a compute file at {}", path.display()));
            }
            compute.push((key, path));
        }
    }
    compute.sort_unstable();
    if expected != [graphics.as_ref().map_or(0, |(_, n)| *n), compute.len()] {
        return Err("Pipeline corpus is missing records. Restage before launching.".into());
    }
    if graphics.is_none() && compute.is_empty() { return Err("Published pipeline corpus is empty".into()); }
    Ok(Some(Corpus { directory: dir, expected, graphics, compute, windows, identity }))
}

impl Corpus {
    pub(crate) fn identity(&self) -> Option<[u8; 32]> { self.identity }

    pub(crate) fn directory(&self) -> &Path { &self.directory }

    pub(crate) fn windows(&self) -> u32 { self.windows }

    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.expected[0], self.expected[1])
    }

    pub(super) fn graphics_path(&self) -> Option<&Path> {
        self.graphics.as_ref().map(|(path, _)| path.as_path())
    }

    pub(super) fn compute_path(&self, key: u64) -> Option<&Path> {
        let i = self.compute.binary_search_by_key(&key, |(key, _)| *key).ok()?;
        Some(self.compute[i].1.as_path())
    }

    pub(super) fn compute_directory(&self) -> Option<&Path> {
        self.compute.first().and_then(|(_, path)| path.parent())
    }

    pub(crate) fn prepare(&self, progress: &LaunchPreparation) -> Result<(), String> {
        self.prepare_with_threads(progress, None)
    }

    pub(super) fn prepare_with_threads(&self, progress: &LaunchPreparation, requested: Option<usize>) -> Result<(), String> {
        let graphics_count = self.graphics.as_ref().map_or(0, |(_, n)| *n);
        let total = graphics_count.checked_add(self.compute.len()).ok_or("Pipeline corpus count overflow")?;
        progress.begin_corpus(total);
        let count = self.windows & !recomp::HEAP_WINDOWS;
        let runtime = crate::gpu::uma::configured_runtime_windows();
        if count as usize != runtime || (self.windows & recomp::HEAP_WINDOWS != 0) != (runtime > 1 && recomp::pso::heap_windows()) {
            return Err(format!("Pipeline corpus memory configuration differs from the runtime ({runtime} windows). Restage for the current configuration or select the matching XCL_UMA_WINDOWS and XCL_HEAP_WINDOWS."));
        }
        let parallel = super::corpus_threads(requested, super::prepare_workers().min(8));
        if let Some((path, n)) = &self.graphics {
            run_graphics(path, *n, count, parallel, progress)?;
        }
        if self.compute.is_empty() {
            progress.check_corpus_complete()?;
            let pipelines = Pipelines::for_adapter(&progress.adapter)?;
            return super::warm_library::publish(self, &pipelines.dev);
        }
        let mut pipelines = Pipelines::for_adapter(&progress.adapter)?;
        let library_path = self.compute_directory().and_then(Path::parent).map(|dir| dir.join("compute.plib.pending"));
        if let Some(path) = &library_path {
            super::pipeline_library::begin(&pipelines.dev, Some(&path.with_extension("")))?;
        }
        pipelines.warm_blobs = super::warm_blobs::Verified::for_device(&pipelines.dev);
        pipelines.max_dxil = 0;
        pipelines.most = parallel;
        let next = AtomicUsize::new(0);
        let completed = Mutex::new(graphics_count);
        let error = Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..parallel.min(self.compute.len()) {
                scope.spawn(|| loop {
                    if progress.cancel.load(Ordering::Acquire) || error.lock().unwrap().is_some() { break; }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some((key,path)) = self.compute.get(i) else { break };
                    let result = std::fs::read(path).map_err(|e| e.to_string()).and_then(|code| {
                        if crate::gpu::shader::fnv64(&code) != *key || !super::is_compute_container(&code) {
                            return Err("Compute shader identity or stage differs from the staged recipe".into());
                        }
                        pipelines.prepare_compute(&code, self.windows)?;
                        let mut done = completed.lock().unwrap();
                        *done += 1;
                        progress.complete_corpus(*done)
                    });
                    if let Err(e) = result {
                        error.lock().unwrap().get_or_insert_with(|| format!("{}: {e}", path.display()));
                        break;
                    }
                });
            }
        });
        if let Some(book) = &pipelines.warm_blobs {
            book.save();
            eprintln!("previously verified native compute pipelines: {}", pipelines.verified.load(Ordering::Relaxed));
        }
        if let Some(e) = error.into_inner().unwrap() {
            super::pipeline_library::abort();
            return Err(e);
        }
        if let Err(e) = progress.check_corpus_complete() {
            super::pipeline_library::abort();
            return Err(e);
        }
        if let Some(path) = library_path.as_deref().filter(|_| super::pipeline_library::building()) {
            let entries = super::pipeline_library::finish(path)?;
            eprintln!("pipeline library: {entries} compute entries");
        }
        super::warm_library::publish(self, &pipelines.dev)
    }
}

fn run_graphics(path: &Path, count: usize, windows: u32, parallel: usize, progress: &LaunchPreparation) -> Result<(), String> {
    let me = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut command = worker_command(&me);
    if std::env::var("XCL_ADAPTER").unwrap_or_default().trim().is_empty() && !progress.adapter.is_empty() {
        command.env("XCL_ADAPTER", &progress.adapter);
    }
    let mut child = command.arg("shaders").arg(path).arg("--graphics-records")
        .arg("--windows").arg(windows.to_string()).env("XCL_PIPELINE_THREADS", parallel.to_string())
        .env("XCL_PIPELINE_LIBRARY_OUT", path.with_file_name("graphics.plib.pending"))
        .spawn().map_err(|e| e.to_string())?;
    let stderr = child.stderr.take().expect("worker_command pipes stderr");
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            if tx.send(line).is_err() { break; }
        }
    });
    let mut summary = None;
    let mut last = String::new();
    let mut made = 0;
    let result = loop {
        if progress.cancel.load(Ordering::Acquire) { break Err("Pipeline preparation cancelled".to_string()); }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Ok(line)) => {
                if let Some(n) = progress_count(&line) {
                    // Independent workers can print their completion counters out of order.
                    made = made.max(n);
                    if let Err(e) = progress.complete_corpus(made) { break Err(e); }
                } else {
                    if let Some(verified) = line.strip_prefix("previously verified native graphics recipes: ")
                        .and_then(|n| n.parse::<usize>().ok()) {
                        eprintln!("Reused {verified} verified native graphics recipes");
                    }
                    if let Some(values) = graphics_summary(&line) { summary = Some(values); }
                    if !line.trim().is_empty() { last = line; }
                }
            }
            Ok(Err(e)) => break Err(e.to_string()),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break Ok(()),
        }
    };
    if result.is_err() { let _ = child.kill(); }
    drop(rx);
    let status = child.wait().map_err(|e| e.to_string())?;
    reader.join().map_err(|_| "Could not read graphics worker output")?;
    result?;
    if !status.success() || !matches!(summary, Some([n, 0, _, _]) if n == count) {
        return Err(format!("Graphics preparation incomplete ({status}): {last}"));
    }
    progress.complete_corpus(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_distinguishes_absent_incomplete_and_ready_compute_corpora() {
        let root = std::env::temp_dir().join(format!("xcl-corpus-discovery-{}", std::process::id()));
        let dir = root.join(".xcl/pso-corpus");
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
        }
        let _cleanup = Cleanup(root.clone());
        assert!(discover(&root).unwrap().is_none());
        std::fs::create_dir_all(dir.join("compute")).unwrap();
        assert!(discover(&root).is_err());
        std::fs::write(dir.join("complete"), "0 1\n").unwrap();
        std::fs::write(dir.join("windows"), "1\n").unwrap();
        assert!(discover(&root).is_err());
        std::fs::write(dir.join("compute/0000000000000010.dxbc"), []).unwrap();
        let corpus = discover(&root).unwrap().unwrap();
        assert_eq!(corpus.compute[0].0, 16);
        assert_eq!(corpus.compute_path(16), Some(dir.join("compute/0000000000000010.dxbc").as_path()));
        assert!(corpus.compute_path(17).is_none());
        assert!(corpus.graphics_path().is_none());
        assert_eq!(corpus.windows, 1);
        assert_eq!(corpus.identity(), None);
        std::fs::write(dir.join("identity"), "00".repeat(32)).unwrap();
        assert_eq!(discover(&root).unwrap().unwrap().identity(), Some([0; 32]));
        std::fs::write(dir.join("identity"), "truncated").unwrap();
        assert!(discover(&root).is_err());
        std::fs::remove_file(dir.join("identity")).unwrap();
        std::fs::write(dir.join("windows"), "255").unwrap();
        assert!(discover(&root).is_err());
        std::fs::write(dir.join("windows"), "0").unwrap();
        assert!(discover(&root).is_err());
    }
}
