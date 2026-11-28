//! A completed corpus's semantic receipt and validated native pipeline-library bytes.

use super::{pipeline_library::NativeLibrary, pso_corpus::Corpus, Pipelines};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use windows::Win32::Graphics::Direct3D12::ID3D12Device;

const VERSION: u32 = 1;
const RECEIPT: &str = "pipeline-libraries-v1.json";
const GRAPHICS: &str = "graphics.plib";
const COMPUTE: &str = "compute.plib";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Semantic {
    version: u32,
    corpus: [u8; 32],
    layout: u64,
    translation: u64,
    settings: u64,
    windows: u32,
    driver: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Receipt {
    semantic: Semantic,
    graphics_bytes: Option<u64>,
    compute_bytes: Option<u64>,
}

#[allow(dead_code)] // Consumed by the stacked lazy runtime integration.
pub(crate) struct Ready {
    pub(crate) graphics: Option<Arc<[u8]>>,
    pub(crate) compute: Option<Arc<[u8]>>,
}

static READY: Mutex<Option<Ready>> = Mutex::new(None);

#[allow(dead_code)] // Consumed by the stacked lazy runtime integration.
pub(crate) fn take_ready() -> Option<Ready> { READY.lock().unwrap().take() }

fn semantic(corpus: [u8; 32], windows: u32, driver: u64) -> Semantic {
    Semantic {
        version: VERSION,
        corpus,
        layout: u64::from_str_radix(env!("XCL_PIPELINE_LAYOUT_ID"), 16).expect("pipeline layout id"),
        translation: crate::gpu::recomp::pso::translation_cache_identity(),
        settings: super::warm_settings_key(),
        windows,
        driver,
    }
}

fn expected(corpus: &Corpus, device: &ID3D12Device) -> Option<Semantic> {
    Some(semantic(
        corpus.identity()?,
        corpus.windows(),
        crate::hle::d3d12x::real::preparation_driver_identity(device)?,
    ))
}

fn paths(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    (dir.join(GRAPHICS), dir.join(COMPUTE), dir.join(RECEIPT))
}

fn pending(path: &Path) -> PathBuf { path.with_extension("plib.pending") }

fn validate(path: &Path, expected: u64, device: &ID3D12Device) -> Option<Arc<[u8]>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() as u64 != expected { return None }
    let bytes: Arc<[u8]> = bytes.into();
    NativeLibrary::open_shared(device, bytes.clone()).ok()?;
    Some(bytes)
}

fn validate_set(dir: &Path, receipt: &Receipt, device: &ID3D12Device) -> Option<Ready> {
    let (graphics, compute, _) = paths(dir);
    Some(Ready {
        graphics: match receipt.graphics_bytes { Some(n) => Some(validate(&graphics, n, device)?), None => None },
        compute: match receipt.compute_bytes { Some(n) => Some(validate(&compute, n, device)?), None => None },
    })
}

pub(crate) fn reuse(corpus: &Corpus, adapter: &str) -> bool {
    match reuse_inner(corpus, adapter) {
        Ok(ready) => {
            *READY.lock().unwrap() = Some(ready);
            true
        }
        Err(reason) => {
            crate::xlog_info!("Pipeline preparation: native library fast path unavailable ({reason})");
            false
        }
    }
}

fn reuse_inner(corpus: &Corpus, adapter: &str) -> Result<Ready, String> {
    let pipelines = Pipelines::for_adapter(adapter)?;
    if !NativeLibrary::supported(&pipelines.dev) { return Err("driver does not support pipeline libraries".into()) }
    let expected = expected(corpus, &pipelines.dev).ok_or("corpus or driver has no semantic identity")?;
    let (_, _, path) = paths(corpus.directory());
    let receipt = super::warm_replay::read_receipt::<Receipt>(&path).ok_or("receipt missing or corrupt")?;
    if receipt.semantic != expected { return Err("receipt semantic identity changed".into()) }
    validate_set(corpus.directory(), &receipt, &pipelines.dev).ok_or("serialized library missing, corrupt or incompatible".into())
}

pub(crate) fn publish(corpus: &Corpus, device: &ID3D12Device) -> Result<(), String> {
    let Some(semantic) = expected(corpus, device) else { return Ok(()) };
    if !NativeLibrary::supported(device) { return Ok(()) }
    let (graphics, compute, receipt_path) = paths(corpus.directory());
    let counts = corpus.counts();
    let graphics_pending = pending(&graphics);
    let compute_pending = pending(&compute);
    let graphics_ready: Option<Arc<[u8]>> = (counts.0 != 0).then(|| std::fs::read(&graphics_pending)).transpose()
        .map_err(|e| format!("Reading prepared graphics library: {e}"))?.map(Into::into);
    let compute_ready: Option<Arc<[u8]>> = (counts.1 != 0).then(|| std::fs::read(&compute_pending)).transpose()
        .map_err(|e| format!("Reading prepared compute library: {e}"))?.map(Into::into);
    let open = |bytes: &Arc<[u8]>| NativeLibrary::open_shared(device, bytes.clone());
    graphics_ready.as_ref().map(open).transpose()
        .map_err(|e| format!("Validating prepared graphics library: {e}"))?;
    compute_ready.as_ref().map(open).transpose()
        .map_err(|e| format!("Validating prepared compute library: {e}"))?;
    if let Some(bytes) = &graphics_ready {
        super::pipeline_library::write_atomic(&graphics, bytes).map_err(|e| e.to_string())?;
    }
    if let Some(bytes) = &compute_ready {
        super::pipeline_library::write_atomic(&compute, bytes).map_err(|e| e.to_string())?;
    }
    let receipt = Receipt {
        semantic,
        graphics_bytes: graphics_ready.as_ref().map(|b| b.len() as u64),
        compute_bytes: compute_ready.as_ref().map(|b| b.len() as u64),
    };
    let encoded = super::warm_replay::receipt_bytes(&receipt).ok_or("Encoding pipeline library receipt")?;
    super::pipeline_library::write_atomic(&receipt_path, &encoded).map_err(|e| e.to_string())?;
    if graphics_ready.is_some() { let _ = std::fs::remove_file(graphics_pending); }
    if compute_ready.is_some() { let _ = std::fs::remove_file(compute_pending); }
    *READY.lock().unwrap() = Some(Ready { graphics: graphics_ready, compute: compute_ready });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::recomp;
    use std::mem::ManuallyDrop;
    use windows::Win32::Graphics::Direct3D12::*;

    #[test]
    fn semantic_inputs_each_invalidate_the_receipt() {
        let base = semantic([1; 32], 4, 7);
        let mut changed = base.clone(); changed.corpus[0] ^= 1; assert_ne!(base, changed);
        let mut changed = base.clone(); changed.layout ^= 1; assert_ne!(base, changed);
        let mut changed = base.clone(); changed.translation ^= 1; assert_ne!(base, changed);
        let mut changed = base.clone(); changed.settings ^= 1; assert_ne!(base, changed);
        let mut changed = base.clone(); changed.windows ^= 1; assert_ne!(base, changed);
        let mut changed = base.clone(); changed.driver ^= 1; assert_ne!(base, changed);
    }

    #[test]
    fn corrupt_and_truncated_receipts_do_not_parse() {
        let dir = std::env::temp_dir().join(format!("xcl-library-receipt-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(RECEIPT);
        std::fs::write(&path, b"{").unwrap();
        assert!(super::super::warm_replay::read_receipt::<Receipt>(&path).is_none());
        std::fs::write(&path, b"not json").unwrap();
        assert!(super::super::warm_replay::read_receipt::<Receipt>(&path).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn incomplete_preparation_never_publishes_a_receipt() {
        let pipelines = match Pipelines::new() {
            Ok(pipelines) => pipelines,
            Err(reason) => { recomp::runner::skip_test(&reason); return; }
        };
        if !NativeLibrary::supported(&pipelines.dev) {
            recomp::runner::skip_test("adapter does not support pipeline libraries");
            return;
        }
        let root = std::env::temp_dir().join(format!("xcl-incomplete-library-{}", std::process::id()));
        struct Cleanup(PathBuf);
        impl Drop for Cleanup { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
        let _cleanup = Cleanup(root.clone());
        let dir = root.join(".xcl/pso-corpus");
        std::fs::create_dir_all(dir.join("compute")).unwrap();
        std::fs::write(dir.join("complete"), "0 1\n").unwrap();
        std::fs::write(dir.join("windows"), "1\n").unwrap();
        std::fs::write(dir.join("identity"), "01".repeat(32)).unwrap();
        std::fs::write(dir.join("compute/0000000000000001.dxbc"), b"fixture").unwrap();
        let corpus = super::super::pso_corpus::discover(&root).unwrap().unwrap();
        assert!(publish(&corpus, &pipelines.dev).is_err());
        assert!(!dir.join(RECEIPT).exists());
    }

    #[test]
    fn cold_publish_makes_an_unchanged_corpus_reusable() {
        if let Err(reason) = recomp::dxc::location() {
            recomp::runner::skip_test(&format!("no DXC: {reason}"));
            return;
        }
        let pipelines = match Pipelines::new() {
            Ok(pipelines) => pipelines,
            Err(reason) => { recomp::runner::skip_test(&reason); return; }
        };
        if !NativeLibrary::supported(&pipelines.dev) {
            recomp::runner::skip_test("adapter does not support pipeline libraries");
            return;
        }
        let root = std::env::temp_dir().join(format!("xcl-warm-library-{}", std::process::id()));
        struct Cleanup(PathBuf);
        impl Drop for Cleanup { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
        let _cleanup = Cleanup(root.clone());
        let dir = root.join(".xcl/pso-corpus");
        std::fs::create_dir_all(dir.join("compute")).unwrap();
        std::fs::write(dir.join("complete"), "0 1\n").unwrap();
        std::fs::write(dir.join("windows"), "1\n").unwrap();
        std::fs::write(dir.join("identity"), "01".repeat(32)).unwrap();
        std::fs::write(dir.join("compute/0000000000000001.dxbc"), b"fixture").unwrap();
        let corpus = super::super::pso_corpus::discover(&root).unwrap().unwrap();

        let source = "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b:register(u0);\n[RootSignature(RS)][numthreads(1,1,1)] void main(){b.Store(0,1);}";
        let dxil = recomp::dxc::compile(source, "main", "cs_6_6", &[]).unwrap();
        let signature: ID3D12RootSignature = unsafe { pipelines.dev.CreateRootSignature(0, &dxil) }.unwrap();
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(signature)),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
            ..Default::default()
        };
        let pso = unsafe { pipelines.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) }.unwrap();
        let _serial = super::super::pipeline_library::BUILD_TEST.lock().unwrap();
        super::super::pipeline_library::abort();
        assert!(super::super::pipeline_library::begin(&pipelines.dev, None).unwrap());
        super::super::pipeline_library::store("fixture.pso".into(), &pso).unwrap();
        super::super::pipeline_library::finish(&dir.join("compute.plib.pending")).unwrap();
        publish(&corpus, &pipelines.dev).unwrap();
        assert!(dir.join(RECEIPT).is_file() && dir.join(COMPUTE).is_file());
        assert!(!dir.join("compute.plib.pending").exists());
        assert!(take_ready().unwrap().compute.is_some());
        std::fs::remove_file(dir.join("compute/0000000000000001.dxbc")).unwrap();
        std::fs::write(root.join("shaders.rpf"), b"invalid pack that the fast path must not read").unwrap();
        let progress = super::super::LaunchPreparation { adapter: "auto".into(), ..Default::default() };
        super::super::prepare_for_launch(&root, &progress).unwrap();
        assert_eq!(progress.done.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(progress.total.load(std::sync::atomic::Ordering::Relaxed), 1);
        super::super::pipeline_library::activate(&pipelines.dev);
        assert!(super::super::pipeline_library::load_compute("fixture.pso", &desc).is_some());
        assert!(super::super::pipeline_library::load_compute("missing.pso", &desc).is_none());
        let final_path = dir.join(COMPUTE);
        let valid = std::fs::read(&final_path).unwrap();
        std::fs::write(&final_path, b"corrupt").unwrap();
        assert!(!reuse(&corpus, "auto"));
        std::fs::remove_file(&final_path).unwrap();
        assert!(!reuse(&corpus, "auto"));
        std::fs::write(&final_path, valid).unwrap();
        std::fs::write(dir.join("identity"), "02".repeat(32)).unwrap();
        let changed = super::super::pso_corpus::summary(&root).unwrap().unwrap();
        assert!(!reuse(&changed, "auto"));
        drop(ManuallyDrop::into_inner(desc.pRootSignature));
    }
}
