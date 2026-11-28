//! Native D3D12 pipeline-library ownership and serialization.
//!
//! D3D12 retains the pointer passed to `CreatePipelineLibrary`, so a reopened library must own
//! its serialized bytes for at least as long as the COM object.

use windows::core::{HSTRING, Interface, Result};
use windows::Win32::Graphics::Direct3D12::*;

struct Builder {
    library: NativeLibrary,
    previous: Option<std::sync::Arc<Previous>>,
    names: std::collections::HashSet<String>,
    imported: std::collections::HashSet<String>,
}

struct Previous {
    library: NativeLibrary,
    names: [std::sync::Mutex<()>; 64],
}

static BUILD: std::sync::Mutex<Option<Builder>> = std::sync::Mutex::new(None);

struct Active {
    graphics: Option<NativeLibrary>,
    compute: Option<NativeLibrary>,
    names: [std::sync::Mutex<()>; 64],
}

static ACTIVE: std::sync::OnceLock<Active> = std::sync::OnceLock::new();

pub(crate) fn activate(device: &ID3D12Device) {
    let Some(ready) = super::warm_library::take_ready() else { return };
    let open = |bytes| NativeLibrary::open_shared(device, bytes);
    let graphics = match ready.graphics.map(open).transpose() {
        Ok(library) => library,
        Err(e) => {
            crate::xlog_warn!("Pipeline library: runtime graphics activation failed ({e})");
            return;
        }
    };
    let compute = match ready.compute.map(open).transpose() {
        Ok(library) => library,
        Err(e) => {
            crate::xlog_warn!("Pipeline library: runtime compute activation failed ({e})");
            return;
        }
    };
    if ACTIVE.set(Active { graphics, compute, names: std::array::from_fn(|_| std::sync::Mutex::new(())) }).is_ok() {
        crate::xlog_info!("Pipeline library: activated prepared PSOs for lazy runtime loading");
    }
}

fn with_name<T>(name: &str, f: impl FnOnce(&Active) -> Option<windows::core::Result<T>>) -> Option<T> {
    let active = ACTIVE.get()?;
    let slot = crate::gpu::shader::fnv64(name.as_bytes()) as usize % active.names.len();
    let _same_name = active.names[slot].lock().unwrap();
    f(active)?.ok()
}

fn from_previous<T>(name: &str, f: impl FnOnce(&NativeLibrary) -> windows::core::Result<T>) -> Option<T> {
    let previous = BUILD.lock().unwrap().as_ref()?.previous.clone()?;
    let slot = crate::gpu::shader::fnv64(name.as_bytes()) as usize % previous.names.len();
    let _same_name = previous.names[slot].lock().unwrap();
    let result = f(&previous.library).ok()?;
    if let Some(builder) = BUILD.lock().unwrap().as_mut() {
        builder.imported.insert(name.into());
    }
    Some(result)
}

pub(crate) fn load_compute(name: &str, desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC) -> Option<ID3D12PipelineState> {
    from_previous(name, |library| library.load_compute(name, desc))
        .or_else(|| with_name(name, |active| active.compute.as_ref().map(|library| library.load_compute(name, desc))))
}

pub(crate) fn load_graphics(name: &str, desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> Option<ID3D12PipelineState> {
    from_previous(name, |library| library.load_graphics(name, desc))
        .or_else(|| with_name(name, |active| active.graphics.as_ref().map(|library| library.load_graphics(name, desc))))
}

pub(crate) fn load_stream(name: &str, desc: &D3D12_PIPELINE_STATE_STREAM_DESC) -> Option<ID3D12PipelineState> {
    from_previous(name, |library| library.load_stream(name, desc))
        .or_else(|| with_name(name, |active| active.graphics.as_ref().map(|library| library.load_stream(name, desc))))
}

pub(crate) fn building() -> bool { BUILD.lock().unwrap().is_some() }

pub(crate) fn begin(device: &ID3D12Device, previous: Option<&std::path::Path>) -> std::result::Result<bool, String> {
    if !NativeLibrary::supported(device) { return Ok(false) }
    let library = NativeLibrary::empty(device).map_err(|e| format!("Creating pipeline library: {e}"))?;
    let previous = previous
        .and_then(|path| std::fs::read(path).ok())
        .and_then(|bytes| NativeLibrary::open(device, bytes).ok())
        .map(|library| std::sync::Arc::new(Previous {
            library,
            names: std::array::from_fn(|_| std::sync::Mutex::new(())),
        }));
    let mut active = BUILD.lock().unwrap();
    if active.is_some() { return Err("A pipeline library build is already active".into()); }
    *active = Some(Builder { library, previous, names: Default::default(), imported: Default::default() });
    Ok(true)
}

pub(crate) fn store(name: String, pipeline: &ID3D12PipelineState) -> std::result::Result<(), String> {
    let mut active = BUILD.lock().unwrap();
    let Some(builder) = active.as_mut() else { return Ok(()) };
    if !builder.names.insert(name.clone()) { return Ok(()) }
    if let Err(e) = builder.library.store(&name, pipeline) {
        builder.names.remove(&name);
        return Err(format!("Storing pipeline {name}: {e}"));
    }
    Ok(())
}

pub(crate) fn abort() { *BUILD.lock().unwrap() = None; }

pub(crate) fn finish(path: &std::path::Path) -> std::result::Result<usize, String> {
    let builder = BUILD.lock().unwrap().take().ok_or("No pipeline library build is active")?;
    let count = builder.names.len();
    let imported = builder.imported.intersection(&builder.names).count();
    let bytes = builder.library.serialize().map_err(|e| format!("Serializing pipeline library: {e}"))?;
    write_atomic(path, &bytes).map_err(|e| format!("Publishing {}: {e}", path.display()))?;
    if imported != 0 { eprintln!("pipeline library: imported {imported} exact entries from the previous library"); }
    Ok(count)
}

pub(crate) fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::windows::ffi::OsStrExt;
    use std::sync::atomic::{AtomicU64, Ordering};
    use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let (temporary, mut file) = loop {
        let candidate = path.with_extension(format!("tmp{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(file) => break (candidate, file),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        let from: Vec<u16> = temporary.as_os_str().encode_wide().chain([0]).collect();
        let to: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    })();
    if result.is_err() { let _ = std::fs::remove_file(temporary); }
    result
}

#[cfg(test)]
pub(crate) static BUILD_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) struct NativeLibrary {
    library: ID3D12PipelineLibrary1,
    _serialized: std::sync::Arc<[u8]>,
}

impl NativeLibrary {
    pub(crate) fn supported(device: &ID3D12Device) -> bool {
        let mut feature = D3D12_FEATURE_DATA_SHADER_CACHE::default();
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_CACHE,
                (&mut feature as *mut D3D12_FEATURE_DATA_SHADER_CACHE).cast(),
                std::mem::size_of_val(&feature) as u32,
            )
        }
        .is_ok()
            && feature.SupportFlags.0 & D3D12_SHADER_CACHE_SUPPORT_LIBRARY.0 != 0
    }

    pub(crate) fn empty(device: &ID3D12Device) -> Result<Self> {
        Self::open_shared(device, std::sync::Arc::from([]))
    }

    pub(crate) fn open(device: &ID3D12Device, serialized: Vec<u8>) -> Result<Self> {
        Self::open_shared(device, serialized.into())
    }

    pub(crate) fn open_shared(device: &ID3D12Device, serialized: std::sync::Arc<[u8]>) -> Result<Self> {
        let device: ID3D12Device1 = device.cast()?;
        let library = if serialized.is_empty() {
            let mut library = std::ptr::null_mut();
            unsafe {
                (Interface::vtable(&device).CreatePipelineLibrary)(
                    Interface::as_raw(&device),
                    std::ptr::null(),
                    0,
                    &ID3D12PipelineLibrary1::IID,
                    &mut library,
                ).ok()?;
                ID3D12PipelineLibrary1::from_raw(library)
            }
        } else {
            unsafe { device.CreatePipelineLibrary::<ID3D12PipelineLibrary1>(&serialized)? }
        };
        Ok(Self { library, _serialized: serialized })
    }

    pub(crate) fn store(&self, name: &str, pipeline: &ID3D12PipelineState) -> Result<()> {
        unsafe { self.library.StorePipeline(&HSTRING::from(name), pipeline) }
    }

    pub(crate) fn load_compute(
        &self,
        name: &str,
        desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
    ) -> Result<ID3D12PipelineState> {
        unsafe { self.library.LoadComputePipeline(&HSTRING::from(name), desc) }
    }

    pub(crate) fn load_graphics(
        &self,
        name: &str,
        desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    ) -> Result<ID3D12PipelineState> {
        unsafe { self.library.LoadGraphicsPipeline(&HSTRING::from(name), desc) }
    }

    pub(crate) fn load_stream(
        &self,
        name: &str,
        desc: &D3D12_PIPELINE_STATE_STREAM_DESC,
    ) -> Result<ID3D12PipelineState> {
        unsafe { self.library.LoadPipeline(&HSTRING::from(name), desc) }
    }

    pub(crate) fn serialize(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; unsafe { self.library.GetSerializedSize() }];
        unsafe { self.library.Serialize(&mut bytes)? };
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::{abort, begin, finish, store, NativeLibrary, BUILD_TEST};
    use crate::gpu::recomp;
    use crate::shaders::Pipelines;
    use std::mem::ManuallyDrop;
    use std::time::Instant;
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Direct3D12::*;

    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup { fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); } }

    fn fixture() -> Option<(Pipelines, Vec<u8>, ID3D12RootSignature)> {
        if let Err(reason) = recomp::dxc::location() {
            recomp::runner::skip_test(&format!("no DXC: {reason}"));
            return None;
        }
        let pipelines = match Pipelines::new() {
            Ok(pipelines) => pipelines,
            Err(reason) => {
                recomp::runner::skip_test(&format!("no D3D12 device: {reason}"));
                return None;
            }
        };
        if !NativeLibrary::supported(&pipelines.dev) {
            recomp::runner::skip_test("adapter does not support D3D12 pipeline libraries");
            return None;
        }
        let source = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer output : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { output.Store(0, 42); }
"#;
        let dxil = recomp::dxc::compile(source, "main", "cs_6_6", &[]).expect("DXC compile");
        let root = unsafe { pipelines.dev.CreateRootSignature(0, &dxil) }.expect("root signature");
        Some((pipelines, dxil, root))
    }

    fn compute_desc(dxil: &[u8], root: &ID3D12RootSignature) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
        D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr().cast(),
                BytecodeLength: dxil.len(),
            },
            ..Default::default()
        }
    }

    fn drop_desc(desc: D3D12_COMPUTE_PIPELINE_STATE_DESC) {
        drop(ManuallyDrop::into_inner(desc.pRootSignature));
    }

    #[test]
    fn compute_pipeline_round_trips_through_serialized_library() {
        let Some((pipelines, dxil, root)) = fixture() else { return };
        let desc = compute_desc(&dxil, &root);
        let pso = unsafe { pipelines.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) }
            .expect("compute pipeline");
        let library = NativeLibrary::empty(&pipelines.dev).expect("empty pipeline library");
        for n in 1..=256 {
            library.store(&format!("compute-{n:04}"), &pso).expect("store pipeline alias");
        }
        assert!(library.store("compute-0001", &pso).is_err(), "duplicate names must be rejected");

        let serialize_at = Instant::now();
        let bytes = library.serialize().expect("serialize pipeline library");
        let serialize_time = serialize_at.elapsed();
        assert!(!bytes.is_empty());

        let open_at = Instant::now();
        let reopened = NativeLibrary::open(&pipelines.dev, bytes.clone()).expect("reopen pipeline library");
        let open_time = open_at.elapsed();
        let loaded = reopened.load_compute("compute-0001", &desc).expect("load compute pipeline");
        assert!(unsafe { loaded.GetCachedBlob() }.is_ok());
        println!(
            "pipeline library: aliases=256 serialized_bytes={} serialize_us={} open_us={}",
            bytes.len(),
            serialize_time.as_micros(),
            open_time.as_micros()
        );
        drop_desc(desc);
    }

    #[test]
    fn distinct_compute_pipelines_scale_and_remain_individually_loadable() {
        const COUNT: usize = 32;
        let Some((pipelines, _, root)) = fixture() else { return };
        let library = NativeLibrary::empty(&pipelines.dev).expect("empty pipeline library");
        let mut programs = Vec::with_capacity(COUNT);
        let build_at = Instant::now();
        for n in 0..COUNT {
            let source = format!(r#"
#define RS "UAV(u0)"
RWByteAddressBuffer output : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {{ output.Store(0, {n}); }}
"#);
            let dxil = recomp::dxc::compile(&source, "main", "cs_6_6", &[]).expect("DXC compile");
            let desc = compute_desc(&dxil, &root);
            let pso = unsafe { pipelines.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) }
                .expect("distinct compute pipeline");
            library.store(&format!("unique-{n:04}"), &pso).expect("store distinct pipeline");
            drop_desc(desc);
            programs.push(dxil);
        }
        let build_time = build_at.elapsed();
        let bytes = library.serialize().expect("serialize distinct pipelines");
        let open_at = Instant::now();
        let reopened = NativeLibrary::open(&pipelines.dev, bytes.clone()).expect("reopen distinct pipelines");
        let open_time = open_at.elapsed();
        let load_at = Instant::now();
        for (n, dxil) in programs.iter().enumerate() {
            let desc = compute_desc(dxil, &root);
            reopened.load_compute(&format!("unique-{n:04}"), &desc).expect("load distinct pipeline");
            drop_desc(desc);
        }
        let load_time = load_at.elapsed();
        println!(
            "pipeline library: unique={COUNT} serialized_bytes={} cold_ms={} open_us={} load_all_us={}",
            bytes.len(), build_time.as_millis(), open_time.as_micros(), load_time.as_micros()
        );
    }

    #[test]
    fn truncated_pipeline_library_is_rejected() {
        let Some((pipelines, dxil, root)) = fixture() else { return };
        let desc = compute_desc(&dxil, &root);
        let pso = unsafe { pipelines.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) }
            .expect("compute pipeline");
        let library = NativeLibrary::empty(&pipelines.dev).expect("empty pipeline library");
        library.store("compute-0001", &pso).expect("store pipeline");
        let mut bytes = library.serialize().expect("serialize pipeline library");
        bytes.truncate(bytes.len() / 2);
        assert!(NativeLibrary::open(&pipelines.dev, bytes).is_err());
        drop_desc(desc);
    }

    #[test]
    fn builder_deduplicates_names_and_atomically_replaces_output() {
        let _serial = BUILD_TEST.lock().unwrap();
        let Some((pipelines, dxil, root)) = fixture() else { return };
        let desc = compute_desc(&dxil, &root);
        let pso = unsafe { pipelines.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) }
            .expect("compute pipeline");
        let path = std::env::temp_dir().join(format!("xcl-pipeline-library-{}.bin", std::process::id()));
        let _cleanup = Cleanup(path.clone());
        abort();
        assert!(begin(&pipelines.dev, None).expect("begin library"));
        store("one".into(), &pso).expect("store one");
        store("one".into(), &pso).expect("deduplicate one");
        assert_eq!(finish(&path).expect("publish library"), 1);
        let first = std::fs::read(&path).expect("published bytes");
        NativeLibrary::open(&pipelines.dev, first).expect("published library");

        assert!(begin(&pipelines.dev, Some(&path)).expect("begin replacement"));
        let changed = recomp::dxc::compile(
            "[RootSignature(\"UAV(u0)\")][numthreads(1,1,1)] void main() {}",
            "main", "cs_6_6", &[],
        ).expect("compile changed compute shader");
        let changed_desc = compute_desc(&changed, &root);
        assert!(super::load_compute("one", &changed_desc).is_none());
        drop_desc(changed_desc);
        let imported = super::load_compute("one", &desc).expect("import previous pipeline");
        assert!(super::load_compute("missing", &desc).is_none());
        store("one".into(), &imported).expect("store imported pipeline");
        store("two".into(), &pso).expect("store replacement");
        assert_eq!(finish(&path).expect("replace library"), 2);
        let replacement = std::fs::read(&path).expect("replacement bytes");
        let library = NativeLibrary::open(&pipelines.dev, replacement).expect("replacement library");
        library.load_compute("one", &desc).expect("load imported pipeline");
        library.load_compute("two", &desc).expect("load replacement");

        std::fs::write(&path, b"not a pipeline library").expect("write corrupt previous library");
        assert!(begin(&pipelines.dev, Some(&path)).expect("ignore corrupt previous library"));
        assert!(super::load_compute("one", &desc).is_none());
        abort();
        drop_desc(desc);
    }

    #[test]
    fn graphics_pipeline_round_trips_through_serialized_library() {
        let _serial = BUILD_TEST.lock().unwrap();
        let Some((pipelines, _, _)) = fixture() else { return };
        let source = r#"
#define RS ""
struct V { float4 p : SV_Position; };
[RootSignature(RS)] V vs(uint id : SV_VertexID) {
    V v; v.p = float4(id == 1 ? 1.0 : -1.0, id == 2 ? 1.0 : -1.0, 0.0, 1.0); return v;
}
[RootSignature(RS)] float4 ps() : SV_Target { return float4(1, 0, 1, 1); }
"#;
        let vs = recomp::dxc::compile(source, "vs", "vs_6_6", &[]).expect("compile VS");
        let ps = recomp::dxc::compile(source, "ps", "ps_6_6", &[]).expect("compile PS");
        let root = unsafe { pipelines.dev.CreateRootSignature(0, &vs) }.expect("graphics root signature");
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root)),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let pso = unsafe { pipelines.dev.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc) }
            .expect("graphics pipeline");
        let library = NativeLibrary::empty(&pipelines.dev).expect("empty pipeline library");
        library.store("graphics-0001", &pso).expect("store graphics pipeline");
        let path = std::env::temp_dir().join(format!("xcl-graphics-library-{}.bin", std::process::id()));
        let _cleanup = Cleanup(path.clone());
        std::fs::write(&path, library.serialize().expect("serialize graphics library")).expect("write graphics library");
        assert!(begin(&pipelines.dev, Some(&path)).expect("begin graphics replacement"));
        let imported = super::load_graphics("graphics-0001", &desc).expect("import graphics pipeline");
        store("graphics-0001".into(), &imported).expect("store imported graphics pipeline");
        assert_eq!(finish(&path).expect("publish graphics replacement"), 1);
        let reopened = NativeLibrary::open(&pipelines.dev, std::fs::read(&path).unwrap()).expect("reopen graphics library");
        reopened.load_graphics("graphics-0001", &desc).expect("load graphics pipeline");
        drop(ManuallyDrop::into_inner(desc.pRootSignature));
    }

    #[repr(C, align(8))]
    struct Sub<T> {
        ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
        value: T,
    }

    #[repr(C)]
    struct MeshStream {
        root: Sub<*mut std::ffi::c_void>,
        mesh: Sub<D3D12_SHADER_BYTECODE>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    }

    #[test]
    fn mesh_pipeline_round_trips_through_serialized_library() {
        let _serial = BUILD_TEST.lock().unwrap();
        let Some((pipelines, _, _)) = fixture() else { return };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        if unsafe {
            pipelines.dev.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS7,
                (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS7).cast(),
                std::mem::size_of_val(&options) as u32,
            )
        }
        .is_err()
            || options.MeshShaderTier.0 < D3D12_MESH_SHADER_TIER_1.0
        {
            recomp::runner::skip_test("adapter does not support mesh shaders");
            return;
        }
        let source = r#"
#define RS ""
struct V { float4 p : SV_Position; };
[RootSignature(RS)]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(out vertices V vertices[3], out indices uint3 triangles[1]) {
    SetMeshOutputCounts(3, 1);
    vertices[0].p = float4(-1, -1, 0, 1);
    vertices[1].p = float4( 0,  1, 0, 1);
    vertices[2].p = float4( 1, -1, 0, 1);
    triangles[0] = uint3(0, 1, 2);
}
"#;
        let mesh = recomp::dxc::compile(source, "main", "ms_6_6", &[]).expect("compile MS");
        let root: ID3D12RootSignature = unsafe { pipelines.dev.CreateRootSignature(0, &mesh) }
            .expect("mesh root signature");
        let mut stream = MeshStream {
            root: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, value: root.as_raw() },
            mesh: Sub {
                ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,
                value: D3D12_SHADER_BYTECODE { pShaderBytecode: mesh.as_ptr().cast(), BytecodeLength: mesh.len() },
            },
            topology: Sub {
                ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                value: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            },
        };
        let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
            SizeInBytes: std::mem::size_of::<MeshStream>(),
            pPipelineStateSubobjectStream: (&mut stream as *mut MeshStream).cast(),
        };
        let device: ID3D12Device2 = pipelines.dev.cast().expect("ID3D12Device2");
        let pso = unsafe { device.CreatePipelineState::<ID3D12PipelineState>(&desc) }.expect("mesh pipeline");
        let library = NativeLibrary::empty(&pipelines.dev).expect("empty pipeline library");
        library.store("mesh-0001", &pso).expect("store mesh pipeline");
        let path = std::env::temp_dir().join(format!("xcl-mesh-library-{}.bin", std::process::id()));
        let _cleanup = Cleanup(path.clone());
        std::fs::write(&path, library.serialize().expect("serialize mesh library")).expect("write mesh library");
        assert!(begin(&pipelines.dev, Some(&path)).expect("begin mesh replacement"));
        let imported = super::load_stream("mesh-0001", &desc).expect("import mesh pipeline");
        store("mesh-0001".into(), &imported).expect("store imported mesh pipeline");
        assert_eq!(finish(&path).expect("publish mesh replacement"), 1);
        let reopened = NativeLibrary::open(&pipelines.dev, std::fs::read(&path).unwrap()).expect("reopen mesh library");
        reopened.load_stream("mesh-0001", &desc).expect("load mesh pipeline");
    }
}
