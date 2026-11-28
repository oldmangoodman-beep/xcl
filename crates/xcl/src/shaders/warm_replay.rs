//! A recipe can be left on disk only after a successful replay identified every
//! native blob it used. The render device still creates the live PSO on first use.
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::os::windows::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Interrupted writes or a damaged index fall back to ordinary preparation.
pub(crate) fn read_receipt<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    let (checksum, payload) = bytes.split_at_checked(8)?;
    if checksum != crate::gpu::shader::fnv64(payload).to_le_bytes() { return None; }
    serde_json::from_slice(payload).ok()
}

pub(crate) fn write_receipt<T: Serialize>(path: &Path, receipt: &T) -> bool {
    let Some(bytes) = receipt_bytes(receipt) else { return false };
    crate::hle::d3d12x::real::driver_cache::write(path, &bytes)
}

pub(crate) fn receipt_bytes<T: Serialize>(receipt: &T) -> Option<Vec<u8>> {
    let mut bytes = vec![0; 8];
    if serde_json::to_writer(&mut bytes, receipt).is_err() { return None; }
    let checksum = crate::gpu::shader::fnv64(&bytes[8..]).to_le_bytes();
    bytes[..8].copy_from_slice(&checksum);
    Some(bytes)
}

pub(crate) type Stamp = [u64; 3];
pub(crate) type Inventory = HashMap<String, Stamp>;

fn stamp(m: &std::fs::Metadata) -> Option<Stamp> {
    (m.is_file() && m.len() != 0 && m.file_attributes() & 0x400 == 0).then(|| [m.len(), m.creation_time(), m.last_write_time()])
}

pub(crate) fn driver_files(dir: &Path) -> Option<Inventory> {
    let mut files = Inventory::new();
    for e in std::fs::read_dir(dir).ok()? {
        let e = e.ok()?;
        let m = e.metadata().ok()?;
        if m.file_attributes() & 0x400 != 0 { continue; }
        let Some(stamp) = stamp(&m) else { continue };
        if let Ok(name) = e.file_name().into_string() {
            if name.ends_with(".pso") { files.insert(name, stamp); }
        }
    }
    Some(files)
}
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct File { name: String, stamp: Stamp }
type Recipes = HashMap<u64, Vec<File>>;

thread_local! { static CAPTURE: RefCell<Option<Vec<File>>> = const { RefCell::new(None) }; }

pub(super) fn capture(f: impl FnOnce() -> bool) -> (bool, Vec<File>) {
    CAPTURE.with(|slot| { *slot.borrow_mut() = Some(Vec::new()); });
    let success = f();
    let files = CAPTURE.with(|slot| slot.take().unwrap_or_default());
    (success, files)
}

/// Record only bytes accepted by the native driver that still occupy this cache file.
pub(crate) fn accepted(path: Option<&Path>, bytes: &[u8]) {
    CAPTURE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(files) = slot.as_mut() else { return };
        let verified = || {
            let path = path?;
            let before = stamp(&path.metadata().ok()?)?;
            if before[0] != bytes.len() as u64 || std::fs::read(path).ok()?.as_slice() != bytes { return None; }
            if stamp(&path.metadata().ok()?)? != before { return None; }
            Some(File { name: path.file_name()?.to_str()?.to_owned(), stamp: before })
        };
        if let Some(file) = verified() { files.push(file); }
    });
}

pub(super) struct Index {
    path: PathBuf,
    entries: Recipes,
    new: Mutex<Recipes>,
}

impl Index {
    pub(super) fn open(dir: &Path, known: &Inventory, key: u64, windows: u32) -> Option<Self> {
        let path = dir.join(format!("warm-recipes-v2-{key:016x}-{windows:02x}.bin"));
        let mut entries: Recipes = read_receipt(&path).unwrap_or_default();
        entries.retain(|_, files| !files.is_empty() && files.iter().all(|f| known.get(&f.name) == Some(&f.stamp)));
        Some(Self { path, entries, new: Default::default() })
    }
    pub(super) fn can_defer(&self, hash: u64) -> bool { self.entries.contains_key(&hash) }
    pub(super) fn add(&self, hash: u64, files: Vec<File>, mesh: bool) {
        let complete = if mesh {
            files.len() == 2 && files.iter().any(|f| f.name.starts_with("tp-ms-"))
                && files.iter().any(|f| f.name.starts_with("tp-cs-"))
        } else {
            files.len() == 1 && files[0].name.starts_with("gfx-")
        };
        if complete { self.new.lock().unwrap().insert(hash, files); }
    }
    pub(super) fn save(mut self) {
        let new = self.new.into_inner().unwrap();
        if new.is_empty() { return; }
        self.entries.extend(new);
        if !write_receipt(&self.path, &self.entries) {
            crate::xlog_warn!("PSO preparation: could not save the verified recipe index");
        }
    }
}

pub(super) fn native_key(dev: &windows::Win32::Graphics::Direct3D12::ID3D12Device) -> Option<u64> {
    let gpu = crate::hle::d3d12x::real::preparation_driver_identity(dev)?;
    let translation = crate::gpu::recomp::pso::translation_cache_identity();
    if translation == 0 || !crate::hle::d3d12x::real::driver_cache::enabled() { return None; }
    let cache = std::path::absolute(crate::gpu::recomp::dxc::cache_dir()).ok()?;
    Some(super::warm_blobs::digest(&(env!("XCL_NATIVE_PREPARATION_ID"), gpu, translation, cache, super::warm_settings_key())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_mesh_needs_both_unchanged_driver_files() {
        let dir = std::env::temp_dir().join(format!("xcl-warm-mesh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let graphics = dir.join("tp-ms-test.pso");
        let compute = dir.join("tp-cs-test.pso");
        std::fs::write(&graphics, b"graphics").unwrap();
        std::fs::write(&compute, b"compute").unwrap();
        let open = || Index::open(&dir, &driver_files(&dir).unwrap(), 1, 4).unwrap();
        let capture_passes = |compute_bytes: &[u8]| capture(|| {
            accepted(Some(&graphics), b"graphics");
            accepted(Some(&compute), compute_bytes);
            true
        }).1;
        let book = open();
        book.add(1, capture_passes(b"wrong"), true);
        book.add(2, capture_passes(b"compute"), true);
        book.save();
        assert!(!open().can_defer(1));
        assert!(open().can_defer(2));
        std::fs::remove_file(&compute).unwrap();
        assert!(!open().can_defer(2));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
