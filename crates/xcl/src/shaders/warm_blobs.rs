//! Successful temporary-device validations. Live PSO creation on the render device
//! still supplies the driver blob and retries normally if the driver refuses it.
use std::collections::{hash_map::DefaultHasher, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;

pub(super) fn digest(value: &impl Hash) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

pub(super) fn stamp(path: &Path, layout: u64, blob: &[u8]) -> [u64; 3] {
    [digest(&path.file_name()), layout, digest(&blob)]
}

pub(super) struct Verified {
    path: PathBuf,
    keys: Mutex<Entries>,
}

#[derive(Default)]
struct Entries {
    known: HashSet<[u64; 3]>,
    unsaved: usize,
    failed_at: usize,
}

fn read_keys(path: &Path) -> Option<HashSet<[u64; 3]>> {
    let bytes = std::fs::read(path).ok()?;
    let (checksum, body) = bytes.split_at_checked(8)?;
    if body.len() % 24 != 0 || checksum != digest(&body).to_le_bytes() { return None; }
    Some(body.chunks_exact(24).map(|row| {
        std::array::from_fn(|i| u64::from_le_bytes(row[i*8..i*8+8].try_into().unwrap()))
    }).collect())
}

fn legacy_path(dir: &Path, native: u64, slice: &str) -> PathBuf {
    let key = digest(&(native, slice));
    dir.join(format!("verified-compute-v1-{key:016x}.bin"))
}

impl Verified {
    pub(super) fn for_device(dev: &ID3D12Device) -> Option<Self> {
        let native = super::warm_replay::native_key(dev)?;
        Some(Self::open(&crate::gpu::recomp::dxc::cache_dir(), native))
    }

    fn open(dir: &Path, native: u64) -> Self {
        let path = dir.join(format!("verified-compute-v2-{native:016x}.bin"));
        let mut entries = Entries::default();
        if let Some(keys) = read_keys(&path) {
            entries.known = keys;
        } else {
            // v1 hashed the worker slice into the filename. Only import this native identity.
            for count in 1..=32 {
                for i in 0..count {
                    let old = legacy_path(dir, native, &format!("{i}/{count}"));
                    if let Some(keys) = read_keys(&old) { entries.known.extend(keys); }
                }
            }
            if let Some(keys) = read_keys(&legacy_path(dir, native, "")) { entries.known.extend(keys); }
            entries.unsaved = entries.known.len();
        }
        Self { path, keys: Mutex::new(entries) }
    }

    pub(super) fn contains(&self, stamp: &[u64; 3]) -> bool {
        self.keys.lock().unwrap().known.contains(stamp)
    }

    pub(super) fn record(&self, stamp: [u64; 3]) {
        let mut entries = self.keys.lock().unwrap();
        if entries.known.insert(stamp) {
            entries.unsaved += 1;
            if entries.unsaved >= entries.failed_at + 64 { self.flush(&mut entries); }
        }
    }

    pub(super) fn save(&self) {
        let mut entries = self.keys.lock().unwrap();
        if entries.unsaved > 0 { self.flush(&mut entries); }
    }

    fn flush(&self, entries: &mut Entries) {
        if let Err(e) = self.merge_and_save(entries) {
            entries.failed_at = entries.unsaved;
            crate::xlog_warn!("Shader preparation: could not save verified compute blob list: {e}");
        }
    }

    fn merge_and_save(&self, entries: &mut Entries) -> std::io::Result<()> {
        // Lock a separate file: atomic replacement changes the receipt's file identity.
        std::fs::create_dir_all(self.path.parent().unwrap())?;
        let lock = std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true)
            .open(self.path.with_extension("lock"))?;
        lock.lock()?;
        if let Some(saved) = read_keys(&self.path) { entries.known.extend(saved); }
        let mut data = vec![0; 8];
        data.reserve(24 * entries.known.len());
        for row in &entries.known {
            for v in row { data.extend_from_slice(&v.to_le_bytes()); }
        }
        let sum = digest(&&data[8..]).to_le_bytes();
        data[..8].copy_from_slice(&sum);
        if !crate::hle::d3d12x::real::driver_cache::write(&self.path, &data) {
            return Err(std::io::Error::other("could not replace receipt"));
        }
        entries.unsaved = 0;
        entries.failed_at = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn warm_receipt_matches_only_the_validated_blob_and_layout() {
        let dir = std::env::temp_dir().join(format!("xcl-warm-receipt-{}", std::process::id()));
        let book = Verified::open(&dir, 1);
        let path = book.path.clone();
        let file = Path::new("123.chain.pso");
        let key = stamp(file, 1, &[3, 4]);
        book.record(key);
        book.save();
        let book = Verified::open(&dir, 1);
        assert!(book.contains(&key));
        assert!(!book.contains(&stamp(file, 2, &[3, 4])));
        assert!(!book.contains(&stamp(file, 1, &[3, 5])));
        assert!(!book.contains(&stamp(Path::new("123.pso"), 1, &[3, 4])));
        let mut contents = std::fs::read(&path).unwrap();
        contents[8] ^= 1;
        std::fs::write(&path, contents).unwrap();
        assert!(!Verified::open(&dir, 1).contains(&key));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn workers_reuse_previous_slices_and_merge_concurrent_checkpoints() {
        let dir = std::env::temp_dir().join(format!("xcl-warm-workers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let native = 41;
        let first = [1, 2, 3];
        let body: Vec<_> = first.iter().flat_map(|v: &u64| v.to_le_bytes()).collect();
        let legacy = [digest(&body.as_slice()).to_le_bytes().as_slice(), &body].concat();
        std::fs::write(legacy_path(&dir, native, "27/28"), legacy).unwrap();
        let a = Verified::open(&dir, native);
        let b = Verified::open(&dir, native);
        assert!(a.contains(&first));
        std::thread::scope(|scope| {
            for (worker, group) in [(a, 2), (b, 3)] {
                scope.spawn(move || {
                    // Opening both workers before either writes exercises stale snapshots.
                    for i in 0..64 { worker.record([group, i, 0]); }
                    worker.save();
                });
            }
        });
        let merged = Verified::open(&dir, native);
        assert!(merged.contains(&first));
        for i in 0..64 {
            assert!(merged.contains(&[2, i, 0]) && merged.contains(&[3, i, 0]));
        }
        let other_device = Verified::open(&dir, native + 1);
        assert!(!other_device.contains(&first) && !other_device.contains(&[2, 0, 0]));
        // An interrupted worker leaves its last complete checkpoint for a later launch.
        for i in 0..64 { merged.record([4, i, 0]); }
        drop(merged);
        assert!(Verified::open(&dir, native).contains(&[4, 63, 0]));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_failed_checkpoint_is_tried_again_after_64_more_entries() {
        let dir = std::env::temp_dir().join(format!("xcl-warm-failed-{}", std::process::id()));
        let book = Verified::open(&dir, 7);
        std::fs::create_dir_all(&book.path).unwrap();
        for i in 0..100 { book.record([5, i, 0]); }
        assert_eq!(book.keys.lock().unwrap().failed_at, 64);
        std::fs::remove_dir(&book.path).unwrap();
        for i in 100..128 { book.record([5, i, 0]); }
        let reopened = Verified::open(&dir, 7);
        assert!(reopened.contains(&[5, 0, 0]) && reopened.contains(&[5, 127, 0]));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
