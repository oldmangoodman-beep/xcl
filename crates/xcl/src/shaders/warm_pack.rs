//! Skip re-emitting graphics stages when their successful pack pass still has all its DXIL.
use super::{warm_blobs::digest, warm_replay::{read_receipt, write_receipt}};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::os::windows::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

type Stamp = [u64; 3];

#[derive(Default, Serialize, Deserialize)]
struct Receipt {
    indices: BTreeSet<usize>,
    outputs: BTreeMap<u64, Stamp>,
}

fn inventory(cache: &Path) -> Option<BTreeMap<u64, Stamp>> {
    let mut outputs = BTreeMap::new();
    for entry in std::fs::read_dir(cache).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "dxil") { continue; }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        if stem.len() != 16 { continue; }
        let Ok(key) = u64::from_str_radix(stem, 16) else { continue };
        let m = entry.metadata().ok()?;
        if m.is_file() && m.file_attributes() & 0x400 == 0 {
            outputs.insert(key, [m.len(), m.creation_time(), m.last_write_time()]);
        }
    }
    Some(outputs)
}

impl Receipt {
    fn valid(&self, count: usize, files: &BTreeMap<u64, Stamp>) -> bool {
        !self.indices.is_empty() && !self.outputs.is_empty() && self.indices.last().is_some_and(|i| *i < count)
            && self.outputs.iter().all(|(key, stamp)| files.get(key) == Some(stamp))
    }
}

pub(super) struct Stages {
    cache: PathBuf,
    path: PathBuf,
    prior: Option<Receipt>,
    fresh: Mutex<Vec<(usize, Vec<(u64, u64)>)>>,
}

impl Stages {
    pub(super) fn open(native: u64, data: &[u8], count: usize, windows: u32, split: bool, graphics_split: bool) -> Option<Self> {
        let cache = crate::gpu::recomp::dxc::cache_dir();
        let key = digest(&(native, digest(&data), count, windows, split, graphics_split));
        let path = cache.join(format!("pack-stages-v1-{key:016x}.bin"));
        let prior = read_receipt::<Receipt>(&path).filter(|r| inventory(&cache).is_some_and(|files| r.valid(count, &files)));
        if let Some(r) = &prior {
            crate::xlog_info!("Shader preparation: reusing verified DXIL for {} graphics records in the pack", r.indices.len());
        }
        Some(Self { cache, path, prior, fresh: Mutex::default() })
    }

    pub(super) fn covered(&self, i: usize) -> bool {
        self.prior.as_ref().is_some_and(|r| r.indices.contains(&i))
    }

    pub(super) fn record(&self, index: usize, outputs: Vec<(u64, u64)>) {
        if !outputs.is_empty() { self.fresh.lock().unwrap().push((index, outputs)); }
    }

    pub(super) fn save(self) {
        let newly = self.fresh.into_inner().unwrap();
        if newly.is_empty() { return; }
        let Some(files) = inventory(&self.cache) else { return };
        let mut receipt = self.prior.unwrap_or_default();
        if receipt.outputs.iter().any(|(k, s)| files.get(k) != Some(s)) { return; }
        for (index, stages) in newly {
            for (key, len) in stages {
                let Some(stamp) = files.get(&key).filter(|s| len != 0 && s[0] == len) else { return };
                receipt.outputs.insert(key, *stamp);
            }
            receipt.indices.insert(index);
        }
        write_receipt(&self.path, &receipt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphics_pack_proof_requires_every_output_and_valid_indices() {
        let receipt = Receipt { indices: BTreeSet::from([0, 3]), outputs: BTreeMap::from([(1, [30, 4, 5]), (9, [60, 7, 8])]) };
        let mut files = receipt.outputs.clone();
        files.insert(11, [40, 2, 3]);
        assert!(receipt.valid(4, &files));
        assert!(!receipt.valid(3, &files));
        files.get_mut(&9).unwrap()[2] += 1;
        assert!(!receipt.valid(4, &files));
        files.remove(&9);
        assert!(!receipt.valid(4, &files));
    }
}
