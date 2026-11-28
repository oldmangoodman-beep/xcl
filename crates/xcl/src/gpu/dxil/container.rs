//! The DXBC container around a DXIL program: feature flags (`SFI0`),
//! signatures (`ISG1`/`OSG1`), pipeline state validation info (`PSV0`), the
//! program (`DXIL`) and the container digest D3D12 checks.
//!
//! The digest is DXBC's retail hash: MD5 over everything after the digest
//! field, with a non-standard final block (bit count in the first and
//! `(bits >> 2) | 1` in the last dword). Computing it here means shipped
//! builds need neither `dxcompiler.dll` nor `dxil.dll`; the validator is a
//! test oracle only.

/// DXIL shader kinds (`DXIL::ShaderKind`, also PSV's stage numbering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Pixel = 0,
    Vertex = 1,
    Geometry = 2,
    Hull = 3,
    Domain = 4,
    Compute = 5,
    Mesh = 13,
    Amplification = 14,
}

impl Stage {
    pub fn profile_prefix(self) -> &'static str {
        match self {
            Stage::Pixel => "ps",
            Stage::Vertex => "vs",
            Stage::Geometry => "gs",
            Stage::Hull => "hs",
            Stage::Domain => "ds",
            Stage::Compute => "cs",
            Stage::Mesh => "ms",
            Stage::Amplification => "as",
        }
    }
}

/// PSV resource types (`PSVResourceType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsvResType {
    Sampler = 1,
    Cbv = 2,
    SrvTyped = 3,
    SrvRaw = 4,
    SrvStructured = 5,
    UavTyped = 6,
    UavRaw = 7,
    UavStructured = 8,
}

/// One bound resource as PSV0 lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsvResource {
    pub res_type: PsvResType,
    pub space: u32,
    pub lower: u32,
    pub upper: u32,
    /// `DXIL::ResourceKind` (11 raw buffer, 12 structured buffer, 2 texture 2D…).
    pub kind: u32,
    pub flags: u32,
}

/// Pipeline state validation info.
#[derive(Debug, Clone)]
pub struct Psv {
    pub stage: Stage,
    /// Wave lane count range (0 / `u32::MAX` when unconstrained).
    pub min_wave: u32,
    pub max_wave: u32,
    pub numthreads: [u32; 3],
    pub resources: Vec<PsvResource>,
    pub entry: String,
}

/// Retail DXBC container digest of `data` (the bytes after the digest field).
pub fn hash(data: &[u8]) -> [u8; 16] {
    let mut st = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let bits = (data.len() as u32).wrapping_mul(8);
    let left = data.len() % 64;
    let full = data.len() - left;
    for blk in data[..full].chunks_exact(64) {
        md5_block(&mut st, blk);
    }
    let rest = &data[full..];
    let mut blk = [0u8; 64];
    if left >= 56 {
        blk[..left].copy_from_slice(rest);
        blk[left] = 0x80;
        md5_block(&mut st, &blk);
        let mut last = [0u8; 64];
        last[..4].copy_from_slice(&bits.to_le_bytes());
        last[60..].copy_from_slice(&((bits >> 2) | 1).to_le_bytes());
        md5_block(&mut st, &last);
    } else {
        blk[..4].copy_from_slice(&bits.to_le_bytes());
        blk[4..4 + left].copy_from_slice(rest);
        blk[4 + left] = 0x80;
        blk[60..].copy_from_slice(&((bits >> 2) | 1).to_le_bytes());
        md5_block(&mut st, &blk);
    }
    let mut out = [0u8; 16];
    for (i, w) in st.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

fn md5_block(st: &mut [u32; 4], blk: &[u8]) {
    const S: [u32; 16] = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];
    let m: Vec<u32> = blk.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let [mut a, mut b, mut c, mut d] = *st;
    for i in 0..64 {
        let (f, g) = match i / 16 {
            0 => ((b & c) | (!b & d), i),
            1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            2 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let k = ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32;
        let f = f.wrapping_add(a).wrapping_add(k).wrapping_add(m[g]);
        a = d;
        d = c;
        c = b;
        b = b.wrapping_add(f.rotate_left(S[(i / 16) * 4 + i % 4]));
    }
    st[0] = st[0].wrapping_add(a);
    st[1] = st[1].wrapping_add(b);
    st[2] = st[2].wrapping_add(c);
    st[3] = st[3].wrapping_add(d);
}

fn fourcc(s: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*s)
}

/// The 24-byte `DXIL` part header: program header + bitcode header.
fn program_header(stage: Stage, sm: (u32, u32), dxil_version: (u32, u32), bitcode_len: usize) -> [u8; 24] {
    assert!(bitcode_len % 4 == 0);
    let words = [
        ((stage as u32) << 16) | (sm.0 << 4) | sm.1,
        ((24 + bitcode_len) / 4) as u32,
        fourcc(b"DXIL"),
        (dxil_version.0 << 8) | dxil_version.1,
        16u32,
        bitcode_len as u32,
    ];
    let mut hdr = [0u8; 24];
    for (i, v) in words.into_iter().enumerate() {
        hdr[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    hdr
}

/// PSV0 with `PSVRuntimeInfo3` (validator 1.8+).
fn psv_part(psv: &Psv) -> Vec<u8> {
    let str_raw = psv.entry.len() + 2;
    let str_len = str_raw.next_multiple_of(4);
    let res_len = if psv.resources.is_empty() { 0 } else { 4 + 24 * psv.resources.len() };
    let exact_len = 68 + res_len + str_len;
    let mut p = Vec::with_capacity(exact_len);
    let w = |p: &mut Vec<u8>, v: u32| p.extend_from_slice(&v.to_le_bytes());
    w(&mut p, 52);
    p.extend_from_slice(&[0; 16]); // stage-specific union (none for compute)
    w(&mut p, psv.min_wave);
    w(&mut p, psv.max_wave);
    p.extend_from_slice(&[psv.stage as u8, 0, 0, 0]); // stage, uses view id, union
    p.extend_from_slice(&[0; 4]); // signature element counts / input vectors
    p.extend_from_slice(&[0; 4]); // output vectors per stream
    for n in psv.numthreads {
        w(&mut p, n);
    }
    w(&mut p, 1); // entry function name: offset 1 in the string table
    w(&mut p, psv.resources.len() as u32);
    if !psv.resources.is_empty() {
        w(&mut p, 24);
        for r in &psv.resources {
            for v in [r.res_type as u32, r.space, r.lower, r.upper, r.kind, r.flags] {
                w(&mut p, v);
            }
        }
    }
    // String table: "" then the entry name, padded to 4 bytes.
    w(&mut p, str_len as u32);
    p.push(0);
    p.extend_from_slice(psv.entry.as_bytes());
    p.push(0);
    p.extend_from_slice(&[0u8; 3][..str_len - str_raw]);
    w(&mut p, 0); // semantic index table
    debug_assert_eq!(p.len(), exact_len);
    p
}

/// An empty input / output signature (`0u32`, `8u32` LE).
const EMPTY_SIGNATURE: [u8; 8] = [0, 0, 0, 0, 8, 0, 0, 0];

/// Assembles and signs a DXBC container directly from borrowed prefix parts and `bitcode`
/// (single bitcode copy, exact allocation).
fn assemble_container(prefix_parts: &[([u8; 4], &[u8])], stage: Stage, sm: (u32, u32), bitcode: &[u8]) -> Vec<u8> {
    let dxil_version = (1, sm.1.min(8));
    let dxil_hdr = program_header(stage, sm, dxil_version, bitcode.len());
    let dxil_len = dxil_hdr.len() + bitcode.len();
    let part_count = prefix_parts.len() + 1;
    let header = 32 + 4 * part_count;
    let total = header + prefix_parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>() + 8 + dxil_len;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(part_count as u32).to_le_bytes());
    let mut off = header;
    for (_, d) in prefix_parts {
        out.extend_from_slice(&(off as u32).to_le_bytes());
        off += 8 + d.len();
    }
    out.extend_from_slice(&(off as u32).to_le_bytes());
    for (cc, d) in prefix_parts {
        out.extend_from_slice(cc);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d);
    }
    out.extend_from_slice(b"DXIL");
    out.extend_from_slice(&(dxil_len as u32).to_le_bytes());
    out.extend_from_slice(&dxil_hdr);
    out.extend_from_slice(bitcode);
    debug_assert_eq!(out.len(), total);
    let digest = hash(&out[20..]);
    out[4..20].copy_from_slice(&digest);
    out
}

/// Builds the signed container for a DXIL program.
pub fn container(stage: Stage, sm: (u32, u32), bitcode: &[u8], features: u64, psv: &Psv) -> Vec<u8> {
    let sfi0 = features.to_le_bytes();
    let psv_bytes = psv_part(psv);
    let prefix = [
        (*b"SFI0", sfi0.as_slice()),
        (*b"ISG1", EMPTY_SIGNATURE.as_slice()),
        (*b"OSG1", EMPTY_SIGNATURE.as_slice()),
        (*b"PSV0", psv_bytes.as_slice()),
    ];
    assemble_container(&prefix, stage, sm, bitcode)
}

// ------------------------------------------------- graphics signatures

/// One row of an `ISG1`/`OSG1`/`PSG1` signature (`DxilProgramSignatureElement`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SigPartElement {
    pub stream: u32,
    pub name: String,
    pub index: u32,
    /// `D3D_NAME` system value.
    pub system_value: u32,
    /// `D3D_REGISTER_COMPONENT_TYPE` (1 uint, 2 sint, 3 float).
    pub comp_type: u32,
    /// Register (row), `u32::MAX` for unallocated system values (depth, coverage…).
    pub register: u32,
    pub mask: u8,
    /// Inputs: components read; outputs: components never written.
    pub rw_mask: u8,
    pub min_precision: u32,
}

/// A signature part. Elements are written in the given order; each name is
/// stored once, in first-use order of `name_order` and then of the elements
/// (DXC writes names in declaration order while sorting elements by register).
pub fn signature_part(elems: &[SigPartElement], name_order: &[&str]) -> Vec<u8> {
    let mut strings: Vec<u8> = Vec::new();
    let mut offs: Vec<(&str, u32)> = Vec::with_capacity(name_order.len() + elems.len());
    let base = 8 + 32 * elems.len() as u32;
    for n in name_order.iter().copied().chain(elems.iter().map(|e| e.name.as_str())) {
        if !offs.iter().any(|&(s, _)| s == n) {
            offs.push((n, base + strings.len() as u32));
            strings.extend_from_slice(n.as_bytes());
            strings.push(0);
        }
    }
    while strings.len() % 4 != 0 {
        strings.push(0);
    }
    let exact_len = 8 + 32 * elems.len() + strings.len();
    let mut p = Vec::with_capacity(exact_len);
    let w = |p: &mut Vec<u8>, v: u32| p.extend_from_slice(&v.to_le_bytes());
    w(&mut p, elems.len() as u32);
    w(&mut p, 8);
    for e in elems {
        let off = offs.iter().find(|&&(s, _)| s == e.name).unwrap().1;
        for v in [e.stream, off, e.index, e.system_value, e.comp_type, e.register] {
            w(&mut p, v);
        }
        p.extend_from_slice(&[e.mask, e.rw_mask, 0, 0]);
        w(&mut p, e.min_precision);
    }
    p.extend_from_slice(&strings);
    debug_assert_eq!(p.len(), exact_len);
    p
}

/// One signature element as PSV0 lists it (`PSVSignatureElement0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsvSigElement {
    /// Empty for system values (PSV stores no name for them).
    pub name: String,
    pub indices: Vec<u32>,
    pub rows: u8,
    pub start_row: u8,
    pub cols: u8,
    pub start_col: u8,
    pub allocated: bool,
    /// `DXIL::SemanticKind`.
    pub kind: u8,
    /// `PSVComponentType` (1 uint32, 2 sint32, 3 float32, 4 uint16, 5 sint16, 6 float16).
    pub comp_type: u8,
    /// `DXIL::InterpolationMode`.
    pub interp: u8,
    pub dynamic_mask: u8,
    pub stream: u8,
}

/// PSV0 for any stage: `PSVRuntimeInfo3` plus signature elements and the
/// input → output dependency tables.
#[derive(Debug, Clone)]
pub struct PsvFull {
    pub stage: Stage,
    /// The 16-byte stage union of `PSVRuntimeInfo0` (VS/PS/MS/... info).
    pub stage_info: [u8; 16],
    pub min_wave: u32,
    pub max_wave: u32,
    pub uses_view_id: bool,
    /// The 2-byte union of `PSVRuntimeInfo1` (GS max vertices / MS prim vectors + topology).
    pub info1: [u8; 2],
    pub numthreads: [u32; 3],
    pub resources: Vec<PsvResource>,
    pub entry: String,
    pub inputs: Vec<PsvSigElement>,
    pub outputs: Vec<PsvSigElement>,
    pub prims: Vec<PsvSigElement>,
    pub input_vectors: u8,
    pub output_vectors: [u8; 4],
    /// Tables after the signature elements, in order: view-ID output masks
    /// (if `uses_view_id`), input → output dependency tables, and the
    /// patch-constant / primitive tables (dwords).
    pub tables: Vec<Vec<u32>>,
}

/// Writes PSV0 for [`PsvFull`].
pub fn psv_part_full(psv: &PsvFull) -> Vec<u8> {
    let total_elems = psv.inputs.len() + psv.outputs.len() + psv.prims.len();
    let elems = || psv.inputs.iter().chain(&psv.outputs).chain(&psv.prims);
    // String table: "" then non-system names in element order (not
    // deduplicated, as DXC writes it), then the entry name.
    let str_raw = 1 + elems().map(|e| if e.name.is_empty() { 0 } else { e.name.len() + 1 }).sum::<usize>() + psv.entry.len() + 1;
    let mut strings = Vec::with_capacity(str_raw.next_multiple_of(4));
    strings.push(0);
    let mut name_off = Vec::with_capacity(total_elems);
    for e in elems() {
        if e.name.is_empty() {
            name_off.push(0);
        } else {
            name_off.push(strings.len() as u32);
            strings.extend_from_slice(e.name.as_bytes());
            strings.push(0);
        }
    }
    let entry_off = strings.len() as u32;
    strings.extend_from_slice(psv.entry.as_bytes());
    strings.push(0);
    while strings.len() % 4 != 0 {
        strings.push(0);
    }
    // Semantic index table: each element's index list, reusing an existing run.
    let mut table: Vec<u32> = Vec::new();
    let mut idx_off = Vec::with_capacity(total_elems);
    for e in elems() {
        let n = e.indices.len();
        let found = if n == 0 || table.len() < n { None } else { (0..=table.len() - n).find(|&i| table[i..i + n] == e.indices[..]) };
        idx_off.push(match found {
            Some(i) => i as u32,
            None => {
                table.extend_from_slice(&e.indices);
                (table.len() - n) as u32
            }
        });
    }
    let res_len = if psv.resources.is_empty() { 0 } else { 4 + 24 * psv.resources.len() };
    let sig_len = if total_elems == 0 { 0 } else { 4 + 16 * total_elems };
    let tbl_len = 4 * psv.tables.iter().map(Vec::len).sum::<usize>();
    let p_len = 68 + res_len + strings.len() + 4 * table.len() + sig_len + tbl_len;
    let mut p = Vec::with_capacity(p_len);
    let w = |p: &mut Vec<u8>, v: u32| p.extend_from_slice(&v.to_le_bytes());
    w(&mut p, 52);
    p.extend_from_slice(&psv.stage_info);
    w(&mut p, psv.min_wave);
    w(&mut p, psv.max_wave);
    p.extend_from_slice(&[psv.stage as u8, psv.uses_view_id as u8, psv.info1[0], psv.info1[1]]);
    p.extend_from_slice(&[psv.inputs.len() as u8, psv.outputs.len() as u8, psv.prims.len() as u8, psv.input_vectors]);
    p.extend_from_slice(&psv.output_vectors);
    for n in psv.numthreads {
        w(&mut p, n);
    }
    w(&mut p, entry_off);
    w(&mut p, psv.resources.len() as u32);
    if !psv.resources.is_empty() {
        w(&mut p, 24);
        for r in &psv.resources {
            for v in [r.res_type as u32, r.space, r.lower, r.upper, r.kind, r.flags] {
                w(&mut p, v);
            }
        }
    }
    w(&mut p, strings.len() as u32);
    p.extend_from_slice(&strings);
    w(&mut p, table.len() as u32);
    for v in &table {
        w(&mut p, *v);
    }
    if total_elems > 0 {
        w(&mut p, 16);
        for (i, e) in elems().enumerate() {
            w(&mut p, name_off[i]);
            w(&mut p, idx_off[i]);
            let cs = (e.cols & 0xF) | ((e.start_col & 3) << 4) | ((e.allocated as u8) << 6);
            p.extend_from_slice(&[e.rows, e.start_row, cs, e.kind, e.comp_type, e.interp, (e.dynamic_mask & 0xF) | ((e.stream & 3) << 4), 0]);
        }
    }
    for t in &psv.tables {
        for v in t {
            w(&mut p, *v);
        }
    }
    debug_assert_eq!(p.len(), p_len);
    p
}

/// Builds the signed container with explicit signature parts (`psg` for
/// mesh primitive outputs) and a prebuilt PSV0.
#[allow(clippy::too_many_arguments)]
pub fn container_with_signatures(stage: Stage, sm: (u32, u32), bitcode: &[u8], features: u64, isg: Vec<u8>, osg: Vec<u8>, psg: Option<Vec<u8>>, psv: Vec<u8>) -> Vec<u8> {
    let sfi0 = features.to_le_bytes();
    match psg.as_deref() {
        Some(p) => assemble_container(
            &[(*b"SFI0", sfi0.as_slice()), (*b"ISG1", isg.as_slice()), (*b"OSG1", osg.as_slice()), (*b"PSG1", p), (*b"PSV0", psv.as_slice())],
            stage, sm, bitcode,
        ),
        None => assemble_container(
            &[(*b"SFI0", sfi0.as_slice()), (*b"ISG1", isg.as_slice()), (*b"OSG1", osg.as_slice()), (*b"PSV0", psv.as_slice())],
            stage, sm, bitcode,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DXC 1.9 output for a 64-thread compute shader storing to two root UAVs
    /// (see `tests/data/dxil/*.hlsl`).
    const DXC_CS: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_cs_raw_uav.dxil");

    #[test]
    fn digest_matches_dxc_signed_container() {
        assert_eq!(hash(&DXC_CS[20..]), DXC_CS[4..20]);
    }

    #[test]
    fn digest_covers_both_tail_layouts() {
        // Lengths with < 56 and >= 56 trailing bytes take different final blocks;
        // both must differ from each other and be stable.
        let a = hash(&[7u8; 64 + 10]);
        let b = hash(&[7u8; 64 + 60]);
        assert_ne!(a, b);
        assert_eq!(a, hash(&[7u8; 64 + 10]));
    }

    #[test]
    fn psv_matches_dxc_layout_for_compute_with_two_raw_uavs() {
        let psv = Psv {
            stage: Stage::Compute,
            min_wave: 64,
            max_wave: 64,
            numthreads: [64, 1, 1],
            resources: (0..2)
                .map(|i| PsvResource { res_type: PsvResType::UavRaw, space: 0, lower: i, upper: i, kind: 11, flags: 0 })
                .collect(),
            entry: "main".into(),
        };
        let ours = psv_part(&psv);
        let theirs = part(DXC_CS, b"PSV0");
        assert_eq!(ours, theirs);
    }

    #[test]
    fn parts_and_containers_are_exact_sized_and_match_reference_layout() {
        let ref_dxbc = |parts: &[([u8; 4], &[u8])]| -> Vec<u8> {
            let hdr = 32 + 4 * parts.len();
            let tot = hdr + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
            let mut o = Vec::with_capacity(tot);
            o.extend_from_slice(b"DXBC");
            o.extend_from_slice(&[0; 16]);
            o.extend_from_slice(&1u16.to_le_bytes());
            o.extend_from_slice(&0u16.to_le_bytes());
            o.extend_from_slice(&(tot as u32).to_le_bytes());
            o.extend_from_slice(&(parts.len() as u32).to_le_bytes());
            let mut off = hdr;
            for (_, d) in parts { o.extend_from_slice(&(off as u32).to_le_bytes()); off += 8 + d.len(); }
            for (cc, d) in parts { o.extend_from_slice(cc); o.extend_from_slice(&(d.len() as u32).to_le_bytes()); o.extend_from_slice(d); }
            let d = hash(&o[20..]);
            o[4..20].copy_from_slice(&d);
            o
        };
        let ref_dxil = |stage: Stage, sm: (u32, u32), bc: &[u8]| -> Vec<u8> {
            let mut p = Vec::new();
            for v in [((stage as u32) << 16) | (sm.0 << 4) | sm.1, ((24 + bc.len()) / 4) as u32, fourcc(b"DXIL"), (1 << 8) | sm.1.min(8), 16, bc.len() as u32] {
                p.extend_from_slice(&v.to_le_bytes());
            }
            p.extend_from_slice(bc);
            p
        };
        let res = PsvResource { res_type: PsvResType::UavRaw, space: 1, lower: 2, upper: 4, kind: 11, flags: 3 };
        let bc = [0xDEu8, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4];

        // 1. psv_part + container across 0 & N resources and all entry padding remainders.
        for n_res in [0usize, 3] {
            for entry in ["", "a", "ab", "abc", "main"] {
                let psv = Psv { stage: Stage::Compute, min_wave: 32, max_wave: 64, numthreads: [8, 4, 1], resources: vec![res.clone(); n_res], entry: entry.into() };
                let p = psv_part(&psv);
                assert_eq!(p.capacity(), p.len(), "psv_part over-allocated for n_res={n_res} entry={entry:?}");
                let s0 = 0x1234_5678_9ABC_DEF0u64.to_le_bytes();
                let dxil = ref_dxil(Stage::Compute, (6, 9), &bc);
                let got = container(Stage::Compute, (6, 9), &bc, 0x1234_5678_9ABC_DEF0, &psv);
                let want = ref_dxbc(&[(*b"SFI0", &s0), (*b"ISG1", &EMPTY_SIGNATURE), (*b"OSG1", &EMPTY_SIGNATURE), (*b"PSV0", &p), (*b"DXIL", &dxil)]);
                assert_eq!(got.capacity(), got.len());
                assert_eq!(got, want);
                assert_eq!(part(&got, b"PSV0"), p.as_slice());
            }
        }

        // 2. signature_part with empty & repeated names + name_order reordering.
        let empty_sig = signature_part(&[], &[]);
        assert_eq!(empty_sig.as_slice(), &EMPTY_SIGNATURE);
        assert_eq!(empty_sig.capacity(), empty_sig.len());
        let mk_sig = |name: &str, idx: u32, reg: u32| SigPartElement {
            stream: 0, name: name.into(), index: idx, system_value: 0, comp_type: 3, register: reg, mask: 0xF, rw_mask: 0x3, min_precision: 0,
        };
        let elems = [mk_sig("TEXCOORD", 0, 0), mk_sig("SV_Position", 0, 1), mk_sig("TEXCOORD", 1, 2)];
        let isg = signature_part(&elems, &["SV_Position", "TEXCOORD"]);
        assert_eq!(isg.capacity(), isg.len());
        let off0 = u32::from_le_bytes(isg[12..16].try_into().unwrap()) as usize;
        let off1 = u32::from_le_bytes(isg[44..48].try_into().unwrap()) as usize;
        let off2 = u32::from_le_bytes(isg[76..80].try_into().unwrap()) as usize;
        assert_eq!(off0, off2, "repeated semantic name must share one string offset");
        assert!(off1 < off0, "name_order must place SV_Position before TEXCOORD");

        // 3. psv_part_full (inputs/outputs/prims, empty & non-empty names, reused index runs, view_id, tables)
        //    + container_with_signatures with PSG1 = Some and None.
        let mk_psv_el = |name: &str, indices: &[u32], row: u8| PsvSigElement {
            name: name.into(), indices: indices.to_vec(), rows: 1, start_row: row, cols: 4, start_col: 0,
            allocated: true, kind: 0, comp_type: 3, interp: 2, dynamic_mask: 0, stream: 0,
        };
        let full = PsvFull {
            stage: Stage::Mesh, stage_info: [7; 16], min_wave: 32, max_wave: 32, uses_view_id: true,
            info1: [1, 2], numthreads: [32, 1, 1], resources: vec![res], entry: "mesh_main".into(),
            inputs: vec![mk_psv_el("", &[0], 0), mk_psv_el("TEXCOORD", &[0, 1], 1)],
            outputs: vec![mk_psv_el("COLOR", &[1], 0)],
            prims: vec![mk_psv_el("PRIM", &[0, 1], 0)],
            input_vectors: 2, output_vectors: [1, 0, 0, 0], tables: vec![vec![0x11, 0x22], vec![0x33]],
        };
        let pfull = psv_part_full(&full);
        // Captured from the pre-refactor serializer. Keep this oracle independent of the
        // container/reference helpers above so a same-length PSV field reordering is visible.
        let golden = concat!(
            "340000000707070707070707070707070707070720000000200000000d010102",
            "0201010201000000200000000100000001000000150000000100000018000000",
            "070000000100000002000000040000000b000000030000002000000000544558",
            "434f4f524400434f4c4f52005052494d006d6573685f6d61696e000003000000",
            "0000000000000000010000001000000000000000000000000100440003020000",
            "010000000100000001014400030200000a000000020000000100440003020000",
            "10000000010000000100440003020000110000002200000033000000",
        );
        let golden: Vec<u8> = golden.as_bytes().chunks_exact(2).map(|b| {
            u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap()
        }).collect();
        assert_eq!(pfull, golden, "full PSV bytes changed from the pre-refactor serializer");
        assert_eq!(pfull.capacity(), pfull.len());
        let pfull_empty = psv_part_full(&PsvFull {
            stage: Stage::Vertex, stage_info: [0; 16], min_wave: 0, max_wave: 0, uses_view_id: false,
            info1: [0; 2], numthreads: [0; 3], resources: vec![], entry: "vs".into(),
            inputs: vec![], outputs: vec![], prims: vec![], input_vectors: 0, output_vectors: [0; 4], tables: vec![],
        });
        assert_eq!(pfull_empty.capacity(), pfull_empty.len());

        let osg = signature_part(&[mk_sig("COLOR", 0, 0)], &[]);
        let psg = signature_part(&[mk_sig("PRIM", 0, 0)], &[]);
        let sfi = 0x55u64.to_le_bytes();
        let dxil = ref_dxil(Stage::Mesh, (6, 6), &bc);
        let with_psg = container_with_signatures(Stage::Mesh, (6, 6), &bc, 0x55, isg.clone(), osg.clone(), Some(psg.clone()), pfull.clone());
        assert_eq!(with_psg.capacity(), with_psg.len());
        assert_eq!(with_psg, ref_dxbc(&[(*b"SFI0", &sfi), (*b"ISG1", &isg), (*b"OSG1", &osg), (*b"PSG1", &psg), (*b"PSV0", &pfull), (*b"DXIL", &dxil)]));
        let without_psg = container_with_signatures(Stage::Mesh, (6, 6), &bc, 0x55, isg.clone(), osg.clone(), None, pfull.clone());
        assert_eq!(without_psg.capacity(), without_psg.len());
        assert_eq!(without_psg, ref_dxbc(&[(*b"SFI0", &sfi), (*b"ISG1", &isg), (*b"OSG1", &osg), (*b"PSV0", &pfull), (*b"DXIL", &dxil)]));
    }

    #[test]
    #[should_panic]
    fn program_header_rejects_unaligned_bitcode() {
        let _ = program_header(Stage::Compute, (6, 6), (1, 6), 6);
    }

    pub(crate) fn part<'a>(c: &'a [u8], cc: &[u8; 4]) -> &'a [u8] {
        let n = u32::from_le_bytes(c[28..32].try_into().unwrap()) as usize;
        for i in 0..n {
            let off = u32::from_le_bytes(c[32 + 4 * i..36 + 4 * i].try_into().unwrap()) as usize;
            if &c[off..off + 4] == cc {
                let len = u32::from_le_bytes(c[off + 4..off + 8].try_into().unwrap()) as usize;
                return &c[off + 8..off + 8 + len];
            }
        }
        panic!("no {cc:?} part")
    }
}
