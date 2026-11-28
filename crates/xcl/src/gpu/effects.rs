//! What a console compute program can change outside itself, read from its code alone.
//!
//! A dispatch XCL does not run (its pipeline is still being built) leaves undone what the
//! program would have written. For memory that is dangerous: a culling pass writes the
//! counts and the arguments an `ExecuteIndirect` reads next, and without them the GPU is
//! given unbounded work. A program that writes nothing but images leaves a stale picture
//! and nothing else, so its dispatches can go on without it while the driver compiles it
//! (`hle::d3d12x::pso_async`).

use super::{rdna2, recomp, xbox};

/// The writes a program can make.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Effects {
    /// memory other programs, the CPU or indirect arguments may read: buffer / flat / global /
    /// scratch stores and atomics, and the global data share (append / consume counters, GWS)
    pub memory: bool,
    /// image stores and image atomics
    pub images: bool,
    /// instructions in the program
    pub instructions: usize,
}

fn update_writes(e: &mut Effects, d: &rdna2::Decoded) {
    let n = d.inst.name;
    let writes = n.contains("_STORE") || n.contains("_ATOMIC");
    match d.enc.name {
        "ENC_MIMG" => e.images |= writes,
        "ENC_MUBUF" | "ENC_MTBUF" | "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" | "ENC_SMEM" => e.memory |= writes,
        // LDS is the group's own and gone after it; the global data share is not
        "ENC_DS" => {
            let gds = rdna2::field_value(d.enc, &d.words, "GDS").is_some_and(|v| v != 0);
            e.memory |= gds || n == "DS_APPEND" || n == "DS_CONSUME" || n == "DS_ORDERED_COUNT" || n.starts_with("DS_GWS_");
        }
        _ => {}
    }
}

fn analyze(code: &[u8], stop_on_memory: bool) -> Option<Effects> {
    let cap = if stop_on_memory { (code.len() / 6).min(256) } else { code.len() / 6 };
    let mut pcs = Vec::with_capacity(cap);
    let mut targets = Vec::new();
    let mut off = 0;
    // the whole program: code can follow an early terminator (a branch to it, or past it); what
    // does not decode (or S_CODE_END padding) after one is taken for trailing data, unless a
    // decoded branch still targets past it or the tail falls through into it
    let mut ended = false;
    let mut last_ok_end = false;
    let mut e = Effects::default();
    while off + 4 <= code.len() {
        let Ok(d) = rdna2::decode(&code[off..], off as u64) else {
            if ended {
                break;
            }
            return None;
        };
        if ended && d.inst.name == "S_CODE_END" {
            break;
        }
        if d.inst.name == "S_SETPC_B64" {
            return analyze_with_far_jumps(code, stop_on_memory);
        }
        if d.inst.flags & (rdna2::F_BRANCH | rdna2::F_INDIRECT) != 0 {
            targets.push(rdna2::branch_target(&d)?);
        }
        off += d.len;
        let term = d.inst.flags & rdna2::F_TERMINATOR != 0;
        ended |= term;
        last_ok_end = term || d.inst.name == "S_BRANCH";
        e.instructions += 1;
        update_writes(&mut e, &d);
        if stop_on_memory && e.memory {
            return None;
        }
        pcs.push(d.pc);
    }
    if !ended || !last_ok_end || !targets.iter().all(|t| pcs.binary_search(t).is_ok()) {
        return None;
    }
    Some(e)
}

fn analyze_with_far_jumps(code: &[u8], stop_on_memory: bool) -> Option<Effects> {
    // Keep early memory-writer rejection cheap even when an initial SETPC selects this slower
    // path; grow only if analysis actually has to retain the rest of a large shader.
    let cap = if stop_on_memory { (code.len() / 6).min(256) } else { code.len() / 6 };
    let mut insts = Vec::with_capacity(cap);
    let mut off = 0;
    let mut ended = false;
    let mut e = Effects::default();
    while off + 4 <= code.len() {
        let Ok(d) = rdna2::decode(&code[off..], off as u64) else {
            if ended {
                break;
            }
            return None;
        };
        if ended && d.inst.name == "S_CODE_END" {
            break;
        }
        off += d.len;
        ended |= d.inst.flags & rdna2::F_TERMINATOR != 0;
        e.instructions += 1;
        update_writes(&mut e, &d);
        if stop_on_memory && e.memory {
            return None;
        }
        insts.push(d);
    }
    if !ended {
        return None;
    }
    rdna2::resolve_far_jumps(&mut insts);
    let last = insts.last()?;
    if last.inst.flags & rdna2::F_TERMINATOR == 0 && last.inst.name != "S_BRANCH" && last.far_target.is_none() {
        return None;
    }
    for d in &insts {
        if d.inst.flags & (rdna2::F_BRANCH | rdna2::F_INDIRECT) != 0
            && rdna2::branch_target(d).is_none_or(|t| insts.binary_search_by_key(&t, |x| x.pc).is_err())
        {
            return None;
        }
    }
    Some(e)
}

/// The writes of the RDNA2 program `code` (one program, as `xbox::Fssm::programs` cuts it).
/// None when the program cannot be decoded through a terminator, falls through past decoded
/// code, contains an unresolved indirect transfer, or branches outside the decoded instructions.
pub fn of(code: &[u8]) -> Option<Effects> {
    analyze(code, false)
}

fn is_big_image_only_code(code: &[u8]) -> bool {
    code.len() >= BIG_CODE && analyze(code, true).is_some()
}

/// Console code a compute program needs to be `big`: 16 KiB, about 4 000 instructions. The
/// programs a retail title's world start waits on longest are 73 – 118 KiB each (22 – 172 s
/// apiece in an RDNA2 driver); what it runs every frame from its first one is far smaller.
pub const BIG_CODE: usize = 16 << 10;

/// Whether the compute shader `blob` is a big program whose only writes are to images: one a
/// title can do without for the frames its pipeline takes to build. False for anything that
/// cannot be read.
pub fn big_and_image_only(blob: &[u8]) -> bool {
    if let Some(f) = recomp::pso::fspx(blob) {
        return is_big_image_only_code(f.code);
    }
    let Some(sh) = xbox::parse(blob) else { return false };
    let Some(f) = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 6)) else { return false };
    if f.error.is_some() {
        return false;
    }
    let Some(p) = f.programs.first() else { return false };
    let Some(end) = p.offset.checked_add(p.len) else { return false };
    f.data.get(p.offset..end).is_some_and(is_big_image_only_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asm(insts: &[(&str, &str, &[(&str, u64)])]) -> Vec<u8> {
        insts.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap_or_else(|e| panic!("{n}: {e}"))).collect()
    }

    const END: (&str, &str, &[(&str, u64)]) = ("S_ENDPGM", "ENC_SOPP", &[]);
    const MOV: (&str, &str, &[(&str, u64)]) = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 129)]);
    const IMAGE_STORE: (&str, &str, &[(&str, u64)]) = ("IMAGE_STORE", "ENC_MIMG", &[("DMASK", 15), ("VADDR", 0), ("VDATA", 4), ("SRSRC", 2)]);
    const IMAGE_LOAD: (&str, &str, &[(&str, u64)]) = ("IMAGE_LOAD", "ENC_MIMG", &[("DMASK", 15), ("VADDR", 0), ("VDATA", 4), ("SRSRC", 2)]);

    #[test]
    fn images_and_memory_are_told_apart() {
        // loads of either kind write nothing
        let load: (&str, &str, &[(&str, u64)]) = ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]);
        assert_eq!(of(&asm(&[MOV, IMAGE_LOAD, load, END])), Some(Effects { memory: false, images: false, instructions: 4 }));
        assert_eq!(of(&asm(&[MOV, IMAGE_STORE, END])), Some(Effects { memory: false, images: true, instructions: 3 }));
        let image_atomic: (&str, &str, &[(&str, u64)]) = ("IMAGE_ATOMIC_ADD", "ENC_MIMG", &[("DMASK", 1), ("VADDR", 0), ("VDATA", 4), ("SRSRC", 2)]);
        assert_eq!(of(&asm(&[image_atomic, END])).map(|e| (e.memory, e.images)), Some((false, true)));
        // every way into memory another pass or an indirect call reads
        let writers: [(&str, &str, &[(&str, u64)]); 7] = [
            ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]),
            ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]),
            ("TBUFFER_STORE_FORMAT_XYZW", "ENC_MTBUF", &[("VDATA", 1), ("SRSRC", 2)]),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 0), ("DATA", 2)]),
            ("FLAT_ATOMIC_ADD", "ENC_FLAT", &[("ADDR", 0), ("DATA", 2)]),
            ("SCRATCH_STORE_DWORD", "ENC_FLAT_SCRATCH", &[("ADDR", 0), ("DATA", 2)]),
            ("DS_APPEND", "ENC_DS", &[("VDST", 1)]),
        ];
        for w in writers {
            assert_eq!(of(&asm(&[IMAGE_STORE, w, END])).map(|e| (e.memory, e.images)), Some((true, true)), "{}", w.0);
        }
        // the group's own LDS is not memory anyone else reads
        let lds: (&str, &str, &[(&str, u64)]) = ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 0), ("DATA0", 1)]);
        assert_eq!(of(&asm(&[lds, IMAGE_STORE, END])).map(|e| (e.memory, e.images)), Some((false, true)));
    }

    #[test]
    fn code_after_an_early_end_counts() {
        // a branch can jump past an early s_endpgm: a store after it is the program's (were it
        // constant data instead, taking it for a store only makes binds wait, as before)
        let store: (&str, &str, &[(&str, u64)]) = ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]);
        assert_eq!(of(&asm(&[IMAGE_STORE, END, store, END])), Some(Effects { memory: true, images: true, instructions: 4 }));
        // what does not decode after the end is trailing data
        let mut code = asm(&[IMAGE_STORE, END]);
        code.extend_from_slice(&[0xff; 8]);
        if rdna2::decode(&[0xff; 8], 0).is_err() {
            assert_eq!(of(&code), Some(Effects { memory: false, images: true, instructions: 2 }));
        }
    }

    #[test]
    fn the_global_data_share_is_memory() {
        // the same LDS add, once with GDS set: other groups and the CPU can see that one
        let lds: (&str, &str, &[(&str, u64)]) = ("DS_ADD_U32", "ENC_DS", &[("ADDR", 0), ("DATA0", 1)]);
        let gds: (&str, &str, &[(&str, u64)]) = ("DS_ADD_U32", "ENC_DS", &[("ADDR", 0), ("DATA0", 1), ("GDS", 1)]);
        assert_eq!(of(&asm(&[lds, IMAGE_STORE, END])).map(|e| e.memory), Some(false));
        assert_eq!(of(&asm(&[gds, IMAGE_STORE, END])).map(|e| e.memory), Some(true));
    }

    #[test]
    fn code_that_does_not_decode_says_nothing() {
        let mut code = asm(&[IMAGE_STORE]);
        code.extend_from_slice(&[0xff; 8]);
        code.extend(asm(&[END]));
        // (only where the decoder really refuses the dwords: otherwise the test states nothing)
        if rdna2::decode(&[0xff; 8], 0).is_err() {
            assert_eq!(of(&code), None);
        }
    }

    #[test]
    fn branch_past_early_end_gap_and_fallthrough_say_nothing() {
        assert!(rdna2::decode(&[0xff; 8], 0).is_err());
        let store: (&str, &str, &[(&str, u64)]) = ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]);
        // 0x00: s_cbranch_scc1 +5 (-> 0x18), 0x04: image_store (8B), 0x0c: s_endpgm,
        // 0x10: undecodable gap (8B), 0x18: buffer_store_dword, 0x20: s_endpgm
        let mut jumped = asm(&[("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 5)]), IMAGE_STORE, END]);
        jumped.extend_from_slice(&[0xff; 8]);
        jumped.extend(asm(&[store, END]));
        assert_eq!(of(&jumped), None);

        // 0x00: s_cbranch_scc1 +1 (-> 0x08), 0x04: s_endpgm, 0x08: image_store falls into gap
        let mut fallthrough = asm(&[("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]), END, IMAGE_STORE]);
        fallthrough.extend_from_slice(&[0xff; 8]);
        fallthrough.extend(asm(&[store, END]));
        assert_eq!(of(&fallthrough), None);

        // missing terminator fails closed; non-default F_TERMINATOR + trailing data still passes
        assert_eq!(of(&asm(&[IMAGE_STORE])), None);
        let mut saved = asm(&[IMAGE_STORE, ("S_ENDPGM_SAVED", "ENC_SOPP", &[])]);
        saved.extend_from_slice(&[0xff; 8]);
        assert_eq!(of(&saved), Some(Effects { memory: false, images: true, instructions: 2 }));
    }

    #[test]
    fn unresolved_indirect_and_oob_branches_fail_closed_while_far_jump_passes() {
        let setpc: (&str, &str, &[(&str, u64)]) = ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 0)]);
        let swappc: (&str, &str, &[(&str, u64)]) = ("S_SWAPPC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)]);
        let oob_branch: (&str, &str, &[(&str, u64)]) = ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 8)]);
        let oob_call: (&str, &str, &[(&str, u64)]) = ("S_CALL_B64", "ENC_SOPK", &[("SDST", 0), ("SIMM16", 8)]);
        for bad in [setpc, swappc, oob_branch, oob_call] {
            assert_eq!(of(&asm(&[IMAGE_STORE, bad, END])), None, "{}", bad.0);
        }

        // resolved far jump (base = 0x04, +16 -> 0x14) past early END to IMAGE_STORE; trailing S_CODE_END padding
        let far_ok = asm(&[
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 0)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 128 + 16)]),
            ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 128)]),
            setpc,
            END,
            IMAGE_STORE,
            END,
            ("S_CODE_END", "ENC_SOPP", &[]),
        ]);
        assert_eq!(of(&far_ok), Some(Effects { memory: false, images: true, instructions: 7 }));

        // resolved far jump whose target lands past the decoded slice still fails closed
        let far_oob = asm(&[
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 0)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 128 + 64)]),
            ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 128)]),
            setpc,
            END,
        ]);
        assert_eq!(of(&far_oob), None);
    }

    #[test]
    fn errored_fssm_is_rejected_by_big_and_image_only() {
        let mut code = Vec::with_capacity(BIG_CODE + 16);
        let mov = asm(&[MOV]);
        while code.len() < BIG_CODE {
            code.extend_from_slice(&mov);
        }
        code.extend(asm(&[IMAGE_STORE, END]));
        let valid = xbox::tests::sample(5, &[xbox::tests::fssm(&code, &[(6, 0)])]);
        assert!(big_and_image_only(&valid));

        // bump XBCH rsize so parse_xbch records f.error while still populating f.programs
        let mut errored = valid.clone();
        let at = errored.windows(4).position(|w| w == b"XBCH").expect("XBCH header");
        let rsize = u32::from_le_bytes(errored[at + 16..at + 20].try_into().unwrap());
        errored[at + 16..at + 20].copy_from_slice(&(rsize + 64).to_le_bytes());
        let sh = xbox::parse(&errored).expect("container still parses");
        assert!(sh.fssm[0].error.is_some() && !sh.fssm[0].programs.is_empty());
        assert!(!big_and_image_only(&errored));
    }

    fn of_reference(code: &[u8]) -> Option<Effects> {
        let (mut insts, mut off, mut ended) = (Vec::new(), 0, false);
        while off + 4 <= code.len() {
            let Ok(d) = rdna2::decode(&code[off..], off as u64) else { if ended { break; } return None; };
            if ended && d.inst.name == "S_CODE_END" { break; }
            off += d.len; ended |= d.inst.flags & rdna2::F_TERMINATOR != 0; insts.push(d);
        }
        if !ended { return None; }
        rdna2::resolve_far_jumps(&mut insts);
        let last = insts.last()?;
        if last.inst.flags & rdna2::F_TERMINATOR == 0 && last.inst.name != "S_BRANCH" && last.far_target.is_none() { return None; }
        let pcs: std::collections::HashSet<u64> = insts.iter().map(|d| d.pc).collect();
        let mut e = Effects::default();
        for d in &insts {
            if d.inst.flags & (rdna2::F_BRANCH | rdna2::F_INDIRECT) != 0 && rdna2::branch_target(d).is_none_or(|t| !pcs.contains(&t)) { return None; }
            e.instructions += 1; update_writes(&mut e, d);
        }
        Some(e)
    }

    fn fspx_blob(code: &[u8]) -> Vec<u8> {
        let mut head = [0u8; 0x20];
        for o in [0x0C, 0x0E, 0x10] { head[o..o + 2].copy_from_slice(&1u16.to_le_bytes()); }
        let mut accc = [b"FSPX".as_slice(), &[0; 4], &2u32.to_le_bytes(), &[0; 4]].concat();
        for w in [0u32, 0x30, 0x20, 0, 6, 0x50, (0x24 + code.len()) as u32, 0] { accc.extend_from_slice(&w.to_le_bytes()); }
        accc.extend_from_slice(&head); accc.extend_from_slice(&[0u8; 0x24]); accc.extend_from_slice(code);
        let mut out = [b"DXBC".as_slice(), &[0; 16]].concat();
        for w in [1u32, (44 + accc.len()) as u32, 1, 36] { out.extend_from_slice(&w.to_le_bytes()); }
        out.extend_from_slice(b"ACCC"); out.extend_from_slice(&(accc.len() as u32).to_le_bytes()); out.extend_from_slice(&accc);
        out
    }

    #[test]
    fn streaming_matches_reference_of_across_edge_cases() {
        let store: (&str, &str, &[(&str, u64)]) = ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]);
        let setpc: (&str, &str, &[(&str, u64)]) = ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 0)]);
        let swappc: (&str, &str, &[(&str, u64)]) = ("S_SWAPPC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)]);
        let call_ok: (&str, &str, &[(&str, u64)]) = ("S_CALL_B64", "ENC_SOPK", &[("SDST", 0), ("SIMM16", 0)]);
        let code_end: (&str, &str, &[(&str, u64)]) = ("S_CODE_END", "ENC_SOPP", &[]);
        let oob_br: (&str, &str, &[(&str, u64)]) = ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 9)]);
        let getpc: (&str, &str, &[(&str, u64)]) = ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 0)]);
        let add16: (&str, &str, &[(&str, u64)]) = ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 128 + 16)]);
        let addc0: (&str, &str, &[(&str, u64)]) = ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 128)]);
        let mut cases = vec![
            Vec::new(),
            [asm(&[store]), vec![0xff; 8]].concat(),
            asm(&[store, oob_br, END]),
            asm(&[store, END, code_end]),
            asm(&[IMAGE_STORE, END, store, END]),
            asm(&[getpc, add16, addc0, setpc, END, IMAGE_STORE, END]),
            asm(&[IMAGE_STORE, END, getpc, ("S_SUB_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 128 + 16)]), ("S_SUBB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 128)]), setpc]),
            asm(&[getpc, ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 128 + 64)]), addc0, setpc, END]),
            asm(&[getpc, add16, addc0, setpc, END, store, END]),
            asm(&[store, getpc, add16, addc0, setpc, END, IMAGE_STORE, END]),
            asm(&[IMAGE_STORE, swappc, END]),
            asm(&[IMAGE_STORE, call_ok, END]),
        ];
        assert_eq!(of(&cases[6]), Some(Effects { memory: false, images: true, instructions: 6 }));
        let pool = [asm(&[MOV]), asm(&[IMAGE_STORE]), asm(&[store]), asm(&[END]), asm(&[code_end]), asm(&[("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0)])]), asm(&[oob_br]), asm(&[setpc])];
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..64 {
            let mut s = Vec::new();
            for _ in 0..16 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; s.extend_from_slice(&pool[(rng as usize) % pool.len()]); }
            cases.push(s);
        }
        for c in &cases {
            let want = of_reference(c);
            assert_eq!(of(c), want);
            assert_eq!(analyze(c, true).is_some(), want.is_some_and(|e| !e.memory));
        }

        let (mov, small) = (asm(&[MOV]), asm(&[IMAGE_STORE, END]));
        assert!(!big_and_image_only(&xbox::tests::sample(5, &[xbox::tests::fssm(&small, &[(6, 0)])])));
        assert!(!big_and_image_only(&fspx_blob(&small)));
        let mut big_img = Vec::with_capacity(BIG_CODE + 16);
        while big_img.len() < BIG_CODE { big_img.extend_from_slice(&mov); }
        big_img.extend(asm(&[IMAGE_STORE, END]));
        let mut big_mem = asm(&[store]);
        while big_mem.len() < BIG_CODE { big_mem.extend_from_slice(&mov); }
        big_mem.extend(asm(&[END]));
        assert!(big_and_image_only(&fspx_blob(&big_img)) && !big_and_image_only(&fspx_blob(&big_mem)) && !big_and_image_only(&fspx_blob(&big_img)[..40]));
        assert!(big_and_image_only(&xbox::tests::sample(5, &[xbox::tests::fssm(&big_img, &[(6, 0)])])) && !big_and_image_only(&xbox::tests::sample(5, &[xbox::tests::fssm(&big_mem, &[(6, 0)])])));

        if !cfg!(debug_assertions) {
            fn bench(iters: u32, mut f: impl FnMut() -> bool) -> (f64, u32) {
                let (t0, mut acc) = (std::time::Instant::now(), 0u32);
                for _ in 0..iters { acc += u32::from(std::hint::black_box(f())); }
                (t0.elapsed().as_micros() as f64 / iters as f64, acc)
            }
            let (mut img_100k, mut mem_100k) = (Vec::with_capacity(100_016), asm(&[store]));
            while img_100k.len() < 100_000 { img_100k.extend_from_slice(&mov); }
            img_100k.extend(asm(&[IMAGE_STORE, END]));
            while mem_100k.len() < 100_000 { mem_100k.extend_from_slice(&mov); }
            mem_100k.extend(asm(&[END]));
            let (blob_img, blob_mem) = (xbox::tests::sample(5, &[xbox::tests::fssm(&img_100k, &[(6, 0)])]), xbox::tests::sample(5, &[xbox::tests::fssm(&mem_100k, &[(6, 0)])]));
            let ref_big = |blob: &[u8]| {
                let sh = xbox::parse(blob)?;
                let f = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 6))?;
                let p = f.programs.first()?;
                let c = f.data.get(p.offset..p.offset.checked_add(p.len)?)?.to_vec();
                Some(c.len() >= BIG_CODE && of_reference(&c).is_some_and(|e| !e.memory))
            };
            let ((ib, a1), (ia, a2)) = (bench(50, || ref_big(&blob_img).unwrap_or(false)), bench(50, || big_and_image_only(&blob_img)));
            let ((mb, b1), (ma, b2)) = (bench(50, || ref_big(&blob_mem).unwrap_or(false)), bench(50, || big_and_image_only(&blob_mem)));
            assert_eq!((a1, a2, b1, b2), (50, 50, 0, 0));
            println!("EFFECTS_BENCH_US img_100k: before={ib:.1} after={ia:.1} | mem_100k: before={mb:.1} after={ma:.1}");
        }
    }
}
