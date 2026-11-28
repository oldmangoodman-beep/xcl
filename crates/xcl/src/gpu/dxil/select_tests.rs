//! GPU equivalence for the guarded-select fold, including inactive lanes.

use super::kernel::{translate, KernelOptions};
use super::module::NO_SELECT_FOLD;
use crate::gpu::rdna2::encode;
use crate::gpu::recomp::runner::{self, KernelState};

const V: u64 = 256;

fn program(list: &[(&str, &str, &[(&str, u64)])]) -> Vec<u8> {
    let mut code = Vec::new();
    for (name, encoding, fields) in list {
        code.extend(encode(name, encoding, fields, None).unwrap_or_else(|e| panic!("{name}: {e}")));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    code
}

fn translate_with_fold(code: &[u8], options: KernelOptions, enabled: bool) -> Vec<u8> {
    NO_SELECT_FOLD.with(|c| c.set(!enabled));
    let result = translate(code, options);
    NO_SELECT_FOLD.with(|c| c.set(false));
    result.unwrap_or_else(|e| panic!("translate (select fold {enabled}): {e}"))
}

#[test]
fn gpu_guarded_select_fold_preserves_inactive_lanes() {
    match (runner::adapter(), runner::wave64_programs_supported()) {
        (Ok(_), Ok(true)) => {}
        (adapter, _) => return runner::skip_test(if adapter.is_err() { "no GPU" } else { "no wave64" }),
    }
    let code = program(&[
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 0)]),
        ("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V), ("VSRC1", 1)]),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 4), ("VSRC1", 2)]),
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 4), ("SRC1", V + 3), ("SRC2", V + 4)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 5), ("VSRC1", 4)]),
        ("V_CMP_GT_F32", "ENC_VOPC", &[("SRC0", V + 4), ("VSRC1", 5)]),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 4), ("VSRC1", 5)]),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 4), ("SRC1", 131)]),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 21), ("SRC0", V + 6), ("SRC1", 137)]),
        ("S_AND_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)]),
        ("V_SUB_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 4), ("VSRC1", 6)]),
        ("V_MAX_F32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V + 4), ("VSRC1", 0)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 4)]),
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 9), ("SRC0", 0xFA), ("VSRC0", 4), ("DPP_CTRL", 0x111), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)]),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 4), ("SRC1", 133)]),
        ("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 6), ("VSRC1", 6)]),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 10)]),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 4), ("VSRC1", 6)]),
        ("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 7), ("VSRC1", 8)]),
    ]);
    const EDGE: [u32; 12] = [0x7FC0_1234, 0, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 1, 0x807F_FFFF, 0x3FC0_0000, 0xC000_0000, 0x3F80_0000, 0x4120_0000, 0xBE80_0000];
    let mut state = KernelState::default();
    let mut random = 0x9E37_79B9_7F4A_7C15u64;
    for (lane, regs) in state.vgpr.iter_mut().enumerate() {
        for word in regs.iter_mut().take(4) {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            *word = if random % 3 == 0 { random as u32 } else { EDGE[(random >> 8) as usize % EDGE.len()] };
        }
        for (reg, word) in regs.iter_mut().enumerate().skip(4).take(8) {
            *word = 0xA000_0000 | (lane as u32) << 8 | reg as u32;
        }
    }
    (state.sgpr[0], state.sgpr[1]) = (0xF0F0_5A3C, 0x0FF0_C3A5);
    (state.sgpr[2], state.sgpr[3]) = (0x3333_FF0F, 0xFFFF_3333);

    let mut ran = 0;
    for (wave64, split) in [(false, false), (true, true), (true, false)] {
        if wave64 && !split && runner::wave64_supported() != Ok(true) {
            continue;
        }
        let options = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let (before, after) = (translate_with_fold(&code, options, false), translate_with_fold(&code, options, true));
        assert!(after.len() < before.len(), "wave64 {wave64} split {split}: nothing folded ({} -> {} B)", before.len(), after.len());
        let a = runner::run(&before, &state, &[]).expect("GPU run before fold");
        let b = runner::run(&after, &state, &[]).expect("GPU run after fold");
        let lanes = if wave64 { 64 } else { 32 };
        for lane in 0..lanes {
            assert_eq!(a.vgpr[lane][..12], b.vgpr[lane][..12], "wave64 {wave64} split {split}: lane {lane}");
            let active = (state.sgpr[lane / 32] >> (lane % 32)) & 1 != 0;
            if !active {
                for reg in [4, 5, 7, 8] {
                    assert_eq!(b.vgpr[lane][reg], state.vgpr[lane][reg], "wave64 {wave64} split {split}: inactive lane {lane} v{reg}");
                }
            }
        }
        assert_eq!(a.sgpr, b.sgpr, "wave64 {wave64} split {split}: SGPRs");
        assert_eq!(a.scc, b.scc, "wave64 {wave64} split {split}: SCC");
        ran += 1;
    }
    assert!(ran >= 2);
}
