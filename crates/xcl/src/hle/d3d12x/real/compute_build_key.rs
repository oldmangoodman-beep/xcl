use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use windows::Win32::Graphics::Direct3D12::D3D12_COMPUTE_PIPELINE_STATE_DESC;

use crate::gpu::recomp::pso::ComputePolicyKey;

/// The translated compute root-signature layout. A version change must change this tag.
const COMPUTE_ABI_SCHEMA: u32 = 1;

/// Canonical identity of a requested compute build, before translation begins.
///
/// `device` is safe identity, rather than an unretained COM pointer: the real backend owns its
/// device reference for the process lifetime. The ABI static-sampler set is transitively and
/// collision-freely identified by the complete shader bytes, translation policy, and sampler
/// base; those bytes also identify the program's intrinsic wave mode. The title root signature
/// and cached PSO are intentionally absent: translation replaces both before calling the PC driver.
#[derive(Clone, Debug)]
pub(in crate::hle::d3d12x) struct ComputeBuildKey {
    device: usize,
    node_mask: u32,
    flags: i32,
    shader: Arc<[u8]>,
    shader_digest: u64,
    shader_len: usize,
    windows: u32,
    split_wave64: bool,
    policy: ComputePolicyKey,
    abi_schema: u32,
    static_sampler_base: u32,
}

impl PartialEq for ComputeBuildKey {
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device
            && self.node_mask == other.node_mask
            && self.flags == other.flags
            && self.windows == other.windows
            && self.split_wave64 == other.split_wave64
            && self.policy == other.policy
            && self.abi_schema == other.abi_schema
            && self.static_sampler_base == other.static_sampler_base
            && self.shader_digest == other.shader_digest
            && self.shader_len == other.shader_len
            && self.shader == other.shader
    }
}

impl Eq for ComputeBuildKey {}

impl Hash for ComputeBuildKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.device.hash(state);
        self.node_mask.hash(state);
        self.flags.hash(state);
        self.windows.hash(state);
        self.split_wave64.hash(state);
        self.policy.hash(state);
        self.abi_schema.hash(state);
        self.static_sampler_base.hash(state);
        self.shader_digest.hash(state);
        self.shader_len.hash(state);
    }
}

impl ComputeBuildKey {
    pub(in crate::hle::d3d12x) fn capture(
        device: usize,
        desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
        shader: Arc<[u8]>,
        split_wave64: bool,
    ) -> Self {
        Self::from_descriptor(
            device,
            desc,
            shader,
            crate::gpu::recomp::pso::mem_windows(),
            split_wave64,
            crate::gpu::recomp::pso::compute_policy_key(),
        )
    }

    fn from_descriptor(
        device: usize,
        desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
        shader: Arc<[u8]>,
        windows: u32,
        split_wave64: bool,
        policy: ComputePolicyKey,
    ) -> Self {
        Self::from_parts(device, desc.NodeMask, desc.Flags.0, shader, windows, split_wave64, policy)
    }

    fn from_parts(
        device: usize,
        node_mask: u32,
        flags: i32,
        shader: Arc<[u8]>,
        windows: u32,
        split_wave64: bool,
        policy: ComputePolicyKey,
    ) -> Self {
        debug_assert_ne!(device, 0);
        let shader_digest = crate::gpu::shader::fnv64(&shader);
        Self::from_parts_with_digest(
            device, node_mask, flags, shader, shader_digest, windows, split_wave64, policy,
        )
    }

    fn from_parts_with_digest(
        device: usize,
        node_mask: u32,
        flags: i32,
        shader: Arc<[u8]>,
        shader_digest: u64,
        windows: u32,
        split_wave64: bool,
        policy: ComputePolicyKey,
    ) -> Self {
        let shader_len = shader.len();
        Self {
            device,
            node_mask,
            flags,
            shader,
            shader_digest,
            shader_len,
            windows,
            split_wave64,
            policy,
            abi_schema: COMPUTE_ABI_SCHEMA,
            static_sampler_base: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash(key: &ComputeBuildKey) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn policy() -> ComputePolicyKey {
        ComputePolicyKey {
            direct: false,
            structured: true,
            precise_x64_masks: true,
            wave64_spin: true,
            safe_barrier: true,
            direct_compact: None,
            direct_compact_nest: None,
            direct_piece_limit: 0,
            per_lane: false,
            disabled_dxil_optimizations: 0,
        }
    }

    fn key() -> ComputeBuildKey {
        ComputeBuildKey::from_parts(1, 2, 3, [4, 5].into(), 6, false, policy())
    }

    #[test]
    fn every_compute_build_input_splits_the_key() {
        let base = key();
        macro_rules! different {
            ($field:ident, $value:expr) => {{
                let mut other = base.clone();
                other.$field = $value;
                assert_ne!(base, other, stringify!($field));
                assert_ne!(hash(&base), hash(&other), stringify!($field));
            }};
        }
        different!(device, 9);
        different!(node_mask, 9);
        different!(flags, 9);
        different!(windows, 9);
        different!(split_wave64, true);
        different!(abi_schema, 9);
        different!(static_sampler_base, 9);

        let mutate_policy = |f: fn(&mut ComputePolicyKey)| {
            let mut other = base.clone();
            f(&mut other.policy);
            assert_ne!(base, other);
            assert_ne!(hash(&base), hash(&other));
        };
        mutate_policy(|p| p.direct = true);
        mutate_policy(|p| p.structured = false);
        mutate_policy(|p| p.precise_x64_masks = false);
        mutate_policy(|p| p.wave64_spin = false);
        mutate_policy(|p| p.safe_barrier = false);
        mutate_policy(|p| p.direct_compact = Some(true));
        mutate_policy(|p| p.direct_compact_nest = Some("outer".into()));
        mutate_policy(|p| p.direct_piece_limit = 1);
        mutate_policy(|p| p.per_lane = true);
        mutate_policy(|p| p.disabled_dxil_optimizations = 1);

        let other = ComputeBuildKey::from_parts(1, 2, 3, [4, 6].into(), 6, false, policy());
        assert_ne!(base, other, "shader");
        assert_ne!(hash(&base), hash(&other), "shader");
    }

    #[test]
    fn equal_keys_have_equal_hashes() {
        assert_eq!(hash(&key()), hash(&key()));
    }

    #[test]
    fn shader_identity_compares_all_bytes_after_a_digest_collision() {
        let base = key();
        let other = ComputeBuildKey::from_parts_with_digest(
            1, 2, 3, [5, 4].into(), base.shader_digest, 6, false, policy(),
        );
        assert_eq!(hash(&base), hash(&other));
        assert_ne!(base, other);
    }

    #[test]
    fn descriptor_padding_is_not_part_of_the_key() {
        let mut a = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
        let mut b = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
        a.NodeMask = 7;
        b.NodeMask = 7;
        a.Flags = windows::Win32::Graphics::Direct3D12::D3D12_PIPELINE_STATE_FLAGS(1);
        b.Flags = windows::Win32::Graphics::Direct3D12::D3D12_PIPELINE_STATE_FLAGS(1);
        let padding = std::mem::offset_of!(D3D12_COMPUTE_PIPELINE_STATE_DESC, NodeMask)
            + std::mem::size_of::<u32>();
        let padding_end = std::mem::offset_of!(D3D12_COMPUTE_PIPELINE_STATE_DESC, CachedPSO);
        assert!(padding_end > padding);
        unsafe {
            std::ptr::write_bytes((&mut b as *mut _ as *mut u8).add(padding), 0xa5, padding_end - padding);
        }
        assert_eq!(
            ComputeBuildKey::from_descriptor(1, &a, [2, 3].into(), 4, false, policy()),
            ComputeBuildKey::from_descriptor(1, &b, [2, 3].into(), 4, false, policy()),
        );
    }

    #[test]
    fn build_key_shares_the_owned_shader_allocation() {
        let shader: Arc<[u8]> = [7, 8, 9].into();
        let pointer = shader.as_ptr();
        let key = ComputeBuildKey::from_parts(1, 0, 0, shader.clone(), 1, false, policy());
        assert_eq!(key.shader.as_ptr(), pointer);
        assert_eq!(Arc::strong_count(&shader), 2);
    }
}
