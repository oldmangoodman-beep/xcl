//! Process and adapter memory against their budgets, split into XCL-owned parts (#2697).

pub fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("XCL_ACTIVITY").is_some())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Video,
    NonLocal,
    Commit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub owner: &'static str,
    pub kind: Kind,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Segment {
    pub usage: u64,
    pub budget: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub commit: Option<u64>,
    pub local: Option<Segment>,
    pub non_local: Option<Segment>,
    pub parts: Vec<Part>,
}

impl Snapshot {
    pub fn counted(&self, kind: Kind) -> u64 {
        self.parts.iter().filter(|p| p.kind == kind).fold(0, |bytes, p| bytes.saturating_add(p.bytes))
    }

    /// A negative remainder exposes aliasing, a stale sample, or a part charged to the wrong side.
    pub fn other(&self, kind: Kind) -> Option<i64> {
        let total = match kind {
            Kind::Video => self.local?.usage,
            Kind::NonLocal => self.non_local?.usage,
            Kind::Commit => self.commit?,
        };
        let difference = i128::from(total) - i128::from(self.counted(kind));
        Some(difference.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64)
    }

    pub fn line(&self) -> String {
        let mb = |b: u64| b >> 20;
        let smb = |b: i64| b / (1 << 20);
        let parts = |kind: Kind| {
            let mut out: Vec<String> = self.parts.iter().filter(|p| p.kind == kind)
                .map(|p| format!("{} {}", p.owner, mb(p.bytes))).collect();
            if let Some(other) = self.other(kind) {
                out.push(format!("other {}", smb(other)));
            }
            out.join(", ")
        };
        let segment = |name: &str, value: Option<Segment>, kind: Kind| match value {
            Some(s) => format!("{name} {} of a {} budget ({})", mb(s.usage), mb(s.budget), parts(kind)),
            None => format!("{name} unknown ({})", parts(kind)),
        };
        let commit = match self.commit {
            Some(bytes) => format!("commit {} ({})", mb(bytes), parts(Kind::Commit)),
            None => format!("commit unknown ({})", parts(Kind::Commit)),
        };
        format!("memory by owner (MB): {commit}; {}; {}",
            segment("video memory", self.local, Kind::Video),
            segment("non-local", self.non_local, Kind::NonLocal))
    }
}

fn process_commit() -> Option<u64> {
    unsafe {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of_val(&c) as u32;
        (K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb) != 0)
            .then(|| u64::try_from(c.PrivateUsage).unwrap_or(u64::MAX))
    }
}

fn video_memory(raw: *mut std::ffi::c_void) -> (Option<Segment>, Option<Segment>) {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::ID3D12Device;
    use windows::Win32::Graphics::Dxgi::*;
    let Some(dev) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return (None, None) };
    unsafe {
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return (None, None) };
        let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter3>(dev.GetAdapterLuid()) else { return (None, None) };
        let query = |group| {
            let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            adapter.QueryVideoMemoryInfo(0, group, &mut i).ok()
                .map(|_| Segment { usage: i.CurrentUsage, budget: i.Budget })
        };
        (query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL), query(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL))
    }
}

/// Adapter-local usage for the pressure sampler, without collecting owner diagnostics.
pub fn local_segment() -> Option<Segment> {
    video_memory(super::real::device_raw()).0
}

pub fn snapshot() -> Snapshot {
    let raw = super::real::device_raw();
    let device = raw as usize;
    let (pages, page_pools) = super::vram::page_bytes_for(device);
    let arenas = super::vram::arena_bytes_for(device);
    let (pools, mapped, retired, sparse_commit) = super::sparse::memory_bytes_for(device);
    let (_, committed_textures) = super::title_textures::committed_bytes_for(device);
    let (local, non_local) = video_memory(raw);
    let uma = crate::gpu::uma::get().map_or(0, |u| u64::try_from(u.size()).unwrap_or(u64::MAX));
    let b = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
    let parts = vec![
        Part { owner: "UMA window", kind: Kind::Commit, bytes: uma },
        Part { owner: "sparse CPU copy", kind: Kind::Commit, bytes: b(sparse_commit) },
        Part { owner: "target pages", kind: Kind::Video, bytes: b(pages) },
        Part { owner: "target pools free", kind: Kind::Video, bytes: b(page_pools.saturating_sub(pages)) },
        Part { owner: "arenas", kind: Kind::Video, bytes: b(arenas) },
        Part { owner: "sparse tiles", kind: Kind::Video, bytes: b(mapped) },
        Part { owner: "sparse pools free", kind: Kind::Video, bytes: b(pools.saturating_sub(mapped)) },
        Part { owner: "sparse pools retired", kind: Kind::Video, bytes: b(retired) },
        Part { owner: "tracked committed sampled textures", kind: Kind::Video, bytes: committed_textures },
    ];
    Snapshot { commit: process_commit(), local, non_local, parts }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remainder_and_partial_query_failure_are_explicit() {
        const M: u64 = 1 << 20;
        let s = Snapshot {
            commit: Some(10_000 * M),
            local: Some(Segment { usage: 9_000 * M, budget: 11_000 * M }),
            non_local: None,
            parts: vec![
                Part { owner: "UMA window", kind: Kind::Commit, bytes: 4_096 * M },
                Part { owner: "target pages", kind: Kind::Video, bytes: 5_000 * M },
                Part { owner: "upload staging", kind: Kind::NonLocal, bytes: 400 * M },
            ],
        };
        assert_eq!(s.other(Kind::Video), Some(4_000 * M as i64));
        assert_eq!(s.other(Kind::NonLocal), None);
        assert_eq!(s.line(), "memory by owner (MB): commit 10000 (UMA window 4096, other 5904); video memory 9000 of a 11000 budget (target pages 5000, other 4000); non-local unknown (upload staging 400)");
        let over = Snapshot { non_local: Some(Segment { usage: 300 * M, budget: 16_000 * M }), ..s };
        assert_eq!(over.other(Kind::NonLocal), Some(-(100 * M as i64)));
        let unknown = Snapshot { commit: None, ..over };
        assert!(unknown.line().contains("commit unknown (UMA window 4096)"));

        let reverse = Snapshot {
            local: None,
            non_local: Some(Segment { usage: 700 * M, budget: 2_000 * M }),
            ..unknown
        };
        let line = reverse.line();
        assert!(line.contains("video memory unknown (target pages 5000)"), "{line}");
        assert!(line.contains("non-local 700 of a 2000 budget (upload staging 400, other 300)"), "{line}");
    }

    #[test]
    fn signed_remainders_truncate_and_saturate() {
        let tiny = Snapshot {
            local: Some(Segment { usage: (1 << 20) - 1, budget: 1 << 20 }),
            parts: vec![Part { owner: "one MiB", kind: Kind::Video, bytes: 1 << 20 }],
            ..Default::default()
        };
        assert_eq!(tiny.other(Kind::Video), Some(-1));
        assert!(tiny.line().contains("other 0"), "{}", tiny.line());

        let large = Snapshot {
            commit: Some(0),
            parts: vec![
                Part { owner: "large", kind: Kind::Commit, bytes: u64::MAX },
                Part { owner: "saturated", kind: Kind::Commit, bytes: 1 },
            ],
            ..Default::default()
        };
        assert_eq!(large.counted(Kind::Commit), u64::MAX);
        assert_eq!(large.other(Kind::Commit), Some(i64::MIN));
    }

    #[test]
    fn device_snapshot_has_budget_and_all_owners() {
        if super::super::backend() != super::super::Backend::Real {
            return crate::gpu::recomp::runner::skip_test("no GPU");
        }
        let s = snapshot();
        assert!(s.local.expect("local video memory").budget > 0, "{}", s.line());
        assert!(s.commit.is_some());
        for owner in ["UMA window", "sparse CPU copy", "target pages", "target pools free", "arenas", "sparse tiles", "sparse pools free", "sparse pools retired", "tracked committed sampled textures"] {
            assert!(s.parts.iter().any(|p| p.owner == owner), "{owner} missing: {}", s.line());
        }
    }
}
