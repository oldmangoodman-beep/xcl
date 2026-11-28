mod support;

use std::hint::black_box;
use support::Criterion;

fn core_benches(c: &mut Criterion) {
    let encoded = xcl::gpu::rdna2::encode(
        "V_ADD_F32",
        "ENC_VOP2",
        &[("VDST", 1), ("SRC0", 258), ("VSRC1", 3)],
        None,
    )
    .expect("the built-in instruction table must encode V_ADD_F32");

    c.bench_function("rdna2/decode_vop2", |b| {
        b.iter(|| {
            black_box(
                xcl::gpu::rdna2::decode(black_box(&encoded), 0)
                    .expect("benchmark instruction must decode"),
            );
        });
    });

    c.bench_function("descriptor/component_mapping", |b| {
        b.iter(|| {
            black_box(xcl::gpu::desc::component_mapping(
                black_box([4, 5, 6, 7]),
                black_box([0, 1, 2, 3]),
            ));
        });
    });
}

fn main() {
    support::run(&[core_benches]);
}
