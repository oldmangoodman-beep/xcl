mod support;

use std::hint::black_box;
use support::{BenchmarkId, Criterion, Throughput};
use xcl::gpu::{desc, rdna2, recomp, shader, xbox};

const VGPR0: u64 = 256;

fn instruction(name: &str, encoding: &str, fields: &[(&str, u64)]) -> Vec<u8> {
    rdna2::encode(name, encoding, fields, None).unwrap()
}

fn arithmetic_program(instructions: usize) -> Vec<u8> {
    let add = instruction(
        "V_ADD_NC_U32",
        "ENC_VOP2",
        &[("VDST", 1), ("SRC0", VGPR0 + 2), ("VSRC1", 3)],
    );
    let xor = instruction(
        "V_XOR_B32",
        "ENC_VOP2",
        &[("VDST", 4), ("SRC0", VGPR0 + 1), ("VSRC1", 5)],
    );
    let mut code = Vec::with_capacity(instructions * 4 + 4);
    for i in 0..instructions {
        code.extend(if i & 1 == 0 { &add } else { &xor });
    }
    code.extend(instruction("S_ENDPGM", "ENC_SOPP", &[]));
    code
}

fn branch_program(blocks: usize) -> Vec<u8> {
    let mov = instruction("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)]);
    let branch = instruction("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)]);
    let add = instruction(
        "V_ADD_NC_U32",
        "ENC_VOP2",
        &[("VDST", 1), ("SRC0", VGPR0 + 2), ("VSRC1", 3)],
    );
    let mut code = Vec::with_capacity(blocks * 12 + 4);
    for _ in 0..blocks {
        code.extend(&mov);
        code.extend(&branch);
        code.extend(&add);
    }
    code.extend(instruction("S_ENDPGM", "ENC_SOPP", &[]));
    code
}

fn graphics_program(instructions: usize, spills: bool) -> Vec<u8> {
    let mut code = Vec::new();
    let mut count = 0usize;
    if spills {
        for lane in 0..48u64 {
            code.extend(instruction(
                "V_WRITELANE_B32",
                "ENC_VOP3",
                &[("VDST", 30), ("SRC0", 32 + lane), ("SRC1", 128 + lane)],
            ));
            count += 1;
        }
        for lane in 0..48u64 {
            code.extend(instruction(
                "V_READLANE_B32",
                "ENC_VOP3",
                &[("VDST", lane), ("SRC0", VGPR0 + 30), ("SRC1", 128 + lane)],
            ));
            count += 1;
        }
    }
    let add = instruction(
        "V_ADD_NC_U32",
        "ENC_VOP2",
        &[("VDST", 1), ("SRC0", VGPR0 + 2), ("VSRC1", 3)],
    );
    for _ in count..instructions {
        code.extend(&add);
    }
    code.extend(instruction("S_ENDPGM", "ENC_SOPP", &[]));
    code
}

fn graphics_options(split_wave64: bool) -> recomp::Options {
    recomp::Options {
        mode: recomp::Mode::Pixel(recomp::PsInterface {
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_ena: 0,
            attrs: None,
        }),
        wave64: true,
        split_wave64,
        ..Default::default()
    }
}

fn rdna2_frontend(c: &mut Criterion) {
    let code = arithmetic_program(1024);
    let mesh_control_flow = branch_program(256);
    let mut group = c.benchmark_group("rdna2_frontend");
    group.throughput(Throughput::Bytes(code.len() as u64));
    group.bench_function("decode_stream", |b| {
        b.iter(|| {
            let mut offset = 0;
            while offset < code.len() {
                let decoded = rdna2::decode(black_box(&code[offset..]), offset as u64).unwrap();
                offset += black_box(decoded.len);
            }
            black_box(offset)
        })
    });
    group.bench_function("disassemble", |b| {
        b.iter(|| black_box(rdna2::disassemble(black_box(&code), 0, false)))
    });
    group.bench_function("wave_hazard_scan", |b| {
        b.iter(|| black_box(recomp::wave_hazards(black_box(&code))))
    });
    group.bench_function("geometry_wave32_hazard_scan", |b| {
        b.iter(|| black_box(recomp::geometry_wave32_hazards(black_box(&code))))
    });
    group.throughput(Throughput::Bytes(mesh_control_flow.len() as u64));
    group.bench_function("mesh_wave_analysis", |b| {
        b.iter(|| black_box(recomp::mesh_wave::analyze(black_box(&mesh_control_flow))))
    });
    group.finish();
}

fn translate(c: &mut Criterion) {
    let mut group = c.benchmark_group("recompiler_translate");
    group.sample_size(20);
    for size in [16usize, 64, 256] {
        let code = arithmetic_program(size);
        group.throughput(Throughput::Elements((size + 1) as u64));
        group.bench_with_input(BenchmarkId::new("linear", size), &code, |b, code| {
            b.iter(|| {
                let translated =
                    recomp::translate(black_box(code), black_box(&recomp::Options::default()))
                        .unwrap();
                black_box((
                    translated.hlsl.len(),
                    translated.instructions,
                    translated.blocks,
                ))
            })
        });
    }
    for blocks in [8usize, 32] {
        let code = branch_program(blocks);
        group.throughput(Throughput::Elements((blocks * 3 + 1) as u64));
        group.bench_with_input(
            BenchmarkId::new("control_flow", blocks),
            &code,
            |b, code| {
                b.iter(|| {
                    let translated =
                        recomp::translate(black_box(code), black_box(&recomp::Options::default()))
                            .unwrap();
                    black_box((
                        translated.hlsl.len(),
                        translated.instructions,
                        translated.blocks,
                    ))
                })
            },
        );
    }
    let plain = graphics_program(256, false);
    let spills = graphics_program(256, true);
    let native = graphics_options(false);
    let split = graphics_options(true);
    for (name, code, options) in [
        ("plain", &plain, &native),
        ("spills_native", &spills, &native),
        ("spills_split", &spills, &split),
    ] {
        group.throughput(Throughput::Elements(257));
        group.bench_with_input(
            BenchmarkId::new("wave64_graphics", name),
            &(code, options),
            |b, (code, options)| {
                b.iter(|| {
                    let translated = recomp::translate(black_box(code), black_box(options)).unwrap();
                    black_box((translated.hlsl.len(), translated.instructions, translated.blocks))
                })
            },
        );
    }
    group.finish();
}

fn descriptors(c: &mut Criterion) {
    let image = desc::xcl_image_for_view(7, 10, desc::ImageType::D2, 1920, 1080, 1, 11);
    let buffer = desc::buffer_words(0x1_2345_6000, 16, 65_536, 74);
    let sampler = [0x0000_0002, 0x0000_0101, 0x3f80_0000, 0x0000_0000];
    let mut group = c.benchmark_group("descriptor_conversion");
    group.bench_function("decode_image", |b| {
        b.iter(|| black_box(desc::image(black_box(&image))))
    });
    group.bench_function("image_to_d3d12", |b| {
        let decoded = desc::image(&image).unwrap();
        b.iter(|| black_box(desc::image_d3d12(black_box(&decoded))))
    });
    group.bench_function("decode_buffer", |b| {
        b.iter(|| black_box(desc::buffer(black_box(&buffer))))
    });
    group.bench_function("sampler_to_d3d12", |b| {
        b.iter(|| black_box(desc::sampler_d3d12(black_box(&sampler), false)))
    });
    group.finish();
}

fn shader_scans(c: &mut Criterion) {
    let program = arithmetic_program(1024);
    let mut blob = vec![0u8; 4096];
    for _ in 0..16 {
        blob.extend_from_slice(&program);
        blob.extend_from_slice(&[0u8; 64]);
    }
    let mut group = c.benchmark_group("shader_scans");
    group.throughput(Throughput::Bytes(blob.len() as u64));
    group.bench_function("find_rdna2_programs", |b| {
        b.iter(|| black_box(rdna2::find_programs(black_box(&blob))))
    });
    group.throughput(Throughput::Bytes(program.len() as u64));
    group.bench_function("classify_raw_rdna2", |b| {
        b.iter(|| black_box(shader::classify(black_box(&program))))
    });
    group.throughput(Throughput::Bytes(blob.len() as u64));
    group.bench_function("fnv64", |b| {
        b.iter(|| black_box(shader::fnv64(black_box(&blob))))
    });
    group.throughput(Throughput::Bytes(program.len() as u64));
    group.bench_function("split_segments", |b| {
        b.iter(|| black_box(xbox::split_segments(black_box(&program))))
    });
    group.finish();
}

fn main() {
    support::run(&[rdna2_frontend, translate, descriptors, shader_scans]);
}
