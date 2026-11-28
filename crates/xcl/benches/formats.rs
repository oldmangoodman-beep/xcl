mod support;

use std::hint::black_box;
use std::time::Duration;
use support::{BenchmarkId, Criterion, Throughput};
use xcl::gpu::{desc, pack, rdna2};
use xcl::pe::{Layout, Pe};

fn instruction() -> Vec<u8> {
    rdna2::encode(
        "V_ADD_F32",
        "ENC_VOP2",
        &[("VDST", 1), ("SRC0", 128), ("VSRC1", 2)],
        None,
    )
    .expect("benchmark instruction must encode")
}

fn rdna2_benches(c: &mut Criterion) {
    let inst = instruction();
    let mut group = c.benchmark_group("rdna2_instruction");
    group.throughput(Throughput::Elements(1));
    group.bench_function("encode", |b| {
        b.iter(|| {
            rdna2::encode(
                black_box("V_ADD_F32"),
                black_box("ENC_VOP2"),
                black_box(&[("VDST", 1), ("SRC0", 128), ("VSRC1", 2)]),
                None,
            )
            .unwrap()
        })
    });
    group.bench_function("decode", |b| {
        b.iter(|| rdna2::decode(black_box(&inst), black_box(0x1_0000)).unwrap())
    });
    let decoded = rdna2::decode(&inst, 0x1_0000).unwrap();
    group.bench_function("format", |b| b.iter(|| rdna2::format(black_box(&decoded))));
    group.bench_function("register_uses", |b| {
        b.iter(|| {
            (
                rdna2::sgpr_uses(black_box(&decoded)),
                rdna2::vgpr_uses(black_box(&decoded)),
            )
        })
    });
    group.finish();

    let mut group = c.benchmark_group("rdna2_stream");
    for instructions in [64usize, 1024, 16_384] {
        let code = inst.repeat(instructions);
        group.throughput(Throughput::Bytes(code.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("disassemble", instructions),
            &code,
            |b, code| b.iter(|| rdna2::disassemble(black_box(code), 0, false)),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("rdna2_program_scan");
    for bytes in [4 * 1024usize, 256 * 1024, 4 * 1024 * 1024] {
        let mut blob = vec![0x5au8; bytes];
        // s_endpgm is enough to exercise the scanner's candidate handling without title data.
        for at in (0..bytes.saturating_sub(4)).step_by(256) {
            blob[at..at + 4].copy_from_slice(&0xbf81_0000u32.to_le_bytes());
        }
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bytes), &blob, |b, blob| {
            b.iter(|| rdna2::find_programs(black_box(blob)))
        });
    }
    group.finish();
}

fn descriptor_benches(c: &mut Criterion) {
    let buffer = desc::buffer_words(0x1_2345_6780, 16, 65_536, 32);
    let image = desc::xcl_image_for_view(17, 10, desc::ImageType::D2, 1920, 1080, 1, 11);
    let sampler = [0x0000_1000, 0x0000_0000, 0x0000_0000, 0x0000_0000];
    let mut group = c.benchmark_group("gpu_descriptors");
    group.throughput(Throughput::Elements(1));
    group.bench_function("decode_buffer", |b| {
        b.iter(|| desc::buffer(black_box(&buffer)))
    });
    group.bench_function("decode_image", |b| {
        b.iter(|| desc::image(black_box(&image)))
    });
    group.bench_function("image_to_d3d12", |b| {
        let decoded = desc::image(&image).unwrap();
        b.iter(|| desc::image_d3d12(black_box(&decoded)))
    });
    group.bench_function("sampler_to_d3d12", |b| {
        b.iter(|| desc::sampler_d3d12(black_box(&sampler), black_box(false)))
    });
    group.finish();
}

fn manifest_and_config_benches(c: &mut Criterion) {
    let languages = (0..64)
        .map(|i| format!(r#"<Resource Language="lang-{i}"/>"#))
        .collect::<String>();
    let xml = format!(
        r#"<Game configVersion="1"><PersistentLocalStorage><SizeMB>4096</SizeMB></PersistentLocalStorage><VirtualMachine><XboxLockhartTitleMemory>Standard</XboxLockhartTitleMemory><XboxAnacondaTitleMemory>Advanced</XboxAnacondaTitleMemory></VirtualMachine><TitleId>12345678</TitleId><Resources>{languages}</Resources></Game>"#
    );
    let config = toml::to_string(&xcl::config::Config::default()).unwrap();
    let mut group = c.benchmark_group("text_formats");
    group.throughput(Throughput::Bytes(xml.len() as u64));
    group.bench_function("manifest_xml", |b| {
        b.iter(|| xcl::manifest::parse(black_box(&xml)))
    });
    group.throughput(Throughput::Bytes(config.len() as u64));
    group.bench_function("config_toml", |b| {
        b.iter(|| toml::from_str::<xcl::config::Config>(black_box(&config)).unwrap())
    });
    group.finish();
}

fn record(name: &str, payload: &[u8]) -> Vec<u8> {
    let compressed = miniz_oxide::deflate::compress_to_vec(payload, 6);
    let mut out = vec![(name.len() + 1) as u8];
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    out.push(50);
    for value in [
        compressed.len() as u32,
        0,
        compressed.len() as u32,
        payload.len() as u32,
    ] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&compressed);
    out
}

fn pack_benches(c: &mut Criterion) {
    let payload = (0..16 * 1024).map(|i| (i % 251) as u8).collect::<Vec<_>>();
    let one = record("CS_SyntheticBenchmark", &payload);
    let mut archive = b"9FPR".to_vec();
    archive.resize(4096, 0xa5);
    for i in 0..256 {
        archive.extend_from_slice(&record(&format!("CS_Synthetic_{i:04}"), &payload));
        archive.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
    }
    archive.extend_from_slice(&[0; 40]);
    let records = pack::scan(&one, 0);
    assert_eq!(records.len(), 1);

    let mut group = c.benchmark_group("rpf_records");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(4));
    group.throughput(Throughput::Bytes(archive.len() as u64));
    group.bench_function("scan_256_records", |b| {
        b.iter(|| pack::scan(black_box(&archive), 0))
    });
    group.throughput(Throughput::Bytes(payload.len() as u64));
    group.bench_function("inflate_16k", |b| {
        b.iter(|| pack::inflate(black_box(&one), black_box(&records[0])).unwrap())
    });
    group.finish();
}

fn pe_benches(c: &mut Criterion) {
    let exe = std::env::current_exe().expect("benchmark executable path");
    let bytes = std::fs::read(exe).expect("benchmark executable bytes");
    let mut group = c.benchmark_group("pe");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("parse_benchmark_executable", |b| {
        b.iter(|| Pe::parse(black_box(&bytes), Layout::File).unwrap())
    });
    group.finish();
}

fn main() {
    support::run(&[
        rdna2_benches,
        descriptor_benches,
        manifest_and_config_benches,
        pack_benches,
        pe_benches,
    ]);
}
