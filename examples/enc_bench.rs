//! Encoder throughput benchmark (× realtime), per mode. The counterpart to
//! `decode_throughput`. Encodes a fixed speech-like signal and reports how
//! much faster than realtime each mode runs.
//!
//!   cargo run --release --example enc_bench --features std

use std::time::Instant;

use opus_native::{Bandwidth, OpusEncoder};

fn bench(label: &str, ch: usize, bw: Bandwidth, br: u32, frames: usize) {
    // A speech/music-like signal: low tone + a high partial, per channel.
    let make = |f: usize| -> Vec<f32> {
        (0..960 * ch)
            .map(|i| {
                let t = (f * 960 + i / ch) as f32 / 48_000.0;
                0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.15 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin()
            })
            .collect()
    };
    let sigs: Vec<Vec<f32>> = (0..200).map(make).collect();

    let mut enc = OpusEncoder::new(ch);
    enc.set_bandwidth(bw);
    enc.set_bitrate(Some(br));
    for s in &sigs {
        let _ = enc.encode_auto(s, 1275); // warm up lazily-created state
    }

    let start = Instant::now();
    let mut bytes = 0usize;
    for _ in 0..frames / 200 {
        for s in &sigs {
            bytes += enc.encode_auto(s, 1275).map_or(0, |p| p.len());
        }
    }
    let secs = start.elapsed().as_secs_f64();
    let audio_s = frames as f64 * 0.02;
    println!(
        "{label:<22} {:>6.0}× realtime  ({:.0} kb/s)",
        audio_s / secs,
        bytes as f64 * 8.0 / audio_s / 1000.0
    );
}

fn main() {
    println!("encoder throughput (release):");
    bench("SILK WB 16k mono", 1, Bandwidth::WideBand, 16_000, 20_000);
    bench("hybrid FB 32k mono", 1, Bandwidth::FullBand, 32_000, 20_000);
    bench("CELT FB 96k mono", 1, Bandwidth::FullBand, 96_000, 20_000);
    bench("hybrid FB 48k stereo", 2, Bandwidth::FullBand, 48_000, 20_000);
}
