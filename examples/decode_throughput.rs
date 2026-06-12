//! Measures CELT decode throughput over an `opus_demo` bitstream file:
//!
//! ```sh
//! cargo run --release --example decode_throughput tests/vectors/testvector01.bit
//! ```
//!
//! Reports decoded audio seconds per wall-clock second (× realtime). Compare
//! backends with `--no-default-features --features std` (built-in FFT) vs the
//! default (`spectrograms` FFT).

use std::time::Instant;

use opus_native::celt::decoder::CeltDecoder;
use opus_native::{Bandwidth, Mode, Packet, RangeDecoder};

fn main() {
    let path = std::env::args().nth(1).expect("usage: decode_throughput <file.bit>");
    let data = std::fs::read(&path).expect("read bitstream file");

    // opus_demo framing: 4-byte BE length, 4-byte BE final range, payload.
    let mut packets = Vec::new();
    let mut off = 0usize;
    while off + 8 <= data.len() {
        let len = u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        off += 8;
        packets.push(&data[off..off + len]);
        off += len;
    }

    let mut decoder = CeltDecoder::new(2);
    let mut samples = 0u64;
    let start = Instant::now();
    for pkt in &packets {
        let parsed = Packet::parse(pkt).expect("valid packet");
        let toc = parsed.toc();
        assert_eq!(toc.mode(), Mode::CeltOnly, "this example decodes CELT-only streams");
        let frame_size = toc.frame_size().samples_per_channel_48k();
        let channels = usize::from(toc.channels());
        let end = match toc.bandwidth() {
            Bandwidth::NarrowBand => 13,
            Bandwidth::MediumBand | Bandwidth::WideBand => 17,
            Bandwidth::SuperWideBand => 19,
            Bandwidth::FullBand => 21,
        };
        for frame in parsed.frames() {
            let mut dec = RangeDecoder::new(frame);
            let pcm = decoder.decode_frame(&mut dec, frame.len(), frame_size, channels, 0, end);
            samples += (pcm.len() / 2) as u64;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let audio_secs = samples as f64 / 48_000.0;
    println!(
        "{}: {} packets, {audio_secs:.1} s audio in {elapsed:.3} s - {:.0}× realtime",
        path,
        packets.len(),
        audio_secs / elapsed
    );
}
