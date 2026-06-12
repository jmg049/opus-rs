//! Round-trip check: our CELT encoder -> our decoder, plus .bit dumps
//! (with recorded final ranges) for opus_demo / libopus verification.
use opus_native::OpusDecoder;
use opus_native::celt::encoder::CeltEncoder;

/// Runs one encoder/decoder round trip and writes an opus_demo-format
/// bitstream dump. Returns the number of final-range mismatches.
fn run(channels: usize, toc: u8, bit_path: &str) -> usize {
    let mut enc = CeltEncoder::with_channels(channels);
    let mut dec = OpusDecoder::new(channels);
    let mut bit = Vec::new();
    let mut in_pcm = Vec::new();
    let mut out_pcm = Vec::new();
    let mut mismatches = 0;
    for f in 0..100 {
        let mut pcm = Vec::with_capacity(960 * channels);
        for i in 0..960 {
            let t = (f * 960 + i) as f32 / 48000.0;
            let left = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 1800.0 * t).sin();
            pcm.push(left);
            if channels == 2 {
                // A decorrelated right channel exercises the mid/side path.
                let right = 0.4 * (2.0 * std::f32::consts::PI * 660.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 2500.0 * t + 0.7).sin();
                pcm.push(right);
            }
        }
        in_pcm.extend_from_slice(&pcm);
        // Cycle the budget: small frames force band skipping and (stereo)
        // intensity clamping in the allocator's explicitly coded decisions.
        let nb_bytes = [159, 25, 47, 159, 80, 251][f % 6];
        let payload = enc.encode_frame(&pcm, nb_bytes);
        let mut packet = vec![toc];
        packet.extend_from_slice(&payload);
        let out = dec.decode_packet(&packet).unwrap();
        out_pcm.extend_from_slice(&out);
        if dec.final_range() != enc.final_range() {
            mismatches += 1;
            if mismatches < 4 {
                println!(
                    "ch{channels} frame {f}: range mismatch enc={} dec={}",
                    enc.final_range(),
                    dec.final_range()
                );
            }
        }
        bit.extend_from_slice(&(packet.len() as u32).to_be_bytes());
        bit.extend_from_slice(&enc.final_range().to_be_bytes());
        bit.extend_from_slice(&packet);
    }
    std::fs::write(bit_path, &bit).unwrap();
    println!("[{channels}ch] range mismatches: {mismatches}/100");
    // SNR vs input at the codec delay (one MDCT overlap, 120 samples),
    // skipping the first frames for warmup.
    let lag = 120 * channels;
    let skip = 4800 * channels;
    let (mut sig, mut noise, mut dot, mut e_out) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in skip..in_pcm.len() - lag {
        let a = f64::from(in_pcm[i]);
        let b = f64::from(out_pcm[i + lag]);
        sig += a * a;
        e_out += b * b;
        dot += a * b;
        noise += (a - b) * (a - b);
    }
    println!(
        "[{channels}ch] snr {:.1} dB corr {:.3} gain {:.3} (at codec delay)",
        10.0 * (sig / noise.max(1e-30)).log10(),
        dot / (sig.sqrt() * e_out.sqrt()).max(1e-30),
        (e_out / sig).sqrt()
    );
    mismatches
}

fn main() {
    // TOC: config 31 (CELT FB 20 ms), code 0; bit 2 set for stereo.
    let m = run(1, 0xF8, "/tmp/ours.bit");
    let s = run(2, 0xFC, "/tmp/ours_stereo.bit");
    assert_eq!(m + s, 0, "encoder/decoder final-range mismatch");
}
