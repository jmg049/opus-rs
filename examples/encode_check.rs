//! Round-trip check: our CELT encoder -> our decoder, plus a .bit dump
//! (with recorded final ranges) for opus_demo / libopus verification.
use opus_native::OpusDecoder;
use opus_native::celt::encoder::CeltEncoder;

fn main() {
    let mut enc = CeltEncoder::new();
    let mut dec = OpusDecoder::new(1);
    let mut bit = Vec::new();
    let mut in_pcm = Vec::new();
    let mut out_pcm = Vec::new();
    let mut mismatches = 0;
    for f in 0..100 {
        let pcm: Vec<f32> = (0..960)
            .map(|i| {
                let t = (f * 960 + i) as f32 / 48000.0;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 1800.0 * t).sin()
            })
            .collect();
        in_pcm.extend_from_slice(&pcm);
        let payload = enc.encode_frame(&pcm, 159);
        // TOC: config 31 (CELT FB 20 ms), mono, code 0.
        let mut packet = vec![0xF8u8];
        packet.extend_from_slice(&payload);
        let out = dec.decode_packet(&packet).unwrap();
        out_pcm.extend_from_slice(&out);
        if dec.final_range() != enc.final_range() {
            mismatches += 1;
            if mismatches < 4 {
                println!(
                    "frame {f}: range mismatch enc={} dec={}",
                    enc.final_range(),
                    dec.final_range()
                );
            }
        }
        bit.extend_from_slice(&(packet.len() as u32).to_be_bytes());
        bit.extend_from_slice(&enc.final_range().to_be_bytes());
        bit.extend_from_slice(&packet);
    }
    std::fs::write("/tmp/ours.bit", &bit).unwrap();
    println!("range mismatches: {mismatches}/100");
    // SNR vs input (skip the first frames for the MDCT warmup).
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for i in 4800..in_pcm.len() {
        sig += f64::from(in_pcm[i]) * f64::from(in_pcm[i]);
        noise += f64::from(out_pcm[i] - in_pcm[i]) * f64::from(out_pcm[i] - in_pcm[i]);
    }
    println!("SNR vs input: {:.1} dB", 10.0 * (sig / noise.max(1e-30)).log10());
    // Alignment/scale diagnostic.
    for lag in [0usize, 120, 240, 480, 960] {
        let (mut sig, mut noise, mut dot, mut e_out) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 4800..in_pcm.len() - lag {
            let a = f64::from(in_pcm[i]);
            let b = f64::from(out_pcm[i + lag]);
            sig += a * a;
            e_out += b * b;
            dot += a * b;
            noise += (a - b) * (a - b);
        }
        println!(
            "lag {lag}: snr {:.1} dB corr {:.3} gain {:.3}",
            10.0 * (sig / noise.max(1e-30)).log10(),
            dot / (sig.sqrt() * e_out.sqrt()).max(1e-30),
            (e_out / sig).sqrt()
        );
    }
}
