//! Public SILK encoder driver (RFC 6716 §5.2; normative `silk/enc_API.c`).
//!
//! [`SilkEncoder`] wraps the per-frame [`SilkChannelEncoder`] with the SILK
//! payload framing: the per-frame VAD flag and the LBRR flag precede the
//! coded frame. This is the mono, single-frame-per-packet path (10 or 20 ms);
//! it produces a range-coded SILK payload that [`crate::silk::SilkDecoder`]
//! (and libopus) decode. Frames are always coded active (no DTX) and without
//! in-band FEC, so the header is a fixed `VAD=1, LBRR=0`.

extern crate alloc;
use alloc::vec::Vec;

use crate::range::RangeEncoder;

use super::super::indices::CondCoding;
use super::frame::SilkChannelEncoder;

/// A SILK encoder for one mono stream.
pub struct SilkEncoder {
    ch: SilkChannelEncoder,
}

impl SilkEncoder {
    /// A new encoder at the given internal rate (`fs_khz` ∈ {8, 12, 16}) and
    /// subframe count (`nb_subfr` = 4 for 20 ms, 2 for 10 ms).
    #[must_use]
    pub fn new(fs_khz: i32, nb_subfr: usize) -> Self {
        SilkEncoder {
            ch: SilkChannelEncoder::new(fs_khz, nb_subfr),
        }
    }

    /// Sets the target bitrate (bps), which maps to the per-frame coding SNR.
    pub fn set_bitrate(&mut self, bps: i32) {
        self.ch.set_bitrate(bps);
    }

    /// Encodes one frame of `input` (i16 PCM at the internal rate,
    /// `nb_subfr * 5 * fs_khz` samples) into a SILK payload.
    ///
    /// # Panics
    ///
    /// Panics if `input` is not exactly one frame, or if the coded frame does
    /// not fit the range coder (it always does for valid inputs).
    #[must_use]
    pub fn encode(&mut self, input: &[i16]) -> Vec<u8> {
        let mut enc = RangeEncoder::new(1275);
        // Header: VAD flag (active) then LBRR flag (no in-band FEC).
        enc.encode_bit_logp(true, 1);
        enc.encode_bit_logp(false, 1);
        self.ch.encode_frame(&mut enc, input, CondCoding::Independently);
        enc.finalize().expect("SILK frame fits the range coder")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::RangeDecoder;
    use crate::silk::api::{DecControl, SilkDecoder};
    use alloc::vec;

    /// A mono SILK payload decodes through the full `SilkDecoder` API and
    /// reproduces the encoder's reconstruction. With the internal rate equal
    /// to the API rate the output resampler is a pure delay, so `out` equals
    /// the encoder's NSQ output `xq` shifted by that (small) delay.
    #[test]
    fn mono_payload_round_trips_through_the_silk_decoder() {
        let (fs_khz, nb_subfr) = (16i32, 4usize);
        let frame_length = nb_subfr * 5 * fs_khz as usize;
        let ltp_mem = 20 * fs_khz as usize;

        let mut seed = 0x7331_u32;
        let input: Vec<i16> = (0..frame_length)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let n = ((seed >> 20) as i32 - 2048) / 4;
                let tone = ((i as f32 * 0.13).sin() * 2000.0) as i32;
                (n + tone).clamp(-30000, 30000) as i16
            })
            .collect();

        let mut e = SilkEncoder::new(fs_khz, nb_subfr);
        e.set_bitrate(24000);
        let bytes = e.encode(&input);
        assert!(!bytes.is_empty());
        let xq_enc: Vec<i16> = e.ch.nsq.xq[ltp_mem..ltp_mem + frame_length].to_vec();

        let ctl = DecControl {
            channels_internal: 1,
            channels_api: 1,
            internal_sample_rate: 16000,
            api_sample_rate: 16000,
            payload_size_ms: 20,
        };
        let mut d = SilkDecoder::new();
        let mut dec = RangeDecoder::new(&bytes);
        let mut out: Vec<i16> = vec![];
        d.decode(&mut dec, &ctl, true, &mut out);

        assert_eq!(out.len(), frame_length, "one frame of output");
        // The output resampler imposes a pure delay; find it and confirm the
        // decoded signal equals the encoder's reconstruction beyond it.
        let delay = (0..=16usize)
            .find(|&d| out[d..] == xq_enc[..frame_length - d])
            .expect("decoded output matches the encoder reconstruction at some small delay");
        assert!(delay <= 16, "unexpected resampler delay {delay}");
        assert!(out[..delay].iter().all(|&v| v == 0), "pre-delay samples are zero");
    }
}
