//! Per-frame SILK encode assembly (RFC 6716 §5.2; normative
//! `silk/float/encode_frame_FLP.c`).
//!
//! [`encode_frame_unvoiced`] is the first end-to-end SILK encode path: it
//! ties together the analysis and quantisation kernels into a single coded
//! frame and is validated by round-tripping through the bit-exact decoder.
//! It targets the simplest legal configuration - unvoiced, no NLSF
//! interpolation, flat noise shaping - which exercises the whole chain
//! (LPC → NLSF VQ → gains → NSQ → index/pulse bitstream) and produces a
//! frame the decoder reconstructs sample-for-sample. Voiced coding, real
//! noise shaping, and the higher-level mode/stereo glue build on top.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::range::RangeEncoder;

use super::super::indices::{
    CondCoding, EcPrevState, MAX_LPC_ORDER, MAX_NB_SUBFR, SideInfoIndices, TYPE_UNVOICED, encode_indices, nlsf_codebook,
};
use super::super::lpc::lpc_analysis_filter;
use super::super::nlsf::nlsf2a;
use super::super::pulses::encode_pulses;
use super::gains::process_gains;
use super::lpc::burg_modified;
use super::nlsf::{a2nlsf, nlsf_encode, nlsf_vq_weights_laroia};
use super::noise_shape::{NoiseShapeConfig, ShapeState, noise_shape_analysis};
use super::nsq::{NsqConfig, NsqState, nsq};

/// One channel's SILK encoder state for the (unvoiced) frame path.
pub(crate) struct SilkChannelEncoder {
    pub nsq: NsqState,
    /// Cross-frame noise-shaping smoothing state (`sShape`).
    pub shape: ShapeState,
    /// Gain-quantiser accumulator (`sShape.LastGainIndex`).
    pub last_gain_index: i8,
    /// Entropy-coding history for [`encode_indices`].
    pub ec_prev: EcPrevState,
    pub fs_khz: i32,
    pub nb_subfr: usize,
}

impl SilkChannelEncoder {
    /// A reset encoder for the given internal rate and subframe count.
    #[must_use]
    pub(crate) fn new(fs_khz: i32, nb_subfr: usize) -> Self {
        SilkChannelEncoder {
            nsq: NsqState::new(),
            shape: ShapeState::default(),
            last_gain_index: 10,
            ec_prev: EcPrevState::default(),
            fs_khz,
            nb_subfr,
        }
    }

    /// Encodes one unvoiced frame of `input` (i16 PCM at the internal rate,
    /// `frame_length` samples) into `enc`. Returns the coded `SideInfoIndices`
    /// (for inspection/round-trip checks).
    pub(crate) fn encode_frame_unvoiced(
        &mut self,
        enc: &mut RangeEncoder,
        input: &[i16],
        cond_coding: CondCoding,
    ) -> SideInfoIndices {
        let order = if self.fs_khz == 16 { 16 } else { 10 };
        let subfr_length = 5 * self.fs_khz as usize;
        let frame_length = self.nb_subfr * subfr_length;
        debug_assert_eq!(input.len(), frame_length);

        // Short-term analysis: Burg LPC over the frame.
        let x_f: Vec<f32> = input.iter().map(|&v| f32::from(v)).collect();
        let mut lpc = [0.0f32; MAX_LPC_ORDER];
        burg_modified(&mut lpc[..order], &x_f, 1.0 / 1e4, frame_length, 1, order);

        // LPC → NLSF → VQ-quantised indices (req. NLSF written back).
        let cb = nlsf_codebook(self.fs_khz);
        let mut nlsf_q15: Vec<i16> = a2nlsf(&lpc[..order]);
        let mut w_q2 = [0i16; MAX_LPC_ORDER];
        nlsf_vq_weights_laroia(&mut w_q2[..order], &nlsf_q15, order);
        let (nlsf_indices, _) = nlsf_encode(&mut nlsf_q15, cb, &w_q2[..order], 1 << 14, 4, TYPE_UNVOICED as usize);

        // Quantised LPC (Q12) for NSQ - exactly what the decoder rebuilds.
        let mut pred_coef = [0i16; 2 * MAX_LPC_ORDER];
        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        nlsf2a(&mut a_q12[..order], &nlsf_q15[..order]);
        pred_coef[..order].copy_from_slice(&a_q12[..order]);
        pred_coef[MAX_LPC_ORDER..MAX_LPC_ORDER + order].copy_from_slice(&a_q12[..order]);

        // Noise-shaping analysis (complexity-0 configuration: no warping, so
        // the plain NSQ sees ordinary shaping coefficients). It produces the
        // AR/tilt/LF shapers and the pre-quantisation per-subframe gains.
        let la_shape = 3 * self.fs_khz as usize;
        let shaping_lpc_order = 12.min(order);
        let shape_win_length = subfr_length + 2 * la_shape;
        let snr_db_q7 = 18 << 7;

        // Sparseness measure uses the LPC residual over the frame as a stand-in
        // for the pitch-analysis residual (the unvoiced path has no pitch pass).
        let mut residual = vec![0i16; frame_length];
        lpc_analysis_filter(&mut residual, input, &a_q12[..order]);
        let pitch_res: Vec<f32> = residual.iter().map(|&r| f32::from(r)).collect();

        // The analysis window needs `la_shape` of history and lookahead; this
        // isolated frame path zero-pads both ends.
        let mut x_buf = vec![0.0f32; frame_length + 2 * la_shape];
        for (i, &v) in input.iter().enumerate() {
            x_buf[la_shape + i] = f32::from(v);
        }
        let shape_cfg = NoiseShapeConfig {
            fs_khz: self.fs_khz,
            nb_subfr: self.nb_subfr,
            subfr_length,
            la_shape,
            shape_win_length,
            shaping_lpc_order,
            warping_q16: 0,
            signal_type: TYPE_UNVOICED,
            snr_db_q7,
            speech_activity_q8: 256,
            input_quality_bands_q15: [32768, 32768],
            use_cbr: true,
            ltp_corr: 0.0,
            pred_gain: 0.0,
            input_tilt_q15: 0,
            pitch_l: [0; MAX_NB_SUBFR],
        };
        let shp = noise_shape_analysis(&mut self.shape, &shape_cfg, &pitch_res, &x_buf);

        // Residual energy on the gain-normalised signal (per `silk_residual_
        // energy_FLP`): ResNrg[k] = Gains[k]^2 * energy(LPC residual of x/Gain).
        let a_f: Vec<f32> = a_q12[..order].iter().map(|&c| f32::from(c) / 4096.0).collect();
        let mut x_hist = vec![0.0f32; order + frame_length];
        for (i, &v) in input.iter().enumerate() {
            x_hist[order + i] = f32::from(v);
        }
        let mut gains = shp.gains;
        let mut res_nrg = [0.0f32; MAX_NB_SUBFR];
        for k in 0..self.nb_subfr {
            let inv_gain = 1.0 / gains[k];
            let base = k * subfr_length;
            let mut nrg = 0.0f64;
            for n in 0..subfr_length {
                let p = base + order + n;
                let mut acc = x_hist[p] * inv_gain;
                for (j, &aj) in a_f.iter().enumerate() {
                    acc -= aj * x_hist[p - 1 - j] * inv_gain;
                }
                nrg += f64::from(acc) * f64::from(acc);
            }
            res_nrg[k] = (f64::from(gains[k]) * f64::from(gains[k]) * nrg) as f32;
        }

        let gres = process_gains(
            &mut gains,
            &res_nrg,
            TYPE_UNVOICED,
            shp.quant_offset_type,
            self.nb_subfr,
            subfr_length,
            snr_db_q7,
            0.0,
            0,
            1,
            256,
            shp.input_quality,
            shp.coding_quality,
            &mut self.last_gain_index,
            cond_coding,
        );

        let ltp_coef = [0i16; 5 * MAX_NB_SUBFR];
        let pitch_l = [0i32; MAX_NB_SUBFR];
        let seed = 0i32;

        let cfg = NsqConfig {
            frame_length,
            subfr_length,
            nb_subfr: self.nb_subfr,
            ltp_mem_length: 20 * self.fs_khz as usize,
            predict_lpc_order: order,
            shaping_lpc_order,
        };
        let mut pulses = vec![0i8; frame_length];
        let lambda_q10 = (gres.lambda * 1024.0) as i32;
        nsq(
            &mut self.nsq,
            &cfg,
            TYPE_UNVOICED,
            gres.quant_offset_type,
            4,
            seed,
            input,
            &mut pulses,
            &pred_coef,
            &ltp_coef,
            &shp.ar_q13,
            &shp.harm_shape_gain_q14,
            &shp.tilt_q14,
            &shp.lf_shp_q14,
            &gres.gains_q16,
            &pitch_l,
            lambda_q10,
            0,
        );

        // Assemble the side info and write the frame.
        let mut indices = SideInfoIndices {
            signal_type: TYPE_UNVOICED as i8,
            quant_offset_type: gres.quant_offset_type as i8,
            nlsf_interp_coef_q2: 4,
            seed: seed as i8,
            ..SideInfoIndices::default()
        };
        indices.gains_indices[..self.nb_subfr].copy_from_slice(&gres.gains_indices[..self.nb_subfr]);
        indices.nlsf_indices[..=order].copy_from_slice(&nlsf_indices[..=order]);

        encode_indices(
            enc,
            &indices,
            self.fs_khz,
            self.nb_subfr,
            false,
            true,
            cond_coding,
            &mut self.ec_prev,
        );
        encode_pulses(enc, TYPE_UNVOICED, gres.quant_offset_type, &pulses, frame_length);
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::decoder::SilkChannelDecoder;
    use super::*;
    use crate::range::RangeDecoder;

    /// End-to-end: an unvoiced SILK frame encoded by our encoder decodes
    /// through the bit-exact decoder, reconstructing the encoder's own
    /// quantised signal and tracking the input.
    #[test]
    fn unvoiced_frame_round_trips_through_the_decoder() {
        let fs_khz = 16i32;
        let nb_subfr = 4usize;
        let subfr = 5 * fs_khz as usize;
        let frame_length = nb_subfr * subfr;

        // A noise-like (unvoiced) input.
        let mut seed = 0x9e37_u32;
        let input: Vec<i16> = (0..frame_length)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let n = ((seed >> 16) as i32 - 32768) / 12;
                let tone = ((i as f32 * 0.11).sin() * 2500.0) as i32;
                (n + tone).clamp(-30000, 30000) as i16
            })
            .collect();

        let mut e = SilkChannelEncoder::new(fs_khz, nb_subfr);
        let mut enc = RangeEncoder::new(512);
        let _ind = e.encode_frame_unvoiced(&mut enc, &input, CondCoding::Independently);
        // The encoder's own reconstruction of this frame.
        let xq_enc: Vec<i16> = e.nsq.xq[20 * fs_khz as usize..20 * fs_khz as usize + frame_length].to_vec();
        let bytes = enc.finalize().expect("frame fits");
        assert!(!bytes.is_empty());

        // Decode it back.
        let mut d = SilkChannelDecoder::new(fs_khz, nb_subfr);
        let mut dec = RangeDecoder::new(&bytes);
        let mut xq = vec![0i16; frame_length];
        d.decode_frame(&mut dec, &mut xq, true, false, CondCoding::Independently);

        // The decoder reproduces the encoder's quantised signal.
        assert_eq!(
            xq, xq_enc,
            "decoder output disagrees with the encoder's NSQ reconstruction"
        );
        // And it tracks the input (lossy, but correlated and bounded).
        let (mut sig, mut dot, mut e_out) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..frame_length {
            let a = f64::from(input[i]);
            let b = f64::from(xq[i]);
            sig += a * a;
            dot += a * b;
            e_out += b * b;
        }
        let corr = dot / (sig.sqrt() * e_out.sqrt()).max(1.0);
        assert!(corr > 0.7, "reconstruction correlation {corr:.3} too low");
    }
}
