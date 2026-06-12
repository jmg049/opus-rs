//! SILK packet-loss concealment and comfort-noise generation
//! (normative `PLC.c`, `CNG.c`, `sum_sqr_shift.c`).
//!
//! Concealment extrapolates the last frame's LTP/LPC model over a noise
//! excitation drawn from the quietest recent subframe, with per-frame
//! attenuation and pitch drift; comfort noise (estimated during inactive
//! good frames) is mixed into concealed output; `glue_frames` fades the
//! first good frame back in when its energy jumps past the concealment's.

use alloc::vec;

use super::decoder::{MAX_FRAME_LENGTH, SilkChannelDecoder, silk_rand};
use super::indices::{MAX_LPC_ORDER, MAX_NB_SUBFR, TYPE_NO_VOICE_ACTIVITY, TYPE_VOICED};
use super::lpc::{lpc_analysis_filter, lpc_inverse_pred_gain};
use super::math::{
    add_rshift_uint, add_sat16, add_sat32, clz32, inverse32_var_q, lshift_sat32, rshift_round, smlabb, smlawb, smulbb,
    smultt, smulwb, smulww, sqrt_approx,
};
use super::nlsf::nlsf2a;
use super::params::{DecoderControl, LTP_ORDER, bwexpander};

/// `HARM_ATT_Q15` (0.99, 0.95) and the random-component attenuations.
const HARM_ATT_Q15: [i32; 2] = [32440, 31130];
const PLC_RAND_ATTENUATE_V_Q15: [i32; 2] = [31130, 26214];
const PLC_RAND_ATTENUATE_UV_Q15: [i32; 2] = [32440, 29491];

/// `BWE_COEF` in Q16 (`SILK_FIX_CONST(0.99, 16)`).
const BWE_COEF_Q16: i32 = 64881;
/// `V_PITCH_GAIN_START_{MIN,MAX}_Q14`.
const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469;
const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565;
/// `RAND_BUF_SIZE` / mask.
const RAND_BUF_SIZE: usize = 128;
/// `LOG2_INV_LPC_GAIN_{HIGH,LOW}_THRES`.
const LOG2_INV_LPC_GAIN_HIGH_THRES: i32 = 3;
const LOG2_INV_LPC_GAIN_LOW_THRES: i32 = 8;
/// `PITCH_DRIFT_FAC_Q16` (0.01).
const PITCH_DRIFT_FAC_Q16: i32 = 655;
/// `MAX_PITCH_LAG_MS`.
const MAX_PITCH_LAG_MS: i32 = 18;
/// `CNG_BUF_MASK_MAX`, `CNG_NLSF_SMTH_Q16`, `CNG_GAIN_SMTH_Q16`,
/// `CNG_GAIN_SMTH_THRESHOLD_Q16` (define.h).
const CNG_BUF_MASK_MAX: i32 = 255;
const CNG_NLSF_SMTH_Q16: i32 = 16348;
const CNG_GAIN_SMTH_Q16: i32 = 4634;
const CNG_GAIN_SMTH_THRESHOLD_Q16: i32 = 46396;

/// `silk_sum_sqr_shift`: energy of an i16 vector and the right shift that
/// makes it fit an i32 with headroom.
pub(crate) fn sum_sqr_shift(x: &[i16]) -> (i32, i32) {
    let len = x.len() as i32;
    let mut shft = 31 - clz32(len);
    // Conservative rounding: start with nrg = len.
    let mut nrg = len as u32;
    let mut i = 0usize;
    while i + 1 < x.len() {
        let mut t = smulbb(i32::from(x[i]), i32::from(x[i]));
        t = smlabb(t, i32::from(x[i + 1]), i32::from(x[i + 1]));
        nrg = add_rshift_uint(nrg, t as u32, shft);
        i += 2;
    }
    if i < x.len() {
        let t = smulbb(i32::from(x[i]), i32::from(x[i]));
        nrg = add_rshift_uint(nrg, t as u32, shft);
    }
    shft = 0.max(shft + 3 - clz32(nrg as i32));
    let mut nrg = 0u32;
    let mut i = 0usize;
    while i + 1 < x.len() {
        let mut t = smulbb(i32::from(x[i]), i32::from(x[i]));
        t = smlabb(t, i32::from(x[i + 1]), i32::from(x[i + 1]));
        nrg = add_rshift_uint(nrg, t as u32, shft);
        i += 2;
    }
    if i < x.len() {
        let t = smulbb(i32::from(x[i]), i32::from(x[i]));
        nrg = add_rshift_uint(nrg, t as u32, shft);
    }
    (nrg as i32, shft)
}

/// Concealment state (`silk_PLC_struct`).
#[derive(Debug, Clone)]
pub(crate) struct PlcState {
    pub pitch_l_q8: i32,
    pub ltp_coef_q14: [i16; LTP_ORDER],
    pub prev_lpc_q12: [i16; MAX_LPC_ORDER],
    pub last_frame_lost: bool,
    pub rand_seed: i32,
    pub rand_scale_q14: i16,
    pub conc_energy: i32,
    pub conc_energy_shift: i32,
    pub prev_ltp_scale_q14: i16,
    pub prev_gain_q16: [i32; 2],
    pub fs_khz: i32,
    pub nb_subfr: usize,
    pub subfr_length: usize,
}

impl Default for PlcState {
    fn default() -> Self {
        PlcState {
            pitch_l_q8: 0,
            ltp_coef_q14: [0; LTP_ORDER],
            prev_lpc_q12: [0; MAX_LPC_ORDER],
            last_frame_lost: false,
            rand_seed: 0,
            rand_scale_q14: 0,
            conc_energy: 0,
            conc_energy_shift: 0,
            prev_ltp_scale_q14: 0,
            prev_gain_q16: [1 << 16; 2],
            fs_khz: 0,
            nb_subfr: 2,
            subfr_length: 20,
        }
    }
}

/// Comfort-noise state (`silk_CNG_struct`).
#[derive(Debug, Clone)]
pub(crate) struct CngState {
    pub exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    pub smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub synth_state: [i32; MAX_LPC_ORDER],
    pub smth_gain_q16: i32,
    pub rand_seed: i32,
    pub fs_khz: i32,
}

impl Default for CngState {
    fn default() -> Self {
        CngState {
            exc_buf_q14: [0; MAX_FRAME_LENGTH],
            smth_nlsf_q15: [0; MAX_LPC_ORDER],
            synth_state: [0; MAX_LPC_ORDER],
            smth_gain_q16: 0,
            rand_seed: 3_176_576,
            fs_khz: 0,
        }
    }
}

impl SilkChannelDecoder {
    /// `silk_PLC_Reset`.
    pub(crate) fn plc_reset(&mut self) {
        self.plc.pitch_l_q8 = (self.frame_length as i32) << 7;
        self.plc.prev_gain_q16 = [1 << 16; 2];
        self.plc.subfr_length = 20;
        self.plc.nb_subfr = 2;
    }

    /// `silk_CNG_Reset`.
    pub(crate) fn cng_reset(&mut self) {
        let step_q15 = i32::from(i16::MAX) / (self.lpc_order as i32 + 1);
        let mut acc_q15 = 0;
        for v in self.cng.smth_nlsf_q15.iter_mut().take(self.lpc_order) {
            acc_q15 += step_q15;
            *v = acc_q15 as i16;
        }
        self.cng.smth_gain_q16 = 0;
        self.cng.rand_seed = 3_176_576;
    }

    /// `silk_PLC`: update on good frames, conceal on lost ones.
    pub(crate) fn plc(&mut self, ctrl: &mut DecoderControl, frame: &mut [i16], lost: bool) {
        if self.fs_khz != self.plc.fs_khz {
            self.plc_reset();
            self.plc.fs_khz = self.fs_khz;
        }
        if lost {
            self.plc_conceal(ctrl, frame);
            self.loss_cnt += 1;
        } else {
            self.plc_update(ctrl);
        }
    }

    /// `silk_PLC_update`: capture the model of the last good frame.
    fn plc_update(&mut self, ctrl: &DecoderControl) {
        self.prev_signal_type = i32::from(self.indices.signal_type);
        let mut ltp_gain_q14 = 0i32;
        if i32::from(self.indices.signal_type) == TYPE_VOICED {
            // The last subframe containing a pitch pulse.
            let mut j = 0usize;
            while (j * self.subfr_length) < ctrl.pitch_l[self.nb_subfr - 1] as usize {
                if j == self.nb_subfr {
                    break;
                }
                let mut temp_q14 = 0i32;
                for i in 0..LTP_ORDER {
                    temp_q14 += i32::from(ctrl.ltp_coef_q14[(self.nb_subfr - 1 - j) * LTP_ORDER + i]);
                }
                if temp_q14 > ltp_gain_q14 {
                    ltp_gain_q14 = temp_q14;
                    let base = (self.nb_subfr - 1 - j) * LTP_ORDER;
                    self.plc
                        .ltp_coef_q14
                        .copy_from_slice(&ctrl.ltp_coef_q14[base..base + LTP_ORDER]);
                    self.plc.pitch_l_q8 = ctrl.pitch_l[self.nb_subfr - 1 - j] << 8;
                }
                j += 1;
            }

            self.plc.ltp_coef_q14 = [0; LTP_ORDER];
            self.plc.ltp_coef_q14[LTP_ORDER / 2] = ltp_gain_q14 as i16;

            // Limit the LT coefficients.
            if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
                let scale_q10 = (V_PITCH_GAIN_START_MIN_Q14 << 10) / ltp_gain_q14.max(1);
                for c in &mut self.plc.ltp_coef_q14 {
                    *c = (smulbb(i32::from(*c), scale_q10) >> 10) as i16;
                }
            } else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
                let scale_q14 = (V_PITCH_GAIN_START_MAX_Q14 << 14) / ltp_gain_q14.max(1);
                for c in &mut self.plc.ltp_coef_q14 {
                    *c = (smulbb(i32::from(*c), scale_q14) >> 14) as i16;
                }
            }
        } else {
            self.plc.pitch_l_q8 = smulbb(self.fs_khz, 18) << 8;
            self.plc.ltp_coef_q14 = [0; LTP_ORDER];
        }

        self.plc.prev_lpc_q12 = ctrl.pred_coef_q12[1];
        self.plc.prev_ltp_scale_q14 = ctrl.ltp_scale_q14 as i16;
        self.plc.prev_gain_q16 = [ctrl.gains_q16[self.nb_subfr - 2], ctrl.gains_q16[self.nb_subfr - 1]];
        self.plc.subfr_length = self.subfr_length;
        self.plc.nb_subfr = self.nb_subfr;
    }

    /// `silk_PLC_conceal`: synthesize one concealed frame into `frame`.
    #[allow(clippy::too_many_lines, reason = "mirrors the reference sequence")]
    fn plc_conceal(&mut self, ctrl: &mut DecoderControl, frame: &mut [i16]) {
        let prev_gain_q10 = [self.plc.prev_gain_q16[0] >> 6, self.plc.prev_gain_q16[1] >> 6];
        if self.params.first_frame_after_reset {
            self.plc.prev_lpc_q12 = [0; MAX_LPC_ORDER];
        }

        // The quietest of the last two subframes drives the noise.
        let (energy1, shift1, energy2, shift2) = {
            let mut exc_buf = vec![0i16; 2 * self.subfr_length];
            for k in 0..2 {
                for i in 0..self.subfr_length {
                    exc_buf[k * self.subfr_length + i] = (smulww(
                        self.exc_q14[i + (k + self.nb_subfr - 2) * self.subfr_length],
                        prev_gain_q10[k],
                    ) >> 8)
                        .clamp(i32::from(i16::MIN), i32::from(i16::MAX))
                        as i16;
                }
            }
            let (e1, s1) = sum_sqr_shift(&exc_buf[..self.subfr_length]);
            let (e2, s2) = sum_sqr_shift(&exc_buf[self.subfr_length..]);
            (e1, s1, e2, s2)
        };
        let rand_base = if (energy1 >> shift2) < (energy2 >> shift1) {
            // First subframe has the lowest energy.
            0.max((self.plc.nb_subfr as i32 - 1) * self.plc.subfr_length as i32 - RAND_BUF_SIZE as i32) as usize
        } else {
            0.max(self.plc.nb_subfr as i32 * self.plc.subfr_length as i32 - RAND_BUF_SIZE as i32) as usize
        };

        let mut b_q14 = self.plc.ltp_coef_q14;
        let mut rand_scale_q14 = i32::from(self.plc.rand_scale_q14);

        // Attenuation gains.
        let att = 1.min(self.loss_cnt as usize);
        let harm_gain_q15 = HARM_ATT_Q15[att];
        let mut rand_gain_q15 = if self.prev_signal_type == TYPE_VOICED {
            PLC_RAND_ATTENUATE_V_Q15[att]
        } else {
            PLC_RAND_ATTENUATE_UV_Q15[att]
        };

        // LPC concealment: bandwidth-expand the previous LPC.
        bwexpander(&mut self.plc.prev_lpc_q12[..self.lpc_order], BWE_COEF_Q16);
        let a_q12 = self.plc.prev_lpc_q12;

        // First lost frame.
        if self.loss_cnt == 0 {
            rand_scale_q14 = 1 << 14;
            if self.prev_signal_type == TYPE_VOICED {
                for &b in &b_q14 {
                    rand_scale_q14 -= i32::from(b);
                }
                rand_scale_q14 = rand_scale_q14.max(3277); // 0.2
                rand_scale_q14 = smulbb(rand_scale_q14, i32::from(self.plc.prev_ltp_scale_q14)) >> 14;
            } else {
                // Reduce noise for unvoiced frames with high LPC gain.
                let inv_gain_q30 = lpc_inverse_pred_gain(&self.plc.prev_lpc_q12[..self.lpc_order]);
                let mut down_scale_q30 = ((1i32 << 30) >> LOG2_INV_LPC_GAIN_HIGH_THRES).min(inv_gain_q30);
                down_scale_q30 = ((1i32 << 30) >> LOG2_INV_LPC_GAIN_LOW_THRES).max(down_scale_q30);
                down_scale_q30 <<= LOG2_INV_LPC_GAIN_HIGH_THRES;
                rand_gain_q15 = smulwb(down_scale_q30, rand_gain_q15) >> 14;
            }
        }

        let mut rand_seed = self.plc.rand_seed;
        let mut lag = rshift_round(self.plc.pitch_l_q8, 8);
        let mut s_ltp_buf_idx = self.ltp_mem_length;

        // Rewhiten the LTP state.
        let mut s_ltp = vec![0i16; self.ltp_mem_length];
        let mut s_ltp_q14 = vec![0i32; self.ltp_mem_length + self.frame_length];
        let idx = self.ltp_mem_length as i32 - lag - self.lpc_order as i32 - (LTP_ORDER / 2) as i32;
        debug_assert!(idx > 0);
        let idx = idx as usize;
        lpc_analysis_filter(
            &mut s_ltp[idx..self.ltp_mem_length],
            &self.out_buf[idx..self.ltp_mem_length],
            &a_q12[..self.lpc_order],
        );
        let inv_gain_q30 = inverse32_var_q(self.plc.prev_gain_q16[1], 46).min(i32::MAX >> 1);
        for i in idx + self.lpc_order..self.ltp_mem_length {
            s_ltp_q14[i] = smulwb(inv_gain_q30, i32::from(s_ltp[i]));
        }

        // LTP synthesis over the noise excitation.
        for _k in 0..self.nb_subfr {
            for _i in 0..self.subfr_length {
                let p = s_ltp_buf_idx - lag as usize + LTP_ORDER / 2;
                let mut ltp_pred_q12 = 2i32;
                for (t, &b) in b_q14.iter().enumerate() {
                    ltp_pred_q12 = smlawb(ltp_pred_q12, s_ltp_q14[p - t], i32::from(b));
                }
                rand_seed = silk_rand(rand_seed);
                let ridx = ((rand_seed >> 25) & (RAND_BUF_SIZE as i32 - 1)) as usize;
                s_ltp_q14[s_ltp_buf_idx] = smlawb(ltp_pred_q12, self.exc_q14[rand_base + ridx], rand_scale_q14) << 2;
                s_ltp_buf_idx += 1;
            }

            // Gradually reduce the LTP and excitation gains; drift the lag.
            for b in &mut b_q14 {
                *b = (smulbb(harm_gain_q15, i32::from(*b)) >> 15) as i16;
            }
            rand_scale_q14 = smulbb(rand_scale_q14, rand_gain_q15) >> 15;
            self.plc.pitch_l_q8 = smlawb(self.plc.pitch_l_q8, self.plc.pitch_l_q8, PITCH_DRIFT_FAC_Q16);
            self.plc.pitch_l_q8 = self.plc.pitch_l_q8.min(smulbb(MAX_PITCH_LAG_MS, self.fs_khz) << 8);
            lag = rshift_round(self.plc.pitch_l_q8, 8);
        }

        // LPC synthesis filtering over the extrapolated excitation.
        let base = self.ltp_mem_length - MAX_LPC_ORDER;
        s_ltp_q14[base..base + MAX_LPC_ORDER].copy_from_slice(&self.slpc_q14_buf);
        for i in 0..self.frame_length {
            let mut lpc_pred_q10 = (self.lpc_order as i32) >> 1;
            for (j, &a) in a_q12.iter().enumerate().take(self.lpc_order) {
                lpc_pred_q10 = smlawb(lpc_pred_q10, s_ltp_q14[base + MAX_LPC_ORDER + i - j - 1], i32::from(a));
            }
            s_ltp_q14[base + MAX_LPC_ORDER + i] =
                add_sat32(s_ltp_q14[base + MAX_LPC_ORDER + i], lshift_sat32(lpc_pred_q10, 4));
            frame[i] = rshift_round(smulww(s_ltp_q14[base + MAX_LPC_ORDER + i], prev_gain_q10[1]), 8)
                .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
        self.slpc_q14_buf
            .copy_from_slice(&s_ltp_q14[base + self.frame_length..base + self.frame_length + MAX_LPC_ORDER]);

        // State updates (B_Q14 aliases the persistent LTP coefficients in
        // the reference, so the per-subframe decay carries across losses).
        self.plc.ltp_coef_q14 = b_q14;
        self.plc.rand_seed = rand_seed;
        self.plc.rand_scale_q14 = rand_scale_q14 as i16;
        for p in &mut ctrl.pitch_l[..MAX_NB_SUBFR] {
            *p = lag;
        }
    }

    /// `silk_PLC_glue_frames`: fades the first good frame back in when its
    /// energy exceeds the concealment's.
    pub(crate) fn plc_glue_frames(&mut self, frame: &mut [i16]) {
        if self.loss_cnt != 0 {
            let (e, s) = sum_sqr_shift(frame);
            self.plc.conc_energy = e;
            self.plc.conc_energy_shift = s;
            self.plc.last_frame_lost = true;
        } else {
            if self.plc.last_frame_lost {
                let (mut energy, energy_shift) = sum_sqr_shift(frame);
                if energy_shift > self.plc.conc_energy_shift {
                    self.plc.conc_energy >>= energy_shift - self.plc.conc_energy_shift;
                } else if energy_shift < self.plc.conc_energy_shift {
                    energy >>= self.plc.conc_energy_shift - energy_shift;
                }
                if energy > self.plc.conc_energy {
                    let lz = clz32(self.plc.conc_energy) - 1;
                    self.plc.conc_energy <<= lz;
                    energy >>= 0.max(24 - lz);
                    let frac_q24 = self.plc.conc_energy / energy.max(1);
                    let mut gain_q16 = sqrt_approx(frac_q24) << 4;
                    let slope_q16 = (((1i32 << 16) - gain_q16) / frame.len() as i32) << 2;
                    for v in frame.iter_mut() {
                        *v = smulwb(gain_q16, i32::from(*v)) as i16;
                        gain_q16 += slope_q16;
                        if gain_q16 > 1 << 16 {
                            break;
                        }
                    }
                }
            }
            self.plc.last_frame_lost = false;
        }
    }

    /// `silk_CNG`: updates the comfort-noise estimate on active-silent good
    /// frames, and mixes comfort noise into concealed ones.
    pub(crate) fn cng(&mut self, ctrl: &DecoderControl, frame: &mut [i16]) {
        let length = self.frame_length;
        if self.fs_khz != self.cng.fs_khz {
            self.cng_reset();
            self.cng.fs_khz = self.fs_khz;
        }
        if self.loss_cnt == 0 && self.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
            // Smooth the NLSFs.
            for i in 0..self.lpc_order {
                self.cng.smth_nlsf_q15[i] += smulwb(
                    i32::from(self.params.prev_nlsf_q15[i]) - i32::from(self.cng.smth_nlsf_q15[i]),
                    CNG_NLSF_SMTH_Q16,
                ) as i16;
            }
            // Buffer excitation from the highest-gain subframe.
            let mut subfr = 0usize;
            let mut max_gain_q16 = 0i32;
            for (i, &g) in ctrl.gains_q16.iter().enumerate().take(self.nb_subfr) {
                if g > max_gain_q16 {
                    max_gain_q16 = g;
                    subfr = i;
                }
            }
            self.cng
                .exc_buf_q14
                .copy_within(0..(self.nb_subfr - 1) * self.subfr_length, self.subfr_length);
            self.cng.exc_buf_q14[..self.subfr_length]
                .copy_from_slice(&self.exc_q14[subfr * self.subfr_length..(subfr + 1) * self.subfr_length]);
            // Smooth the gains.
            for &g in ctrl.gains_q16.iter().take(self.nb_subfr) {
                self.cng.smth_gain_q16 += smulwb(g - self.cng.smth_gain_q16, CNG_GAIN_SMTH_Q16);
                if smulww(self.cng.smth_gain_q16, CNG_GAIN_SMTH_THRESHOLD_Q16) > g {
                    self.cng.smth_gain_q16 = g;
                }
            }
        }

        if self.loss_cnt != 0 {
            // Generate comfort noise and add it to the concealed frame.
            let mut gain_q16 = smulww(i32::from(self.plc.rand_scale_q14), self.plc.prev_gain_q16[1]);
            if gain_q16 >= 1 << 21 || self.cng.smth_gain_q16 > 1 << 23 {
                gain_q16 = smultt(gain_q16, gain_q16);
                gain_q16 = smultt(self.cng.smth_gain_q16, self.cng.smth_gain_q16).wrapping_sub(gain_q16 << 5);
                gain_q16 = sqrt_approx(gain_q16) << 16;
            } else {
                gain_q16 = smulww(gain_q16, gain_q16);
                gain_q16 = smulww(self.cng.smth_gain_q16, self.cng.smth_gain_q16).wrapping_sub(gain_q16 << 5);
                gain_q16 = sqrt_approx(gain_q16) << 8;
            }
            let gain_q10 = gain_q16 >> 6;

            // Excitation drawn from the buffered samples.
            let mut sig_q14 = vec![0i32; length + MAX_LPC_ORDER];
            {
                let mut exc_mask = CNG_BUF_MASK_MAX;
                while exc_mask > length as i32 {
                    exc_mask >>= 1;
                }
                let mut seed = self.cng.rand_seed;
                for v in sig_q14[MAX_LPC_ORDER..].iter_mut() {
                    seed = silk_rand(seed);
                    let idx = ((seed >> 24) & exc_mask) as usize;
                    *v = self.cng.exc_buf_q14[idx];
                }
                self.cng.rand_seed = seed;
            }

            // CNG LPC from the smoothed NLSFs.
            let mut a_q12 = [0i16; MAX_LPC_ORDER];
            nlsf2a(&mut a_q12[..self.lpc_order], &self.cng.smth_nlsf_q15[..self.lpc_order]);

            sig_q14[..MAX_LPC_ORDER].copy_from_slice(&self.cng.synth_state);
            for i in 0..length {
                let mut lpc_pred_q10 = (self.lpc_order as i32) >> 1;
                for (j, &a) in a_q12.iter().enumerate().take(self.lpc_order) {
                    lpc_pred_q10 = smlawb(lpc_pred_q10, sig_q14[MAX_LPC_ORDER + i - j - 1], i32::from(a));
                }
                sig_q14[MAX_LPC_ORDER + i] = add_sat32(sig_q14[MAX_LPC_ORDER + i], lshift_sat32(lpc_pred_q10, 4));
                frame[i] = add_sat16(
                    frame[i],
                    rshift_round(smulww(sig_q14[MAX_LPC_ORDER + i], gain_q10), 8)
                        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
                );
            }
            self.cng
                .synth_state
                .copy_from_slice(&sig_q14[length..length + MAX_LPC_ORDER]);
        } else {
            self.cng.synth_state[..self.lpc_order].fill(0);
        }
    }
}
