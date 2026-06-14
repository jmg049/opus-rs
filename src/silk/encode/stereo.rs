//! Stereo prediction quantisation for the SILK encoder (RFC 6716 §5.2;
//! normative `silk/stereo_quant_pred.c`, `silk/stereo_encode_pred.c`).
//!
//! The mid/side stereo predictor weights are vector-quantised on a
//! sub-stepped grid of the shared `STEREO_PRED_QUANT_Q13` levels and coded
//! as a joint index plus two per-predictor refinements. [`stereo_quant_pred`]
//! is the exact inverse of the decoder's `stereo_decode_pred`, so the
//! quantised weights round-trip; [`stereo_encode_pred`] writes the indices
//! the decoder reads.

use crate::range::RangeEncoder;

use super::super::math::{smlabb, smulwb};
use super::super::tables::{
    STEREO_ONLY_CODE_MID_ICDF, STEREO_PRED_JOINT_ICDF, STEREO_PRED_QUANT_Q13, UNIFORM3_ICDF, UNIFORM5_ICDF,
};

const STEREO_QUANT_TAB_SIZE: usize = 16;
const STEREO_QUANT_SUB_STEPS: i32 = 5;
/// `SILK_FIX_CONST(0.5 / STEREO_QUANT_SUB_STEPS, 16)`.
const HALF_SUB_STEP_Q16: i32 = 6554;

/// `silk_stereo_quant_pred`: quantise the two predictor weights (Q13) in
/// place, returning the codebook indices `ix[2][3]`. On return `pred_q13[0]`
/// holds the first weight minus the second (the form the NSQ/decoder use).
pub(crate) fn stereo_quant_pred(pred_q13: &mut [i32; 2]) -> [[i8; 3]; 2] {
    let mut ix = [[0i8; 3]; 2];
    for n in 0..2 {
        let mut err_min_q13 = i32::MAX;
        let mut quant_pred_q13 = 0i32;
        'outer: for i in 0..STEREO_QUANT_TAB_SIZE - 1 {
            let low_q13 = i32::from(STEREO_PRED_QUANT_Q13[i]);
            let step_q13 = smulwb(i32::from(STEREO_PRED_QUANT_Q13[i + 1]) - low_q13, HALF_SUB_STEP_Q16);
            for j in 0..STEREO_QUANT_SUB_STEPS {
                let lvl_q13 = smlabb(low_q13, step_q13, 2 * j + 1);
                let err_q13 = (pred_q13[n] - lvl_q13).abs();
                if err_q13 < err_min_q13 {
                    err_min_q13 = err_q13;
                    quant_pred_q13 = lvl_q13;
                    ix[n][0] = i as i8;
                    ix[n][1] = j as i8;
                } else {
                    // The error is monotone away from the best level.
                    break 'outer;
                }
            }
        }
        ix[n][2] = ix[n][0] / 3;
        ix[n][0] -= ix[n][2] * 3;
        pred_q13[n] = quant_pred_q13;
    }
    pred_q13[0] -= pred_q13[1];
    ix
}

/// `silk_stereo_encode_pred`: code the predictor indices (joint index then
/// two uniform refinements per predictor).
pub(crate) fn stereo_encode_pred(enc: &mut RangeEncoder, ix: &[[i8; 3]; 2]) {
    let n = 5 * ix[0][2] + ix[1][2];
    debug_assert!(n < 25);
    enc.encode_icdf(n as usize, &STEREO_PRED_JOINT_ICDF, 8);
    for row in ix {
        enc.encode_icdf(row[0] as usize, &UNIFORM3_ICDF, 8);
        enc.encode_icdf(row[1] as usize, &UNIFORM5_ICDF, 8);
    }
}

/// `silk_stereo_encode_mid_only`: code the mid-only flag.
pub(crate) fn stereo_encode_mid_only(enc: &mut RangeEncoder, mid_only_flag: i8) {
    enc.encode_icdf(mid_only_flag as usize, &STEREO_ONLY_CODE_MID_ICDF, 8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::{RangeDecoder, RangeEncoder};
    use crate::silk::stereo::{stereo_decode_mid_only, stereo_decode_pred};

    /// Quantised predictor weights round-trip exactly through the decoder's
    /// `stereo_decode_pred`, and the mid-only flag round-trips.
    #[test]
    fn stereo_pred_round_trips_through_the_decoder() {
        for &(p0, p1, mid_only) in &[
            (0i32, 0i32, 0i8),
            (4096, -2048, 1),
            (8000, 3000, 0),
            (-7000, 6000, 1),
            (1234, -5678, 0),
        ] {
            let mut pred = [p0, p1];
            let ix = stereo_quant_pred(&mut pred);
            // `pred` is now the quantised (difference, second) pair.

            let mut enc = RangeEncoder::new(16);
            stereo_encode_pred(&mut enc, &ix);
            stereo_encode_mid_only(&mut enc, mid_only);
            let bytes = enc.finalize().expect("fits");

            let mut dec = RangeDecoder::new(&bytes);
            let dec_pred = stereo_decode_pred(&mut dec);
            let dec_mid_only = stereo_decode_mid_only(&mut dec);

            assert_eq!(dec_pred, pred, "predictor weights disagree for ({p0},{p1})");
            assert_eq!(dec_mid_only, mid_only == 1, "mid-only flag disagrees");
        }
    }
}
