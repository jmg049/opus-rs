//! Subframe gain dequantisation (RFC 6716 §4.2.7.4), uniform on a log scale.
//!
//! The decoded gain indices accumulate into `prev_ind` (delta coding with a
//! double-step extension above a threshold), clamp to the 64 quantisation
//! levels, and convert to linear Q16 via [`log2lin`](super::math::log2lin).

#![allow(dead_code, reason = "consumed incrementally as the SILK decoder stages land")]

use super::indices::MAX_NB_SUBFR;
use super::math::{lin2log, log2lin, smulwb};

/// `N_LEVELS_QGAIN`.
const N_LEVELS_QGAIN: i32 = 64;
/// `MAX_DELTA_GAIN_QUANT` / `MIN_DELTA_GAIN_QUANT`.
const MAX_DELTA_GAIN_QUANT: i32 = 36;
const MIN_DELTA_GAIN_QUANT: i32 = -4;
/// `OFFSET`: `(MIN_QGAIN_DB * 128) / 6 + 16 * 128`.
const OFFSET: i32 = (2 * 128) / 6 + 16 * 128;
/// `INV_SCALE_Q16`: `(65536 * ((MAX-MIN)_DB * 128 / 6)) / (N_LEVELS - 1)`.
const INV_SCALE_Q16: i32 = (65536 * (((88 - 2) * 128) / 6)) / (N_LEVELS_QGAIN - 1);
/// `SCALE_Q16`: the encode-side reciprocal of `INV_SCALE_Q16`.
const SCALE_Q16: i32 = (65536 * (N_LEVELS_QGAIN - 1)) / (((88 - 2) * 128) / 6);

/// Quantise per-subframe linear Q16 gains to indices
/// (delta-coded against `prev_ind`, with the double-step extension), writing
/// the requantised gains back into `gain_q16`. The exact inverse of
/// [`gains_dequant`].
pub(crate) fn gains_quant(
    ind: &mut [i8; MAX_NB_SUBFR],
    gain_q16: &mut [i32; MAX_NB_SUBFR],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr {
        // Log scale, scale, floor.
        ind[k] = smulwb(SCALE_Q16, lin2log(gain_q16[k]) - OFFSET) as i8;
        // Round towards the previous index (hysteresis).
        if i32::from(ind[k]) < i32::from(*prev_ind) {
            ind[k] = ind[k].wrapping_add(1);
        }
        ind[k] = i32::from(ind[k]).clamp(0, N_LEVELS_QGAIN - 1) as i8;

        if k == 0 && !conditional {
            // Full index, limited so it can't drop too far.
            ind[k] = i32::from(ind[k]).clamp(i32::from(*prev_ind) + MIN_DELTA_GAIN_QUANT, N_LEVELS_QGAIN - 1) as i8;
            *prev_ind = ind[k];
        } else {
            // Delta index.
            ind[k] = (i32::from(ind[k]) - i32::from(*prev_ind)) as i8;
            let double_step = 2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + i32::from(*prev_ind);
            if i32::from(ind[k]) > double_step {
                ind[k] = (double_step + ((i32::from(ind[k]) - double_step + 1) >> 1)) as i8;
            }
            ind[k] = i32::from(ind[k]).clamp(MIN_DELTA_GAIN_QUANT, MAX_DELTA_GAIN_QUANT) as i8;
            if i32::from(ind[k]) > double_step {
                *prev_ind =
                    (i32::from(*prev_ind) + (i32::from(ind[k]) << 1) - double_step).min(N_LEVELS_QGAIN - 1) as i8;
            } else {
                *prev_ind = (i32::from(*prev_ind) + i32::from(ind[k])) as i8;
            }
            // Shift to make non-negative.
            ind[k] = (i32::from(ind[k]) - MIN_DELTA_GAIN_QUANT) as i8;
        }

        gain_q16[k] = log2lin((smulwb(INV_SCALE_Q16, i32::from(*prev_ind)) + OFFSET).min(3967));
    }
}

/// Per-subframe linear Q16 gains from the decoded
/// indices; `prev_ind` is the cross-frame accumulator (`LastGainIndex`).
pub(crate) fn gains_dequant(
    ind: &[i8; MAX_NB_SUBFR],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) -> [i32; MAX_NB_SUBFR] {
    let mut gain_q16 = [0i32; MAX_NB_SUBFR];
    for k in 0..nb_subfr {
        if k == 0 && !conditional {
            // The index may not drop more than 16 steps (~21.8 dB).
            *prev_ind = i32::from(ind[k]).max(i32::from(*prev_ind) - 16) as i8;
        } else {
            let ind_tmp = i32::from(ind[k]) + MIN_DELTA_GAIN_QUANT;
            // Accumulate deltas, with double steps above the threshold.
            let double_step_size_threshold = 2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + i32::from(*prev_ind);
            if ind_tmp > double_step_size_threshold {
                *prev_ind = (i32::from(*prev_ind) + ((ind_tmp << 1) - double_step_size_threshold)) as i8;
            } else {
                *prev_ind = (i32::from(*prev_ind) + ind_tmp) as i8;
            }
        }
        *prev_ind = i32::from(*prev_ind).clamp(0, N_LEVELS_QGAIN - 1) as i8;

        // Scale and convert to linear (3967 = 31 in Q7).
        gain_q16[k] = log2lin((smulwb(INV_SCALE_Q16, i32::from(*prev_ind)) + OFFSET).min(3967));
    }
    gain_q16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference pins for `gains_dequant` with these exact sequences.
    #[test]
    fn dequant_matches_reference_pins() {
        // Independent first frame: indices [32, 20, 20, 25], prev starts 10.
        let mut prev = 10i8;
        let gains = gains_dequant(&[32, 20, 20, 25], &mut prev, false, 4);
        assert_eq!(gains, [12_713_984, 158_334_976, 1_686_110_208, 1_686_110_208]);
        assert_eq!(prev, 63);

        // Conditional follow-up frame: deltas [4, 0, 5, 8].
        let gains = gains_dequant(&[4, 0, 5, 8], &mut prev, true, 4);
        assert_eq!(gains, [1_686_110_208, 897_581_056, 1_048_576_000, 1_686_110_208]);
        assert_eq!(prev, 63);

        // Independent restart from a low accumulator.
        let mut prev = 0i8;
        let gains = gains_dequant(&[40, 0, 0, 0], &mut prev, false, 4);
        assert_eq!(gains, [44_826_624, 23_855_104, 12_713_984, 6_782_976]);
        assert_eq!(prev, 28);
    }

    /// `gains_quant` and `gains_dequant` are exact inverses: the requantised
    /// gains and the prev-index accumulator agree after a round trip, for
    /// both an independent and a conditional frame.
    #[test]
    fn quant_dequant_round_trip() {
        for &conditional in &[false, true] {
            for target in [
                [12_000_000i32, 160_000_000, 1_600_000_000, 800_000_000],
                [500_000, 50_000_000, 5_000_000, 900_000_000],
                [1_000_000, 1_000_000, 1_000_000, 1_000_000],
            ] {
                let mut prev_e = 20i8;
                let mut ind = [0i8; MAX_NB_SUBFR];
                let mut gain = target;
                gains_quant(&mut ind, &mut gain, &mut prev_e, conditional, 4);

                // The decoder reconstructs the same gains from the indices.
                let mut prev_d = 20i8;
                let dec = gains_dequant(&ind, &mut prev_d, conditional, 4);
                assert_eq!(dec, gain, "gains differ (conditional={conditional})");
                assert_eq!(prev_d, prev_e, "prev_ind differs (conditional={conditional})");
                // Delta indices are non-negative as coded.
                let start = usize::from(!conditional);
                for &i in &ind[start..4] {
                    assert!(i >= 0, "negative coded index {i}");
                }
            }
        }
    }
}
