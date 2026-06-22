//! The Laplace-distributed symbol coder used for CELT coarse energy
//! (RFC 6716 §4.3.2.1).
//!
//! Coarse energy prediction errors approximately follow a Laplace
//! distribution. The coder models it with a geometric decay: symbol 0 has
//! frequency `fs0`, symbols ±1 have a frequency derived from `fs0` and
//! `decay`, and each further magnitude decays by `decay/2^15`, down to a
//! guaranteed minimum probability for the tails. All arithmetic is bit-exact
//! integer math in a total frequency of 2^15.

use crate::range::{RangeDecoder, RangeEncoder};

/// Minimum probability of an energy delta (out of 32768).
const LAPLACE_LOG_MINP: u32 = 0;
const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;

/// Minimum number of guaranteed representable energy deltas in one direction.
const LAPLACE_NMIN: u32 = 16;

/// Total frequency: the coder always works in 15 bits.
const FTB: u32 = 15;
const FT: u32 = 1 << FTB;

/// Frequency of the first non-zero magnitude, derived from `fs0` and `decay`.
///
/// `decay` is positive and at most 11456 when called by CELT.
fn freq1(fs0: u32, decay: u32) -> u32 {
    let ft = FT - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    (ft * (16384 - decay)) >> 15
}

/// Encodes `value` with a Laplace model; returns the value actually coded,
/// which may have been saturated to the supported tail range.
///
/// `fs` is the frequency of zero (out of 2^15) and `decay` the geometric
/// decay factor (out of 2^15), both from the CELT energy probability model.
pub fn ec_laplace_encode(enc: &mut RangeEncoder, value: i32, fs: u32, decay: u32) -> i32 {
    let mut fl = 0u32;
    let mut fs = fs;
    let mut value = value;

    if value != 0 {
        // s = -1 for negative values; (val + s) ^ s = |val|.
        let s = -i32::from(value < 0);
        let val = (value + s) ^ s;

        fl = fs;
        fs = freq1(fs, decay);

        // Search the geometrically decaying part of the PDF.
        let mut i = 1;
        while fs > 0 && i < val {
            fs *= 2;
            fl += fs + 2 * LAPLACE_MINP;
            fs = (fs * decay) >> 15;
            i += 1;
        }

        if fs == 0 {
            // Everything beyond the decayed range has probability MINP; the
            // value saturates to the largest representable delta. Note `s` is
            // -1 or 0, so `2*di + 1 + s` is 2*di for negative values and
            // 2*di + 1 for positive ones.
            let ndi_max = ((FT - fl + LAPLACE_MINP - 1) >> LAPLACE_LOG_MINP) as i32;
            let ndi_max = (ndi_max - s) >> 1;
            let di = (val - i).min(ndi_max - 1);
            fl = (fl as i32 + (2 * di + 1 + s) * LAPLACE_MINP as i32) as u32;
            fs = LAPLACE_MINP.min(FT - fl);
            value = (i + di + s) ^ s;
        } else {
            fs += LAPLACE_MINP;
            if s == 0 {
                fl += fs;
            }
        }
        debug_assert!(fl + fs <= FT);
        debug_assert!(fs > 0);
    }

    enc.encode_bin(fl, fl + fs, FTB);
    value
}

/// Decodes one Laplace-coded value; mirror of [`ec_laplace_encode`].
pub fn ec_laplace_decode(dec: &mut RangeDecoder, fs: u32, decay: u32) -> i32 {
    let mut val = 0i32;
    let mut fl = 0u32;
    let mut fs = fs;
    let fm = dec.decode_bin(FTB);

    if fm >= fs {
        val += 1;
        fl = fs;
        fs = freq1(fs, decay) + LAPLACE_MINP;

        // Search the geometrically decaying part of the PDF.
        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = ((fs - 2 * LAPLACE_MINP) * decay) >> 15;
            fs += LAPLACE_MINP;
            val += 1;
        }

        // Everything beyond that has probability MINP.
        if fs <= LAPLACE_MINP {
            let di = (fm - fl) >> (LAPLACE_LOG_MINP + 1);
            val += di as i32;
            fl += 2 * di * LAPLACE_MINP;
        }

        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }

    debug_assert!(fl < FT);
    debug_assert!(fs > 0);
    debug_assert!(fl <= fm);
    debug_assert!(fm < (fl + fs).min(FT));

    dec.update(fl, (fl + fs).min(FT), FT);
    val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::{RangeDecoder, RangeEncoder};

    /// Representative (fs, decay) pairs in the range the CELT energy
    /// probability model produces (`e_prob_model` entries are scaled by
    /// `<<7` for fs and `<<6` for decay).
    const MODELS: [(u32, u32); 4] = [
        (9 << 7, 100 << 6),
        (42 << 7, 80 << 6),
        (110 << 7, 60 << 6),
        (66 << 7, 120 << 6),
    ];

    #[test]
    fn round_trips_in_range_values() {
        for &(fs, decay) in &MODELS {
            let values: alloc::vec::Vec<i32> = (-20..=20).collect();
            let mut enc = RangeEncoder::new(256);
            let mut coded = alloc::vec::Vec::new();
            for &v in &values {
                coded.push(ec_laplace_encode(&mut enc, v, fs, decay));
            }
            // Small magnitudes are never saturated.
            assert_eq!(coded, values, "fs={fs} decay={decay}");
            let enc_rng = enc.range_size();
            let buf = enc.finalize().expect("within budget");

            let mut dec = RangeDecoder::new(&buf);
            for &v in &values {
                assert_eq!(ec_laplace_decode(&mut dec, fs, decay), v, "fs={fs} decay={decay}");
            }
            assert_eq!(dec.range_size(), enc_rng, "encoder/decoder rng agreement");
        }
    }

    #[test]
    fn saturates_extreme_values_consistently() {
        // A huge delta saturates on encode; the decoder must return exactly
        // the saturated value the encoder actually coded.
        let (fs, decay) = MODELS[0];
        let mut enc = RangeEncoder::new(64);
        let coded = ec_laplace_encode(&mut enc, 30_000, fs, decay);
        assert!(coded < 30_000, "value must saturate");
        let buf = enc.finalize().expect("within budget");
        let mut dec = RangeDecoder::new(&buf);
        assert_eq!(ec_laplace_decode(&mut dec, fs, decay), coded);
    }

    extern crate alloc;
}
