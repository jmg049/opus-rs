//! PVQ codeword enumeration - "coding with replacement and signs"
//! (RFC 6716 §4.3.4.2; normative reference `cwrs.c`).
//!
//! CELT codes each band's normalized shape as an N-dimensional vector of K
//! signed unit pulses, transmitted as a single uniformly distributed integer
//! index in `0..V(N, K)`, where
//!
//! - `V(N, K)` is the number of N-dimensional pulse vectors with K pulses
//!   (signs included), and
//! - `U(N, K) = (V(N-1, K-1) + V(N, K-1)) / 2`, the enumeration's working
//!   function, symmetric in its arguments and obeying
//!   `U(N, K) = U(N-1, K) + U(N, K-1) + U(N-1, K-1)`.
//!
//! This implementation ports the table-free (`SMALL_FOOTPRINT`) variant of
//! the reference: rows of `U` are computed on the fly with the recurrence in
//! O(K) memory. It is mathematically identical to the table-driven fast path
//! and produces the same indices bit-for-bit; the precomputed-table
//! optimization can be layered on later without changing any output.

use alloc::vec;
use alloc::vec::Vec;

use crate::range::{RangeDecoder, RangeEncoder};

/// Advances `u` from row N-1 to row N of the recurrence
/// `u[i][j] = u[i-1][j] + u[i][j-1] + u[i-1][j-1]`, with `ui0` the new row's
/// base case. Mirrors `unext()`.
fn unext(u: &mut [u32], mut ui0: u32) {
    for j in 1..u.len() {
        let ui1 = u[j].wrapping_add(u[j - 1]).wrapping_add(ui0);
        u[j - 1] = ui0;
        ui0 = ui1;
    }
    *u.last_mut().expect("len >= 2") = ui0;
}

/// Inverse of [`unext`]: steps `u` back one row. Mirrors `uprev()`.
fn uprev(u: &mut [u32], mut ui0: u32) {
    for j in 1..u.len() {
        let ui1 = u[j].wrapping_sub(u[j - 1]).wrapping_sub(ui0);
        u[j - 1] = ui0;
        ui0 = ui1;
    }
    *u.last_mut().expect("len >= 2") = ui0;
}

/// Computes `V(n, k)` and fills `u[i] = U(n, i)` for `i in 0..=k+1`.
/// Mirrors `ncwrs_urow()`; requires `n >= 2`, `k >= 1`, `u.len() == k + 2`.
fn ncwrs_urow(n: usize, k: usize, u: &mut [u32]) -> u32 {
    debug_assert!(n >= 2 && k >= 1);
    debug_assert_eq!(u.len(), k + 2);
    u[0] = 0;
    u[1] = 1;
    // Row 2: U(2, k) = 2k - 1.
    for (i, v) in u.iter_mut().enumerate().skip(2) {
        *v = (i as u32) * 2 - 1;
    }
    for _ in 2..n {
        unext(&mut u[1..], 1);
    }
    u[k] + u[k + 1]
}

/// Decodes codeword index `i` into the pulse vector `y` (length N, K pulses).
/// `u` must contain row N of `U` for `0..=k+1`; destroyed in the process.
/// Mirrors the `SMALL_FOOTPRINT` `cwrsi()`.
fn cwrsi(k: usize, mut i: u32, y: &mut [i32], u: &mut [u32]) {
    debug_assert!(!y.is_empty());
    let mut k = k;
    for yj_out in y.iter_mut() {
        let p = u[k + 1];
        // s = -1 when the pulses in this dimension are negative.
        let s = -i32::from(i >= p);
        if s != 0 {
            i -= p;
        }
        let k0 = k;
        let mut p = u[k];
        while p > i {
            k -= 1;
            p = u[k];
        }
        i -= p;
        let yj = (k0 - k) as i32;
        *yj_out = (yj + s) ^ s;
        uprev(&mut u[..k + 2], 0);
    }
}

/// Computes the codeword index of pulse vector `y` and `V(n, k)`.
/// `u` is scratch of length `k + 2`. Mirrors the `SMALL_FOOTPRINT` `icwrs()`.
fn icwrs(k: usize, y: &[i32], u: &mut [u32]) -> (u32, u32) {
    let n = y.len();
    debug_assert!(n >= 2);
    u[0] = 0;
    for (idx, v) in u.iter_mut().enumerate().skip(1) {
        *v = (idx as u32) * 2 - 1;
    }

    // N = 1 base case on the last element.
    let mut i = u32::from(y[n - 1] < 0);
    let mut kk = y[n - 1].unsigned_abs() as usize;

    let mut j = n - 2;
    i += u[kk];
    kk += y[j].unsigned_abs() as usize;
    if y[j] < 0 {
        i += u[kk + 1];
    }
    while j > 0 {
        j -= 1;
        unext(u, 0);
        i += u[kk];
        kk += y[j].unsigned_abs() as usize;
        if y[j] < 0 {
            i += u[kk + 1];
        }
    }
    debug_assert_eq!(kk, k, "sum of |y| must equal K");
    (i, u[kk] + u[kk + 1])
}

/// The size of the PVQ codebook: the number of N-dimensional vectors of K
/// signed pulses, `V(N, K)`.
///
/// Requires `n >= 2` and `k >= 1`; the result must fit in 32 bits, which
/// holds for every (N, K) pair CELT's bit allocation can produce.
#[must_use]
pub fn pvq_codebook_size(n: usize, k: usize) -> u32 {
    let mut u = vec![0u32; k + 2];
    ncwrs_urow(n, k, &mut u)
}

/// Decodes K signed unit pulses into `y` (RFC 6716 §4.3.4.2,
/// `decode_pulses()`).
///
/// `y.len()` is the band size N (≥ 2); `k` ≥ 1. Returns `None` when the
/// uniformly coded index is out of range, which indicates frame corruption.
#[must_use]
pub fn decode_pulses(dec: &mut RangeDecoder, y: &mut [i32], k: usize) -> Option<()> {
    debug_assert!(y.len() >= 2 && k >= 1);
    let mut u = vec![0u32; k + 2];
    let v = ncwrs_urow(y.len(), k, &mut u);
    let i = dec.decode_uint(v)?;
    cwrsi(k, i, y, &mut u);
    Some(())
}

/// Encodes the pulse vector `y` (sum of magnitudes K ≥ 1, length N ≥ 2);
/// mirror of [`decode_pulses`] (`encode_pulses()`).
pub fn encode_pulses(enc: &mut RangeEncoder, y: &[i32], k: usize) {
    debug_assert!(y.len() >= 2 && k >= 1);
    let mut u: Vec<u32> = vec![0u32; k + 2];
    let (i, v) = icwrs(k, y, &mut u);
    enc.encode_uint(i, v);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// V(N, K) for N, K < 10, from the reference implementation's own
    /// documentation table in `cwrs.c`.
    const V_TABLE: [[u32; 10]; 10] = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 4, 8, 12, 16, 20, 24, 28, 32, 36],
        [1, 6, 18, 38, 66, 102, 146, 198, 258, 326],
        [1, 8, 32, 88, 192, 360, 608, 952, 1408, 1992],
        [1, 10, 50, 170, 450, 1002, 1970, 3530, 5890, 9290],
        [1, 12, 72, 292, 912, 2364, 5336, 10836, 20256, 35436],
        [1, 14, 98, 462, 1666, 4942, 12642, 28814, 59906, 115598],
        [1, 16, 128, 688, 2816, 9424, 27008, 68464, 157184, 332688],
        [1, 18, 162, 978, 4482, 16722, 53154, 148626, 374274, 864146],
    ];

    #[test]
    fn codebook_sizes_match_reference_table() {
        for (n, row) in V_TABLE.iter().enumerate().skip(2) {
            for (k, &expected) in row.iter().enumerate().skip(1) {
                assert_eq!(pvq_codebook_size(n, k), expected, "V({n}, {k})");
            }
        }
    }

    /// The enumeration is a bijection: every index in `0..V(N, K)` decodes to
    /// a distinct vector with exactly K pulses, and re-encodes to itself.
    #[test]
    fn exhaustive_index_bijection_small_nk() {
        for n in 2..=6usize {
            for k in 1..=6usize {
                let v = pvq_codebook_size(n, k);
                for i in 0..v {
                    let mut u = vec![0u32; k + 2];
                    ncwrs_urow(n, k, &mut u);
                    let mut y = vec![0i32; n];
                    cwrsi(k, i, &mut y, &mut u);

                    let pulses: u32 = y.iter().map(|x| x.unsigned_abs()).sum();
                    assert_eq!(pulses, k as u32, "N={n} K={k} i={i}: pulse count");

                    let mut scratch = vec![0u32; k + 2];
                    let (back, nc) = icwrs(k, &y, &mut scratch);
                    assert_eq!(back, i, "N={n} K={k}: index round-trip");
                    assert_eq!(nc, v, "N={n} K={k}: V agreement");
                }
            }
        }
    }

    /// Pulse vectors survive an actual range coder round trip, and the
    /// encoder/decoder `rng` states agree afterwards.
    #[test]
    fn range_coder_round_trip() {
        // A deterministic spread of shapes, including larger N and K. All
        // (N, K) pairs keep V(N, K) within 32 bits, the invariant CELT's bit
        // allocation guarantees (V(24, 10) or V(96, 6) would overflow - the
        // allocation can never produce those).
        let cases: [(usize, usize); 6] = [(2, 1), (4, 3), (8, 8), (16, 4), (24, 5), (96, 3)];

        let mut enc = RangeEncoder::new(1024);
        let mut vectors = alloc::vec::Vec::new();
        for &(n, k) in &cases {
            // Deterministic pulse pattern: alternate signs, spread across dims.
            let mut y = vec![0i32; n];
            for p in 0..k {
                let at = (p * 7) % n;
                y[at] += if p % 2 == 0 { 1 } else { -1 };
            }
            // Fix up: ensure the sum of magnitudes is exactly k (collisions of
            // opposite sign would cancel; regenerate deterministically).
            let total: u32 = y.iter().map(|x| x.unsigned_abs()).sum();
            if total != k as u32 {
                y = vec![0i32; n];
                for p in 0..k {
                    y[p % n] += 1;
                }
            }
            encode_pulses(&mut enc, &y, k);
            vectors.push((n, k, y));
        }
        let enc_rng = enc.range_size();
        let buf = enc.finalize().expect("within budget");

        let mut dec = RangeDecoder::new(&buf);
        for (n, k, expected) in vectors {
            let mut y = vec![0i32; n];
            decode_pulses(&mut dec, &mut y, k).expect("in range");
            assert_eq!(y, expected, "N={n} K={k}");
        }
        assert_eq!(dec.range_size(), enc_rng);
    }

    extern crate alloc;
}
