//! Range encoder (RFC 6716 §5.1).

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use super::{CODE_BOT, CODE_SHIFT, CODE_TOP, SYM_BITS, SYM_MAX, WINDOW_SIZE, ilog};

/// The encoder ran out of buffer space: range-coder data (front) and raw bits
/// (back) collided inside the fixed-size frame buffer.
///
/// The bit-allocation logic of a correct encoder prevents this by construction
/// (it budgets with [`RangeEncoder::tell`]); hitting it indicates a caller bug
/// or an undersized frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeEncoderError;

impl fmt::Display for RangeEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("range encoder overflowed its frame buffer")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RangeEncoderError {}

/// The Opus range encoder.
///
/// Writes into a fixed-size frame buffer chosen at construction: range-coded
/// symbols grow from the front, raw bits grow from the back, and
/// [`finalize`](Self::finalize) terminates the stream so that the
/// [`RangeDecoder`](super::RangeDecoder) decodes the intended values
/// regardless of the gap/overlap in the middle (RFC 6716 §5.1.5).
///
/// The fixed size mirrors the codec's design: Opus frames are budgeted to an
/// exact byte length up front, and the bit-allocation logic consults
/// [`tell`](Self::tell) to stay within it.
///
/// State per RFC 6716 §5.1: the four-tuple `(val, rng, rem, ext)`, initialized
/// to `(0, 2^31, none, 0)`. After encoding a symbol sequence, `rng` exactly
/// matches the decoder's `rng` after decoding the same sequence.
#[derive(Debug, Clone)]
pub struct RangeEncoder {
    buf: Vec<u8>,
    /// Front bytes written (range-coder direction).
    offs: usize,
    /// Bytes written from the end (raw-bits direction).
    end_offs: usize,
    /// Raw-bits window: bits awaiting flush to the end of the buffer.
    end_window: u32,
    /// Number of valid bits in `end_window`.
    nend_bits: u32,
    /// Low end of the current range.
    val: u32,
    /// Size of the current range.
    rng: u32,
    /// Buffered output byte (a value < 255), or `None` before the first
    /// carry-out. Buffered because a later carry may increment it.
    rem: Option<u8>,
    /// Count of buffered 0xFF bytes, which propagate carries.
    ext: u32,
    /// Conservative upper bound on whole bits produced so far
    /// (RFC 6716 §5.1.6).
    nbits_total: u32,
    /// Sticky overflow flag; reported by [`finalize`](Self::finalize).
    error: bool,
}

impl RangeEncoder {
    /// Creates an encoder for a frame of exactly `size` bytes.
    ///
    /// `size` is the total space shared by range-coder data and raw bits; the
    /// finished frame returned by [`finalize`](Self::finalize) is exactly this
    /// long, zero-padded between the two regions per RFC 6716 §5.1.5.
    #[must_use]
    pub fn new(size: usize) -> Self {
        RangeEncoder {
            buf: vec![0; size],
            offs: 0,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            val: 0,
            rng: CODE_TOP,
            rem: None,
            ext: 0,
            // Matches the decoder's convention of 33 after its initial
            // renormalization (RFC 6716 §4.1.6), so tell() values agree.
            nbits_total: 33,
            error: false,
        }
    }

    /// Writes one byte in the front (range-coder) direction.
    fn write_byte(&mut self, b: u8) {
        if self.offs + self.end_offs >= self.buf.len() {
            self.error = true;
        } else {
            self.buf[self.offs] = b;
            self.offs += 1;
        }
    }

    /// Writes one byte in the back (raw-bits) direction.
    fn write_byte_at_end(&mut self, b: u8) {
        if self.offs + self.end_offs >= self.buf.len() {
            self.error = true;
        } else {
            self.end_offs += 1;
            let at = self.buf.len() - self.end_offs;
            self.buf[at] = b;
        }
    }

    /// Carry propagation and output buffering (RFC 6716 §5.1.1.2,
    /// `ec_enc_carry_out`).
    ///
    /// Takes a 9-bit value: 8 data bits plus a carry bit. A 0xFF data byte
    /// cannot be finalized yet (a later carry would propagate through it), so
    /// it is only counted in `ext`; anything else flushes the pending bytes
    /// with the carry applied.
    fn carry_out(&mut self, c: u32) {
        if c == SYM_MAX {
            self.ext += 1;
        } else {
            let carry = (c >> SYM_BITS) as u8;
            if let Some(rem) = self.rem.take() {
                self.write_byte(rem + carry);
            }
            if self.ext > 0 {
                // All 0xFF if no carry; all 0x00 if the carry rippled through.
                let sym = (SYM_MAX + u32::from(carry)) as u8;
                for _ in 0..self.ext {
                    self.write_byte(sym);
                }
                self.ext = 0;
            }
            self.rem = Some((c & SYM_MAX) as u8);
        }
    }

    /// Renormalization (RFC 6716 §5.1.1.1): restores `rng > 2^23`, emitting
    /// the top 9 bits of `val` (8 data bits + carry) per iteration.
    fn normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.carry_out(self.val >> CODE_SHIFT);
            self.val = (self.val << SYM_BITS) & 0x7FFF_FFFF;
            self.rng <<= SYM_BITS;
            self.nbits_total += SYM_BITS;
        }
    }

    /// Encodes a symbol with the three-tuple `(fl, fh, ft)`
    /// (RFC 6716 §5.1.1, `ec_encode`). The tuple must satisfy
    /// `0 <= fl < fh <= ft <= 65535`.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        debug_assert!(fl < fh && fh <= ft && ft <= u32::from(u16::MAX));
        let r = self.rng / ft;
        if fl > 0 {
            self.val += self.rng - r * (ft - fl);
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * (ft - fh);
        }
        self.normalize();
    }

    /// Like [`encode`](Self::encode) with `ft = 1 << ftb`, avoiding the
    /// division (RFC 6716 §5.1.2.1, `ec_encode_bin`).
    pub fn encode_bin(&mut self, fl: u32, fh: u32, ftb: u32) {
        debug_assert!(ftb <= 16);
        let ft = 1u32 << ftb;
        debug_assert!(fl < fh && fh <= ft);
        let r = self.rng >> ftb;
        if fl > 0 {
            self.val += self.rng - r * (ft - fl);
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * (ft - fh);
        }
        self.normalize();
    }

    /// Encodes one binary symbol whose probability of being "1" is
    /// `1 / 2^logp` (RFC 6716 §5.1.2.2, `ec_enc_bit_logp`).
    pub fn encode_bit_logp(&mut self, bit: bool, logp: u32) {
        let r = self.rng;
        let s = r >> logp;
        let r = r - s;
        if bit {
            self.val += r;
        }
        self.rng = if bit { s } else { r };
        self.normalize();
    }

    /// Encodes symbol `k` against the same "inverse" CDF table the decoder
    /// uses (RFC 6716 §5.1.2.3, `ec_enc_icdf`); see
    /// [`RangeDecoder::decode_icdf`](super::RangeDecoder::decode_icdf).
    pub fn encode_icdf(&mut self, k: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if k > 0 {
            self.val += self.rng - r * u32::from(icdf[k - 1]);
            self.rng = r * u32::from(icdf[k - 1] - icdf[k]);
        } else {
            self.rng -= r * u32::from(icdf[k]);
        }
        self.normalize();
    }

    /// Appends `bits` raw bits (LSB-first) at the end of the frame
    /// (RFC 6716 §5.1.3, `ec_enc_bits`). `bits` must be at most 24.
    pub fn encode_raw_bits(&mut self, value: u32, bits: u32) {
        debug_assert!(bits > 0 && bits <= WINDOW_SIZE - SYM_BITS);
        debug_assert!(value >> bits == 0 || bits == 32);
        if self.nend_bits + bits > WINDOW_SIZE {
            while self.nend_bits >= SYM_BITS {
                self.write_byte_at_end((self.end_window & SYM_MAX) as u8);
                self.end_window >>= SYM_BITS;
                self.nend_bits -= SYM_BITS;
            }
        }
        self.end_window |= value << self.nend_bits;
        self.nend_bits += bits;
        self.nbits_total += bits;
    }

    /// Encodes `t`, one of `ft` equiprobable values in `0..ft`
    /// (RFC 6716 §5.1.4, `ec_enc_uint`). `ft` must be at least 2 and need not
    /// be a power of two.
    pub fn encode_uint(&mut self, t: u32, ft: u32) {
        debug_assert!(ft > 1);
        debug_assert!(t < ft);
        let ftb = ilog(ft - 1);
        if ftb <= 8 {
            self.encode(t, t + 1, ft);
        } else {
            let ft_hi = ((ft - 1) >> (ftb - 8)) + 1;
            let t_hi = t >> (ftb - 8);
            self.encode(t_hi, t_hi + 1, ft_hi);
            self.encode_raw_bits(t & ((1 << (ftb - 8)) - 1), ftb - 8);
        }
    }

    /// Conservative upper bound on the whole number of bits produced so far
    /// (RFC 6716 §5.1.6); must agree exactly with the decoder's
    /// [`tell`](super::RangeDecoder::tell) after the same symbols.
    #[inline]
    #[must_use]
    pub fn tell(&self) -> u32 {
        self.nbits_total - ilog(self.rng)
    }

    /// Like [`tell`](Self::tell) to fractional 1/8th-bit precision.
    #[must_use]
    pub fn tell_frac(&self) -> u32 {
        super::decoder::tell_frac(self.nbits_total, self.rng)
    }

    /// The current range size; see
    /// [`RangeDecoder::range_size`](super::RangeDecoder::range_size).
    #[inline]
    #[must_use]
    pub fn range_size(&self) -> u32 {
        self.rng
    }

    /// Terminates the stream and returns the finished frame
    /// (RFC 6716 §5.1.5, `ec_enc_done`).
    ///
    /// Chooses the value in `[val, val + rng)` with the most trailing zeros -
    /// so trailing bits can hold raw-bits data without desynchronizing the
    /// range coder - flushes all carries and the raw-bits window, and
    /// zero-pads the gap between the two regions. The returned buffer is
    /// exactly the size given to [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// Returns [`RangeEncoderError`] if the front and back regions collided at
    /// any point ("busting the budget").
    pub fn finalize(mut self) -> Result<Vec<u8>, RangeEncoderError> {
        // Bits of val that must be output to disambiguate the final range.
        let mut l: i32 = (super::CODE_BITS - ilog(self.rng)) as i32;
        let mut msk = (CODE_TOP - 1) >> l;
        let mut end = self.val.wrapping_add(msk) & !msk;

        // If rounding val up to the next multiple of (msk + 1) cannot keep
        // all the don't-care trailing bits inside the range, use one more bit.
        if (end | msk) >= self.val + self.rng {
            l += 1;
            msk >>= 1;
            end = self.val.wrapping_add(msk) & !msk;
        }

        while l > 0 {
            self.carry_out(end >> CODE_SHIFT);
            end = (end << SYM_BITS) & (CODE_TOP - 1);
            l -= SYM_BITS as i32;
        }

        // Flush any pending carry chain into the output.
        if self.rem.is_some() || self.ext > 0 {
            self.carry_out(0);
        }

        // Flush whole bytes of the raw-bits window.
        while self.nend_bits >= SYM_BITS {
            self.write_byte_at_end((self.end_window & SYM_MAX) as u8);
            self.end_window >>= SYM_BITS;
            self.nend_bits -= SYM_BITS;
        }

        if !self.error {
            // Zero the gap between the range-coder data and the raw bits.
            let gap = self.offs..self.buf.len() - self.end_offs;
            self.buf[gap].fill(0);

            if self.nend_bits > 0 {
                if self.end_offs >= self.buf.len() {
                    // No room at all for the leftover raw bits.
                    self.error = true;
                } else {
                    // The leftover raw bits share a byte with the final range
                    // coder output. `-l` is the number of low bits in that
                    // byte the range coder does not care about; if the regions
                    // have met and the leftover bits exceed it, the stream is
                    // busted (the range data takes precedence).
                    let spare = (-l) as u32;
                    if self.offs + self.end_offs >= self.buf.len() && spare < self.nend_bits {
                        self.end_window &= (1u32 << spare) - 1;
                        self.error = true;
                    }
                    let at = self.buf.len() - self.end_offs - 1;
                    self.buf[at] |= self.end_window as u8;
                }
            }
        }

        if self.error {
            Err(RangeEncoderError)
        } else {
            Ok(self.buf)
        }
    }
}
