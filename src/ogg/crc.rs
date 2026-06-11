//! The Ogg page CRC (RFC 3533 §6, item 7).
//!
//! A direct (non-reflected) CRC-32 with generator polynomial `0x04C11DB7`,
//! zero initial value, and no final XOR - *not* the IEEE/zlib CRC-32, which
//! reflects its input and inverts its output. The checksum covers the entire
//! page with the CRC field itself set to zero.

/// Generator polynomial from RFC 3533 §6.
const POLY: u32 = 0x04C1_1DB7;

/// Byte-at-a-time lookup table, computed at compile time.
const TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut r = (i as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            r = if r & 0x8000_0000 != 0 { (r << 1) ^ POLY } else { r << 1 };
            bit += 1;
        }
        table[i] = r;
        i += 1;
    }
    table
};

/// Computes the Ogg CRC of `data` starting from `crc` (zero for a fresh page).
#[must_use]
pub(crate) const fn update(mut crc: u32, data: &[u8]) -> u32 {
    let mut i = 0;
    while i < data.len() {
        crc = (crc << 8) ^ TABLE[((crc >> 24) as u8 ^ data[i]) as usize];
        i += 1;
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::update;

    #[test]
    fn known_vectors() {
        // Independently computed with the reference algorithm (bitwise,
        // poly 0x04C11DB7, no reflection, zero init, no final xor).
        assert_eq!(update(0, b""), 0);
        assert_eq!(update(0, &[0]), 0);
        // A single 0x01 byte shifts the polynomial straight through.
        assert_eq!(update(0, &[1]), super::POLY);
        assert_eq!(update(0, b"OggS"), 0x5FB0_A94F);
        // The standard CRC catalogue check string.
        assert_eq!(update(0, b"123456789"), 0x89A1_897F);
    }

    #[test]
    fn incremental_equals_oneshot() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let oneshot = update(0, data);
        let split = update(update(0, &data[..17]), &data[17..]);
        assert_eq!(oneshot, split);
    }
}
