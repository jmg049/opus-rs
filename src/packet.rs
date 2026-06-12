//! Opus packet framing (RFC 6716 §3).
//!
//! An Opus packet carries one or more frames sharing a common configuration
//! (mode, bandwidth, frame size, channel count), described by the leading
//! [TOC byte](Toc). The framing is *not* self-delimiting: the transport (RTP,
//! Ogg, Matroska, ...) supplies the total packet length, and the framing uses
//! it to minimise overhead.
//!
//! [`Packet::parse`] validates every constraint the RFC labels [R1]-[R7]
//! (§3.4); packets violating any of them are rejected as malformed, exactly as
//! the spec requires ("a receiver MUST NOT process packets that violate any of
//! the rules above as normal Opus packets").

use alloc::vec::Vec;
use core::fmt;
use core::time::Duration;

/// Maximum length in bytes of a single Opus frame ([R2], RFC 6716 §3.2.1).
///
/// This is the largest value the one/two-byte length coding can represent
/// (`255*4 + 255`), and the limit every implicit frame length must also obey
/// to allow repacketization by gateways.
pub const MAX_FRAME_LEN: usize = 1275;

/// Maximum audio duration of one packet in tenths of a millisecond
/// ([R5], RFC 6716 §3.2.5): 120 ms.
const MAX_PACKET_DURATION_TENTH_MS: u32 = 1200;

/// Operating mode of an Opus frame (RFC 6716 §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// LP (SILK) layer only: low-bitrate speech, NB through WB.
    SilkOnly,
    /// SILK below 8 kHz plus CELT above: SWB/FB speech at medium bitrates.
    Hybrid,
    /// MDCT (CELT) layer only: music and low-delay use, NB through FB.
    CeltOnly,
}

/// Audio bandwidth of an Opus frame (RFC 6716 §2.1.3, §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Bandwidth {
    /// 4 kHz audio bandwidth, 8 kHz effective sample rate.
    NarrowBand,
    /// 6 kHz audio bandwidth, 12 kHz effective sample rate.
    MediumBand,
    /// 8 kHz audio bandwidth, 16 kHz effective sample rate.
    WideBand,
    /// 12 kHz audio bandwidth, 24 kHz effective sample rate.
    SuperWideBand,
    /// 20 kHz audio bandwidth, 48 kHz effective sample rate.
    FullBand,
}

impl Bandwidth {
    /// The effective sample rate in Hz for this bandwidth.
    #[must_use]
    pub const fn sample_rate_hz(self) -> u32 {
        match self {
            Bandwidth::NarrowBand => 8_000,
            Bandwidth::MediumBand => 12_000,
            Bandwidth::WideBand => 16_000,
            Bandwidth::SuperWideBand => 24_000,
            Bandwidth::FullBand => 48_000,
        }
    }

    /// The audio bandwidth in Hz (the highest frequency reproduced).
    #[must_use]
    pub const fn audio_bandwidth_hz(self) -> u32 {
        match self {
            Bandwidth::NarrowBand => 4_000,
            Bandwidth::MediumBand => 6_000,
            Bandwidth::WideBand => 8_000,
            Bandwidth::SuperWideBand => 12_000,
            Bandwidth::FullBand => 20_000,
        }
    }
}

/// Duration of one Opus frame (RFC 6716 §3.1, Table 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FrameSize {
    /// 2.5 ms (CELT-only modes).
    Ms2_5,
    /// 5 ms (CELT-only modes).
    Ms5,
    /// 10 ms.
    Ms10,
    /// 20 ms.
    Ms20,
    /// 40 ms (SILK-only modes).
    Ms40,
    /// 60 ms (SILK-only modes).
    Ms60,
}

impl FrameSize {
    /// Frame duration in tenths of a millisecond (exact for 2.5 ms).
    #[must_use]
    pub const fn tenth_ms(self) -> u32 {
        match self {
            FrameSize::Ms2_5 => 25,
            FrameSize::Ms5 => 50,
            FrameSize::Ms10 => 100,
            FrameSize::Ms20 => 200,
            FrameSize::Ms40 => 400,
            FrameSize::Ms60 => 600,
        }
    }

    /// Frame duration as a [`Duration`].
    #[must_use]
    pub const fn duration(self) -> Duration {
        Duration::from_micros(self.tenth_ms() as u64 * 100)
    }

    /// Number of samples per channel in one frame at 48 kHz, the rate every
    /// Opus decoder ultimately operates at.
    #[must_use]
    pub const fn samples_per_channel_48k(self) -> usize {
        // 48 samples per ms = 4.8 per tenth-ms; tenth_ms is always a
        // multiple of 5, so this stays exact: 4.8 * 5 = 24.
        (self.tenth_ms() / 5) as usize * 24
    }
}

/// The table-of-contents byte heading every Opus packet (RFC 6716 §3.1).
///
/// Wraps the raw byte and exposes the configuration it encodes. All 256 byte
/// values are valid TOCs; malformed-ness only arises from the framing that
/// follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Toc(u8);

impl Toc {
    /// Interprets `byte` as a TOC byte.
    #[must_use]
    pub const fn new(byte: u8) -> Self {
        Toc(byte)
    }

    /// Builds a TOC byte from its three fields.
    ///
    /// # Panics
    ///
    /// Panics if `config > 31` or `frame_count_code > 3`.
    #[must_use]
    pub fn from_parts(config: u8, stereo: bool, frame_count_code: u8) -> Self {
        assert!(config < 32, "config must be 0..32");
        assert!(frame_count_code < 4, "frame count code must be 0..4");
        Toc((config << 3) | (u8::from(stereo) << 2) | frame_count_code)
    }

    /// The raw TOC byte.
    #[must_use]
    pub const fn byte(self) -> u8 {
        self.0
    }

    /// The configuration number (0..32): the top five bits.
    #[must_use]
    pub const fn config(self) -> u8 {
        self.0 >> 3
    }

    /// `true` for stereo, `false` for mono.
    #[must_use]
    pub const fn stereo(self) -> bool {
        (self.0 >> 2) & 1 == 1
    }

    /// The number of channels (1 or 2).
    #[must_use]
    pub const fn channels(self) -> u8 {
        1 + ((self.0 >> 2) & 1)
    }

    /// The frame count code `c` (0..4): the bottom two bits.
    #[must_use]
    pub const fn frame_count_code(self) -> u8 {
        self.0 & 0x3
    }

    /// The operating mode for this configuration (RFC 6716 Table 2).
    #[must_use]
    pub const fn mode(self) -> Mode {
        match self.config() {
            0..=11 => Mode::SilkOnly,
            12..=15 => Mode::Hybrid,
            _ => Mode::CeltOnly,
        }
    }

    /// The audio bandwidth for this configuration (RFC 6716 Table 2).
    #[must_use]
    pub const fn bandwidth(self) -> Bandwidth {
        match self.config() {
            0..=3 => Bandwidth::NarrowBand,
            4..=7 => Bandwidth::MediumBand,
            8..=11 => Bandwidth::WideBand,
            12..=13 => Bandwidth::SuperWideBand,
            14..=15 => Bandwidth::FullBand,
            // CELT-only: NB, WB, SWB, FB in blocks of four (no MB).
            16..=19 => Bandwidth::NarrowBand,
            20..=23 => Bandwidth::WideBand,
            24..=27 => Bandwidth::SuperWideBand,
            _ => Bandwidth::FullBand,
        }
    }

    /// The frame size for this configuration (RFC 6716 Table 2).
    #[must_use]
    pub const fn frame_size(self) -> FrameSize {
        let config = self.config();
        if config < 12 {
            // SILK-only: 10, 20, 40, 60 ms.
            match config & 0x3 {
                0 => FrameSize::Ms10,
                1 => FrameSize::Ms20,
                2 => FrameSize::Ms40,
                _ => FrameSize::Ms60,
            }
        } else if config < 16 {
            // Hybrid: 10, 20 ms.
            if config & 0x1 == 0 {
                FrameSize::Ms10
            } else {
                FrameSize::Ms20
            }
        } else {
            // CELT-only: 2.5, 5, 10, 20 ms.
            match config & 0x3 {
                0 => FrameSize::Ms2_5,
                1 => FrameSize::Ms5,
                2 => FrameSize::Ms10,
                _ => FrameSize::Ms20,
            }
        }
    }
}

/// Why a packet failed to parse; each variant names the RFC 6716 §3.4
/// requirement it violates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PacketError {
    /// The packet is empty \[R1\].
    Empty,
    /// An implicit frame length exceeds [`MAX_FRAME_LEN`] \[R2\].
    FrameTooLarge,
    /// A code 1 packet's payload is not evenly divisible in two \[R3\].
    Code1UnevenPayload,
    /// A frame-length field is truncated or overruns the remaining payload
    /// \[R4\]\[R7\].
    InvalidFrameLength,
    /// A code 3 packet signals zero frames, or more than 120 ms of audio
    /// \[R5\].
    InvalidFrameCount,
    /// A code 3 packet's padding overruns the packet \[R6\]\[R7\].
    InvalidPadding,
    /// A CBR code 3 packet's payload is not an exact multiple of the frame
    /// count \[R6\].
    CbrPayloadNotDivisible,
}

impl fmt::Display for PacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            PacketError::Empty => "packet is empty [R1]",
            PacketError::FrameTooLarge => "frame length exceeds 1275 bytes [R2]",
            PacketError::Code1UnevenPayload => "code 1 packet payload length is odd [R3]",
            PacketError::InvalidFrameLength => "truncated or overrunning frame length [R4/R7]",
            PacketError::InvalidFrameCount => "frame count is zero or exceeds 120 ms [R5]",
            PacketError::InvalidPadding => "padding overruns the packet [R6/R7]",
            PacketError::CbrPayloadNotDivisible => "CBR payload is not a multiple of the frame count [R6]",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PacketError {}

/// A parsed Opus packet: the TOC plus borrowed slices of each frame's
/// compressed data.
///
/// Produced by [`Packet::parse`], which enforces all of RFC 6716 §3.4's
/// well-formedness rules. A frame slice may be empty: zero-length frames are
/// valid in any mode and signal DTX or a dropped frame (§3.2.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet<'a> {
    toc: Toc,
    frames: Vec<&'a [u8]>,
    padding: usize,
}

impl<'a> Packet<'a> {
    /// Parses one Opus packet (RFC 6716 §3.2), validating \[R1\]-\[R7\].
    ///
    /// # Errors
    ///
    /// Returns the [`PacketError`] naming the violated requirement.
    pub fn parse(data: &'a [u8]) -> Result<Self, PacketError> {
        let (&toc_byte, mut rest) = data.split_first().ok_or(PacketError::Empty)?;
        let toc = Toc::new(toc_byte);

        let mut frames = Vec::new();
        let mut padding = 0usize;

        match toc.frame_count_code() {
            // One frame, occupying the whole payload.
            0 => {
                check_frame_len(rest.len())?;
                frames.push(rest);
            },
            // Two frames of equal size.
            1 => {
                if rest.len() % 2 != 0 {
                    return Err(PacketError::Code1UnevenPayload);
                }
                let half = rest.len() / 2;
                check_frame_len(half)?;
                let (a, b) = rest.split_at(half);
                frames.push(a);
                frames.push(b);
            },
            // Two frames; explicit length for the first.
            2 => {
                let n1 = read_frame_len(&mut rest)?;
                let frame1 = take(&mut rest, n1)?;
                check_frame_len(rest.len())?;
                frames.push(frame1);
                frames.push(rest);
            },
            // Signaled number of frames, optional padding, CBR or VBR.
            _ => {
                let (&count_byte, body) = rest.split_first().ok_or(PacketError::InvalidFrameLength)?;
                rest = body;
                let vbr = count_byte >> 7 == 1;
                let has_padding = (count_byte >> 6) & 1 == 1;
                let frame_count = usize::from(count_byte & 0x3F);

                if frame_count == 0 || frame_count as u32 * toc.frame_size().tenth_ms() > MAX_PACKET_DURATION_TENTH_MS {
                    return Err(PacketError::InvalidFrameCount);
                }

                if has_padding {
                    // Padding length coding (§3.2.5): 255 means "254 plus the
                    // value of the next byte", chainable to reach any size.
                    loop {
                        let (&b, body) = rest.split_first().ok_or(PacketError::InvalidPadding)?;
                        rest = body;
                        if b == 255 {
                            padding += 254;
                        } else {
                            padding += usize::from(b);
                            break;
                        }
                    }
                    if padding > rest.len() {
                        return Err(PacketError::InvalidPadding);
                    }
                    // The padding bytes trail the packet; the decoder MUST
                    // accept any value for them.
                    rest = &rest[..rest.len() - padding];
                }

                if vbr {
                    // M-1 explicit lengths; the final frame takes the rest.
                    let mut lengths = Vec::with_capacity(frame_count - 1);
                    for _ in 0..frame_count - 1 {
                        lengths.push(read_frame_len(&mut rest)?);
                    }
                    for n in lengths {
                        frames.push(take(&mut rest, n)?);
                    }
                    check_frame_len(rest.len())?;
                    frames.push(rest);
                } else {
                    // CBR: equal sizes inferred from the remaining payload.
                    if rest.len() % frame_count != 0 {
                        return Err(PacketError::CbrPayloadNotDivisible);
                    }
                    let size = rest.len() / frame_count;
                    check_frame_len(size)?;
                    for _ in 0..frame_count {
                        frames.push(take(&mut rest, size)?);
                    }
                }
            },
        }

        Ok(Packet { toc, frames, padding })
    }

    /// Parses one **self-delimited** packet from the front of `data`
    /// (RFC 6716 Appendix B): every frame length is explicit, so the
    /// packet does not need to span the whole buffer. Returns the packet
    /// and the number of bytes it occupied - the framing used for all but
    /// the last stream of a multistream payload.
    ///
    /// # Errors
    ///
    /// As [`parse`](Self::parse), plus `InvalidFrameLength` when an
    /// explicit length overruns `data`.
    pub fn parse_self_delimited(data: &'a [u8]) -> Result<(Self, usize), PacketError> {
        let (&toc_byte, mut rest) = data.split_first().ok_or(PacketError::Empty)?;
        let toc = Toc::new(toc_byte);

        let mut frames = Vec::new();
        let mut padding = 0usize;

        match toc.frame_count_code() {
            // One frame with an explicit length.
            0 => {
                let n = read_frame_len(&mut rest)?;
                frames.push(take(&mut rest, n)?);
            },
            // Two frames, both of the signalled size.
            1 => {
                let n = read_frame_len(&mut rest)?;
                frames.push(take(&mut rest, n)?);
                frames.push(take(&mut rest, n)?);
            },
            // Two explicit lengths.
            2 => {
                let n1 = read_frame_len(&mut rest)?;
                let n2 = read_frame_len(&mut rest)?;
                frames.push(take(&mut rest, n1)?);
                frames.push(take(&mut rest, n2)?);
            },
            // Signalled count; the last frame's length is explicit too.
            _ => {
                let (&count_byte, body) = rest.split_first().ok_or(PacketError::InvalidFrameLength)?;
                rest = body;
                let vbr = count_byte >> 7 == 1;
                let has_padding = (count_byte >> 6) & 1 == 1;
                let frame_count = usize::from(count_byte & 0x3F);

                if frame_count == 0 || frame_count as u32 * toc.frame_size().tenth_ms() > MAX_PACKET_DURATION_TENTH_MS {
                    return Err(PacketError::InvalidFrameCount);
                }

                if has_padding {
                    loop {
                        let (&b, body) = rest.split_first().ok_or(PacketError::InvalidPadding)?;
                        rest = body;
                        if b == 255 {
                            padding += 254;
                        } else {
                            padding += usize::from(b);
                            break;
                        }
                    }
                }

                if vbr {
                    let mut lengths = Vec::with_capacity(frame_count);
                    for _ in 0..frame_count {
                        lengths.push(read_frame_len(&mut rest)?);
                    }
                    for n in lengths {
                        frames.push(take(&mut rest, n)?);
                    }
                } else {
                    let size = read_frame_len(&mut rest)?;
                    for _ in 0..frame_count {
                        frames.push(take(&mut rest, size)?);
                    }
                }
                // The padding bytes trail the frames within this packet's
                // region of the buffer.
                if padding > rest.len() {
                    return Err(PacketError::InvalidPadding);
                }
                rest = &rest[padding..];
            },
        }

        let consumed = data.len() - rest.len();
        Ok((Packet { toc, frames, padding }, consumed))
    }

    /// The packet's TOC byte.
    #[must_use]
    pub const fn toc(&self) -> Toc {
        self.toc
    }

    /// The compressed frames, in order. Never empty; individual frames may be
    /// zero-length (DTX).
    #[must_use]
    pub fn frames(&self) -> &[&'a [u8]] {
        &self.frames
    }

    /// Bytes of Opus padding the packet carried (code 3 only).
    #[must_use]
    pub const fn padding(&self) -> usize {
        self.padding
    }

    /// Total audio duration of the packet.
    #[must_use]
    pub fn duration(&self) -> Duration {
        Duration::from_micros(self.frames.len() as u64 * u64::from(self.toc.frame_size().tenth_ms()) * 100)
    }
}

/// Enforces \[R2\] on an implicit frame length.
fn check_frame_len(len: usize) -> Result<(), PacketError> {
    if len > MAX_FRAME_LEN {
        Err(PacketError::FrameTooLarge)
    } else {
        Ok(())
    }
}

/// Reads a one/two-byte frame length (RFC 6716 §3.2.1): `0` = no frame (DTX),
/// `1..=251` literal, `252..=255` = `second_byte*4 + first_byte`.
fn read_frame_len(rest: &mut &[u8]) -> Result<usize, PacketError> {
    let (&b0, body) = rest.split_first().ok_or(PacketError::InvalidFrameLength)?;
    *rest = body;
    let len = match b0 {
        0..=251 => usize::from(b0),
        _ => {
            let (&b1, body) = rest.split_first().ok_or(PacketError::InvalidFrameLength)?;
            *rest = body;
            usize::from(b1) * 4 + usize::from(b0)
        },
    };
    // By construction len <= 255*4 + 255 = 1275, so [R2] always holds here.
    debug_assert!(len <= MAX_FRAME_LEN);
    Ok(len)
}

/// Splits `n` bytes off the front of `rest`, failing with \[R4\]/\[R7\] if
/// they are not present.
fn take<'a>(rest: &mut &'a [u8], n: usize) -> Result<&'a [u8], PacketError> {
    if n > rest.len() {
        return Err(PacketError::InvalidFrameLength);
    }
    let (head, tail) = rest.split_at(n);
    *rest = tail;
    Ok(head)
}
