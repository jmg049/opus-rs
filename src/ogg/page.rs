//! Ogg page parsing, packet reassembly, and page writing (RFC 3533 §5-6).

use alloc::vec::Vec;
use core::fmt;

use super::crc;

/// The four-byte capture pattern heading every Ogg page.
pub const CAPTURE_PATTERN: [u8; 4] = *b"OggS";

/// Fixed page-header length before the segment table.
pub(crate) const HEADER_LEN: usize = 27;

/// Maximum number of segments per page, and therefore the maximum body size
/// (255 segments × 255 bytes = 65 025 body bytes; 65 307 total).
pub(crate) const MAX_SEGMENTS: usize = 255;

/// The granule position value meaning "no packet completes on this page"
/// (RFC 3533 §6, item 4: -1 in two's complement).
pub const NO_GRANULE: u64 = u64::MAX;

/// Why a page failed to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OggError {
    /// The capture pattern `OggS` was not found.
    MissingCapturePattern,
    /// The stream-structure version is not 0.
    UnsupportedVersion(u8),
    /// Fewer bytes available than the header and segment table declare.
    Truncated,
    /// The page checksum did not match its contents.
    BadCrc,
}

impl fmt::Display for OggError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OggError::MissingCapturePattern => f.write_str("missing OggS capture pattern"),
            OggError::UnsupportedVersion(v) => write!(f, "unsupported Ogg version {v}"),
            OggError::Truncated => f.write_str("page truncated"),
            OggError::BadCrc => f.write_str("page CRC mismatch"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OggError {}

/// One parsed Ogg page: borrowed header fields plus the raw body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page<'a> {
    /// Header-type flag 0x01: the first packet continues from the previous
    /// page.
    pub continued: bool,
    /// Header-type flag 0x02: first page of the logical bitstream.
    pub bos: bool,
    /// Header-type flag 0x04: last page of the logical bitstream.
    pub eos: bool,
    /// Codec-defined position marker; [`NO_GRANULE`] when no packet completes
    /// on this page.
    pub granule_position: u64,
    /// Serial number identifying the logical bitstream.
    pub serial: u32,
    /// Per-logical-bitstream page counter, for loss detection.
    pub sequence: u32,
    /// Lacing values: the segment table.
    pub segments: &'a [u8],
    /// The page body (all segments concatenated).
    pub body: &'a [u8],
}

impl<'a> Page<'a> {
    /// Parses one page from the start of `data`, verifying its CRC.
    ///
    /// Returns the page and the total number of bytes it occupies.
    ///
    /// # Errors
    ///
    /// [`OggError::MissingCapturePattern`] if `data` does not begin with
    /// `OggS`; [`OggError::Truncated`] if the declared length overruns
    /// `data`; [`OggError::BadCrc`] on checksum mismatch;
    /// [`OggError::UnsupportedVersion`] for any version other than 0.
    pub fn parse(data: &'a [u8]) -> Result<(Self, usize), OggError> {
        if data.len() < HEADER_LEN {
            return Err(
                if data.starts_with(&CAPTURE_PATTERN) || CAPTURE_PATTERN.starts_with(data) {
                    OggError::Truncated
                } else {
                    OggError::MissingCapturePattern
                },
            );
        }
        if data[0..4] != CAPTURE_PATTERN {
            return Err(OggError::MissingCapturePattern);
        }
        if data[4] != 0 {
            return Err(OggError::UnsupportedVersion(data[4]));
        }

        let n_segments = usize::from(data[26]);
        let body_start = HEADER_LEN + n_segments;
        if data.len() < body_start {
            return Err(OggError::Truncated);
        }
        let segments = &data[HEADER_LEN..body_start];
        let body_len: usize = segments.iter().map(|&v| usize::from(v)).sum();
        let total = body_start + body_len;
        if data.len() < total {
            return Err(OggError::Truncated);
        }

        // CRC covers the whole page with the checksum field zeroed.
        let declared_crc = u32::from_le_bytes([data[22], data[23], data[24], data[25]]);
        let mut actual = crc::update(0, &data[..22]);
        actual = crc::update(actual, &[0, 0, 0, 0]);
        actual = crc::update(actual, &data[26..total]);
        if actual != declared_crc {
            return Err(OggError::BadCrc);
        }

        let flags = data[5];
        Ok((
            Page {
                continued: flags & 0x01 != 0,
                bos: flags & 0x02 != 0,
                eos: flags & 0x04 != 0,
                granule_position: u64::from_le_bytes([
                    data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13],
                ]),
                serial: u32::from_le_bytes([data[14], data[15], data[16], data[17]]),
                sequence: u32::from_le_bytes([data[18], data[19], data[20], data[21]]),
                segments,
                body: &data[body_start..total],
            },
            total,
        ))
    }
}

/// Iterates the pages of a physical Ogg bitstream held in memory.
///
/// On corruption, the reader resynchronizes by scanning forward for the next
/// `OggS` capture pattern with a valid checksum - the recovery behaviour the
/// capture pattern exists for (RFC 3533 §6).
#[derive(Debug, Clone)]
pub struct PageReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> PageReader<'a> {
    /// Creates a reader over a complete physical bitstream.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        PageReader { data, pos: 0 }
    }

    /// The current byte offset into the stream.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }
}

impl<'a> Iterator for PageReader<'a> {
    type Item = Page<'a>;

    fn next(&mut self) -> Option<Page<'a>> {
        while self.pos < self.data.len() {
            match Page::parse(&self.data[self.pos..]) {
                Ok((page, consumed)) => {
                    self.pos += consumed;
                    return Some(page);
                },
                // Truncated final page: nothing more to read.
                Err(OggError::Truncated) => return None,
                // Resync: scan for the next capture pattern.
                Err(_) => {
                    let from = self.pos + 1;
                    match find_capture(&self.data[from..]) {
                        Some(off) => self.pos = from + off,
                        None => return None,
                    }
                },
            }
        }
        None
    }
}

/// Finds the next `OggS` offset in `data`.
fn find_capture(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|w| w == CAPTURE_PATTERN)
}

/// A packet reassembled from one logical bitstream, with its page context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OggPacket {
    /// The packet payload.
    pub data: Vec<u8>,
    /// Granule position of the page on which this packet *completed*, or
    /// [`NO_GRANULE`] when it was not the page's last completed packet.
    pub granule_position: u64,
    /// This packet completed on a page with the end-of-stream flag.
    pub eos: bool,
    /// This packet is the last one completing on its page.
    pub completes_page: bool,
}

/// Reassembles the packets of **one** logical bitstream (selected by serial
/// number) from a physical Ogg stream.
///
/// Implements the continuity rules of RFC 3533 §5 and RFC 7845 §3: a packet
/// spanning pages is dropped - never partially decoded - when the continued
/// flag is missing, a sequence number is skipped, or the stream ends
/// mid-packet.
#[derive(Debug, Clone)]
pub struct PacketReader<'a> {
    pages: PageReader<'a>,
    serial: u32,
    /// Buffered partial packet continuing onto the next page.
    partial: Vec<u8>,
    /// Whether `partial` is valid (a lacing value of 255 ended the last page).
    have_partial: bool,
    /// The in-flight packet exceeded the defensive size cap and will be
    /// discarded when it completes.
    poisoned: bool,
    /// Last seen sequence number, for gap detection.
    last_sequence: Option<u32>,
    /// Queue of packets completed on the current page.
    ready: alloc::collections::VecDeque<OggPacket>,
}

/// Defensive cap on a single reassembled packet (16 MiB). Real Opus packets
/// are ≤ 61 440 bytes (RFC 7845 §6); anything near this cap is garbage input.
const MAX_PACKET_LEN: usize = 16 * 1024 * 1024;

impl<'a> PacketReader<'a> {
    /// Creates a reader for the logical bitstream identified by `serial`.
    #[must_use]
    pub fn new(data: &'a [u8], serial: u32) -> Self {
        PacketReader {
            pages: PageReader::new(data),
            serial,
            partial: Vec::new(),
            have_partial: false,
            poisoned: false,
            last_sequence: None,
            ready: alloc::collections::VecDeque::new(),
        }
    }

    /// Splits one page's body into packets, honouring continuation state.
    fn ingest(&mut self, page: &Page<'a>) {
        // Sequence gap: anything buffered is unrecoverable (RFC 7845 §3).
        let consecutive = self
            .last_sequence
            .is_none_or(|prev| page.sequence == prev.wrapping_add(1));
        self.last_sequence = Some(page.sequence);

        if !consecutive || page.continued != self.have_partial {
            // Sequence gap, a continued first packet without matching state,
            // or stale buffered state without a continued flag: whatever is
            // buffered is unrecoverable (RFC 7845 §3).
            self.partial.clear();
            self.have_partial = false;
            self.poisoned = false;
        }

        let mut offset = 0usize;
        let mut last_complete_idx: Option<usize> = None;

        let mut iter = page.segments.iter().peekable();
        while let Some(&lacing) = iter.next() {
            let len = usize::from(lacing);
            if self.partial.len() + len > MAX_PACKET_LEN {
                // Defensive cap: drop the whole packet once it completes
                // rather than surface a silently truncated one.
                self.poisoned = true;
                self.partial.clear();
            }
            if !self.poisoned {
                self.partial.extend_from_slice(&page.body[offset..offset + len]);
            }
            offset += len;
            self.have_partial = true;

            if lacing < 255 {
                // Packet complete.
                let data = core::mem::take(&mut self.partial);
                let poisoned = core::mem::take(&mut self.poisoned);
                self.have_partial = false;
                if !poisoned {
                    self.ready.push_back(OggPacket {
                        data,
                        granule_position: NO_GRANULE,
                        eos: page.eos,
                        completes_page: iter.peek().is_none(),
                    });
                    last_complete_idx = Some(self.ready.len() - 1);
                }
            }
        }

        // The page's granule position belongs to the last packet completed on
        // it (RFC 7845 §4).
        if let Some(idx) = last_complete_idx {
            self.ready[idx].granule_position = page.granule_position;
        }
    }
}

impl<'a> Iterator for PacketReader<'a> {
    type Item = OggPacket;

    fn next(&mut self) -> Option<OggPacket> {
        loop {
            if let Some(pkt) = self.ready.pop_front() {
                return Some(pkt);
            }
            let page = self.pages.find(|p| p.serial == self.serial)?;
            self.ingest(&page);
        }
    }
}

/// Builds Ogg pages for one logical bitstream.
///
/// Packets are appended with [`push`](Self::push); pages are emitted into an
/// output buffer when full (255 segments) or when explicitly
/// [`flush`](Self::flush)ed. The writer maintains the sequence numbers and
/// the continued/bos/eos flags.
#[derive(Debug, Clone)]
pub struct PageWriter {
    serial: u32,
    sequence: u32,
    /// Segment table of the page being assembled.
    segments: Vec<u8>,
    /// Body of the page being assembled.
    body: Vec<u8>,
    /// Granule position of the last packet completed on the page being
    /// assembled; [`NO_GRANULE`] when none has.
    page_granule: u64,
    /// The page being assembled starts mid-packet.
    continued: bool,
    /// The next emitted page is the first of the stream.
    first: bool,
}

impl PageWriter {
    /// Creates a writer for a logical bitstream with the given serial number.
    #[must_use]
    pub const fn new(serial: u32) -> Self {
        PageWriter {
            serial,
            sequence: 0,
            segments: Vec::new(),
            body: Vec::new(),
            page_granule: NO_GRANULE,
            continued: false,
            first: true,
        }
    }

    /// Appends one packet, emitting full pages into `out` as needed.
    ///
    /// `granule_position` is the codec-defined position as of the end of this
    /// packet; it is recorded on the page where the packet completes
    /// (RFC 7845 §4: the page granule belongs to the last packet completed on
    /// it). When `end_of_stream` is set, the page the packet completes on is
    /// flagged EOS and flushed immediately.
    pub fn push(&mut self, out: &mut Vec<u8>, packet: &[u8], granule_position: u64, end_of_stream: bool) {
        let mut remaining = packet;
        loop {
            // Lacing: full 255-byte segments, then a final short one. A
            // packet that is an exact multiple of 255 bytes loops once more
            // with zero remaining bytes and emits the terminating 0 lacing
            // value (RFC 3533 §5).
            let take = remaining.len().min(255);
            self.segments.push(take as u8);
            self.body.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];

            let packet_done = take < 255;
            if packet_done {
                self.page_granule = granule_position;
            }

            if self.segments.len() == MAX_SEGMENTS {
                self.emit(out, end_of_stream && packet_done);
                self.continued = !packet_done;
            }

            if packet_done {
                break;
            }
        }

        if end_of_stream && !self.segments.is_empty() {
            self.emit(out, true);
        }
    }

    /// Emits the partially filled page, if any.
    ///
    /// RFC 7845 §3 requires the ID header to sit alone on the first page and
    /// the comment header to finish its final page; call this after each.
    pub fn flush(&mut self, out: &mut Vec<u8>) {
        if !self.segments.is_empty() {
            self.emit(out, false);
        }
    }

    /// Serialises the assembled page and resets the assembly buffers.
    fn emit(&mut self, out: &mut Vec<u8>, eos: bool) {
        let mut flags = 0u8;
        if self.continued {
            flags |= 0x01;
        }
        if self.first {
            flags |= 0x02;
        }
        if eos {
            flags |= 0x04;
        }

        let header_start = out.len();
        out.extend_from_slice(&CAPTURE_PATTERN);
        out.push(0); // version
        out.push(flags);
        out.extend_from_slice(&self.page_granule.to_le_bytes());
        out.extend_from_slice(&self.serial.to_le_bytes());
        out.extend_from_slice(&self.sequence.to_le_bytes());
        let crc_at = out.len();
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.push(self.segments.len() as u8);
        out.extend_from_slice(&self.segments);
        out.extend_from_slice(&self.body);

        let crc = crc::update(0, &out[header_start..]);
        out[crc_at..crc_at + 4].copy_from_slice(&crc.to_le_bytes());

        self.sequence = self.sequence.wrapping_add(1);
        self.segments.clear();
        self.body.clear();
        self.page_granule = NO_GRANULE;
        self.continued = false;
        self.first = false;
    }
}
