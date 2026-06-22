//! The multistream decoder (RFC 7845 §5.1.1 layouts): N elementary Opus
//! streams per packet -
//! the first `coupled` decoded as stereo, the rest mono - routed to output
//! channels through a mapping table. All streams but the last use
//! self-delimited framing (RFC 6716 Appendix B).

use alloc::vec;
use alloc::vec::Vec;

use crate::decoder::OpusDecoder;
use crate::packet::{Packet, PacketError};

/// A multistream Opus decoder.
pub struct MultistreamDecoder {
    decoders: Vec<OpusDecoder>,
    channels: usize,
    coupled: usize,
    mapping: Vec<u8>,
    fs: u32,
}

impl MultistreamDecoder {
    /// Creates a decoder for `streams` elementary streams (the first
    /// `coupled` stereo) mapped onto `mapping.len()` output channels;
    /// `mapping[ch]` selects a decoded channel index or 255 for silence.
    ///
    /// # Panics
    ///
    /// Panics on invalid layouts (RFC 7845 §5.1.1 limits) or rates.
    #[must_use]
    pub fn with_rate(fs_hz: u32, streams: usize, coupled: usize, mapping: &[u8]) -> Self {
        assert!(streams >= 1 && coupled <= streams && streams + coupled <= 255);
        let decoded_channels = (streams + coupled) as u8;
        assert!(
            mapping.iter().all(|&m| m == 255 || m < decoded_channels),
            "mapping entry out of range"
        );
        let mut decoders = Vec::with_capacity(streams);
        for s in 0..streams {
            decoders.push(OpusDecoder::with_rate(fs_hz, if s < coupled { 2 } else { 1 }));
        }
        MultistreamDecoder {
            decoders,
            channels: mapping.len(),
            coupled,
            mapping: mapping.to_vec(),
            fs: fs_hz,
        }
    }

    /// Output channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Decodes one multistream packet to interleaved f32 at the decoder
    /// rate.
    ///
    /// # Errors
    ///
    /// Returns the packet-layer error for malformed payloads (including
    /// streams of differing durations).
    pub fn decode_packet(&mut self, data: &[u8]) -> Result<Vec<f32>, PacketError> {
        let streams = self.decoders.len();
        let mut rest = data;

        // Parse and decode every elementary stream.
        let mut stream_pcm: Vec<Vec<f32>> = Vec::with_capacity(streams);
        let mut duration = None;
        for s in 0..streams {
            let packet = if s != streams - 1 {
                let (packet, used) = Packet::parse_self_delimited(rest)?;
                rest = &rest[used..];
                packet
            } else {
                Packet::parse(rest)?
            };
            let frames = packet.frames().len();
            let dur48 = frames * packet.toc().frame_size().samples_per_channel_48k();
            if *duration.get_or_insert(dur48) != dur48 {
                return Err(PacketError::InvalidFrameCount);
            }
            stream_pcm.push(self.decoders[s].decode_parsed(&packet));
        }
        let n = duration.unwrap_or(0) * self.fs as usize / 48_000;

        // Route decoded channels to the output layout.
        let mut out = vec![0.0f32; n * self.channels];
        for (ch, &m) in self.mapping.iter().enumerate() {
            if m == 255 {
                continue; // silence
            }
            let m = usize::from(m);
            let (s, sub, sch) = if m < 2 * self.coupled {
                (m / 2, m % 2, 2)
            } else {
                (self.coupled + (m - 2 * self.coupled), 0, 1)
            };
            let src = &stream_pcm[s];
            for i in 0..n {
                out[i * self.channels + ch] = src[i * sch + sub];
            }
        }
        Ok(out)
    }
}
