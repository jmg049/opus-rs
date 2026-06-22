//! Python enums mirroring `opus_native`'s public packet enums.
//!
//! Each is a thin value-type wrapper over the corresponding core enum
//! (`crate::packet::{Mode, Bandwidth, FrameSize}`), kept here so the core stays
//! dependency-free. `From` impls convert in both directions, and the
//! enum-specific helper methods delegate to the core implementation, so Python
//! and Rust expose identical behaviour under identical names.

use pyo3::prelude::*;
use std::time::Duration;

/// Operating mode of an Opus frame (RFC 6716 §3.1).
#[pyclass(eq, eq_int, frozen, hash, from_py_object, module = "opus_native", name = "Mode")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// LP (SILK) layer only: low-bitrate speech, narrowband through wideband.
    SilkOnly,
    /// SILK below 8 kHz plus CELT above: super-wideband/fullband speech.
    Hybrid,
    /// MDCT (CELT) layer only: music and low-delay use, narrowband to fullband.
    CeltOnly,
}

impl From<crate::packet::Mode> for Mode {
    fn from(m: crate::packet::Mode) -> Self {
        match m {
            crate::packet::Mode::SilkOnly => Mode::SilkOnly,
            crate::packet::Mode::Hybrid => Mode::Hybrid,
            crate::packet::Mode::CeltOnly => Mode::CeltOnly,
        }
    }
}

impl From<Mode> for crate::packet::Mode {
    fn from(m: Mode) -> Self {
        match m {
            Mode::SilkOnly => crate::packet::Mode::SilkOnly,
            Mode::Hybrid => crate::packet::Mode::Hybrid,
            Mode::CeltOnly => crate::packet::Mode::CeltOnly,
        }
    }
}

/// Audio bandwidth of an Opus frame (RFC 6716 §2.1.3, §3.1).
#[pyclass(
    eq,
    eq_int,
    frozen,
    hash,
    ord,
    from_py_object,
    module = "opus_native",
    name = "Bandwidth"
)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[pymethods]
impl Bandwidth {
    /// The effective sample rate in Hz for this bandwidth.
    #[getter]
    fn sample_rate_hz(&self) -> u32 {
        crate::packet::Bandwidth::from(*self).sample_rate_hz()
    }

    /// The audio bandwidth in Hz (the highest frequency reproduced).
    #[getter]
    fn audio_bandwidth_hz(&self) -> u32 {
        crate::packet::Bandwidth::from(*self).audio_bandwidth_hz()
    }
}

impl From<crate::packet::Bandwidth> for Bandwidth {
    fn from(b: crate::packet::Bandwidth) -> Self {
        match b {
            crate::packet::Bandwidth::NarrowBand => Bandwidth::NarrowBand,
            crate::packet::Bandwidth::MediumBand => Bandwidth::MediumBand,
            crate::packet::Bandwidth::WideBand => Bandwidth::WideBand,
            crate::packet::Bandwidth::SuperWideBand => Bandwidth::SuperWideBand,
            crate::packet::Bandwidth::FullBand => Bandwidth::FullBand,
        }
    }
}

impl From<Bandwidth> for crate::packet::Bandwidth {
    fn from(b: Bandwidth) -> Self {
        match b {
            Bandwidth::NarrowBand => crate::packet::Bandwidth::NarrowBand,
            Bandwidth::MediumBand => crate::packet::Bandwidth::MediumBand,
            Bandwidth::WideBand => crate::packet::Bandwidth::WideBand,
            Bandwidth::SuperWideBand => crate::packet::Bandwidth::SuperWideBand,
            Bandwidth::FullBand => crate::packet::Bandwidth::FullBand,
        }
    }
}

/// Duration of one Opus frame (RFC 6716 §3.1, Table 2).
#[pyclass(
    eq,
    eq_int,
    frozen,
    hash,
    ord,
    from_py_object,
    module = "opus_native",
    name = "FrameSize"
)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[pymethods]
impl FrameSize {
    /// Frame duration in tenths of a millisecond (exact for 2.5 ms).
    #[getter]
    fn tenth_ms(&self) -> u32 {
        crate::packet::FrameSize::from(*self).tenth_ms()
    }

    /// Number of samples per channel in one frame at 48 kHz.
    #[getter]
    fn samples_per_channel_48k(&self) -> usize {
        crate::packet::FrameSize::from(*self).samples_per_channel_48k()
    }

    /// Frame duration as a `datetime.timedelta`.
    #[getter]
    fn duration(&self) -> Duration {
        crate::packet::FrameSize::from(*self).duration()
    }
}

impl From<crate::packet::FrameSize> for FrameSize {
    fn from(f: crate::packet::FrameSize) -> Self {
        match f {
            crate::packet::FrameSize::Ms2_5 => FrameSize::Ms2_5,
            crate::packet::FrameSize::Ms5 => FrameSize::Ms5,
            crate::packet::FrameSize::Ms10 => FrameSize::Ms10,
            crate::packet::FrameSize::Ms20 => FrameSize::Ms20,
            crate::packet::FrameSize::Ms40 => FrameSize::Ms40,
            crate::packet::FrameSize::Ms60 => FrameSize::Ms60,
        }
    }
}

impl From<FrameSize> for crate::packet::FrameSize {
    fn from(f: FrameSize) -> Self {
        match f {
            FrameSize::Ms2_5 => crate::packet::FrameSize::Ms2_5,
            FrameSize::Ms5 => crate::packet::FrameSize::Ms5,
            FrameSize::Ms10 => crate::packet::FrameSize::Ms10,
            FrameSize::Ms20 => crate::packet::FrameSize::Ms20,
            FrameSize::Ms40 => crate::packet::FrameSize::Ms40,
            FrameSize::Ms60 => crate::packet::FrameSize::Ms60,
        }
    }
}
