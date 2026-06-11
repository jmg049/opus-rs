//! The CELT layer (RFC 6716 §4.3) - under construction.
//!
//! CELT is the MDCT half of Opus: band energies coded with a Laplace model
//! plus fine refinement, and band shapes coded as PVQ pulse vectors. The
//! decode pipeline (§4.3) is
//!
//! ```text
//! coarse energy → fine energy → bit allocation → PVQ shapes →
//! anti-collapse → denormalization → inverse MDCT → post-filter
//! ```
//!
//! Implemented so far, bottom-up - each kernel fully tested in isolation
//! before the pipeline is assembled:
//!
//! | Module | Reference | Contents |
//! |--------|-----------|----------|
//! | [`laplace`] | `laplace.c`, §4.3.2.1 | the Laplace coder for coarse energy deltas |
//! | [`cwrs`] | `cwrs.c`, §4.3.4.2 | PVQ codeword enumeration (pulse vectors ↔ indices) |

pub mod cwrs;
pub mod laplace;
