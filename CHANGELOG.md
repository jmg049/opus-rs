# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Packet framing layer (RFC 6716 §3): TOC byte introspection (mode/bandwidth/frame size per Table 2), frame packing codes 0-3, padding, and full [R1]-[R7] malformed-packet validation
- Range decoder and encoder (RFC 6716 §4.1/§5.1): symbol, binary, ICDF, raw-bits, and uniform-integer coding with `tell`/`tell_frac`, verified by encoder/decoder `rng`-agreement round-trips
