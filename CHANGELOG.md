# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Range decoder and encoder (RFC 6716 §4.1/§5.1): symbol, binary, ICDF, raw-bits, and uniform-integer coding with `tell`/`tell_frac`, verified by encoder/decoder `rng`-agreement round-trips
