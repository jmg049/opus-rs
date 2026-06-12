# opus_native

A pure-Rust implementation of the [Opus audio codec](https://opus-codec.org/)
([RFC 6716](https://www.rfc-editor.org/rfc/rfc6716)).

**No FFI. No unsafe code. No dependencies.**

> **Status: pre-release, under active development.** The decoder is
> feature-complete and passes the official conformance criterion: all twelve
> test vectors at 99.2-100% `opus_compare` quality, bit-exact final ranges on
> every packet, sample-identical SILK PCM. Loss concealment, in-band FEC, and
> DTX are validated sample-exact against libopus; output rates 8-48 kHz;
> surround (multistream) files decode end to end. The encoder is next.
> Nothing here is API-stable yet.

## Why

Every Rust project that touches Opus today links `libopus` through FFI. There is
no complete, conformant, pure-Rust implementation - this crate exists to be that:
a decoder (first) and encoder (second) that pass the official Opus test vectors,
compile to any target Rust reaches (including `wasm32` and embedded `no_std`),
and can be embedded by higher-level audio crates without a C toolchain.

`opus_native` is deliberately standalone: it depends on nothing and exposes
plain `&[u8]`/`&[i16]`/`&[f32]` interfaces, so it can sit underneath any audio
framework.

## Design principles

- **Bit-exact**: all entropy-coder arithmetic follows RFC 6716 exactly; the
  encoder's range state is verified to match the decoder's symbol-for-symbol.
- **Decode-first**: the decoder is the normative half of the spec and the half
  that conformance vectors exercise. Encoder work follows decoder work at every
  layer.
- **`forbid(unsafe_code)`**: enforced at the crate level.
- **`no_std` + `alloc`**: the `std` feature (on by default) only adds
  `std::error::Error` impls and conveniences.
- **Fast by default, zero-dep by choice**: the codec targets near-real-time
  streaming, so the default build routes the MDCT's inner FFT through the
  [`spectrograms`](https://crates.io/crates/spectrograms) crate's planned
  FFTs (~10× faster decode). The FFT sits behind a seam: disabling the
  `spectrograms` feature leaves a dependency-free build (the built-in
  evaluation) for embedded/wasm-minimal targets. Everything else in the
  crate is dependency-free either way.

## Layout

| Module | RFC 6716 | Status |
|--------|----------|--------|
| `range` | §4.1, §5.1 | Range decoder + encoder, raw bits, uniform ints, `tell`/`tell_frac` |
| `packet` | §3 | TOC parsing, frame packing codes 0-3, padding, R1-R7 validation |
| `lpc` | §4.2/§5.2 groundwork | Levinson-Durbin, LP analysis/synthesis, pitch estimation, LTP |
| `experimental` | - | pre-conformance frame codec, mode detection, crossover, mid/side (feature `experimental-codec`) |
| `decoder` | §4 | the Opus decoder: TOC dispatch, hybrid, redundancy, transitions, PLC/FEC/DTX, 8-48 kHz output - all twelve official vectors bit-exact on the final-range oracle |
| `multistream` | RFC 7845 §5.1.1 | surround layouts: self-delimited demux, N decoders, channel mapping |
| `celt` | §4.3 | complete decoder (RFC 8251 updates included) |
| `silk` | §4.2 | complete decoder for the normal path (PLC/CNG pending); pure SILK vectors decode sample-identically |
| `ogg` | RFC 3533 + RFC 7845 | Ogg pages (CRC, lacing, resync), packet reassembly, `OpusHead`/`OpusTags`, granule/pre-skip/end-trim timing, stream reader + writer |

## Performance

The default build decodes far beyond realtime (one core, release build,
official conformance vectors):

| Stream | `spectrograms` FFT (default) | built-in FFT (zero-dep) |
|--------|------------------------------|-------------------------|
| testvector01 (CELT stereo SWB/FB) | ~410× realtime | ~10× realtime |
| testvector07 (CELT stereo, all bandwidths) | ~730× realtime | - |

Reproduce with
`cargo run --release --example decode_throughput tests/vectors/testvector01.bit`.

## Conformance

The decoder **passes the official conformance criterion**: every one of
the twelve vectors scores 99.2-100% on the `opus_compare` quality metric
(pass bar: ≥ 0%), with per-packet final ranges bit-exact across the whole
suite. It is built against the official
[Opus test vectors](https://opus-codec.org/testvectors/) (RFC 8251 set).
Fetch them with `tools/fetch-testvectors.sh` (~121 MB, not committed); the
conformance tests in `tests/conformance.rs` skip cleanly when absent. The
packet layer validates against every packet of all twelve vectors. The
CELT-only vectors (testvector01/07/11) decode with per-packet final-range
equality - the bit-exactness oracle - and the synthesized PCM scores
83-104 dB SNR against the reference decode, far beyond the official
`opus_compare` criterion. The harness grows the remaining vectors as the
SILK decoder lands.

## License

MIT, see [LICENSE](LICENSE).

The Opus codec itself is royalty-free; see the
[Opus IPR statements](https://datatracker.ietf.org/ipr/search/?rfc=6716&submit=rfc).
