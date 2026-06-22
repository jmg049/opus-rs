# opus_native

A pure-Rust implementation of the [Opus audio codec](https://opus-codec.org/)
([RFC 6716](https://www.rfc-editor.org/rfc/rfc6716)) - decoder **and** encoder,
no C and no FFI.

**Pure Rust · `unsafe` only in a few SIMD kernels, every one checked under
[Miri](https://github.com/rust-lang/miri) · runs anywhere Rust runs, including
`wasm32` and `no_std`.**

> Pre-release, not yet API-stable. The decoder passes the official Opus
> conformance vectors; the encoder produces standard Opus that libopus and
> ffmpeg decode. Both run at hundreds of × realtime on one core.

## Why it matters

Every Rust project that touches Opus today links `libopus` over FFI - a C
toolchain in the build, a non-Rust blob in the binary, and no `wasm` or
`no_std`. `opus_native` is the missing pure-Rust codec:

- **No C toolchain, no FFI.** Pure Rust top to bottom; compiles to every target
  Rust reaches. Plain `&[u8]`/`&[i16]`/`&[f32]` interfaces drop under any audio
  stack.
- **Minimal, audited `unsafe`.** The crate denies `unsafe` by default. The only
  exceptions are a handful of `std::arch` SIMD hot loops, each with a
  `// SAFETY:` justification ([`docs/unsafe.md`](docs/unsafe.md)) and
  **machine-checked for undefined behaviour by Miri** - on both the SSE2 and
  AVX2 paths - via `tools/miri.sh`. No `portable_simd`, no inline asm.
- **Zero-dependency core.** The default build adds one optional FFT dependency
  for fast decoding; `default-features = false` is fully dependency-free for
  embedded and `wasm`-minimal targets.

## Use

```toml
[dependencies]
opus_native = "0.1"
```

```rust
use opus_native::{OpusDecoder, OpusEncoder};

// Decode Opus packets to interleaved f32 PCM.
let mut dec = OpusDecoder::new(2); // channels
let pcm = dec.decode_packet(&packet)?;

// Encode 48 kHz PCM (one 20 ms frame = 960 samples/channel, interleaved).
let mut enc = OpusEncoder::new(1);
enc.set_bitrate(Some(24_000));
let packet = enc.encode_auto(&pcm_960, 4000)?;
```

```rust
// Whole Ogg Opus files.
let (pcm, head) = opus_native::decode_ogg_opus(&bytes)?;
let ogg = opus_native::encode_ogg_opus(&pcm, 2, 96_000);
```

## Performance

Measured against **libopus 1.6.1** (SIMD-enabled C) on identical data, one core,
pinned to a single performance core: `cargo bench --bench vs_libopus --features
std`. × realtime; "ratio" is `opus_native ÷ libopus`.

**Decode**

| Mode | opus_native | libopus | ratio |
|------|-------------|---------|-------|
| SILK wideband 16 kb/s | **2095×** | 1171× | **1.79×** |
| hybrid fullband 32 kb/s | **1199×** | 850× | **1.41×** |
| CELT fullband 64 kb/s | 1389× | 1566× | 0.89× |

Pure-Rust decode beats SIMD libopus on speech; CELT trails only on the MDCT,
where libopus's SIMD still wins.

**Encode** (matched complexity)

| Mode | opus_native | libopus | ratio |
|------|-------------|---------|-------|
| SILK wideband 16 kb/s | 615× | 743× | 0.83× |
| hybrid fullband 32 kb/s | 482× | 559× | 0.86× |
| CELT fullband 64 kb/s | 778× | 1092× | 0.71× |

Every mode encodes far beyond realtime. At matched complexity it runs at
0.7-0.9× of libopus and is still being tuned; against libopus's *default*
complexity it is 1.3-2.7× faster (it doesn't yet spend cycles on
delayed-decision NSQ or warped noise shaping).

## Conformance

Passes the official Opus conformance criterion: all twelve
[RFC 8251 test vectors](https://opus-codec.org/testvectors/) score 99.2-100% on
`opus_compare`, with per-packet final ranges bit-exact. Fetch the vectors with
`tools/fetch-testvectors.sh` (~121 MB, not committed); the conformance tests
skip cleanly without them.

## License

MIT, see [LICENSE](LICENSE). The Opus format is royalty-free; see the
[Opus IPR statements](https://datatracker.ietf.org/ipr/search/?rfc=6716&submit=rfc).
