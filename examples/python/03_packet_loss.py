"""Packet-loss concealment (PLC) and forward error correction (FEC).

When a packet is lost, the decoder can conceal it from prior state with
`decode_lost`, or - if the *next* received packet carries in-band FEC (LBRR)
data - partially recover it with `decode_fec`.

Run: python examples/python/03_packet_loss.py
"""

import numpy as np

from opus_native import OpusDecoder, OpusEncoder

SR = 48000
FRAME = 960  # 20 ms


def make_frame(i: int) -> np.ndarray:
    t = (np.arange(FRAME) + i * FRAME) / SR
    return (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32).reshape(FRAME, 1)


def main() -> None:
    enc = OpusEncoder(1, bitrate=48000)
    packets = [enc.encode(make_frame(i)) for i in range(5)]

    dec = OpusDecoder(1)

    # Decode packets 0 and 1 normally.
    for i in (0, 1):
        dec.decode_packet(packets[i])

    # Packet 2 is "lost": conceal it from the decoder's state. The concealed
    # frame is the duration of the last good packet (20 ms -> 960 samples).
    concealed = dec.decode_lost(FRAME)
    print(f"PLC: concealed a lost packet -> {concealed.shape} {concealed.dtype}")

    # `decode_fec` recovers a lost frame from the *next* packet's FEC data.
    # Plain packets carry no LBRR, so this falls back to concealment - the call
    # shape is the same either way.
    recovered = dec.decode_fec(packets[3], FRAME)
    print(f"FEC: requested recovery from the next packet -> {recovered.shape}")
    print("(plain packets have no LBRR, so this fell back to concealment)")

    # Resume normal decoding.
    dec.decode_packet(packets[3])
    dec.decode_packet(packets[4])
    print("resumed normal decoding after the gap")


if __name__ == "__main__":
    main()
