#!/usr/bin/env python3
import json
import math
import wave
from pathlib import Path

import struct

# 基本参数
MAPPED_FILE = "ethanol_mapped_params.json"
OUT_WAV = "ethanol_track.wav"

SAMPLE_RATE = 48000        # CD级采样率
DURATION_SEC = 16          # 整段声音时长（可改）
MASTER_GAIN = 0.7          # 总音量

# 电子 / techno 感的节奏参数
BPM = 120                  # 120BPM
BEAT_FREQ = BPM / 60.0     # 2Hz, 4拍感
PULSE_DEPTH = 0.8          # 0~1，脉冲感强度


def load_mapped_peaks(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["mapped_peaks"]


def main():
    root = Path(".")
    peaks = load_mapped_peaks(root / MAPPED_FILE)

    if not peaks:
        print("No mapped peaks found.")
        return

    # 只取前 N 个强峰，避免太糊
    # 可以按 intensity_norm 排序
    peaks_sorted = sorted(peaks, key=lambda p: p["intensity_norm"], reverse=True)
    TOP_N = 4
    use_peaks = peaks_sorted[:TOP_N]

    print("Using peaks:")
    for p in use_peaks:
        print(
            f"  {p['wavenumber_cm-1']} cm-1 | "
            f"int={p['intensity_norm']} | "
            f"freq={p['audio_freq_hz']} Hz"
        )

    num_samples = int(SAMPLE_RATE * DURATION_SEC)
    samples = []

    for n in range(num_samples):
        t = n / SAMPLE_RATE

        # 全局 techno 呼吸 / 脉冲 envelope
        # envelope 在 1-PULSE_DEPTH ~ 1 之间波动
        pulse = 0.5 * (1.0 + math.sin(2.0 * math.pi * BEAT_FREQ * t))
        envelope = (1.0 - PULSE_DEPTH) + PULSE_DEPTH * pulse

        # 混合所有峰：每个峰一个正弦
        v = 0.0
        for p in use_peaks:
            freq = float(p["audio_freq_hz"])
            amp = float(p["audio_amp"])

            # 简单的 amplitude scaling，防止爆
            amp_scaled = amp / TOP_N

            v += amp_scaled * math.sin(2.0 * math.pi * freq * t)

        # 应用节奏 envelope
        v *= envelope

        # Master gain
        v *= MASTER_GAIN

        # 限幅 [-1, 1]
        if v > 1.0:
            v = 1.0
        elif v < -1.0:
            v = -1.0

        samples.append(v)

    # 写成 16bit PCM WAV，单声道
    with wave.open(str(root / OUT_WAV), "w") as wf:
        wf.setnchannels(1)          # mono
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(SAMPLE_RATE)

        frames = bytearray()
        for v in samples:
            # float(-1~1) → int16
            s = int(v * 32767.0)
            frames += struct.pack("<h", s)

        wf.writeframes(frames)

    print(f"✅ Wrote audio to {OUT_WAV}")


if __name__ == "__main__":
    main()