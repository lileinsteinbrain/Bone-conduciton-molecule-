import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def load_spectrum(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    peaks = data.get("peaks", [])
    if not peaks:
        raise ValueError("JSON 里没有 'peaks' 字段或为空")

    # 保证每个 peak 有 intensity，缺省为 1.0
    for p in peaks:
        if "intensity" not in p:
            p["intensity"] = 1.0

    return data, peaks


def map_wavenumber_to_freqs(wavenumber, wn_min, wn_max):
    """
    把波数区间 [wn_min, wn_max] 映射到：
      - 音轨: 220–880 Hz
      - 触觉: 40–220 Hz
    """
    if wn_max == wn_min:
        x = 0.5
    else:
        x = (wavenumber - wn_min) / (wn_max - wn_min)
        x = max(0.0, min(1.0, x))

    audio_f = 220.0 + x * (880.0 - 220.0)
    tactile_f = 40.0 + x * (220.0 - 40.0)
    return audio_f, tactile_f


def make_envelope(n_samples):
    """
    简单的 ADS-ish 包络：前 10% 线性起，后 30% 线性衰，中间保持。
    """
    n_attack = int(n_samples * 0.1)
    n_release = int(n_samples * 0.3)
    n_sustain = max(0, n_samples - n_attack - n_release)

    env = np.ones(n_samples, dtype=np.float32)

    if n_attack > 0:
        env[:n_attack] = np.linspace(0.0, 1.0, n_attack, endpoint=False)

    if n_release > 0:
        env[-n_release:] = np.linspace(1.0, 0.0, n_release, endpoint=True)

    return env


def build_tracks(peaks, total_duration=12.0, sample_rate=44100):
    """
    核心合成逻辑：
      - 整体时长 total_duration（默认 12 秒）
      - 每个峰占一段时间（平均切分）
      - 峰的波数 → audio/tactile 频率
      - 峰的强度 → 振幅
      - 若 JSON 里有 mapping.rhythm → 作为该段 tremolo 节奏因子
    """
    n_samples = int(total_duration * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)
    tactile = np.zeros(n_samples, dtype=np.float32)

    # 提取波数 & 强度范围
    wns = np.array([p["wavenumber_cm-1"] for p in peaks], dtype=np.float32)
    intensities = np.array([p.get("intensity", 1.0) for p in peaks], dtype=np.float32)

    wn_min = float(wns.min())
    wn_max = float(wns.max())
    if intensities.max() == 0:
        intensities_norm = np.ones_like(intensities)
    else:
        intensities_norm = intensities / intensities.max()

    n_peaks = len(peaks)
    seg_samples = n_samples // n_peaks

    for i, peak in enumerate(peaks):
        start = i * seg_samples
        end = n_samples if i == n_peaks - 1 else (i + 1) * seg_samples
        seg_len = end - start
        if seg_len <= 0:
            continue

        t = np.linspace(0.0, (seg_len - 1) / sample_rate, seg_len, dtype=np.float32)

        wn = peak["wavenumber_cm-1"]
        audio_f, tactile_f = map_wavenumber_to_freqs(wn, wn_min, wn_max)

        amp = float(intensities_norm[i])

        # 如果 JSON 里有 mapping.rhythm，则用来调节该段的 tremolo 频率
        rhythm_factor = 1.0
        mapping = peak.get("mapping")
        if isinstance(mapping, dict) and "rhythm" in mapping:
            try:
                rhythm_factor = float(mapping["rhythm"])
            except Exception:
                rhythm_factor = 1.0

        # 在这一段里做一点节奏性的 tremolo
        # tremolo 频率由 rhythm_factor 映射到 0.6–2.0 Hz
        rhythm_clamped = min(max(rhythm_factor, 0.0), 2.0)
        trem_f = 0.6 + rhythm_clamped * 0.7
        trem = 0.5 * (1.0 + np.sin(2 * np.pi * trem_f * t))

        env = make_envelope(seg_len)
        seg_audio = amp * env * trem * np.sin(2 * np.pi * audio_f * t)
        seg_tactile = amp * env * trem * np.sin(2 * np.pi * tactile_f * t)

        audio[start:end] += seg_audio
        tactile[start:end] += seg_tactile

    # 归一化，避免削波
    max_val = max(float(np.max(np.abs(audio))), 1e-6)
    audio /= max_val + 1e-6

    # 触觉信号整体小一点，给放大器留 headroom
    max_val_t = max(float(np.max(np.abs(tactile))), 1e-6)
    tactile /= (max_val_t + 1e-6)
    tactile *= 0.6

    return audio, tactile, sample_rate


def main():
    # 默认找 ethanol_ir.json；如果你传别的文件名就用那个
    json_name = "ethanol_ir.json"
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if not arg.startswith("-"):
                json_name = arg
                break

    json_path = Path(json_name)
    if not json_path.exists():
        raise SystemExit(f"找不到 {json_path}，请确认文件名。")

    data, peaks = load_spectrum(json_path)
    molecule = data.get("molecule", "unknown")

    print(f"Loaded {len(peaks)} peaks for {molecule} from {json_path}")

    audio, tactile, sr = build_tracks(peaks)

    base = json_path.with_suffix("").name
    audio_path = base + "_audio.wav"
    tactile_path = base + "_tactile.wav"

    sf.write(audio_path, audio, sr, subtype="PCM_16")
    sf.write(tactile_path, tactile, sr, subtype="PCM_16")

    print(f"Done! Generated {audio_path} (for speakers/headphones)")
    print(f"      and {tactile_path} (for bone conduction amp).")


if __name__ == "__main__":
    main()
