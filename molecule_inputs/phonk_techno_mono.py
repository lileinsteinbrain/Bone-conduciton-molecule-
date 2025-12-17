import json
import numpy as np
from scipy.io.wavfile import write

SR = 44100
DUR = 16.0  # seconds


def load_molecular_rhythm(json_path="ethanol_mapped_params.json"):
    """
    从 ethanol_mapped_params.json 里：
    - 读出每个峰的波数 wavenumber_cm-1
    - 读出 audio_amp（相对强度）
    然后把波数映射到 0.6–2.0 Hz 的分子节奏频率。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    peaks = data["mapped_peaks"]

    wavenumbers = np.array(
        [p["wavenumber_cm-1"] for p in peaks], dtype=float
    )
    amps = np.array(
        [p.get("audio_amp", 1.0) for p in peaks], dtype=float
    )

    # 归一化振幅
    if amps.max() > 0:
        amps = amps / amps.max()

    w_min = float(wavenumbers.min())
    w_max = float(wavenumbers.max())
    if w_max == w_min:
        wn_norm = np.ones_like(wavenumbers) * 0.5
    else:
        wn_norm = (wavenumbers - w_min) / (w_max - w_min)

    # 把波数映射到 0.6–2.0 Hz 的分子 rhythm
    rhythm_hz = 0.6 + 1.4 * wn_norm

    return rhythm_hz, amps


def build_phonk_techno_mono():
    t = np.linspace(0.0, DUR, int(SR * DUR), endpoint=False)

    # === 1. 分子节奏 ===
    rhythm_hz, amps = load_molecular_rhythm()

    # 用所有分子 rhythm 叠加出一个总的分子 envelope
    mol_env = np.zeros_like(t)
    for a, r in zip(amps, rhythm_hz):
        mol_env += a * (0.5 + 0.5 * np.sin(2 * np.pi * r * t))
    if np.max(np.abs(mol_env)) > 0:
        mol_env = mol_env / np.max(np.abs(mol_env))

    # 句子级淡入淡出，让开头结尾不要太突兀
    fade_in = np.clip(t / 2.0, 0.0, 1.0)
    fade_out = np.clip((DUR - t) / 2.0, 0.0, 1.0)
    phrase_env = np.minimum(fade_in, fade_out)

    # === 2. Techno / Phonk 主体 ===
    bpm = 145.0
    beat_len = 60.0 / bpm

    # 2.1 四拍 kick（60 Hz 带一点 pitch drop）
    kick = np.zeros_like(t)
    for k in range(int(DUR / beat_len) + 2):
        start = k * beat_len
        phase = t - start
        mask = phase >= 0
        env = np.exp(-phase * 35.0)
        env[~mask] = 0.0

        # 简单的 pitch drop：80 -> 55 Hz
        f0 = 80.0 - 25.0 * np.clip(phase, 0.0, 0.15) / 0.15
        sig = np.sin(2 * np.pi * f0 * t)
        kick += 1.3 * sig * env

    # 2.2 sub-bass（55 Hz），跟 off-beat 走
    bass = np.sin(2 * np.pi * 55.0 * t)
    bass_gate = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * (bpm / 120.0) * t))
    bass = bass * bass_gate * 0.8

    # 2.3 phonk-ish 中频 texture（简单的多声部）
    lead = np.zeros_like(t)
    base_freq = 220.0
    for i in range(min(4, len(rhythm_hz))):
        f = base_freq + 40.0 * i
        r = rhythm_hz[i]
        a = amps[i]
        # 利用分子 rhythm 做一点 FM 抖动
        lead += a * np.sin(
            2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * r * t)
        )
    lead *= 0.25  # 不要盖住 kick 和 bass

    music_core = kick + bass + lead

    # === 3. 把分子 rhythm 嵌进骨震调制里 ===
    # 分子 envelope 作为整体音量的 40% 调制
    track = music_core * (0.6 + 0.4 * mol_env) * phrase_env

    # 再做一次归一化，防止爆音
    max_val = np.max(np.abs(track)) + 1e-9
    track = track / max_val

    return track.astype(np.float32)


if __name__ == "__main__":
    audio = build_phonk_techno_mono()
    out = (audio * 32767).astype(np.int16)
    write("ethanol_phonk_techno_mono.wav", SR, out)
    print("✔ wrote ethanol_phonk_techno_mono.wav")