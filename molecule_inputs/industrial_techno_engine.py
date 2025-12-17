import json
import numpy as np
from rhythm_engine import build_rhythm_envelopes  # 你已有的节奏引擎

SR = 44100


def load_molecule_params(path="ethanol_mapped_params.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_global_params(mapped):
    """从 mapped_peaks 里压缩出几个全局控制参数。"""
    peaks = mapped["mapped_peaks"]
    wns = np.array([p["wavenumber_cm-1"] for p in peaks], dtype=float)
    ints = np.array([p["intensity_norm"] for p in peaks], dtype=float)

    # 0–1 归一
    w_norm = (wns - wns.min()) / (wns.max() - wns.min() + 1e-9)
    i_norm = (ints - ints.min()) / (ints.max() - ints.min() + 1e-9)

    brightness = float(w_norm.mean())          # 越高 → 越亮、越快
    density = float(i_norm.mean())            # 越高 → 越密集
    contrast = float(i_norm.std() * 2.0)      # 峰强度差异 → 动态对比

    # tempo：140–180 BPM
    bpm = 140.0 + 40.0 * brightness

    # 失真：0.3–0.9
    drive = 0.3 + 0.6 * density

    # sidechain pump：0.25–0.8
    pump = 0.25 + 0.55 * contrast
    pump = max(0.25, min(0.8, pump))

    return {
        "bpm": bpm,
        "drive": drive,
        "pump": pump,
        "brightness": brightness,
        "density": density,
    }


def _make_time(duration_sec, sr=SR):
    n = int(duration_sec * sr)
    t = np.linspace(0.0, duration_sec, n, endpoint=False)
    return t, n


def _phonk_kick_track(t, bpm, punch=1.0):
    """4-on-the-floor 踢 + 一点 phonk 风格的低频拖尾。"""
    sr = SR
    beat_hz = bpm / 60.0
    n = t.shape[0]
    kick = np.zeros_like(t)

    # 每个整拍触发一次
    samples_per_beat = int(sr / beat_hz)
    positions = np.arange(0, n, samples_per_beat)

    for pos in positions:
        length = int(0.25 * sr)  # 250ms
        end = min(n, pos + length)
        tt = np.linspace(0, 1, end - pos, endpoint=False)
        # 低频 + 指数衰减
        wave = np.sin(2 * np.pi * (45 + 10 * punch) * tt)
        env = np.exp(-8 * tt)  # 快速 decays
        kick[pos:end] += wave * env

    return kick


def _phonk_bass_track(t, bpm, brightness):
    """跟着 kick 的 sub bass，带一点滑音。"""
    sr = SR
    beat_hz = bpm / 60.0
    base_freq = 55.0 + 25.0 * brightness  # 大概 A1–C2 区间
    bass = np.zeros_like(t)

    # 每拍一个 note，偶数拍上滑一点
    samples_per_beat = int(sr / beat_hz)
    n = t.shape[0]

    for i, pos in enumerate(range(0, n, samples_per_beat)):
        end = min(n, pos + samples_per_beat)
        tt = np.linspace(0, 1, end - pos, endpoint=False)

        if i % 2 == 0:
            f0 = base_freq
            f1 = base_freq * 1.1
        else:
            f0 = base_freq * 0.9
            f1 = base_freq

        freq = f0 + (f1 - f0) * tt
        phase = 2 * np.pi * np.cumsum(freq) / sr
        wave = np.sin(phase)
        env = 0.9 * np.exp(-2 * tt)
        bass[pos:end] += wave * env

    return bass


def _phonk_cowbell_track(t, bpm, density, brightness):
    """phonk 标志性 cowbell / lead，密度由分子决定。"""
    sr = SR
    beat_hz = bpm / 60.0
    sixteenth = 1.0 / (beat_hz * 4.0)  # 十六分音符长度（秒）
    n = t.shape[0]
    cow = np.zeros_like(t)

    # 频率稍微飘一点
    base_f1 = 600.0 + 200.0 * brightness
    base_f2 = 800.0 + 300.0 * brightness

    # 密度：每拍触发 1–4 个
    trig_per_beat = int(1 + density * 3)
    step_sec = sixteenth / max(1, trig_per_beat)
    step_samples = int(step_sec * sr)

    for pos in range(0, n, step_samples):
        if np.random.rand() > (0.3 + 0.5 * density):
            continue

        length = int(0.12 * sr)
        end = min(n, pos + length)
        tt = np.linspace(0, 1, end - pos, endpoint=False)

        f1 = base_f1 * (0.9 + 0.2 * np.random.rand())
        f2 = base_f2 * (0.9 + 0.2 * np.random.rand())
        wave = 0.6 * np.sin(2 * np.pi * f1 * tt) + 0.4 * np.sin(2 * np.pi * f2 * tt)
        env = np.exp(-10 * tt)
        cow[pos:end] += wave * env

    return cow


def _noise_percussion(t, bpm, density):
    """背景噪音鼓点，填满 techno 感。"""
    sr = SR
    beat_hz = bpm / 60.0
    n = t.shape[0]
    noise = np.random.randn(n).astype(np.float32)

    # 高通 + gating
    # 简单做法：用正弦当 gate
    gate = 0.5 + 0.5 * np.sin(2 * np.pi * beat_hz * 2.0 * t)
    gate = np.power(gate, 3)

    # 密度控制整体音量
    return noise * gate * (0.2 + 0.4 * density)


def _tanh_distort(x, drive):
    return np.tanh(x * (1.0 + 5.0 * drive))


def render_phonk_techno(mapped_params, duration_sec=16.0, sr=SR):
    """
    主函数：给你 phonk-techno 风格的分子轨。

    输入： mapped_params = ethanol_mapped_params.json 的内容（dict）
    输出：audio (np.float32), sr
    """
    t, n = _make_time(duration_sec, sr)
    global_ctrl = _derive_global_params(mapped_params)
    bpm = global_ctrl["bpm"]
    drive = global_ctrl["drive"]
    pump = global_ctrl["pump"]
    brightness = global_ctrl["brightness"]
    density = global_ctrl["density"]

    # 1) 分子驱动的节奏包络（多峰 → 多层 envelopes）
    peaks = mapped_params["mapped_peaks"]
    envs = build_rhythm_envelopes(peaks, duration_sec=duration_sec, sr=sr, bpm=bpm)
    molecular_env = envs.mean(axis=0)  # [n]

    # 2) 各个音轨
    kick = _phonk_kick_track(t, bpm, punch=pump)
    bass = _phonk_bass_track(t, bpm, brightness)
    cow = _phonk_cowbell_track(t, bpm, density, brightness)
    noise = _noise_percussion(t, bpm, density)

    # 3) 初步混音
    mix = (
        1.2 * kick
        + 0.9 * bass
        + 0.7 * cow
        + 0.5 * noise
    )

    # 4) 分子 envelope 作为整体动态 + 失真 drive
    mix = mix * molecular_env
    mix = _tanh_distort(mix, drive)

    # 5) sidechain-like pump（让一切跟 kick 一起呼吸）
    # 简易做法：用 kick 的包络做压缩
    kick_env = np.abs(kick)
    kick_env = kick_env / (kick_env.max() + 1e-9)
    sidechain = 1.0 - pump * kick_env
    mix = mix * sidechain

    # 6) 归一化
    max_val = np.max(np.abs(mix)) + 1e-9
    audio = (mix / max_val * 0.9).astype(np.float32)
    return audio, sr