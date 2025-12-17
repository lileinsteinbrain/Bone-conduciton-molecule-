import numpy as np


def _compute_rhythm_hz_for_peaks(peaks):
    """
    rhythm_hz（0.6–2.0 Hz）
    
    """
    wavenumbers = [p["wavenumber_cm-1"] for p in peaks]
    w_min, w_max = min(wavenumbers), max(wavenumbers)

    rhythm_list = []
    for p in peaks:
        wn = p["wavenumber_cm-1"]
        if w_max == w_min:
            wn_norm = 0.5
        else:
            wn_norm = (wn - w_min) / (w_max - w_min)

        
        rhythm_hz = 0.6 + 1.4 * wn_norm
        rhythm_list.append(rhythm_hz)

    return rhythm_list


def build_rhythm_envelopes(peaks, duration_sec=16.0, sr=44100,
                           use_global_pulse=True, bpm=120.0):
    """
    输入：mapped_peaks 列表（包含 wavenumber_cm-1 等字段）
    输出：envelopes, shape = [num_peaks, num_samples]

    对每个峰做三层东西：
      1) phrase_env: 整段的淡入 / 淡出（头尾不要突兀）
      2) global_env: 全局 4-on-the-floor 呼吸（按 bpm）
      3) peak_lfo: 每个峰自己的 LFO (rhythm_hz，根据波数来)

    最后 env_i(t) = phrase_env * global_env * peak_lfo_i
    """

    n_samples = int(duration_sec * sr)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)

    # —— 1) phrase envelope：2 秒淡入 + 2 秒淡出 ——
    fade_in = np.clip(t / 2.0, 0.0, 1.0)  # 0→1
    fade_out = np.clip((duration_sec - t) / 2.0, 0.0, 1.0)  # 1→0
    phrase_env = np.minimum(fade_in, fade_out)
    # 不要太低，保底 0.2
    phrase_env = np.clip(phrase_env, 0.2, 1.0)

    # —— 2) global pulse：按 bpm 做一个全局脉冲 ——
    if use_global_pulse:
        beat_hz = bpm / 60.0           # 120 bpm → 2 Hz
        global_env = 0.7 + 0.3 * np.sin(2 * np.pi * beat_hz * t)
    else:
        global_env = np.ones_like(t)

    # —— 3) 每个峰自己的 LFO ——
    rhythm_hz_list = _compute_rhythm_hz_for_peaks(peaks)
    envelopes = []

    for idx, peak in enumerate(peaks):
        r_hz = rhythm_hz_list[idx]
      
        phase_offset = idx * np.pi / 4.0

        peak_lfo = 0.5 + 0.5 * np.sin(2 * np.pi * r_hz * t + phase_offset)
        env = phrase_env * global_env * peak_lfo

        envelopes.append(env.astype(np.float32))

    envelopes = np.stack(envelopes, axis=0)
    return envelopes


# ---------------------------------------------------------
# 供 generate_audio_with_rhythm.py 调用的简单接口
# ---------------------------------------------------------
def apply_rhythm_modulation(waveform, base_freq, intensity):
    """
    最小可用版本：
    给一条 waveform 加一个 0.5–3 Hz 左右的节奏脉冲（AM 调制）。

    参数：
      waveform : np.ndarray, shape [N]，原始音频
      base_freq : float，没怎么用，只是保留接口（以后可以用来微调）
      intensity : 0–1，强度 → 决定脉冲速度和深度

    返回：
      调制后的 waveform（同样 shape）
    """
    sr = 44100
    n = len(waveform)
    t = np.linspace(0, n / sr, n, endpoint=False)

    # 把 0–1 的 intensity 映射到 0.5–3 Hz 的脉冲频率
    rhythm_hz = 0.5 + 2.5 * float(np.clip(intensity, 0.0, 1.0))

    # 0–1 的 AM 调制波：慢慢“呼吸”起伏
    mod = 0.5 * (1.0 + np.sin(2 * np.pi * rhythm_hz * t))

    return waveform * mod.astype(np.float32)
