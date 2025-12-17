import json
import numpy as np
import soundfile as sf

MAPPED_JSON = "ethanol_mapped_params.json"
SR = 44100
DURATION_SEC = 16.0


def load_mapped_params(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["mapped_peaks"]


def synth_sine(freq_hz, amp, duration_sec, sr):
    n = int(duration_sec * sr)
    t = np.linspace(0.0, duration_sec, n, endpoint=False)
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def main():
    peaks = load_mapped_params(MAPPED_JSON)

    n = int(DURATION_SEC * SR)

    speaker_wave = np.zeros(n, dtype=np.float32)
    bone_wave = np.zeros(n, dtype=np.float32)

    for p in peaks:
        # —— Speaker 轨：用 audio_freq_hz / audio_amp ——
        f_audio = float(p.get("audio_freq_hz", 220.0))
        a_audio = float(p.get("audio_amp", 0.2))
        speaker_wave += synth_sine(f_audio, a_audio, DURATION_SEC, SR)

        # —— Bone 轨：用 tactile_freq_hz / tactile_amp ——
        f_tac = float(p.get("tactile_freq_hz", 120.0))
        a_tac = float(p.get("tactile_amp", 0.3))
        bone_wave += synth_sine(f_tac, a_tac, DURATION_SEC, SR)

    # 简单防削波：整体归一化到 0.9
    def normalize(x):
        peak = np.max(np.abs(x))
        if peak < 1e-6:
            return x
        return (0.9 * x / peak).astype(np.float32)

    speaker_wave = normalize(speaker_wave)
    bone_wave = normalize(bone_wave)

    sf.write("ethanol_speaker.wav", speaker_wave, SR)
    sf.write("ethanol_bone.wav", bone_wave, SR)

    print("Done: wrote ethanol_speaker.wav & ethanol_bone.wav")


if __name__ == "__main__":
    main()