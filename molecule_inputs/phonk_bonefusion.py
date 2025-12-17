import numpy as np
from scipy.io.wavfile import write
import json

SR = 44100
duration = 8.0


# ------------------------------------------------
# Load molecule data
# ------------------------------------------------
with open("ethanol_mapped_params.json", "r") as f:
    mol = json.load(f)

peaks = mol["mapped_peaks"]
peak_freqs = np.array([p["audio_freq_hz"] for p in peaks])
peak_amps  = np.array([p["audio_amp"] for p in peaks])

def auto_rhythms(n):
    return np.linspace(0.6, 2.0, n) if n > 1 else [1.0]

peak_rhythms = auto_rhythms(len(peaks))


# ------------------------------------------------
# PHONK + TECHNO MUSIC LAYER (可听)
# ------------------------------------------------
def make_music():
    t = np.linspace(0, duration, int(SR*duration), endpoint=False)

    # Phonk kick
    kick_env = (np.sin(2*np.pi*2*t) > 0.9).astype(float)
    kick = 1.3 * kick_env * np.exp(-t*20)

    # Bassline
    bass = 0.9 * np.tanh(np.sin(2*np.pi*55*t) * 3)

    # Hats
    hats = 0.15 * (np.random.rand(len(t))*2 - 1) * (np.sin(2*np.pi*12*t) > 0.4)

    # Molecular melodic accents
    accents = np.zeros_like(t)
    for f, a, r in zip(peak_freqs, peak_amps, peak_rhythms):
        accents += 0.28 * a * np.sin(2*np.pi*(f + 4*np.sin(2*np.pi*r*t))*t)

    music = kick + bass + hats + accents
    music /= np.max(np.abs(music) + 1e-9)
    return music


# ------------------------------------------------
# BONE CONDUCTION SUB LAYER (深振)
# ------------------------------------------------
def make_bone():
    t = np.linspace(0, duration, int(SR*duration), endpoint=False)

    # deep buzz
    base = 0.9 * np.sin(2*np.pi*45*t)

    macro = np.zeros_like(t)
    for a, r in zip(peak_amps, peak_rhythms):
        macro += (0.5 + 0.5*np.sin(2*np.pi*r*t)) * a

    vib = base * macro
    vib *= 4.0               # 增强振动
    vib = np.clip(vib, -1, 1)
    return vib


# ------------------------------------------------
# FINAL MIX (单轨)
# ------------------------------------------------
music = make_music()
vib   = make_bone()

out = music * 0.75 + vib * 0.45
out /= np.max(np.abs(out) + 1e-9)

write("ethanol_phonk_bonefusion.wav", SR, (out * 32767).astype(np.int16))
print("✔ WROTE ethanol_phonk_bonefusion.wav — single exciter version")