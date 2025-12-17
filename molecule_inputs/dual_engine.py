import numpy as np
from scipy.io.wavfile import write
import json

SR = 44100

# ------------------------------------------------
# Load mapped ethanol peaks
# ------------------------------------------------
with open("ethanol_mapped_params.json", "r") as f:
    mol = json.load(f)

peaks = mol["mapped_peaks"]

# Extract audio freqs + amps
peak_freqs = [p["audio_freq_hz"] for p in peaks]
peak_amps  = [p["audio_amp"] for p in peaks]

# Auto-generate rhythm based on peak index (0.6–2.0 Hz)
def generate_auto_rhythms(n):
    if n <= 1:
        return [1.0]
    return list(np.linspace(0.6, 2.0, n))

peak_rhythms = generate_auto_rhythms(len(peaks))


# ------------------------------------------------
# Techno Music Track (Track A)
# ------------------------------------------------
def techno_music_track(duration=8.0):
    t = np.linspace(0, duration, int(SR*duration), endpoint=False)

    # 4/4 kick
    kick = np.zeros_like(t)
    beat_hz = 2  # 120 BPM
    kick_trig = (np.sin(2*np.pi*beat_hz*t) > 0.95).astype(float)
    kick = kick_trig * np.exp(-t*18)

    # Hats
    hats = 0.15 * (np.random.rand(len(t))*2 - 1)
    hats *= (np.sin(2*np.pi*8*t) > 0.3)

    # Rumble (bass)
    rumble = 0.35 * np.sin(2*np.pi*50*t)

    # Molecular accents
    accents = np.zeros_like(t)
    for f, a, r in zip(peak_freqs, peak_amps, peak_rhythms):
        accents += a * np.sin(2*np.pi*(f + 3*np.sin(2*np.pi*r*t))*t)

    music = kick + hats + rumble + accents*0.25
    music /= np.max(np.abs(music) + 1e-9)
    return music


# ------------------------------------------------
# Bone Track (Track B)
# ------------------------------------------------
def bone_track(duration=8.0):
    t = np.linspace(0, duration, int(SR*duration), endpoint=False)

    deep_pulse = np.sin(2*np.pi*55*t)

    macro = np.zeros_like(t)
    for r in peak_rhythms:
        macro += 0.5 + 0.5*np.sin(2*np.pi*r*t)

    vib = deep_pulse * macro
    vib /= np.max(np.abs(vib) + 1e-9)

    return vib


# ------------------------------------------------
# Render
# ------------------------------------------------
duration = 8.0
music = techno_music_track(duration)
bone  = bone_track(duration)

write("trackA_music.wav", SR, (music*32767).astype(np.int16))
write("trackB_bone.wav",  SR, (bone*32767).astype(np.int16))

combined = 0.7*music + 0.3*bone
write("preview_combined.wav", SR, (combined*32767).astype(np.int16))

print("✔ WROTE:")
print(" - trackA_music.wav")
print(" - trackB_bone.wav")
print(" - preview_combined.wav")