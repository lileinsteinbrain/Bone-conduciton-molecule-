import soundfile as sf
import numpy as np

SPEAKER = "ethanol_phonk_speaker.wav"
BONE = "ethanol_phonk_bone.wav"
OUT = "ethanol_phonk_dual_stereo.wav"

def main():
    spk, sr1 = sf.read(SPEAKER)
    bone, sr2 = sf.read(BONE)

    if spk.ndim > 1:
        spk = spk[:, 0]
    if bone.ndim > 1:
        bone = bone[:, 0]

    if sr1 != sr2:
        raise RuntimeError(f"sample rate mismatch: {sr1} vs {sr2}")

    n = min(len(spk), len(bone))
    spk = spk[:n]
    bone = bone[:n]

    stereo = np.stack([spk, bone], axis=1).astype("float32")
    sf.write(OUT, stereo, sr1)
    print("✅ wrote", OUT)

if __name__ == "__main__":
    main()