import json
import numpy as np
import soundfile as sf
from rhythm_engine import apply_rhythm_modulation

MAPPED_JSON = "ethanol_mapped_params.json"

SAMPLE_RATE = 44100
DURATION = 5.0   # seconds


def synth_peak(freq, amp):
    """Create a sine wave for one spectral peak."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    wave = amp * np.sin(2 * np.pi * freq * t)
    return wave


def main():
    # -------------------------------
    # Load mapped molecule parameters
    # -------------------------------
    with open(MAPPED_JSON, "r", encoding="utf-8") as f:
        mapped = json.load(f)

    peaks = mapped["mapped_peaks"]

    # --------------------------------
    # Track A (audio): additive synthesis
    # --------------------------------
    audio = np.zeros(int(SAMPLE_RATE * DURATION))

    for p in peaks:
        freq = p["audio_freq_hz"]
        amp = p["audio_amp"]

        component = synth_peak(freq, amp)

        # Apply rhythm engine
        component = apply_rhythm_modulation(
            waveform=component,
            base_freq=freq,
            intensity=p["intensity_norm"],
        )

        audio += component

    # Normalize track A
    audio /= np.max(np.abs(audio)) + 1e-6

    # --------------------------------
    # Track B (tactile vibration): low frequency only
    # --------------------------------
    tactile = np.zeros_like(audio)

    for p in peaks:
        f = p["tactile_freq_hz"]
        a = p["tactile_amp"] * 0.8

        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        wave = a * np.sin(2 * np.pi * f * t)

        # Rhythm modulation again
        wave = apply_rhythm_modulation(wave, f, p["intensity_norm"])

        tactile += wave

    tactile /= np.max(np.abs(tactile)) + 1e-6

    # ---------------------
    # Export WAV files
    # ---------------------
    sf.write("ethanol_track_rhythm.wav", audio, SAMPLE_RATE)
    sf.write("ethanol_tactile_rhythm.wav", tactile, SAMPLE_RATE)

    print("✔ Generated ethanol_track_rhythm.wav and ethanol_tactile_rhythm.wav")


if __name__ == "__main__":
    main()