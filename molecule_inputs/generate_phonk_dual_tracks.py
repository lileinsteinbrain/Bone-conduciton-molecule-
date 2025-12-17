import json
import soundfile as sf
import numpy as np

from industrial_techno_engine import load_molecule_params, render_phonk_techno

SR = 44100
DURATION_SEC = 16.0

MAPPED_JSON = "ethanol_mapped_params.json"
TACTILE_SOURCE = "ethanol_tactile_rhythm.wav"  # 你之前生成的带节奏 tactile 轨

SPEAKER_OUT = "ethanol_phonk_speaker.wav"
BONE_OUT = "ethanol_phonk_bone.wav"


def main():
    # 1) 加载分子参数
    mapped = load_molecule_params(MAPPED_JSON)

    # 2) 生成 phonk-techno speaker 轨
    audio_speaker, sr = render_phonk_techno(mapped, duration_sec=DURATION_SEC, sr=SR)

    # 3) 载入已有的 tactile 轨（分子驱动的骨传导）
    bone, sr_bone = sf.read(TACTILE_SOURCE)
    if bone.ndim > 1:
        bone = bone[:, 0]  # 取单声道
    if sr_bone != SR:
        # 简单粗暴的重采样（最近邻），对测试足够
        old_t = np.linspace(0, len(bone) / sr_bone, len(bone), endpoint=False)
        new_t = np.linspace(0, len(bone) / sr, len(audio_speaker), endpoint=False)
        bone = np.interp(new_t, old_t, bone).astype(np.float32)

    # 对齐长度
    n = min(len(audio_speaker), len(bone))
    audio_speaker = audio_speaker[:n]
    bone = bone[:n]

    # 4) 写文件
    sf.write(SPEAKER_OUT, audio_speaker, SR)
    sf.write(BONE_OUT, bone, SR)

    print("✅ Done.")
    print(f"  Speaker track : {SPEAKER_OUT}")
    print(f"  Bone track    : {BONE_OUT}")


if __name__ == "__main__":
    main()