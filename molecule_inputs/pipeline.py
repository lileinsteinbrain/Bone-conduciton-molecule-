#!/usr/bin/env python3
import json
from pathlib import Path

# --------- 配置：文件名（你可以按需要改） ---------

IR_FILE = "ethanol_ir.json"
RAMAN_FILE = "ethanol_raman.json"
BASE_FILE = "mapping_base.json"
OVERRIDE_FILE = "ethanol_override.json"
OUT_FILE = "ethanol_mapped_params.json"


# --------- 一些小工具函数 ---------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_range(x, in_min, in_max, out_min, out_max):
    """线性映射，把 [in_min, in_max] → [out_min, out_max]"""
    if in_max == in_min:
        return (out_min + out_max) / 2.0
    t = (x - in_min) / (in_max - in_min)
    t = max(0.0, min(1.0, t))  # clamp
    return out_min + t * (out_max - out_min)


def normalize_intensities(peaks):
    """把某一组 peaks 的 intensity 归一化到 0–1"""
    vals = [p["intensity"] for p in peaks]
    if not vals:
        return peaks
    min_i = min(vals)
    max_i = max(vals)
    if max_i == min_i:
        for p in peaks:
            p["intensity_norm"] = 0.5
        return peaks
    for p in peaks:
        p["intensity_norm"] = (p["intensity"] - min_i) / (max_i - min_i)
    return peaks


def wavenumber_to_tactile_band(wn):
    """简单粗暴地根据波数分成 low / mid / high 三段"""
    if wn < 1300:
        return "low"
    elif wn < 2500:
        return "mid"
    else:
        return "high"


def center_of_range(r):
    """给 [min, max] 算中点"""
    return (r[0] + r[1]) / 2.0


# --------- 主逻辑：加载 → 合并 → 映射 ---------

def main():
    root = Path(".")  # 当前文件夹

    # 1. 载入基础 mapping & override
    base = load_json(root / BASE_FILE)
    override = load_json(root / OVERRIDE_FILE)

    audio_cfg = base["audio_mapping"]
    tactile_cfg = base["tactile_mapping"]
    peak_bias = override.get("overrides", {}).get("peak_bias", {})
    tactile_shape = override.get("overrides", {}).get(
        "tactile_shape", tactile_cfg.get("default_shape", "smooth")
    )
    vib_profile = override.get("overrides", {}).get("vibration_profile", {})

    # 2. 载入 IR / Raman 光谱（有的话就用，没有可以关掉 Raman）
    ir = load_json(root / IR_FILE)
    ir_peaks = ir.get("peaks", [])
    for p in ir_peaks:
        p["source"] = "IR"

    # Raman 如果暂时没有，可以把这一块注释掉，然后 all_peaks = ir_peaks
    if (root / RAMAN_FILE).exists():
        raman = load_json(root / RAMAN_FILE)
        raman_peaks = raman.get("peaks", [])
        for p in raman_peaks:
            p["source"] = "Raman"
    else:
        raman_peaks = []

    # 把两种光谱拼在一起
    all_peaks = ir_peaks + raman_peaks

    if not all_peaks:
        print("No peaks found in IR/Raman JSONs.")
        return

    # 3. 各自归一化强度
    all_peaks = normalize_intensities(all_peaks)

    wns = [p["wavenumber_cm-1"] for p in all_peaks]
    wn_min, wn_max = min(wns), max(wns)

    freq_min = audio_cfg["freq_min_hz"]
    freq_max = audio_cfg["freq_max_hz"]
    amp_min = audio_cfg["amplitude_min"]
    amp_max = audio_cfg["amplitude_max"]
    bw_min = audio_cfg["timbre_bandwidth_min"]
    bw_max = audio_cfg["timbre_bandwidth_max"]


    mapped = []
    for p in all_peaks:
        wn = p["wavenumber_cm-1"]
        inten_norm = p["intensity_norm"]

       
        key = str(int(round(wn)))
        bias = peak_bias.get(key, 0.5)

        # combined_weight 让「真实强度」和「设计偏好」各占一半
        combined_weight = 0.5 * inten_norm + 0.5 * bias

        audio_freq = map_range(combined_weight, 0.0, 1.0, freq_min, freq_max)
        # timbre 带宽随强度变化
        timbre_bw = map_range(inten_norm, 0.0, 1.0, bw_min, bw_max)
        # 振幅用 gamma 提一下弱峰
        amp = amp_min + (inten_norm ** 0.8) * (amp_max - amp_min)

        # 5.2 tactile band + 频率（骨传导）
        band = wavenumber_to_tactile_band(wn)
        if band == "low":
            t_range = tactile_cfg["vibration_low_hz"]
        elif band == "mid":
            t_range = tactile_cfg["vibration_mid_hz"]
        else:
            t_range = tactile_cfg["vibration_high_hz"]

        tactile_center = center_of_range(t_range)
        tactile_amp = amp  # 暂时用同一个尺度，之后可独立调

        mapped.append(
            {
                "source": p.get("source", "IR"),
                "wavenumber_cm-1": wn,
                "label": p.get("label", ""),
                "intensity_raw": p["intensity"],
                "intensity_norm": round(float(inten_norm), 3),
                # Audio 参数
                "audio_freq_hz": round(float(audio_freq), 2),
                "audio_amp": round(float(amp), 3),
                "audio_timbre_bw_hz": round(float(timbre_bw), 1),
                # Tactile 参数
                "tactile_band": band,
                "tactile_freq_hz": round(float(tactile_center), 2),
                "tactile_amp": round(float(tactile_amp), 3),
                "tactile_shape": tactile_shape,
                # 分子个性：振动 profile
                "vibration_profile": vib_profile,
            }
        )

    # 6. 输出到一个合并好的 JSON，后面 Ableton / Max / Arduino 都可以直接吃
    out = {
        "molecule": ir.get("molecule", "Unknown"),
        "formula": ir.get("formula", ""),
        "mapped_peaks": mapped,
    }

    with open(root / OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Wrote {len(mapped)} peaks to {OUT_FILE}")
    print("Example peak:")
    print(json.dumps(mapped[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
