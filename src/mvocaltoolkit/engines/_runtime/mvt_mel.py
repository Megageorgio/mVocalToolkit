"""Mel contracts shared by bank inspection and isolated rendering; no tensor dependencies."""
import math


def compatibility_errors(acoustic, vocoder):
    errors = []
    integers = {"sample_rate", "hop_size", "win_size", "fft_size", "num_mel_bins"}
    for source, config in (("acoustic", acoustic), ("vocoder", vocoder)):
        for key in integers & config.keys():
            value = config[key]
            if type(value) is not int or value <= 0:
                errors.append(f"{source}.{key} must be a positive integer")
        for key in ("mel_fmin", "mel_fmax"):
            value = config.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                errors.append(f"{source}.{key} must be finite and non-negative")
        if str(config.get("mel_base", "e")) not in {"e", "10"}:
            errors.append(f"{source}.mel_base must be e or 10")
        if config.get("mel_scale", "slaney") not in {"slaney", "htk"}:
            errors.append(f"{source}.mel_scale must be slaney or htk")
    if errors:
        return errors
    for key in (*sorted(integers), "mel_fmin", "mel_fmax", "mel_scale"):
        if key not in acoustic or key not in vocoder:
            continue
        a, v = acoustic[key], vocoder[key]
        if key == "mel_fmax":
            if a is None and acoustic.get("sample_rate"):
                a = acoustic["sample_rate"] / 2
            if v is None and vocoder.get("sample_rate"):
                v = vocoder["sample_rate"] / 2
        if a is None or v is None:
            continue
        same = a == v if key == "mel_scale" else math.isclose(a, v, rel_tol=0, abs_tol=1e-5)
        if not same:
            errors.append(f"Vocoder and acoustic {key} differ: {v} vs {a}")
    return errors
