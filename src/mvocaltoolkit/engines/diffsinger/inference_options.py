"""Complete missing variance parameters even when some others are explicitly replaced."""
PREDICTIONS = ("dur", "pitch", "energy", "breathiness", "voicing", "tension")


def prediction_tags(config: dict, segments: list[dict], requested: list[str]) -> list[str]:
    enabled = {key for key in PREDICTIONS if config.get("predict_" + key)}
    unsupported = set(requested) - enabled
    if unsupported:
        raise ValueError("This variance model cannot predict: " + ", ".join(sorted(unsupported)))
    result = set(requested)
    for key in enabled:
        field = {"dur": "ph_dur", "pitch": "f0_seq"}.get(key, key)
        if any(not segment.get(field) for segment in segments):
            result.add(key)
    return sorted(result)
