"""Describe a bank chain without scanning audio, starting workers or probing a GPU."""


def describe_workflow(req, overlays):
    plan, notices = [], []

    def add(identifier, kind=None, **details):
        plan.append(dict(id=identifier, kind=kind or identifier, **details))

    for i, source in enumerate(req.sources):
        actions = []
        if req.pipeline.get("separation"): actions.append("separation")
        if req.pipeline.get("alignment"): actions.append("alignment")
        if req.prepare:
            actions.append("labels")
            if req.preparation.get("segment", True): actions.append("segmentation")
            if req.preparation.get("f0_method", "auto") != "none": actions.append("f0")
            if req.preparation.get("estimate_notes", bool(overlays.get("variance", {}).get("predict_pitch"))):
                actions.append("notes")
        add(f"prepare_{i}", "preparation" if actions else "input", source=source.folder,
            speaker=source.speaker, language=source.language, actions=actions)
    add("dataset")
    if req.checks.get("detailed_audio"): notices.append("audio_heuristics")
    if req.review_before_training:
        add("review")
        notices.append("review_pause")
    if req.vocoder_training is not None:
        config = req.vocoder_training
        default_steps = req.prototype_steps if req.prototype else (100 if (config.get("preset") or {}).get("goal", "check") == "check" else 2000)
        steps = int(config.get("steps", default_steps))
        add("vocoder", steps=min(steps, req.prototype_steps) if req.prototype else steps)
        add("export_vocoder")
    for i, config in enumerate(req.g2p_training):
        steps = int(config.get("max_steps") or 0)
        if req.prototype: steps = min(steps or req.prototype_steps, req.prototype_steps)
        add(f"g2p_{i}", "g2p", language=config.get("language", req.sources[0].language), steps=steps or None)
        add(f"export_g2p_{i}", "export_g2p")
    for kind in ("acoustic", "variance"):
        if kind in overlays: add(kind, steps=overlays[kind].get("max_updates"))
    if req.export:
        for kind in overlays: add("export_" + kind)
        components = {}
        if not req.acoustic: components["acoustic"] = req.acoustic_export
        if not req.variance and req.variance_export: components["variance"] = req.variance_export
        if req.vocoder and req.vocoder_training is None: components["vocoder"] = req.vocoder
        add("bank", components=components)
        if req.auto_test_phrase and not req.render_ds: add("test_phrase")
        if req.render_ds or req.render_test_phrase: add("bank_render")
        if not req.variance and not req.variance_export: notices.append("acoustic_only")
        if not req.vocoder and req.vocoder_training is None: notices.append("standard_vocoder")
    else:
        notices.append("no_bank")
    if req.prototype: notices.append("prototype")
    if not req.prepare: notices.append("prepared_inputs")
    if req.device == "cpu" and (overlays or req.vocoder_training is not None or req.g2p_training):
        notices.append("cpu_training")
    if req.g2p_training: notices.extend(["g2p_dictionary", "legacy_g2p_stage"])
    return dict(plan=plan, notices=notices, device=req.device, source=req.source)
