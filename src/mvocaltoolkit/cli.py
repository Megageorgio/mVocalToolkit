"""Command line interface: `mvt serve`, and the same operations as the API without a server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__, formats
from .settings import Home


def _toolkit(args):
    from .toolkit import Toolkit  # noqa: PLC0415

    return Toolkit(Home(args.home))


def _print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


async def _run_job(tk, kind: str, func) -> Any:
    """Runs an operation as a job in-process, printing progress."""
    job = tk.jobs.submit(kind, func)
    queue = job.subscribe()
    last_line = ""
    while True:
        event = await queue.get()
        name = event.get("event")
        if name == "progress":
            line = f"[{event.get('progress', 0) * 100:5.1f}%] {event.get('stage', '')} {event.get('message', '')}"
            if line != last_line:
                print(line[:160], file=sys.stderr)
                last_line = line
        elif name == "log":
            print("  " + str(event.get("message", ""))[:200], file=sys.stderr)
        elif name == "item_error":
            print(f"  ! {event.get('item')}: {event.get('error')}", file=sys.stderr)
        elif name == "paused":
            print("The job paused for review, which isn't supported in the CLI. Cancelling.", file=sys.stderr)
            job.cancel()
        elif name == "finished":
            break
    info = job.info
    if info.status.value != "done":
        print(f"Job {info.status.value}: {info.error or ''}", file=sys.stderr)
        if info.message and info.status.value == "failed":
            print(info.message, file=sys.stderr)
        raise SystemExit(1)
    return info.result


def cmd_serve(args) -> None:
    import uvicorn  # noqa: PLC0415

    from .server.app import create_app  # noqa: PLC0415
    from .toolkit import Toolkit  # noqa: PLC0415

    home = Home(args.home)
    settings = home.load_settings()
    if settings.auto_update and not getattr(args, "no_update", False):
        from .self_update import EXIT_UPDATING, hand_over, pending_update  # noqa: PLC0415

        source = pending_update(home.root, settings.github_token)
        if source and hand_over(source, home.root, sys.argv[1:]):
            from .self_update import update_log  # noqa: PLC0415

            print(f"A newer mVocalToolkit is available; updating from {source} and starting again.", flush=True)
            print(f"Update log: {update_log(home.root)}", flush=True)
            raise SystemExit(EXIT_UPDATING)
    host = args.host or settings.host
    port = args.port or settings.port
    if host not in ("127.0.0.1", "localhost", "::1"):
        settings = home.ensure_token(settings)
        print(f"Remote access enabled. Token: {settings.token}")
    print(f"mVocalToolkit {__version__} | home: {home.root} | http://{host}:{port}/docs")
    tk = Toolkit(home, settings)
    # the folder it was started from stays free (it can be moved or deleted while the toolkit runs)
    try:
        os.chdir(home.root)
    except OSError:
        pass
    unused = float(getattr(args, "exit_when_unused", 0) or 0)
    if unused > 0:
        print(f"Stops by itself {int(unused)} s after the last program using it closes.", flush=True)
    uvicorn.run(create_app(tk, exit_when_unused=unused), host=host, port=port, log_level="info")


def cmd_engines(args) -> None:
    tk = _toolkit(args)
    if args.action == "list":
        for e in tk.engines.list():
            state = "installed" if e["installed"] else "not installed"
            if e.get("outdated"):
                state += " (update needed)"
            print(f"{e['name']:<12} {state:<26} {', '.join(e['capabilities']):<28} {e['title']}")
        return
    spec = tk.engines.spec(args.name)
    if args.action == "install":
        asyncio.run(tk.engines.envs.install(spec, log=lambda m: print(m, file=sys.stderr), force=args.force))
    elif args.action == "remove":
        tk.engines.envs.uninstall(spec)
        print(f"{args.name} removed")
    elif args.action == "info":
        async def info():
            try:
                return await tk.engines.call(args.name, "ping", log=lambda m: print(m, file=sys.stderr))
            finally:
                await tk.engines.stop_all()

        _print(asyncio.run(info()))


def cmd_models(args) -> None:
    tk = _toolkit(args)
    if args.action == "list":
        from .models.store import model_listing  # noqa: PLC0415

        for model in model_listing(tk.models, task=args.task, engine=args.engine, language=args.lang):
            if args.installed and not model["installed"]:
                continue
            mark = "*" if model["installed"] else " "
            kind = " [pack]" if model.get("type") == "pack" else ""
            print(f"{mark} {model['id']:<38} {model['engine']:<10} {','.join(model.get('languages') or []):<10} "
                  f"{model.get('name', '')}{kind}")
        print("(* = installed; other models are downloaded on first use)", file=sys.stderr)
    elif args.action == "download":
        entry = tk.catalog.get(args.id)
        if entry is None:
            raise SystemExit(f"{args.id} is not in the catalogs")

        def progress(value, message, _detail=None):
            pct = f"{value * 100:5.1f}%" if value is not None else "      "
            print(f"\r{pct} {message[:100]:<100}", end="", file=sys.stderr)

        installed = asyncio.run(tk.models.download(entry, progress))
        print(file=sys.stderr)
        print("Installed: " + ", ".join(m.id for m in installed))
    elif args.action == "remove":
        print("Removed" if tk.models.remove(args.id) else "Not installed")
    elif args.action == "import":
        installed = tk.models.import_local(
            args.engine, args.path, args.id, args.name, args.lang.split(",") if args.lang else None, author=args.author
        )
        for model in installed:
            print(f"Imported {model.id}: {model.layout}")


def cmd_datasets(args) -> None:
    from .datasets import LicenseNotAccepted, ManualDownload, license_text  # noqa: PLC0415

    tk = _toolkit(args)
    if args.action == "list":
        for d in tk.datasets.listing(args.lang):
            mark = "*" if d["installed"] else " "
            hours = f"{d['hours']:g} h" if d.get("hours") else "?"
            lic = d["license"]
            terms = ("commercial" if lic["commercial"] else "non-commercial") + ", models " + (
                "can be published" if lic["publish_models"] else "not to be published")
            manual = ", download by hand" if d.get("manual_download") else ""
            print(f"{mark} {d['id']:<16} {','.join(d['languages']):<26} {hours:>7}  {d['name']}")
            print(f"  {'':<16} {lic['name']}: {terms}{manual}")
        print("(* = downloaded; `mvt datasets info <id>` shows the license)", file=sys.stderr)
        return
    entry = tk.catalog.datasets.get(args.id or "")
    if entry is None:
        raise SystemExit(f"{args.id} is not in the catalogs")
    if args.action == "info":
        print(license_text(entry))
        if tk.datasets.is_installed(entry.id):
            print(f"Downloaded to {tk.datasets.folder(entry.id)}")
    elif args.action == "download":
        accepted = args.accept_license
        if not accepted:
            print(license_text(entry))
            if not sys.stdin.isatty():
                raise SystemExit("Accept the license with --accept-license to download.")
            try:
                accepted = input("Accept this license and download? [y/N] ").strip().lower() in ("y", "yes")
            except EOFError:
                accepted = False
            if not accepted:
                raise SystemExit("Not downloaded.")

        def progress(value, message, _detail=None):
            pct = f"{value * 100:5.1f}%" if value is not None else "      "
            print(f"\r{pct} {message[:100]:<100}", end="", file=sys.stderr)

        try:
            path = asyncio.run(tk.datasets.download(entry, accepted, progress))
        except (LicenseNotAccepted, ManualDownload) as e:
            raise SystemExit(str(e)) from e
        print(file=sys.stderr)
        print(f"Downloaded to {path} (the license is in LICENSE.txt there)")
    elif args.action == "remove":
        print("Removed" if tk.datasets.remove(entry.id) else "Not downloaded")


def cmd_train(args) -> None:
    """Training runs: start from a spec file, follow, stop, resume, export, render."""
    from .training.runs import RunSpec  # noqa: PLC0415

    tk = _toolkit(args)

    def show(info) -> None:
        line = f"{info.id}  {info.status.value:<9} {info.spec.kind:<22} step {info.step}"
        if info.max_steps:
            line += f"/{info.max_steps}"
        if info.speed:
            line += f"  {info.speed * 60:.1f} steps/min"
        if info.eta:
            line += f"  {int(info.eta // 3600)}:{int(info.eta % 3600 // 60):02d} left"
        if info.val_loss is not None:
            line += f"  val {info.val_loss:.4f}"
        if info.message:
            line += f"  {info.message[:80]}"
        print(line)
        if info.error:
            print("  " + info.error.replace("\n", "\n  "))

    if args.action == "list":
        for info in tk.training.list():
            show(info)
        return
    if args.action == "start":
        spec_data = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        spec = RunSpec.model_validate(spec_data)
        if args.dir:
            spec.dir = args.dir
        if args.device:
            spec.device = args.device
        from .training.recipes import FAMILIES, snapshot as recipe_snapshot
        if spec.kind in FAMILIES:
            expected = FAMILIES[spec.kind]["engine"]
            if spec.engine and spec.engine != expected or spec.device == "mps":
                raise ValueError("Choose the matching engine and CPU or CUDA")
            spec.engine = expected
            config = {**spec.config, "allow_download": tk.settings.auto_download_models}
            spec.config = {**config, **recipe_snapshot(spec.kind, config, tk.home.dir("training") / "recipe-data")}
        if spec.kind == "g2p_rnnt":
            from .training.g2p import snapshot
            prepared = snapshot(spec.config, tk.home.dir("training") / "g2p-data")
            spec.config = {**{k: v for k, v in spec.config.items() if k != "prepared"}, **prepared}
        if spec.kind == "rvc_voice":
            if spec.engine_name() != "rvc" or spec.device == "mps":
                raise ValueError("RVC training requires the rvc engine and CPU or CUDA")
            from .training.rvc import snapshot
            prepared = snapshot(spec.config, tk.home.dir("training") / "rvc-data")
            spec.config = {**spec.config, **prepared}
        if spec.kind == "singingvocoders_vocoder":
            from .training.vocoder import build_overlay as vocoder_overlay, resolve_assets as vocoder_assets, scan_sources

            overlay = vocoder_overlay({**spec.config, "device": spec.device})
            report = scan_sources(spec.config)
            errors = [p["problem"] for p in report["problems"] if p["severity"] == "error"]
            if errors:
                raise ValueError("; ".join(errors[:5]))
            spec.config = {**spec.config, "overlay": overlay, "data_report": report, "assets": vocoder_assets(spec.config)}
        if spec.kind.startswith("diffsinger"):
            from .training.diffsinger import build_overlay, resolve_assets  # noqa: PLC0415
            from .training.dataset import validate_report  # noqa: PLC0415

            overlay = build_overlay(spec.kind, {**spec.config, "device": spec.device})
            validate_report(str(spec.config.get("data_report") or ""), bool(overlay.get("predict_pitch")), bool(overlay.get("predict_dur")))
            spec.config = {**spec.config, "overlay": overlay, "assets": resolve_assets(spec.config)}

        async def main():
            await tk.start()
            run = tk.training.start(spec)
            print(f"Run {run.id} queued in {run.dir}")
            if not args.wait:
                print("The toolkit must keep running for the run to be prepared and started: use `mvt serve`, or --wait.")
                await tk.stop()
                return
            queue = run.subscribe()
            while True:
                event = await queue.get()
                kind = event.get("event")
                if kind == "prepare":
                    pct = f"{event['progress'] * 100:5.1f}%" if event.get("progress") is not None else "      "
                    print(f"\r{pct} {event.get('stage', '')}: {str(event.get('message', ''))[:90]:<90}", end="", file=sys.stderr)
                elif kind == "launched":
                    print(f"\nTrainer started (pid {event.get('pid')}); log: {run.dir / 'train.log'}", file=sys.stderr)
                elif kind == "step":
                    info = run.info
                    losses = " ".join(f"{k}={v:.4f}" for k, v in list(info.losses.items())[:3])
                    eta = f" {int(info.eta // 60)} min left" if info.eta else ""
                    print(f"\rstep {info.step}/{info.max_steps or '?'}  {losses}{eta}        ", end="", file=sys.stderr)
                elif kind == "validation":
                    print(f"\nvalidation at {event.get('step')}: val_loss {event.get('val_loss')}", file=sys.stderr)
                elif kind == "checkpoint":
                    print(f"\ncheckpoint {event.get('step')}", file=sys.stderr)
                elif kind == "samples":
                    print(f"\nsamples at {event.get('step')}: {len(event.get('files') or [])} files", file=sys.stderr)
                elif kind == "warning":
                    print(f"\nwarning: {event.get('text')}", file=sys.stderr)
                elif kind == "end":
                    print(file=sys.stderr)
                    show(run.info)
                    break
            await tk.stop()

        asyncio.run(main())
        return
    run = tk.training.get(args.id or "")
    if run is None:
        raise SystemExit(f"Unknown run {args.id}")
    if args.action == "status":
        show(run.info)
        if run.info.checkpoints:
            print("checkpoints: " + ", ".join(str(c.step) for c in run.info.checkpoints))
        for step, files in sorted(run.info.samples.items(), key=lambda kv: int(kv[0])):
            print(f"samples {step}: {len(files)} files")
    elif args.action == "log":
        for line in run.log_tail(args.lines or 100):
            print(line)
    elif args.action == "events":
        for event in run.read_events_since(args.since or 0):
            print(json.dumps(event, ensure_ascii=False))
    elif args.action == "stop":
        run.stop_file.write_text("cli", encoding="utf-8")
        print("STOP written: a trainer that is running saves a checkpoint and ends (a running toolkit records the stop)")
    elif args.action == "resume":
        print(f"A resume needs the running toolkit: POST /training/runs/{run.id}/resume")
    elif args.action == "export":
        out = args.out or str(run.dir / "export" / (str(args.step) if args.step else "latest"))
        result = _run(tk, "training_export", lambda job: tk.engines.call(
            run.info.spec.engine_name(), "export",
            {"run_dir": str(run.dir), "kind": run.info.spec.kind, "out_dir": out, "step": args.step,
             "export_spk": args.spk or [], "freeze_spk": None},
            on_progress=lambda d: job.progress(d.get("progress"), stage=d.get("stage") or "export", message=d.get("message")),
            log=job.log))
        _print(result)
    elif args.action == "render":
        if not args.ds:
            raise SystemExit("render needs --ds <file.ds>")
        out = args.out or str(run.dir / "renders")
        result = _run(tk, "training_render", lambda job: tk.engines.call(
            run.info.spec.engine_name(), "render",
            {"run_dir": str(run.dir), "kind": run.info.spec.kind, "ds": args.ds, "out_dir": out, "step": args.step,
             "spk": args.spk[0] if args.spk else None, "lang": args.lang, "predict": []},
            on_progress=lambda d: job.progress(d.get("progress"), stage=d.get("stage") or "render", message=d.get("message")),
            log=job.log))
        _print(result)


def cmd_voicebank(args) -> None:
    """Assembles an OpenUtau DiffSinger voicebank from export folders."""
    from .training.openutau import VoicebankSpec, assemble  # noqa: PLC0415

    result = assemble(VoicebankSpec(name=args.name, out_dir=args.out, acoustic=args.acoustic,
                                    variance=args.variance or [], vocoder=args.vocoder or "",
                                    licenses=args.license or [], overwrite=args.overwrite))
    print(f"{result.dir}: {', '.join(result.models)}; languages {', '.join(result.languages) or '-'}; "
          f"speakers {', '.join(result.speakers) or '-'}; {len(result.files)} files")
    for p in result.problems:
        print(f"  problem: {p['file']}: {p['problem']}")
    if not result.problems:
        print("  the linter found nothing")


def cmd_dataset(args) -> None:
    """Builds a DiffSinger raw dataset from folders of labelled recordings."""
    from .training.dataset import SourceSpec, build_dataset, spec_from_result  # noqa: PLC0415

    sources = []
    for folder in args.folders:
        speaker = ""
        if "=" in folder and not Path(folder).exists():
            speaker, folder = folder.split("=", 1)
        sources.append(SourceSpec(folder=folder, speaker=speaker, language=args.lang or ""))
    dictionaries = {}
    for item in args.dictionary or []:
        lang, _, path = item.partition("=")
        dictionaries[lang] = path

    def progress(value, message):
        print(f"\r{value * 100:5.1f}% {message[:90]:<90}", end="", file=sys.stderr)

    result = build_dataset(sources, Path(args.out), dictionaries or None, args.validation, progress)
    print(file=sys.stderr)
    print(f"{result.files} files, {result.seconds / 60:.1f} min, speakers: " + ", ".join(s["speaker"] for s in result.speakers))
    for lang, phones in result.phonemes.items():
        print(f"{lang}: {len(phones)} phonemes: {' '.join(phones)}")
    for problem in result.problems:
        print("problem: " + json.dumps(problem, ensure_ascii=False))
    spec = {"kind": args.kind, "config": spec_from_result(result, args.kind, {"goal": args.goal, "gpu_gb": args.gpu_gb, "size": args.size})}
    spec_path = Path(args.out) / f"spec-{args.kind}.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Run spec written to {spec_path}: `mvt train start {spec_path} --wait`")


def cmd_languages(args) -> None:
    """Languages with models for a task: what a GUI shows as "language -> model"."""
    from .models.store import model_listing  # noqa: PLC0415
    from .text.languages import language_info  # noqa: PLC0415

    groups: dict[str, list[str]] = {}
    universal: list[str] = []
    for model in model_listing(_toolkit(args).models, task=args.task):
        langs = [lang for lang in model.get("languages") or [] if lang != "*"]
        mark = "*" if model["installed"] else ""
        for lang in langs:
            groups.setdefault(lang, []).append(model["id"] + mark)
        if not langs:
            universal.append(model["id"] + mark)
    for code, ids in sorted(groups.items()):
        info = language_info(code)
        print(f"{code:<5} {info['name']:<12} {', '.join(ids)}")
    if universal:
        print(f"{'*':<5} {'any':<12} {', '.join(universal)}")


def _run(tk, kind: str, func) -> Any:
    async def main():
        try:
            return await _run_job(tk, kind, func)
        finally:
            await tk.stop()

    return asyncio.run(main())


def _input_spec(paths: list[str]):
    from .api_models import InputItem, InputSpec  # noqa: PLC0415

    spec = InputSpec()
    for p in paths:
        path = Path(p)
        if path.is_dir():
            if spec.folder:
                raise SystemExit("Only one folder per run")
            spec.folder = str(path)
        else:
            spec.items.append(InputItem(path=str(path)))
    return spec


def cmd_label(args) -> None:
    from .api_models import AlignRequest, OutputSpec, PostprocessOptions, TranscribeOptions  # noqa: PLC0415
    from .pipelines.label import run_align  # noqa: PLC0415

    tk = _toolkit(args)
    transcribe = None
    if not args.no_transcribe:
        transcribe = TranscribeOptions(model=args.whisper, language=args.lang, batch_size=args.batch_size)
    req = AlignRequest(
        input=_input_spec(args.paths),
        model=args.model,
        language=args.lang,
        extra_languages=args.also_lang.split(",") if args.also_lang else [],
        optional_breaths=args.breaths,
        split_silence=args.split,
        mode=args.mode,
        ap_detector="none" if args.no_breath else "loudness_spectral_centroid",
        dictionary=args.dictionary,
        transcribe=transcribe,
        skip_unknown_words=args.skip_unknown,
        postprocess=PostprocessOptions(rule_sets=args.rules.split(",") if args.rules else []),
        output=OutputSpec(formats=args.formats.split(","), dir=args.out, return_labels=False),
        refine=_refine_options(args),
    )

    async def main():
        try:
            return await _run_job(tk, "label", lambda job: run_align(tk, job, req))
        finally:
            await tk.stop()

    result = asyncio.run(main())
    ok = sum(1 for i in result["items"] if i.get("ok"))
    print(f"Done: {ok}/{len(result['items'])} files labeled with {result['model']}")
    for item in result["items"]:
        if not item.get("ok"):
            print(f"  failed {item['name']}: {item.get('error')}")


def cmd_transcribe(args) -> None:
    from .api_models import TranscribeRequest  # noqa: PLC0415
    from .pipelines.label import run_transcribe  # noqa: PLC0415

    tk = _toolkit(args)
    req = TranscribeRequest(
        input=_input_spec(args.paths), model=args.model, language=args.lang, batch_size=args.batch_size,
        save_txt=not args.no_save, output_dir=args.out,
    )

    async def main():
        try:
            return await _run_job(tk, "transcribe", lambda job: run_transcribe(tk, job, req))
        finally:
            await tk.stop()

    result = asyncio.run(main())
    for item in result["items"]:
        print(f"{item['name']}: {item.get('text') if item.get('ok') else 'ERROR ' + str(item.get('error'))}")


def cmd_segment(args) -> None:
    from .api_models import OutputSpec, SegmentRequest  # noqa: PLC0415
    from .pipelines.extra import run_segment  # noqa: PLC0415

    tk = _toolkit(args)
    req = SegmentRequest(input=_input_spec(args.paths), model=args.model, lang_id=args.lang_id,
                         output=OutputSpec(formats=args.formats.split(","), dir=args.out, return_labels=False),
                         refine=_refine_options(args))

    async def main():
        try:
            return await _run_job(tk, "segment", lambda job: run_segment(tk, job, req))
        finally:
            await tk.stop()

    result = asyncio.run(main())
    print(f"Done: {sum(1 for i in result['items'] if i.get('ok'))}/{len(result['items'])}")


def _refine_options(args):
    from .api_models import RefineOptions  # noqa: PLC0415

    if not getattr(args, "refine", None):
        return None
    return RefineOptions(model=args.refine, mode=args.refine_mode)


def cmd_refine(args) -> None:
    from .api_models import OutputSpec, RefineRequest  # noqa: PLC0415
    from .pipelines.refine import run_refine  # noqa: PLC0415

    tk = _toolkit(args)
    req = RefineRequest(input=_input_spec(args.paths), model=args.model, mode=args.mode,
                        output=OutputSpec(formats=args.formats.split(","), dir=args.out, layout="beside",
                                          return_labels=False))

    async def main():
        try:
            return await _run_job(tk, "refine", lambda job: run_refine(tk, job, req))
        finally:
            await tk.stop()

    result = asyncio.run(main())
    for item in result["items"]:
        info = item.get("data", {}).get("refine", {})
        print(f"{item['name']}: " + (f"{info.get('moved')}/{info.get('boundaries')} boundaries moved ({info.get('mode')})"
                                     if item.get("ok") else "ERROR " + str(item.get("error"))))


def cmd_midi(args) -> None:
    from .api_models import MidiRequest  # noqa: PLC0415
    from .pipelines.extra import run_midi  # noqa: PLC0415

    tk = _toolkit(args)
    tempo: Any = "auto" if args.tempo == "auto" else float(args.tempo)
    req = MidiRequest(input=_input_spec(args.paths), model=args.model, language=args.lang, tempo=tempo,
                      output_formats=args.formats.split(","), output_dir=args.out)

    async def main():
        try:
            return await _run_job(tk, "midi", lambda job: run_midi(tk, job, req))
        finally:
            await tk.stop()

    result = asyncio.run(main())
    for item in result["items"]:
        print(f"{item['name']}: {item.get('files') if item.get('ok') else 'ERROR ' + str(item.get('error'))}")


def cmd_separate(args) -> None:
    from .api_models import SeparateRequest  # noqa: PLC0415
    from .pipelines.extra import run_separate  # noqa: PLC0415

    tk = _toolkit(args)
    req = SeparateRequest(input=_input_spec(args.paths), model=args.model,
                          stems=args.stems.split(",") if args.stems else None, output_format=args.format,
                          output_dir=args.out)
    result = _run(tk, "separate", lambda job: run_separate(tk, job, req))
    for item in result["items"]:
        print(f"{item['name']}: {item.get('files') if item.get('ok') else 'ERROR ' + str(item.get('error'))}")


def cmd_pitch(args) -> None:
    from .api_models import PitchRequest  # noqa: PLC0415
    from .pipelines.extra import run_pitch  # noqa: PLC0415

    tk = _toolkit(args)
    req = PitchRequest(input=_input_spec(args.paths), model=args.model, hop=args.hop,
                       output_formats=args.formats.split(","), output_dir=args.out, return_curve=False)
    result = _run(tk, "pitch", lambda job: run_pitch(tk, job, req))
    for item in result["items"]:
        print(f"{item['name']}: {item.get('files') if item.get('ok') else 'ERROR ' + str(item.get('error'))}")


def cmd_fix(args) -> None:
    from .api_models import FixLabelsRequest  # noqa: PLC0415
    from .pipelines.fix import fix_labels  # noqa: PLC0415

    target = Path(args.path)
    req = FixLabelsRequest(folder=str(target) if target.is_dir() else None,
                           paths=[] if target.is_dir() else [str(target)],
                           rule_sets=args.rules.split(","), backup=not args.no_backup, dry_run=args.dry_run)
    result = fix_labels(req)
    print(f"Changed {result['changed']} of {len(result['files'])} files" + (f", backup: {result['backup']}"
                                                                           if result["backup"] else ""))
    for item in result["files"]:
        if not item["ok"]:
            print(f"  failed {item['path']}: {item['error']}")


def cmd_g2p(args) -> None:
    from .api_models import TextRequest  # noqa: PLC0415
    from .pipelines.extra import run_text  # noqa: PLC0415

    tk = _toolkit(args)
    req = TextRequest(texts=args.text, language=args.language, model=args.model)
    result = asyncio.run(run_text(tk, None, req, validate_only=args.validate))
    if args.json:
        _print(result)
        return
    for item in result["items"]:
        if args.validate:
            unknown = item.get("unknown_words", [])
            print(f"{item['text']}: " + (", ".join(unknown) if unknown else "all words are known"))
            continue
        guessed = item.get("guessed", {})
        for token, phonemes in zip(item.get("tokens", []), item.get("phonemes", [])):
            mark = " (guessed)" if guessed.get(token) else ""
            print(f"{token}\t{' '.join(phonemes) if phonemes else '?'}{mark}")


def cmd_words(args) -> None:
    from .text.g2p import load_user_words, save_user_words  # noqa: PLC0415

    home = Home(args.home)
    words = load_user_words(home, args.model)
    if args.action == "add":
        words[args.word] = args.phonemes
        save_user_words(home, args.model, words)
    elif args.action == "remove":
        words.pop(args.word, None)
        save_user_words(home, args.model, words)
    for word, phonemes in sorted(load_user_words(home, args.model).items()):
        print(f"{word}\t{' '.join(phonemes)}")


def cmd_convert(args) -> None:
    label = formats.read(Path(args.input), args.from_format)
    fmt = args.to_format or formats.detect_format(Path(args.output))
    formats.write(label, Path(args.output), fmt)
    print(f"Written {args.output}")


def cmd_config(args) -> None:
    home = Home(args.home)
    settings = home.load_settings()
    if args.action == "show":
        _print(settings.to_dict())
    elif args.action == "set":
        value: Any = args.value
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            pass
        if hasattr(settings, args.key):
            setattr(settings, args.key, value)
        else:
            settings.extra[args.key] = value
        home.save_settings(settings)
        print(f"{args.key} = {value!r}")
    elif args.action == "token":
        settings.token = ""
        settings = home.ensure_token(settings)
        print(settings.token)


def cmd_update(args) -> None:
    from .update import check_update, self_update  # noqa: PLC0415

    info = asyncio.run(check_update())
    _print(info)
    if args.apply and info.get("update_available"):
        raise SystemExit(self_update())


def cmd_storage(args) -> None:
    from . import storage  # noqa: PLC0415

    home = Home(args.home)
    if args.action == "cleanup":
        _print(storage.cleanup(home, None if args.all else float(args.days)))
    else:
        _print(storage.usage(home))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvt", description="mVocalToolkit")
    parser.add_argument("--home", help="Toolkit folder (default: MVT_HOME or ~/mVocalToolkit)")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="Run the API server")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--no-update", action="store_true", help="Don't check for a newer toolkit at start")
    p.add_argument("--exit-when-unused", type=float, default=0, metavar="SECONDS",
                   help="Stop by itself this long after the last program using it (see /clients) has closed")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("engines", help="Manage engines (isolated environments)")
    p.add_argument("action", choices=["list", "install", "remove", "info"])
    p.add_argument("name", nargs="?")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_engines)

    p = sub.add_parser("models", help="Manage models")
    p.add_argument("action", choices=["list", "download", "remove", "import"])
    p.add_argument("id", nargs="?", help="model id (download/remove)")
    p.add_argument("--task", help="align, transcribe, segment, midi, tempo, pitch, separate")
    p.add_argument("--engine")
    p.add_argument("--lang")
    p.add_argument("--installed", action="store_true")
    p.add_argument("--path", help="import: folder or archive")
    p.add_argument("--name")
    p.add_argument("--author", help="import: who made the model (shown next to its name)")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("datasets", help="Public singing datasets for training, with their licenses")
    p.add_argument("action", choices=["list", "info", "download", "remove"])
    p.add_argument("id", nargs="?", help="dataset id (info/download/remove)")
    p.add_argument("--lang", help="list: datasets of this language")
    p.add_argument("--accept-license", action="store_true", help="download: the license was read and is accepted")
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("voicebank", help="Assemble an OpenUtau DiffSinger voicebank from export folders")
    p.add_argument("out", help="folder where the bank is made (a subfolder named after the singer)")
    p.add_argument("--name", required=True, help="the singer's name")
    p.add_argument("--acoustic", required=True, help="export folder of the acoustic model")
    p.add_argument("--variance", action="append", help="export folder with dur/pitch/variance models (repeatable)")
    p.add_argument("--vocoder", help="folder with a custom vocoder (ONNX + vocoder.yaml)")
    p.add_argument("--license", action="append", help="a license line for readme.txt (repeatable)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing bank folder")
    p.set_defaults(func=cmd_voicebank)

    p = sub.add_parser("train", help="Training runs (DiffSinger): start from a spec, status, log, export, render")
    p.add_argument("action", choices=["start", "list", "status", "log", "events", "stop", "resume", "export", "render"])
    p.add_argument("spec", nargs="?", help="start: the run spec JSON ({kind, config}); other actions: the run id", default=None)
    p.add_argument("--dir", help="start: the run folder (default: <home>/training/<id>)")
    p.add_argument("--device", help="start: auto, cuda, cuda:1, cpu")
    p.add_argument("--wait", action="store_true", help="start: run the queue here and follow the run until it ends")
    p.add_argument("--lines", type=int, help="log: lines to show")
    p.add_argument("--since", type=int, help="events: from this event number")
    p.add_argument("--step", type=int, help="export/render: checkpoint step")
    p.add_argument("--out", "-o", help="export/render: output folder")
    p.add_argument("--spk", action="append", help="export: speakers to export; render: the speaker")
    p.add_argument("--lang", help="render: the language")
    p.add_argument("--ds", help="render: the .ds file")
    p.set_defaults(func=cmd_train, id=None)

    p = sub.add_parser("dataset", help="Build a DiffSinger raw dataset from folders of labelled recordings")
    p.add_argument("action", choices=["build"])
    p.add_argument("folders", nargs="+", help="folders with recordings and labels; speaker=folder names the speaker")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--lang", help="language of the recordings (default: xx)")
    p.add_argument("--dictionary", "-d", action="append", help="lang=path; missing ones are made from the labels")
    p.add_argument("--validation", type=int, default=3, help="validation files per speaker")
    p.add_argument("--kind", default="diffsinger_acoustic", help="run spec to write: diffsinger_acoustic or diffsinger_variance")
    p.add_argument("--goal", default="full", help="full, check, preview, finetune")
    p.add_argument("--size", default="medium", help="small, medium, large")
    p.add_argument("--gpu-gb", type=float, default=16)
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("languages", help="Languages and their models for a task (the GUI model choice)")
    p.add_argument("--task", default="align")
    p.set_defaults(func=cmd_languages)

    p = sub.add_parser("label", help="Transcribe (if needed) and align audio files (SOFA, HubertFA or TIFA)")
    p.add_argument("paths", nargs="+", help="audio files or a folder")
    p.add_argument("--model", "-m", required=True, help="aligner model id (see: mvt languages) or path:")
    p.add_argument("--lang", "-l")
    p.add_argument("--also-lang", "-L", help="TIFA: other languages in the texts, e.g. en for English words in zh")
    p.add_argument("--breaths", action="store_true",
                   help="TIFA: optional breaths (AP) at the start, at SP marks and after punctuation")
    p.add_argument("--split", action="store_true",
                   help="TIFA, SOFA: align long files again by segments cut at pauses (silence or a breath)")
    p.add_argument("--mode", choices=["force", "match"], default="force")
    p.add_argument("--dictionary", help="custom dictionary file")
    p.add_argument("--no-breath", action="store_true", help="don't detect breaths (AP)")
    p.add_argument("--formats", "-f", default="htk")
    p.add_argument("--out", "-o")
    p.add_argument("--rules", help="rule sets, e.g. en_fixes")
    p.add_argument("--whisper", default="whisper-large-v3-turbo")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-transcribe", action="store_true")
    p.add_argument("--skip-unknown", action="store_true")
    _refine_args(p)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("transcribe", help="Transcribe audio files (writes .txt next to them)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--model", "-m", default="whisper-large-v3-turbo")
    p.add_argument("--lang", "-l")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", "-o")
    p.add_argument("--no-save", action="store_true")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("segment", help="Phoneme segmentation without text (WFL-ASR)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--lang-id", type=int)
    p.add_argument("--formats", "-f", default="htk")
    p.add_argument("--out", "-o")
    _refine_args(p)
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("refine", help="Refine the phoneme boundaries of ready labels (the label file next to each audio)")
    p.add_argument("paths", nargs="+", help="audio files or a folder (labels: .lab, .TextGrid or .json next to them)")
    p.add_argument("--model", "-m", default="mrefiner-ru-v0.1.0")
    p.add_argument("--mode", choices=["auto", "normal", "safe"], default="auto")
    p.add_argument("--formats", "-f", default="htk")
    p.add_argument("--out", "-o", required=True, help="output folder (the input labels are not overwritten)")
    p.set_defaults(func=cmd_refine)

    p = sub.add_parser("midi", help="Extract notes (GAME)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--model", "-m", default="game-large")
    p.add_argument("--lang", "-l")
    p.add_argument("--tempo", default="120", help="BPM or auto")
    p.add_argument("--formats", "-f", default="mid,csv")
    p.add_argument("--out", "-o")
    p.set_defaults(func=cmd_midi)

    p = sub.add_parser("separate", help="Separate vocals (only when you need it: it can degrade clean audio)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--model", "-m", default="auto")
    p.add_argument("--stems", help="keep only these stems, e.g. vocals")
    p.add_argument("--format", default="wav", choices=["wav", "flac", "mp3"])
    p.add_argument("--out", "-o")
    p.set_defaults(func=cmd_separate)

    p = sub.add_parser("pitch", help="Extract f0 curves (RMVPE, FCPE, Parselmouth)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--model", "-m", default="rmvpe")
    p.add_argument("--hop", type=float, default=0.01)
    p.add_argument("--formats", "-f", default="csv")
    p.add_argument("--out", "-o")
    p.set_defaults(func=cmd_pitch)

    p = sub.add_parser("fix", help="Apply fixes to existing label files (in place, with a backup)")
    p.add_argument("path", help="label file or folder")
    p.add_argument("--rules", "-r", required=True, help="rule sets, e.g. dx,merge_uh_r,merge_duplicates")
    p.add_argument("--no-backup", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_fix)

    p = sub.add_parser("convert", help="Convert label formats")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--from", dest="from_format")
    p.add_argument("--to", dest="to_format")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("g2p", help="Words to phonemes with an aligner model's dictionary, own words and G2P")
    p.add_argument("text", nargs="+", help="texts (one argument per text)")
    p.add_argument("--model", "-m", required=True, help="aligner model id")
    p.add_argument("--language", "-l")
    p.add_argument("--validate", action="store_true", help="only list the words missing in the dictionary")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_g2p)

    p = sub.add_parser("words", help="Own words of an aligner model, used by every request with it")
    p.add_argument("action", choices=["list", "add", "remove"])
    p.add_argument("model", help="aligner model id")
    p.add_argument("word", nargs="?")
    p.add_argument("phonemes", nargs="*")
    p.set_defaults(func=cmd_words)

    p = sub.add_parser("config", help="Show or change settings")
    p.add_argument("action", choices=["show", "set", "token"])
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("storage", help="Disk space used by models, engines and job leftovers; remove old leftovers")
    p.add_argument("action", choices=["show", "cleanup"], nargs="?", default="show")
    p.add_argument("--days", type=float, default=0, help="cleanup: only what is older than this many days")
    p.add_argument("--all", action="store_true", help="cleanup: everything, whatever its age")
    p.set_defaults(func=cmd_storage)

    p = sub.add_parser("update", help="Check for a new version")
    p.add_argument("--apply", action="store_true", help="upgrade (uv tool installs)")
    p.set_defaults(func=cmd_update)
    return parser


def _refine_args(p) -> None:
    p.add_argument("--refine", metavar="MODEL", help="refine the boundaries with this refiner model, e.g. mrefiner-ru-v0.1.0")
    p.add_argument("--refine-mode", choices=["auto", "normal", "safe"], default="auto")


def main(argv: list[str] | None = None) -> None:
    from .nowindow import hide_child_consoles  # noqa: PLC0415

    hide_child_consoles()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "models" and args.action == "import" and (not args.engine or not args.path):
        parser.error("models import needs --engine and --path (the positional id is optional)")
    if getattr(args, "command", "") == "train" and args.action != "start":
        args.id = args.spec
    args.func(args)


if __name__ == "__main__":
    main()
