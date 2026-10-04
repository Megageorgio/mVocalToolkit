"""Command line interface: `mvt serve`, and the same operations as the API without a server."""

from __future__ import annotations

import argparse
import asyncio
import json
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
    host = args.host or settings.host
    port = args.port or settings.port
    if host not in ("127.0.0.1", "localhost", "::1"):
        settings = home.ensure_token(settings)
        print(f"Remote access enabled. Token: {settings.token}")
    print(f"mVocalToolkit {__version__} | home: {home.root} | http://{host}:{port}/docs")
    tk = Toolkit(home, settings)
    uvicorn.run(create_app(tk), host=host, port=port, log_level="info")


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

        def progress(value, message):
            pct = f"{value * 100:5.1f}%" if value is not None else "      "
            print(f"\r{pct} {message[:100]:<100}", end="", file=sys.stderr)

        installed = asyncio.run(tk.models.download(entry, progress))
        print(file=sys.stderr)
        print("Installed: " + ", ".join(m.id for m in installed))
    elif args.action == "remove":
        print("Removed" if tk.models.remove(args.id) else "Not installed")
    elif args.action == "import":
        installed = tk.models.import_local(
            args.engine, args.path, args.id, args.name, args.lang.split(",") if args.lang else None
        )
        for model in installed:
            print(f"Imported {model.id}: {model.layout}")


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
        mode=args.mode,
        ap_detector="none" if args.no_breath else "loudness_spectral_centroid",
        dictionary=args.dictionary,
        transcribe=transcribe,
        skip_unknown_words=args.skip_unknown,
        postprocess=PostprocessOptions(rule_sets=args.rules.split(",") if args.rules else []),
        output=OutputSpec(formats=args.formats.split(","), dir=args.out, return_labels=False),
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
                         output=OutputSpec(formats=args.formats.split(","), dir=args.out, return_labels=False))

    async def main():
        try:
            return await _run_job(tk, "segment", lambda job: run_segment(tk, job, req))
        finally:
            await tk.stop()

    result = asyncio.run(main())
    print(f"Done: {sum(1 for i in result['items'] if i.get('ok'))}/{len(result['items'])}")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvt", description="mVocalToolkit")
    parser.add_argument("--home", help="Toolkit folder (default: MVT_HOME or ~/mVocalToolkit)")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="Run the API server")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
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
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("languages", help="Languages and their models for a task (the GUI model choice)")
    p.add_argument("--task", default="align")
    p.set_defaults(func=cmd_languages)

    p = sub.add_parser("label", help="Transcribe (if needed) and align audio files (SOFA or HubertFA)")
    p.add_argument("paths", nargs="+", help="audio files or a folder")
    p.add_argument("--model", "-m", required=True, help="aligner model id (see: mvt languages) or path:")
    p.add_argument("--lang", "-l")
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
    p.set_defaults(func=cmd_segment)

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
    p.add_argument("--model", "-m", default="separation-vocals-bs-roformer")
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

    p = sub.add_parser("convert", help="Convert label formats")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--from", dest="from_format")
    p.add_argument("--to", dest="to_format")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("config", help="Show or change settings")
    p.add_argument("action", choices=["show", "set", "token"])
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("update", help="Check for a new version")
    p.add_argument("--apply", action="store_true", help="upgrade (uv tool installs)")
    p.set_defaults(func=cmd_update)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "models" and args.action == "import" and (not args.engine or not args.path):
        parser.error("models import needs --engine and --path (the positional id is optional)")
    args.func(args)


if __name__ == "__main__":
    main()
