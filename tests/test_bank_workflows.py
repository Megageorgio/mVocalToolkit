import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from conftest import make_wav
from mvocaltoolkit.api_models import RecordingPreparationRequest
from mvocaltoolkit.labels import Label
from mvocaltoolkit.settings import Home
from mvocaltoolkit.training.workflows import BankWorkflowRequest, BankWorkflows, check_workflow
from mvocaltoolkit.training.runs import Run, RunInfo, RunStatus, RunSpec
from mvocaltoolkit.training.preparation import add_groups, prepare_recordings
from mvocaltoolkit.training.preparation_cache import inventory, remove
from mvocaltoolkit.training.migration import inspect_project, import_project, stage_external
from mvocaltoolkit.training.phoneme_preview import preview, suggest_merges


class Job:
    id = "fixture"
    def check_cancelled(self): pass
    def progress(self, *args, **kwargs): pass
    def log(self, *args): pass
    def item_error(self, item, error): raise AssertionError(f"{item}: {error}")


def corpus(root):
    for i in range(2):
        path = make_wav(root / f"clip{i}.wav", rate=16000 + i)
        path.with_suffix(".lab").write_text("0 10000000 a\n")
    return root


def test_preparation_cache_reuses_verified_outputs_and_invalidates_changes(tmp_path):
    root = corpus(tmp_path / "source")
    calls = []
    async def call(tk, job, engine, method, params):
        calls.append(engine)
        for part in params["slices"]:
            target = Path(part["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(params["audio"]).read_bytes())
        return {}
    req = RecordingPreparationRequest(folder=str(root), out_dir=str(tmp_path / "a"), language="ru", f0_method="none")
    first = asyncio.run(prepare_recordings(None, Job(), req, call))
    assert first["cache_hits"] == 0 and len(calls) == 2
    second = asyncio.run(prepare_recordings(None, Job(), req.model_copy(update={"out_dir": str(tmp_path / "b")}), call))
    assert second["cache_hits"] == 2 and len(calls) == 2
    (root / "clip0.lab").write_text("0 10000000 i\n")
    third = asyncio.run(prepare_recordings(None, Job(), req.model_copy(update={"out_dir": str(tmp_path / "c")}), call))
    assert third["cache_hits"] == 1 and len(calls) == 3
    assert "i" in (tmp_path / "c/transcriptions.csv").read_text()
    cache_root = Path(first["cache_dir"])
    entries = inventory(cache_root)
    assert len(entries) == 3
    with pytest.raises(ValueError):
        remove(cache_root, ["../source"])
    assert remove(cache_root, [entries[0]["key"]]) == 1
    assert (root / "clip0.wav").is_file()


def test_context_liquid_rules_override_generic_list_and_preserve_existing_groups():
    def grouped(rule):
        label = Label.from_segments([(0, .1, "ru/t"), (.1, .2, "ru/r"), (.2, .5, "ru/a")])
        add_groups(label, ["a"], ["r"], {"ru/r": rule})
        return label
    assert len(grouped(["t"]).words) == 2
    assert grouped(["t"]).words[-1].start == .1
    assert grouped(["d"]).words[-1].start == .2
    assert grouped(False).words[-1].start == .2
    label = grouped(["t"])
    old = label.words.copy()
    add_groups(label, ["a"], ["r"], {"ru/r": False})
    assert label.words == old


class FakeTraining:
    def __init__(self): self.runs = {}; self.starts = 0
    def _busy(self): return False
    @asynccontextmanager
    async def inference_slot(self): yield
    def start(self, spec):
        self.starts += 1
        run = Run(RunInfo(id=str(self.starts), spec=spec, dir=spec.dir, status=RunStatus.done, step=200))
        self.runs[run.id] = run
        return run
    def get(self, key): return self.runs.get(key)
    async def resume(self, key): self.runs[key].info.status = RunStatus.done
    async def stop(self, key): self.runs[key].info.status = RunStatus.stopped
    def _save_index(self): pass


def fake_toolkit(tmp_path):
    return SimpleNamespace(home=Home(tmp_path / "home"), training=FakeTraining(),
        models=SimpleNamespace(get_installed=lambda key: None), engines=SimpleNamespace(specs={}))


def test_workflow_resumes_completed_training_after_export_failure_without_retraining(tmp_path):
    tk = fake_toolkit(tmp_path)
    manager = BankWorkflows(tk)
    root = corpus(tmp_path / "source")
    req = BankWorkflowRequest(project_dir=str(tmp_path / "project"), name="Voice", sources=[dict(folder=str(root), speaker="s", language="ru")],
        prepare=False, variance=False, training={"preset": {"goal": "full"}, "overrides": {"max_updates": 999999}})
    export_calls = 0
    async def call(tk, job, engine, method, params):
        nonlocal export_calls
        assert method == "export"
        export_calls += 1
        if export_calls == 1:
            raise ValueError("fixture exporter failure")
        out = Path(params["out_dir"])
        out.mkdir()
        (out / "acoustic.onnx").write_bytes(b"synthetic export; not a model")
        (out / "phonemes.json").write_text(json.dumps(["SP", "AP", "a"]))
        (out / "languages.json").write_text(json.dumps({"ru": 0}))
        (out / "dictionary-ru.txt").write_text("a\ta\n")
        (out / "dsconfig.yaml").write_text(yaml.safe_dump(dict(acoustic="acoustic.onnx", phonemes="phonemes.json", languages="languages.json")))
        return dict(out_dir=str(out))
    async def execute():
        state = manager.start(req, call)
        await manager.tasks[state["id"]]
        assert state["status"] == "failed" and "fixture" in state["error"]
        assert tk.training.starts == 1
        run = next(iter(tk.training.runs.values()))
        assert run.info.spec.config["overlay"]["max_updates"] == 200
        manager.resume(state["id"], call)
        await manager.tasks[state["id"]]
        assert state["status"] == "done", state["error"]
        assert tk.training.starts == 1 and export_calls == 2
        assert Path(state["results"]["bank"]["dir"], "character.yaml").is_file()
        assert Path(state["results"]["bank"]["dir"]).parent == tmp_path / "project/exports"
    asyncio.run(execute())
    restored = BankWorkflows(tk)
    restored.load()
    assert next(iter(restored.states.values()))["status"] == "done"


def test_review_gate_and_checkpoint_source_selection(tmp_path):
    tk = fake_toolkit(tmp_path)
    manager = BankWorkflows(tk)
    root = corpus(tmp_path / "source")
    req = BankWorkflowRequest(project_dir=str(tmp_path / "project"), name="Voice", sources=[dict(folder=str(root), speaker="s", language="ru")],
        prepare=False, acoustic=False, variance=False, export=False, review_before_training=True)
    async def call(*args): raise AssertionError("no engine is needed")
    async def execute():
        state = manager.start(req, call)
        await manager.tasks[state["id"]]
        assert state["status"] == "review"
        manager.resume(state["id"], call)
        await manager.tasks[state["id"]]
        assert state["status"] == "done"
    asyncio.run(execute())
    assert RunSpec(kind="diffsinger_acoustic", config={"source": "benchmark"}).engine_name() == "diffsinger_benchmark"
    with pytest.raises(ValueError, match="source"):
        RunSpec(kind="diffsinger_acoustic", config={"source": "arbitrary"}).engine_name()
    with pytest.raises(ValueError, match="paths"):
        check_workflow(req.model_copy(update={"training": {"overrides": {"work_dir": "elsewhere"}}}))


def test_migration_copies_configs_checkpoints_and_data_without_loading_pickle(tmp_path):
    source = tmp_path / "difftrainer"
    raw = corpus(source / "data/singer")
    cfg_dir = source / "checkpoints/voice"
    cfg_dir.mkdir(parents=True)
    dictionary = source / "dictionary-ru.txt"
    dictionary.write_text("a\ta\n")
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(dict(datasets=[dict(raw_data_dir=str(raw), speaker="s", language="ru")],
        dictionaries={"ru": str(dictionary)}, audio_sample_rate=44100)))
    checkpoint = cfg_dir / "model_ckpt_steps_123.ckpt"
    checkpoint.write_bytes(b"fixture bytes; never unpickled")
    before = checkpoint.read_bytes()
    report = inspect_project(str(source))
    assert len(report["configs"]) == 1
    tk = fake_toolkit(tmp_path)
    imported = import_project(tk, str(source), str(tmp_path / "project"))
    assert len(imported["runs"]) == 1
    run = next(iter(tk.training.runs.values()))
    assert run.info.status == RunStatus.stopped
    assert (run.dir / checkpoint.name).read_bytes() == before == checkpoint.read_bytes()
    assert len(list((run.dir / "dataset").rglob("*.wav"))) == 2
    staging = stage_external(str(checkpoint), str(cfg_dir / "config.yaml"), tmp_path / "preview")
    assert (staging / checkpoint.name).read_bytes() == before


def test_phone_preview_and_ipa_suggestions_do_not_modify_sources(tmp_path):
    wav = make_wav(tmp_path / "clip.wav")
    lab = wav.with_suffix(".lab")
    lab.write_text("0 5000000 ru/ax\n5000000 10000000 ru/py\n")
    before = lab.read_bytes()
    result = preview(dict(path=str(wav), language="ru", profiles={"ru": "ru-openutau"}, alphabet="wfl-native"))
    assert result["examples"][0]["source"] == "ru/ax" and result["examples"][0]["target"] == "ru/a"
    assert result["report"]["lossy"] and lab.read_bytes() == before
    result = suggest_merges([dict(phone="ru/m", ipa="m"), dict(phone="en/m", ipa="m"), dict(phone="ru/my", ipa="mʲ")])
    assert result["groups"] == [["en/m", "ru/m"]]


def test_pipeline_cache_preserves_labels_and_never_imports_intermediate_stems(tmp_path, monkeypatch):
    from mvocaltoolkit.training.workflow_pipeline import preprocess
    from mvocaltoolkit.training.dataset import SourceSpec
    from mvocaltoolkit.pipelines import extra
    tk = fake_toolkit(tmp_path)
    root = corpus(tmp_path / "source")
    calls = []
    async def model(*args): return "fixture.onnx", str(tmp_path), "fixture"
    monkeypatch.setattr(extra, "_separation_model", model)
    async def call(tk, job, engine, method, params):
        calls.append((engine, method))
        assert params["stems"] is None
        item = params["items"][0]
        out = Path(item["output_dir"])
        out.mkdir(parents=True)
        stems = {}
        for stem in ("Vocals", "Instrumental"):
            target = out / (stem + ".wav")
            target.write_bytes(Path(item["audio"]).read_bytes())
            stems[stem] = str(target)
        return [dict(files=stems)]
    source = SourceSpec(str(root), speaker="s", language="ru")
    opts = {"separation": [{"model": "fixture"}]}
    async def run():
        assert await preprocess(tk, Job(), source, tmp_path / "unused", {"separation": []}, [], call) == str(root)
        first = Path(await preprocess(tk, Job(), source, tmp_path / "a", opts, [], call))
        second = Path(await preprocess(tk, Job(), source, tmp_path / "b", opts, [], call))
        assert len(calls) == 2
        assert len(list(first.rglob("*.wav"))) == len(list(second.rglob("*.wav"))) == 2
        assert len(list(second.rglob("*.json"))) == 2
        assert len(list(root.glob("*.wav"))) == 2
        assert all(p.name.endswith("-pass0.wav") for p in second.glob("*.wav"))
    asyncio.run(run())


def test_workflow_stop_waits_for_cancellation_and_retry_still_requires_review(tmp_path, monkeypatch):
    from mvocaltoolkit.training import preparation
    tk = fake_toolkit(tmp_path)
    manager = BankWorkflows(tk)
    root = corpus(tmp_path / "source")
    req = BankWorkflowRequest(project_dir=str(tmp_path / "project"), name="Voice", sources=[dict(folder=str(root), speaker="s", language="ru")],
        acoustic=False, variance=False, export=False, review_before_training=True)
    entered, finished = asyncio.Event(), asyncio.Event()
    attempts = 0
    async def prep(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            entered.set()
            try: await asyncio.Event().wait()
            finally: finished.set()
        return dict(out_dir=str(root), conversions=[])
    monkeypatch.setattr(preparation, "prepare_recordings", prep)
    async def call(*args): raise AssertionError("no engine is needed")
    async def run():
        state = manager.start(req, call)
        await entered.wait()
        await manager.stop(state["id"])
        assert finished.is_set() and state["status"] == "stopped"
        manager.resume(state["id"], call)
        await manager.tasks[state["id"]]
        assert state["status"] == "review" and not state.get("approved")
        await manager.close()
    asyncio.run(run())


def test_review_resume_rechecks_edited_dataset_and_preserves_originals(tmp_path):
    tk = fake_toolkit(tmp_path)
    manager = BankWorkflows(tk)
    root = corpus(tmp_path / "source")
    req = BankWorkflowRequest(project_dir=str(tmp_path / "project"), name="Voice", sources=[dict(folder=str(root), speaker="s", language="ru")],
        prepare=False, acoustic=False, variance=False, export=False, review_before_training=True)
    async def call(*args): raise AssertionError("no engine is needed")
    async def run():
        state = manager.start(req, call)
        await manager.tasks[state["id"]]
        assert state["status"] == "review"
        folder = Path(state["results"]["dataset"]["spec"]["datasets"][0]["raw_data_dir"])
        for wav in folder.rglob("*.wav"):
            wav.write_bytes(b"invalid recording after review")
        manager.resume(state["id"], call)
        await manager.tasks[state["id"]]
        assert state["status"] == "failed" and "Dataset check blocks" in state["error"]
        assert all(p.stat().st_size > 44 for p in root.glob("*.wav"))
    asyncio.run(run())


def test_expression_only_variance_does_not_require_notes_or_word_groups(tmp_path):
    tk = fake_toolkit(tmp_path)
    manager = BankWorkflows(tk)
    root = corpus(tmp_path / "source")
    req = BankWorkflowRequest(project_dir=str(tmp_path / "project"), name="Voice", sources=[dict(folder=str(root), speaker="s", language="ru")],
        prepare=False, acoustic=False, variance=True, export=False, variance_settings={"variance": {"energy": True}})
    async def call(*args): raise AssertionError("no inference engine is needed")
    async def run():
        state = manager.start(req, call)
        await manager.tasks[state["id"]]
        assert state["status"] == "done", state["error"]
        overlay = next(iter(tk.training.runs.values())).info.spec.config["overlay"]
        assert overlay["predict_energy"] and not overlay["predict_pitch"] and not overlay["predict_dur"]
    asyncio.run(run())


def test_workflow_optional_vocoder_and_g2p_are_bounded_and_feed_acoustic(tmp_path):
    tk=fake_toolkit(tmp_path); manager=BankWorkflows(tk)
    source=tmp_path/"audio"
    for i in range(2):
        wav=make_wav(source/f"clip{i}.wav",seconds=1+i*.01,rate=44100)
        wav.with_suffix(".lab").write_text("0 10000000 a\n")
    dictionary=tmp_path/"dict.txt";dictionary.write_text("alpha\ta\nbeta\ta\n")
    req=BankWorkflowRequest(project_dir=str(tmp_path/"project"),name="Voice",sources=[dict(folder=str(source),speaker="s",language="ru")],
        prepare=False,variance=False,export=False,prototype_steps=5,vocoder_training={"accept_license":True,"steps":100},
        g2p_training=[dict(dictionary=str(dictionary),language="ru",max_steps=100)])
    calls=[]
    async def call(tk,job,engine,method,params):
        calls.append((engine,method));out=Path(params["out_dir"]);out.mkdir(exist_ok=True,parents=True)
        if engine=="singingvocoders":
            (out/"model.ckpt").write_bytes(b"fixture")
            return dict(out_dir=str(out),checkpoint=str(out/"model.ckpt"))
        return dict(out_dir=str(out),files=[])
    async def execute():
        state=manager.start(req,call);await manager.tasks[state["id"]]
        assert state["status"]=="done",state["error"]
        runs={run.info.spec.kind:run for run in tk.training.runs.values()}
        assert len(runs)==3 and runs["singingvocoders_vocoder"].info.spec.config["steps"]==5
        assert runs["g2p_rnnt"].info.spec.config["max_steps"]==5
        acoustic=runs["diffsinger_acoustic"].info.spec.config
        assert acoustic["assets"]["vocoder"].endswith("model.ckpt") and acoustic["overlay"]["max_updates"]==5
        assert calls==[("singingvocoders","export"),("diffsinger","export_vocoder"),("g2p","export")]
    asyncio.run(execute())
