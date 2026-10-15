"""Lightweight plan/preflight contracts; no audio fixtures or model workers."""
from pathlib import Path

import pytest

from mvocaltoolkit.training.workflows import BankWorkflowRequest, check_workflow


def request(tmp_path, **options):
    source = tmp_path / "recordings"
    source.mkdir(exist_ok=True)
    return BankWorkflowRequest(project_dir=str(tmp_path / "project"), name="Voice",
        sources=[dict(folder=str(source), speaker="singer", language="ru")], **options)


def test_legacy_boundary_refinement_is_removed_from_bank_workflows_without_mutating_input(tmp_path):
    prep = dict(refine=dict(model="old-refiner"), segment=False)
    pipeline = dict(alignment=dict(model="aligner", refine=dict(model="old-refiner")))
    req = request(tmp_path, preparation=prep, pipeline=pipeline)
    assert req.preparation == {"segment": False}
    assert req.pipeline == {"alignment": {"model": "aligner"}}
    assert prep["refine"] == {"model": "old-refiner"} and "refine" in pipeline["alignment"]


@pytest.mark.parametrize("settings", [dict(cpu_threads=0), dict(cpu_threads=True), dict(cpu_threads=33), dict(cache_models="yes")])
def test_render_resource_settings_fail_preflight_without_models(tmp_path, settings):
    with pytest.raises(ValueError):
        check_workflow(request(tmp_path, render_options=settings))


def test_plan_includes_optional_stages_in_execution_order_without_creating_outputs(tmp_path):
    req = request(tmp_path, prototype_steps=9, review_before_training=True,
        vocoder_training=dict(accept_license=True, steps=100),
        g2p_training=[dict(language="ru", max_steps=40)], render_test_phrase=True)
    checked = check_workflow(req)
    assert [row["id"] for row in checked["plan"]] == ["prepare_0", "dataset", "review", "vocoder", "export_vocoder",
        "g2p_0", "export_g2p_0", "acoustic", "variance", "export_acoustic", "export_variance", "bank", "test_phrase", "bank_render"]
    trainers = [row for row in checked["plan"] if row["kind"] in {"vocoder", "g2p", "acoustic", "variance"}]
    assert all(row["steps"] == 9 for row in trainers)
    assert {"review_pause", "prototype", "g2p_dictionary"} <= set(checked["notices"])
    assert "notes" in checked["plan"][0]["actions"]
    assert not Path(req.project_dir).exists()
    assert not list((tmp_path / "recordings").iterdir())


def test_prepared_dataset_only_plan_omits_bank_and_unused_test_settings(tmp_path):
    checked = check_workflow(request(tmp_path, prepare=False, acoustic=False, variance=False, export=False))
    assert [row["kind"] for row in checked["plan"]] == ["input", "dataset"]
    assert {"prepared_inputs", "no_bank"} <= set(checked["notices"])


@pytest.mark.parametrize("settings, message", [
    (dict(acoustic=False), "acoustic ONNX export"),
    (dict(variance=False, variance_export="missing-variance"), "variance ONNX export"),
    (dict(export=False, render_test_phrase=True), "Enable bank export"),
    (dict(auto_test_phrase=False, render_test_phrase=True), "Enable test phrase generation"),
    (dict(render_test_phrase=True), "local ONNX vocoder"),
    (dict(vocoder="missing-vocoder"), "vocoder.yaml"),
    (dict(test_phrase=dict(notes="H4")), "Notes must"),
    (dict(test_phrase=dict(mode="manual", phonemes=" | ")), "1–512 test phonemes"),
    (dict(g2p_training=[dict(max_steps=-1)]), "G2P max_steps"),
])
def test_incompatible_plan_fails_before_any_project_output(tmp_path, settings, message):
    req = request(tmp_path, **settings)
    with pytest.raises(ValueError, match=message): check_workflow(req)
    assert not Path(req.project_dir).exists()


def test_existing_ds_takes_priority_over_generated_phrase_and_its_unused_settings(tmp_path):
    ds = tmp_path / "phrase.ds"
    ds.write_text("[]")
    vocoder = tmp_path / "vocoder"
    vocoder.mkdir()
    (vocoder / "vocoder.yaml").write_text("model: vocoder.onnx")
    checked = check_workflow(request(tmp_path, render_ds=str(ds), vocoder=str(vocoder), test_phrase=dict(notes="H4")))
    kinds = [row["kind"] for row in checked["plan"]]
    assert "test_phrase" not in kinds and kinds[-1] == "bank_render"


def test_full_vocoder_default_matches_full_training_preset(tmp_path):
    checked = check_workflow(request(tmp_path, prototype=False, vocoder_training=dict(accept_license=True, preset=dict(goal="full"))))
    assert next(row["steps"] for row in checked["plan"] if row["kind"] == "vocoder") == 2000
