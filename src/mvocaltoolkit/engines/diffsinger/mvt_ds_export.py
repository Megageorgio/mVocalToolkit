"""Runs DiffSinger's scripts/export.py with one fix applied in this process only: onnxsim (0.6+) writes
"onnxsim.*" metadata into every graph it simplifies, and onnx.compose.merge_models refuses to join two graphs whose
metadata differ ("Can't merge models with different values for the same model metadata property"). The metadata is
dropped before merging. Nothing upstream is modified.

    python mvt_ds_export.py acoustic --exp <run dir> --out <dir> [--ckpt N] ...   (the same arguments as export.py)
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

SOURCE = Path(os.environ.get("MVT_SOURCE_DIR") or os.getcwd()).resolve()


def main() -> None:
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    import onnx.compose  # noqa: PLC0415

    original = onnx.compose.merge_models

    def merge_without_tool_metadata(m1, m2, *args, **kwargs):  # type: ignore[no-untyped-def]
        for model in (m1, m2):
            keep = [p for p in model.metadata_props if not p.key.startswith("onnxsim.")]
            del model.metadata_props[:]
            model.metadata_props.extend(keep)
        return original(m1, m2, *args, **kwargs)

    onnx.compose.merge_models = merge_without_tool_metadata

    # the exporter names its files after exp_name, which is the run folder's absolute path here (the run folder is the
    # experiment folder): give it a plain model name instead, once the config is loaded
    name = os.environ.get("MVT_EXPORT_NAME", "").strip()
    if name:
        import importlib  # noqa: PLC0415

        # (`import utils.hparams` would give the dict: the utils package re-exports it under the same name)
        hp = importlib.import_module("utils.hparams")
        original_set_hparams = hp.set_hparams

        def set_hparams_named(*a, **kw):  # type: ignore[no-untyped-def]
            result = original_set_hparams(*a, **kw)
            hp.hparams["exp_name"] = name
            return result

        hp.set_hparams = set_hparams_named
    script = SOURCE / "scripts" / "export.py"
    sys.argv = [str(script)] + sys.argv[1:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
