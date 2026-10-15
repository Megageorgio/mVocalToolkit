"""Device selection for the pinned DiffSinger CLI, without changing its source."""
import os
import runpy
import sys
from pathlib import Path

source = Path(os.environ["MVT_SOURCE_DIR"])
sys.path.insert(0, str(source))
from basics.base_svs_infer import BaseSVSInfer

original = BaseSVSInfer.__init__
chosen = os.environ.get("MVT_INFER_DEVICE", "auto")


def initialize(self, device=None):
    original(self, device=(chosen if chosen != "auto" else device))


BaseSVSInfer.__init__ = initialize
runpy.run_path(str(source / "scripts" / "infer.py"), run_name="__main__")
