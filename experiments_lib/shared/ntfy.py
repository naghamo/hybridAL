"""Thin wrapper around the existing `_ntfy_push.py` script. Re-exports its
push helpers so future aggregators only depend on `experiments_lib`."""
import importlib.util
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_path = _root / "_ntfy_push.py"

spec = importlib.util.spec_from_file_location("_ntfy_push_compat", _path)
if spec is None or spec.loader is None:
    raise ImportError(f"could not load {_path}")
_module = importlib.util.module_from_spec(spec)
sys.modules["_ntfy_push_compat"] = _module
spec.loader.exec_module(_module)

push_message = _module.push_message
push_file = _module.push_file
load_cfg = _module.load_cfg
