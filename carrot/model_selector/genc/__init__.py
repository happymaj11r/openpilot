"""Gen C (comma master state-queue supercombo) engine for the carrot model selector.

Gen C models (`new_img` + `state_*_q` inputs, `next_state_*` outputs) need the
tinygrad generation that comma pins on master, which is newer than the tinygrad
vendored at the repo root (used by the Gen A engine and dmonitoringmodeld).
That tinygrad is vendored under `genc/tinygrad_repo/` and activated **only** in
the Gen C modeld process and the Gen C compile subprocesses, so every other
process keeps the system tinygrad.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

GENC_DIR = Path(__file__).resolve().parent
TINYGRAD_DIR = GENC_DIR / "tinygrad_repo"
_PIN_FILE = TINYGRAD_DIR / "PIN"
TINYGRAD_PIN = _PIN_FILE.read_text().split()[0] if _PIN_FILE.is_file() else ""

COMPILE_ONNX_SCRIPT = TINYGRAD_DIR / "examples" / "openpilot" / "compile_onnx.py"
COMPILE_WARP_SCRIPT = TINYGRAD_DIR / "examples" / "openpilot" / "compile_warp.py"


_VENDORED_TOP_LEVEL = ("tinygrad", "examples")


def _loaded_from_vendor(module) -> bool:
    file = getattr(module, "__file__", None) or ""
    try:
        return Path(file).resolve().is_relative_to(TINYGRAD_DIR)
    except (OSError, ValueError):
        return False


def activate_tinygrad() -> None:
    """Put the vendored tinygrad first on sys.path.

    Must run before any `import tinygrad` in the process. If the system tinygrad
    (or the root tinygrad_repo `examples` package) was already imported — e.g.
    inherited through the manager fork — those modules are dropped from
    sys.modules so every later import resolves to the vendored generation.
    Mixing the two generations in one process would load pickled JITs with the
    wrong implementation.
    """
    path = str(TINYGRAD_DIR)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    for top in _VENDORED_TOP_LEVEL:
        mod = sys.modules.get(top)
        if mod is None or _loaded_from_vendor(mod):
            continue
        for name in [n for n in sys.modules if n == top or n.startswith(top + ".")]:
            del sys.modules[name]


def subprocess_env(base: dict[str, str]) -> dict[str, str]:
    """Environment for Gen C compile subprocesses: vendored tinygrad wins on PYTHONPATH."""
    env = dict(base)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(TINYGRAD_DIR) + (os.pathsep + existing if existing else "")
    return env
