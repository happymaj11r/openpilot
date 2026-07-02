"""Dispatcher entrypoint registered in place of `selfdrive.modeld.modeld`.

Decides at startup which engine serves the driving model:

* `/data/models` holds a unified supercombo pkl (new architecture)
    → upstream `modeld.main()` with `MODELD_MODELS_DIR=/data/models` so
      `helpers.modeld_pkl_path()` loads the custom pkl (engine code unchanged)
* `/data/models` holds a valid legacy (vision+policy) model set
    → `carrot_modeld.main()` (own 2/3-model engine)
* otherwise
    → upstream `modeld.main()` with the built-in model

There is no runtime patching of the upstream module — the env var is read by
`selfdrive/modeld/helpers.py::modeld_pkl_path()` itself, and the legacy engine
is fully independent, so upstream can evolve without us having to re-merge.

Crash-loop breaker: a corrupt or incompatible custom model (truncated pkl,
tinygrad version skew after a fork update, missing output slices, ...) makes
the engine die during load, and the process manager restarts modeld forever.
To recover without manual intervention, each custom-engine launch increments
`/data/models/.load_attempts`; the counter is cleared once the process
survives LOAD_SETTLE_SECONDS (load failures crash within seconds).  After
MAX_LOAD_ATTEMPTS consecutive early crashes the custom model is quarantined
to /data/models_quarantined and the built-in default engine takes over.
"""
from __future__ import annotations

import argparse
import os
import shutil
import threading
import time
from pathlib import Path

from openpilot.common.swaglog import cloudlog

from .config import (
    MODELD_MODELS_DIR_ENV,
    MODELS_DIR as CUSTOM_MODELS_DIR,
    PARAM_DRIVING_MODEL_NAME,
)
from .validator import describe, is_valid_legacy_model_dir, is_valid_supercombo_model_dir

STATUS_FILE = "/data/model_selector_status"
QUARANTINE_DIR = Path("/data/models_quarantined")
LOAD_ATTEMPTS_NAME = ".load_attempts"
MAX_LOAD_ATTEMPTS = 3
LOAD_SETTLE_SECONDS = 90.0


def _write_status(engine: str, desc: str) -> None:
    try:
        with open(STATUS_FILE, "w") as f:
            f.write(f"engine={engine}\npid={os.getpid()}\nstarted={int(time.time())}\ndescribe={desc}\n")
    except OSError:
        pass


def _attempts_file() -> Path:
    return CUSTOM_MODELS_DIR / LOAD_ATTEMPTS_NAME


def _read_attempts() -> int:
    try:
        return int(_attempts_file().read_text().strip())
    except (OSError, ValueError):
        return 0


def _write_attempts(n: int) -> None:
    try:
        _attempts_file().write_text(str(n))
    except OSError:
        pass


def _clear_attempts() -> None:
    try:
        _attempts_file().unlink(missing_ok=True)
    except OSError:
        pass


def _quarantine_custom_model() -> None:
    """Move the repeatedly-crashing custom model aside so the built-in
    default engine can run.  Kept on disk (not deleted) for diagnosis."""
    try:
        if QUARANTINE_DIR.exists():
            shutil.rmtree(QUARANTINE_DIR, ignore_errors=True)
        CUSTOM_MODELS_DIR.rename(QUARANTINE_DIR)
        cloudlog.error(f"[MODEL_SELECTOR] quarantined crashing custom model to {QUARANTINE_DIR}")
    except OSError as e:
        cloudlog.error(f"[MODEL_SELECTOR] quarantine failed: {e}")
        return
    try:
        from openpilot.common.params import Params
        Params().remove(PARAM_DRIVING_MODEL_NAME)
    except Exception:
        pass


def _select_engine() -> str:
    """Returns one of: 'upstream_custom', 'carrot_legacy', 'upstream_default'."""
    desc = describe(CUSTOM_MODELS_DIR)
    if is_valid_supercombo_model_dir(CUSTOM_MODELS_DIR):
        engine, status = "upstream_custom", "upstream_modeld_custom"
        msg = f"[MODEL_SELECTOR] running upstream modeld (custom supercombo) — {desc}"
    elif is_valid_legacy_model_dir(CUSTOM_MODELS_DIR):
        engine, status = "carrot_legacy", "carrot_modeld"
        msg = f"[MODEL_SELECTOR] running carrot_modeld (custom legacy) — {desc}"
    else:
        engine, status = "upstream_default", "upstream_modeld"
        msg = f"[MODEL_SELECTOR] running upstream modeld (default) — {desc}"
    print(msg, flush=True)
    cloudlog.warning(msg)
    _write_status(status, desc)
    return engine


def _arm_crash_loop_breaker(engine: str) -> str:
    """Count consecutive early-crash launches of a custom engine; after
    MAX_LOAD_ATTEMPTS quarantine the model and fall back to the default.
    Returns the (possibly downgraded) engine to run."""
    if engine == "upstream_default":
        return engine

    attempts = _read_attempts()
    if attempts >= MAX_LOAD_ATTEMPTS:
        msg = f"[MODEL_SELECTOR] custom model crashed {attempts}x during load — falling back to default"
        print(msg, flush=True)
        cloudlog.error(msg)
        _quarantine_custom_model()
        _write_status("upstream_modeld", f"quarantined after {attempts} failed loads")
        return "upstream_default"

    _write_attempts(attempts + 1)
    # Surviving the settle window means the model loaded fine; only
    # deterministic load-time failures should ever trip the breaker.
    timer = threading.Timer(LOAD_SETTLE_SECONDS, _clear_attempts)
    timer.daemon = True
    timer.start()
    return engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="A boolean for demo mode.")
    args, _ = parser.parse_known_args()

    engine = _arm_crash_loop_breaker(_select_engine())
    if engine == "carrot_legacy":
        from openpilot.carrot.model_selector import carrot_modeld
        carrot_modeld.main(demo=args.demo)
    else:
        if engine == "upstream_custom":
            # Must be set before the upstream import so every helpers call —
            # including the USBGPU probe in main() — sees the custom dir.
            os.environ[MODELD_MODELS_DIR_ENV] = str(CUSTOM_MODELS_DIR)
        from openpilot.selfdrive.modeld import modeld as upstream_modeld
        upstream_modeld.main(demo=args.demo)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cloudlog.warning("got SIGINT")
