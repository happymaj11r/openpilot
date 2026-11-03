"""modeld_runner engine selection with the Gen C artifact set in the mix."""
from __future__ import annotations

import sys
import types

import pytest


class _Cloudlog:
  def __getattr__(self, _name):
    return lambda *args, **kwargs: None


# Keep the runner importable without the native openpilot runtime (same pattern as test_recompile).
swaglog_stub = types.ModuleType("openpilot.common.swaglog")
swaglog_stub.cloudlog = _Cloudlog()
sys.modules.setdefault("openpilot.common.swaglog", swaglog_stub)
params_stub = types.ModuleType("openpilot.common.params")
params_stub.Params = object
sys.modules.setdefault("openpilot.common.params", params_stub)

from carrot.model_selector import modeld_runner


def _arrange(monkeypatch, *, genc=False, supercombo=False, legacy=False, current=True):
  monkeypatch.setattr(modeld_runner, "describe", lambda _d: "desc")
  monkeypatch.setattr(modeld_runner, "is_valid_genc_model_dir", lambda _d: genc)
  monkeypatch.setattr(modeld_runner, "is_valid_supercombo_model_dir", lambda _d: supercombo)
  monkeypatch.setattr(modeld_runner, "is_valid_legacy_model_dir", lambda _d: legacy)
  monkeypatch.setattr(modeld_runner, "model_compile_env_is_current", lambda _d: current)
  statuses = []
  monkeypatch.setattr(modeld_runner, "_write_status", lambda engine, desc: statuses.append(engine))
  return statuses


def test_genc_takes_priority_over_other_artifacts(monkeypatch):
  statuses = _arrange(monkeypatch, genc=True, supercombo=True, legacy=True)
  assert modeld_runner._select_engine() == "genc_custom"
  assert statuses == ["genc_modeld"]


@pytest.mark.parametrize("flags, expected", [
  ({"supercombo": True}, "upstream_custom"),
  ({"legacy": True}, "carrot_legacy"),
  ({}, "upstream_default"),
])
def test_existing_selection_unchanged(monkeypatch, flags, expected):
  _arrange(monkeypatch, **flags)
  assert modeld_runner._select_engine() == expected


def test_stale_genc_artifacts_fall_back_to_default(monkeypatch):
  statuses = _arrange(monkeypatch, genc=True, current=False)
  assert modeld_runner._select_engine() == "upstream_default"
  assert statuses == ["upstream_modeld"]


def test_main_dispatches_to_genc_modeld(monkeypatch):
  calls = []
  fake = types.ModuleType("carrot.model_selector.genc_modeld")
  fake.main = lambda demo=False: calls.append(demo)
  monkeypatch.setitem(sys.modules, "carrot.model_selector.genc_modeld", fake)
  monkeypatch.setattr(modeld_runner, "_select_engine", lambda: "genc_custom")
  monkeypatch.setattr(modeld_runner, "_arm_crash_loop_breaker", lambda engine: engine)
  monkeypatch.setattr(sys, "argv", ["modeld_runner", "--demo"])
  modeld_runner.main()
  assert calls == [True]


def test_crash_loop_breaker_counts_genc_launches(monkeypatch, tmp_path):
  monkeypatch.setattr(modeld_runner, "CUSTOM_MODELS_DIR", tmp_path)
  monkeypatch.setattr(modeld_runner, "_write_status", lambda *a, **k: None)
  monkeypatch.setattr(modeld_runner, "_quarantine_custom_model", lambda: None)
  monkeypatch.setattr(modeld_runner.threading, "Timer", lambda *a, **k: types.SimpleNamespace(start=lambda: None, daemon=False))
  assert modeld_runner._arm_crash_loop_breaker("genc_custom") == "genc_custom"
  assert modeld_runner._read_attempts() == 1
  modeld_runner._write_attempts(modeld_runner.MAX_LOAD_ATTEMPTS)
  assert modeld_runner._arm_crash_loop_breaker("genc_custom") == "upstream_default"
