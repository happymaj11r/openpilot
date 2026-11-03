"""installer routes a single-onnx model by its ONNX generation:
Gen A → upstream compile_modeld path, Gen C → genc pipeline, anything else → InstallError."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Cloudlog:
  def __getattr__(self, _name):
    return lambda *args, **kwargs: None


swaglog_stub = types.ModuleType("openpilot.common.swaglog")
swaglog_stub.cloudlog = _Cloudlog()
params_stub = types.ModuleType("openpilot.common.params")
params_stub.Params = object
camera_stub = types.ModuleType("openpilot.common.transformations.camera")
camera_stub._ar_ox_fisheye = SimpleNamespace(width=1928, height=1208)
camera_stub._os_fisheye = SimpleNamespace(width=1344, height=760)
hardware_stub = types.ModuleType("openpilot.system.hardware")
hardware_stub.TICI = False

_missing = object()
_stubs = {
  "openpilot.common.swaglog": swaglog_stub,
  "openpilot.common.params": params_stub,
  "openpilot.common.transformations.camera": camera_stub,
  "openpilot.system.hardware": hardware_stub,
}
_saved_modules = {name: sys.modules.get(name, _missing) for name in _stubs}
try:
  sys.modules.update(_stubs)
  from carrot.model_selector import installer
finally:
  for name, previous in _saved_modules.items():
    if previous is _missing:
      sys.modules.pop(name, None)
    else:
      sys.modules[name] = previous
  for parent_name, child_name, module_name in (
    ("openpilot.common", "swaglog", "openpilot.common.swaglog"),
    ("openpilot.common", "params", "openpilot.common.params"),
    ("openpilot.common.transformations", "camera", "openpilot.common.transformations.camera"),
    ("openpilot.system", "hardware", "openpilot.system.hardware"),
  ):
    parent = sys.modules.get(parent_name)
    if parent is not None and _saved_modules[module_name] is _missing and getattr(parent, child_name, None) is _stubs[module_name]:
      delattr(parent, child_name)

from carrot.model_selector.genc import compile as genc_compile

GEN_A_IO = {"inputs": {n: [1, 2] for n in ("img", "big_img", "desire_pulse", "traffic_convention", "action_t", "features_buffer")},
            "outputs": {"outputs": [1, 10]}, "output_slices": {}}
GEN_C_IO = {"inputs": {"new_img": [2, 6, 128, 256], "desire": [8], "traffic_convention": [1, 2], "action_t": [1, 2],
                       "state_img_q": [2, 5, 6, 128, 256], "state_desire_q": [100, 1, 8], "state_feat_q": [96, 1, 512]},
            "outputs": {"outputs": [1, 2580], "next_state_img_q": [2, 5, 6, 128, 256],
                        "next_state_desire_q": [100, 1, 8], "next_state_feat_q": [96, 1, 512]},
            "output_slices": {"plan": [0, 1]}}
GEN_B_IO = {"inputs": {n: [1] for n in ("warped", "state_img_q", "state_big_img_q", "desire", "state_desire_q",
                                        "state_prev_feat", "state_feat_q", "traffic_convention", "action_t")},
            "outputs": {"outputs": [1, 10]}, "output_slices": {}}


@pytest.fixture
def routed(monkeypatch):
  calls = []
  monkeypatch.setattr(installer, "_compile_supercombo", lambda tmp_dir, env: calls.append(("A", tmp_dir)))

  def fake_compile_genc(tmp_dir, onnx, env, camera_configs, run, python="python3", io=None):
    calls.append(("C", tmp_dir, tuple(camera_configs), io))

  monkeypatch.setattr(genc_compile, "compile_genc", fake_compile_genc)
  return calls


def _probe(io):
  return lambda onnx, work_dir, env, run, python="python3": io


def test_gen_a_uses_upstream_pipeline(monkeypatch, tmp_path: Path, routed):
  monkeypatch.setattr(genc_compile, "probe_onnx_io", _probe(GEN_A_IO))
  installer._compile_supercombo_by_generation(tmp_path, {"DEV": "QCOM"})
  assert routed == [("A", tmp_path)]


def test_gen_c_uses_genc_pipeline_with_probed_io(monkeypatch, tmp_path: Path, routed):
  monkeypatch.setattr(genc_compile, "probe_onnx_io", _probe(GEN_C_IO))
  installer._compile_supercombo_by_generation(tmp_path, {"DEV": "QCOM"})
  assert routed == [("C", tmp_path, installer.GENC_CAMERA_CONFIGS, GEN_C_IO)]


def test_gen_b_is_rejected_before_any_compile(monkeypatch, tmp_path: Path, routed):
  monkeypatch.setattr(genc_compile, "probe_onnx_io", _probe(GEN_B_IO))
  with pytest.raises(installer.InstallError, match="unsupported supercombo generation.*state_big_img_q"):
    installer._compile_supercombo_by_generation(tmp_path, {"DEV": "QCOM"})
  assert routed == []


def test_probe_failure_is_reported(monkeypatch, tmp_path: Path, routed):
  def failing_probe(*_a, **_k):
    raise installer.InstallError("python3 onnx_io.py failed: boom")

  monkeypatch.setattr(genc_compile, "probe_onnx_io", failing_probe)
  with pytest.raises(installer.InstallError, match="cannot read supercombo graph I/O"):
    installer._compile_supercombo_by_generation(tmp_path, {"DEV": "QCOM"})
  assert routed == []


def test_genc_validation_error_becomes_install_error(monkeypatch, tmp_path: Path):
  monkeypatch.setattr(genc_compile, "probe_onnx_io", _probe(GEN_C_IO))

  def rejecting_compile(*_a, **_k):
    raise genc_compile.GencCompileError("new_img shape (2, 6, 64, 128) != (2, 6, 128, 256)")

  monkeypatch.setattr(genc_compile, "compile_genc", rejecting_compile)
  with pytest.raises(installer.InstallError, match="new_img shape"):
    installer._compile_supercombo_by_generation(tmp_path, {"DEV": "QCOM"})
