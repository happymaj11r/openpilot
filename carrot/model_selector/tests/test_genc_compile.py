"""Gen C compile wrapper: command assembly, I/O validation, orchestration (no real compile)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carrot.model_selector.config import GENC_METADATA_NAME, GENC_PKL_NAME, GENC_WARP_PKL_FMT
from carrot.model_selector.genc import COMPILE_ONNX_SCRIPT, COMPILE_WARP_SCRIPT, TINYGRAD_DIR
from carrot.model_selector.genc import compile as genc_compile
from carrot.model_selector.genc.compile import (GencCompileError, compile_genc, genc_compile_commands, onnx_io_command,
                                                validate_genc_io)

CAMS = ((1928, 1208), (1344, 760))

# SSHv4 / comma master (2026-11-03) real I/O document as produced by genc/onnx_io.py
SSHV4_IO = {
  "inputs": {
    "new_img": [2, 6, 128, 256], "desire": [8], "traffic_convention": [1, 2], "action_t": [1, 2],
    "state_img_q": [2, 5, 6, 128, 256], "state_desire_q": [100, 1, 8], "state_feat_q": [96, 1, 512],
  },
  "outputs": {
    "outputs": [1, 2580], "next_state_img_q": [2, 5, 6, 128, 256], "next_state_desire_q": [100, 1, 8],
    "next_state_feat_q": [96, 1, 512],
  },
  "metadata": {"model_checkpoint": "6a7d09ad"},
  "output_slices": {
    "meta": [0, 55], "desire_pred": [55, 87], "pose": [87, 99], "wide_from_device_euler": [99, 105],
    "road_transform": [105, 117], "lane_lines": [117, 645], "lane_lines_prob": [645, 653], "road_edges": [653, 917],
    "lead": [917, 1061], "lead_prob": [1061, 1064], "hidden_state": [1064, 1576], "plan": [1576, 2566],
    "desire_state": [2566, 2574], "pad": [2574, 2576], "traffic_units": [2576, 2580],
  },
}


def test_sshv4_io_validates():
  validate_genc_io(SSHV4_IO)


@pytest.mark.parametrize("mutate, needle", [
  (lambda io: io["inputs"].pop("desire"), "not a Gen C"),
  (lambda io: io["outputs"].pop("next_state_feat_q"), "not a Gen C"),
  (lambda io: io["inputs"].__setitem__("new_img", [2, 6, 64, 128]), "new_img shape"),
  (lambda io: io["inputs"].__setitem__("desire", [7]), "desire shape"),
  (lambda io: io["inputs"].__setitem__("action_t", [1, 3]), "action_t shape"),
  (lambda io: io["outputs"].__setitem__("next_state_feat_q", [95, 1, 512]), "next_state_feat_q shape"),
  (lambda io: io["outputs"].__setitem__("outputs", [2580]), "outputs shape"),
  (lambda io: io["output_slices"].pop("hidden_state"), "output_slices missing"),
])
def test_invalid_io_is_rejected(mutate, needle):
  io = copy.deepcopy(SSHV4_IO)
  mutate(io)
  with pytest.raises(GencCompileError, match=needle):
    validate_genc_io(io)


def test_gen_a_io_is_rejected_with_input_list():
  io = {"inputs": {n: [1] for n in ("img", "big_img", "desire_pulse", "traffic_convention", "action_t", "features_buffer")},
        "outputs": {"outputs": [1, 10]}, "output_slices": {}}
  with pytest.raises(GencCompileError, match="generation=A"):
    validate_genc_io(io)


def test_compile_commands_mirror_comma_sconscript(tmp_path: Path):
  onnx = tmp_path / "driving_supercombo.onnx"
  cmds = genc_compile_commands(onnx, tmp_path, CAMS, python="py")
  assert cmds[0] == ["py", str(COMPILE_ONNX_SCRIPT), str(onnx), str(tmp_path / GENC_PKL_NAME),
                     "--out-of-band", "--benchmark-runs", "1"]
  assert len(cmds) == 1 + len(CAMS)
  warp_tici = cmds[1]
  assert warp_tici[:2] == ["py", str(COMPILE_WARP_SCRIPT)]
  # 1928x1208 NV12 (VENUS, nv12_info): stride 2048, y 1216, uv 608 -> copy size 2048*(1216+608)
  assert warp_tici[warp_tici.index("--frame") + 1] == "1928,1208,2048,1216,608,3735552"
  assert warp_tici[warp_tici.index("--warp-to") + 1] == "512x256"
  assert warp_tici[warp_tici.index("--layout") + 1] == "yuv420"
  assert warp_tici[warp_tici.index("--frames") + 1] == "2"
  assert warp_tici[-1] == str(tmp_path / GENC_WARP_PKL_FMT.format(w=1928, h=1208))
  assert cmds[2][-1] == str(tmp_path / GENC_WARP_PKL_FMT.format(w=1344, h=760))
  assert str(COMPILE_ONNX_SCRIPT).startswith(str(TINYGRAD_DIR))


def test_onnx_io_command(tmp_path: Path):
  cmd = onnx_io_command(tmp_path / "m.onnx", tmp_path / "io.json", python="py")
  assert cmd[0] == "py" and cmd[1].endswith("genc/onnx_io.py") and cmd[-2:] == ["--json", str(tmp_path / "io.json")]


def test_compile_genc_orchestration(tmp_path: Path, monkeypatch):
  onnx = tmp_path / "driving_supercombo.onnx"
  onnx.write_bytes(b"onnx")
  calls: list[tuple[list[str], dict]] = []

  def fake_run(cmd, env):
    calls.append((cmd, env))
    if cmd[1].endswith("onnx_io.py"):
      Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(SSHV4_IO))
    elif "--output" in cmd:
      Path(cmd[cmd.index("--output") + 1]).write_bytes(b"warp")
    else:
      Path(cmd[3]).write_bytes(b"model")

  io = compile_genc(tmp_path, onnx, {"DEV": "QCOM", "PYTHONPATH": "/data/openpilot"}, CAMS, fake_run, python="py")
  assert io["inputs"]["new_img"] == [2, 6, 128, 256]
  assert len(calls) == 1 + 1 + len(CAMS)
  # vendored tinygrad must win on PYTHONPATH for every subprocess, DEV flags preserved
  for _, env in calls:
    assert env["PYTHONPATH"].split(":")[0] == str(TINYGRAD_DIR)
    assert env["PYTHONPATH"].endswith("/data/openpilot")
    assert env["DEV"] == "QCOM"
  meta = json.loads((tmp_path / GENC_METADATA_NAME).read_text())
  assert meta["generation"] == "C" and meta["cameras"] == [[1928, 1208], [1344, 760]]
  assert meta["compile_dev"] == "QCOM"
  assert meta["output_slices"]["plan"] == [1576, 2566]
  assert not (tmp_path / ".genc_onnx_io.json").exists()


def test_compile_genc_reuses_probed_io(tmp_path: Path):
  onnx = tmp_path / "driving_supercombo.onnx"
  calls = []

  def fake_run(cmd, env):
    calls.append(cmd)
    assert not cmd[1].endswith("onnx_io.py"), "I/O probe must not run again"
    if "--output" in cmd:
      Path(cmd[cmd.index("--output") + 1]).write_bytes(b"warp")
    else:
      Path(cmd[3]).write_bytes(b"model")

  compile_genc(tmp_path, onnx, {"DEV": "CPU:LLVM"}, CAMS, fake_run, io=copy.deepcopy(SSHV4_IO))
  assert len(calls) == 1 + len(CAMS)
  assert json.loads((tmp_path / GENC_METADATA_NAME).read_text())["compile_dev"] == "CPU:LLVM"


def test_probe_onnx_io_cleans_up(tmp_path: Path):
  def fake_run(cmd, env):
    assert env["PYTHONPATH"].split(":")[0] == str(TINYGRAD_DIR)
    Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(SSHV4_IO))

  io = genc_compile.probe_onnx_io(tmp_path / "m.onnx", tmp_path, {}, fake_run)
  assert io["inputs"]["new_img"] == [2, 6, 128, 256]
  assert not (tmp_path / ".genc_onnx_io.json").exists()


def test_compile_genc_rejects_before_compiling(tmp_path: Path):
  onnx = tmp_path / "driving_supercombo.onnx"
  bad = copy.deepcopy(SSHV4_IO)
  bad["inputs"].pop("state_feat_q")
  calls = []

  def fake_run(cmd, env):
    calls.append(cmd)
    Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(bad))

  with pytest.raises(GencCompileError, match="not a Gen C"):
    compile_genc(tmp_path, onnx, {}, CAMS, fake_run)
  assert len(calls) == 1  # only the I/O probe ran


def test_compile_genc_detects_missing_artifact(tmp_path: Path):
  onnx = tmp_path / "driving_supercombo.onnx"

  def fake_run(cmd, env):
    if cmd[1].endswith("onnx_io.py"):
      Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(SSHV4_IO))
    # compile steps "succeed" without writing anything

  with pytest.raises(GencCompileError, match="produced no output"):
    compile_genc(tmp_path, onnx, {}, CAMS, fake_run)
  assert not (tmp_path / GENC_METADATA_NAME).exists()


def test_module_constants_match_config():
  assert genc_compile.NEW_IMG_SHAPE == (2, 6, 128, 256)
  assert genc_compile.MODEL_INPUT_SIZE == (512, 256)
