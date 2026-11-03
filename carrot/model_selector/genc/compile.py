"""Gen C compile pipeline (comma master flow): compile_onnx.py + compile_warp.py per camera.

Pure helpers (command assembly, metadata validation) are kept free of side
effects so they can be unit-tested; `compile_genc` wires them to the installer's
subprocess runner.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

from ..config import GENC_METADATA_NAME, GENC_PKL_NAME, GENC_WARP_PKL_FMT
from . import COMPILE_ONNX_SCRIPT, COMPILE_WARP_SCRIPT, GENC_DIR, TINYGRAD_PIN, subprocess_env
from .generation import GEN_C_IMAGE_INPUT, GEN_C_STATE_INPUTS, describe_inputs, detect_generation, next_state_name

ONNX_IO_SCRIPT = GENC_DIR / "onnx_io.py"

# MEDMODEL_INPUT_SIZE (w, h); `new_img` is (2 cams, 6 planes, h/2, w/2).
MODEL_INPUT_SIZE = (512, 256)
NEW_IMG_SHAPE = (2, 6, MODEL_INPUT_SIZE[1] // 2, MODEL_INPUT_SIZE[0] // 2)
DESIRE_LEN = 8
TRAFFIC_CONVENTION_LEN = 2
ACTION_T_LEN = 2
# Slices the carrot fill_model_msg / parser path reads unconditionally.
REQUIRED_OUTPUT_SLICES = ("plan", "meta", "pose", "lane_lines", "lane_lines_prob", "road_edges", "lead",
                          "lead_prob", "desire_state", "desire_pred", "hidden_state", "road_transform",
                          "wide_from_device_euler")


class GencCompileError(ValueError):
    """Raised for unsupported/invalid Gen C models; the installer wraps it into InstallError."""


def validate_genc_io(io: dict) -> None:
    """Reject anything the Gen C runtime (genc/runtime.py) cannot run, before compiling."""
    inputs = {k: tuple(v) for k, v in (io.get("inputs") or {}).items()}
    outputs = {k: tuple(v) for k, v in (io.get("outputs") or {}).items()}
    slices = io.get("output_slices") or {}

    gen = detect_generation(inputs, outputs)
    if gen != "C":
        raise GencCompileError(
            f"not a Gen C supercombo (generation={gen}); inputs: {describe_inputs(inputs)}; outputs: {describe_inputs(outputs)}"
        )
    img = inputs[GEN_C_IMAGE_INPUT]
    if img != NEW_IMG_SHAPE:
        raise GencCompileError(f"{GEN_C_IMAGE_INPUT} shape {img} != {NEW_IMG_SHAPE} (model input must be {MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]})")
    for name, expected in (("desire", DESIRE_LEN), ("traffic_convention", TRAFFIC_CONVENTION_LEN), ("action_t", ACTION_T_LEN)):
        shape = inputs[name]
        if not shape or int(shape[-1]) != expected:
            raise GencCompileError(f"{name} shape {shape} incompatible (last dim must be {expected})")
    for state in GEN_C_STATE_INPUTS:
        nxt = next_state_name(state)
        if outputs.get(nxt) != inputs[state]:
            raise GencCompileError(f"{nxt} shape {outputs.get(nxt)} != {state} shape {inputs[state]}")
    out = outputs.get("outputs")
    if not out or len(out) != 2 or int(out[0]) != 1:
        raise GencCompileError(f"outputs shape {out} must be (1, N)")
    missing = [s for s in REQUIRED_OUTPUT_SLICES if s not in slices]
    if missing:
        raise GencCompileError(f"output_slices missing {missing}")


def onnx_io_command(onnx: Path, json_out: Path, python: str = "python3") -> list[str]:
    return [python, str(ONNX_IO_SCRIPT), str(onnx), "--json", str(json_out)]


def genc_compile_commands(onnx: Path, out_dir: Path, camera_configs: Sequence[tuple[int, int]],
                          python: str = "python3") -> list[list[str]]:
    """Commands mirroring comma's openpilot/selfdrive/modeld/SConscript for the driving model."""
    model_w, model_h = MODEL_INPUT_SIZE
    cmds = [[python, str(COMPILE_ONNX_SCRIPT), str(onnx), str(out_dir / GENC_PKL_NAME),
             "--out-of-band", "--benchmark-runs", "1"]]
    for cam_w, cam_h in camera_configs:
        stride, y_height, uv_height, _ = get_nv12_info(cam_w, cam_h)
        frame = f"{cam_w},{cam_h},{stride},{y_height},{uv_height},{stride * (y_height + uv_height)}"
        cmds.append([python, str(COMPILE_WARP_SCRIPT),
                     "--frame", frame, "--warp-to", f"{model_w}x{model_h}",
                     "--layout", "yuv420", "--frames", "2",
                     "--output", str(out_dir / GENC_WARP_PKL_FMT.format(w=cam_w, h=cam_h))])
    return cmds


def genc_artifacts(out_dir: Path, camera_configs: Sequence[tuple[int, int]]) -> list[Path]:
    return [out_dir / GENC_PKL_NAME] + [out_dir / GENC_WARP_PKL_FMT.format(w=w, h=h) for w, h in camera_configs]


def write_genc_metadata(out_dir: Path, io: dict, camera_configs: Sequence[tuple[int, int]], compile_dev: str) -> Path:
    path = out_dir / GENC_METADATA_NAME
    doc = {
        "generation": "C",
        "tinygrad_pin": TINYGRAD_PIN,
        # The runtime replays the pickled JITs on this backend (mirrors tg_compiled_flags.json
        # of the legacy engine): QCOM on device, CPU:LLVM for PC compiles.
        "compile_dev": compile_dev,
        "model_input_size": list(MODEL_INPUT_SIZE),
        "cameras": [list(c) for c in camera_configs],
        "inputs": io.get("inputs", {}),
        "outputs": io.get("outputs", {}),
        "output_slices": io.get("output_slices", {}),
        "model_metadata": io.get("metadata", {}),
    }
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


def probe_onnx_io(onnx: Path, work_dir: Path, env: dict[str, str],
                  run: Callable[[list[str], dict[str, str]], None], python: str = "python3") -> dict:
    """Read graph I/O names/shapes + output_slices in a vendored-tinygrad subprocess."""
    io_json = work_dir / ".genc_onnx_io.json"
    try:
        run(onnx_io_command(onnx, io_json, python), subprocess_env(env))
        return json.loads(io_json.read_text())
    finally:
        io_json.unlink(missing_ok=True)


def compile_genc(tmp_dir: Path, onnx: Path, env: dict[str, str], camera_configs: Sequence[tuple[int, int]],
                 run: Callable[[list[str], dict[str, str]], None], python: str = "python3",
                 io: dict | None = None) -> dict:
    """Validate + compile a Gen C model into tmp_dir. `run(cmd, env)` must raise on failure.

    `io` is the document from `probe_onnx_io` when the caller already probed the model
    (the installer does, to pick the generation). Returns the validated I/O document.
    """
    genv = subprocess_env(env)
    if io is None:
        io = probe_onnx_io(onnx, tmp_dir, env, run, python)
    validate_genc_io(io)
    for cmd in genc_compile_commands(onnx, tmp_dir, camera_configs, python):
        run(cmd, genv)
    missing = [p.name for p in genc_artifacts(tmp_dir, camera_configs) if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise GencCompileError(f"compile produced no output: {missing}")
    write_genc_metadata(tmp_dir, io, camera_configs, compile_dev=genv.get("DEV", ""))
    return io
