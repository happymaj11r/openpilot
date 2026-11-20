#!/usr/bin/env python3
"""PC check for the state-queue (Gen B / Gen C) pipeline: compile a supercombo with the vendored
tinygrad, load it through `genc.runtime.ModelState`, run a few synthetic frames, and compare the
raw model output of every frame against a plain evaluation of the ONNX graph on the same inputs.

    PYTHONPATH=<repo root> python3 carrot/model_selector/genc/pc_check.py \
        --onnx /path/driving_supercombo.onnx --out /path/out_dir [--camera 1928x1208] [--runs 3]

Exit status 0 only when every check passes. Uses DEV=CPU:LLVM unless --dev is given.

The reference comparison covers what the smoke run cannot see: the runtime reuses buffers to
feed each `next_state_*` output back as the next frame's `state_*` input, and packs the host
inputs into one upload. A wrong offset, a stale state, or a state tensor overwritten before its
last read still gives finite, changing outputs — but not the ones the model computes when it
is run frame by frame with ordinary copies. (This is how the in-place state feedback inherited
from comma was found to be wrong for Gen B.) It runs on the backend given by --dev, CPU by
default; a pass there says nothing about QCOM.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from carrot.model_selector.config import GENC_METADATA_NAME
from carrot.model_selector.genc import compile as genc_compile
from carrot.model_selector.validator import has_genc

PLAN_SHAPE = (1, 33, 15)
# JIT replay and direct graph evaluation run the same float32 kernels on the same backend,
# so they agree to rounding; a state/packing bug shows up at the scale of the outputs themselves.
REFERENCE_RTOL = 1e-3
REFERENCE_ATOL = 1e-3


def parse_size(value: str) -> tuple[int, int]:
  w, h = value.lower().split('x')
  return int(w), int(h)


def compile_model(onnx: Path, out: Path, cameras, dev: str, python: str) -> None:
  env = os.environ.copy()
  env['DEV'] = dev

  def run(cmd, run_env):
    print(f"  $ {' '.join(cmd[1:3])} ...", flush=True)
    start = time.monotonic()
    # dump_pickle() writes its temp file into cwd; keep that inside the output dir.
    subprocess.run(cmd, env=run_env, cwd=str(out), check=True, timeout=4 * 3600)
    print(f"    ok ({time.monotonic() - start:.0f}s)", flush=True)

  io = genc_compile.compile_genc(out, onnx, env, cameras, run, python=python)
  print(f"compiled: inputs={sorted(io['inputs'])} outputs={sorted(io['outputs'])}", flush=True)


def reference_run(onnx: Path, model, feeds: list[dict[str, np.ndarray]]) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
  """Per frame: the raw `outputs` vector and the state tensors left for the next frame, from
  evaluating the ONNX graph directly on the captured inputs.

  No JIT, no packed host buffer, and every state tensor is a plain copy of the previous
  frame's `next_state_*` result, so the answer does not depend on how the runtime wires its
  buffers. Uses the tinygrad already active in this process (the vendored one).
  """
  from tinygrad import Device, Tensor
  from tinygrad.nn.onnx import OnnxRunner

  runner = OnnxRunner(onnx)
  state = {name: np.zeros(*model.input_shapes[name]) for name in model.state_pairs}
  raws, states = [], []
  for feed in feeds:
    out = runner({name: Tensor(value).to(Device.DEFAULT) for name, value in (feed | state).items()})
    Tensor.realize(*out.values())
    raws.append(out['outputs'].numpy().reshape(-1).astype(np.float32))
    state = {name: out[next_name].numpy().copy() for name, next_name in model.state_pairs.items()}
    states.append(state)
  return raws, states


def same_values(got: np.ndarray, want: np.ndarray) -> bool:
  if got.shape != want.shape:
    return False
  if np.issubdtype(want.dtype, np.integer):
    return bool(np.array_equal(got, want))  # image history is uint8: exact, and no unsigned wrap-around in a difference
  return bool(np.allclose(got, want, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL))


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--onnx', type=Path, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--camera', type=parse_size, default=(1928, 1208))
  ap.add_argument('--cameras', type=parse_size, nargs='*', default=None,
                  help='camera sizes to compile warps for (default: all installer configs)')
  ap.add_argument('--runs', type=int, default=3)
  ap.add_argument('--dev', default='CPU:LLVM')
  ap.add_argument('--skip-compile', action='store_true', help='reuse artifacts already in --out')
  ap.add_argument('--no-reference', action='store_true', help='skip the comparison against direct ONNX evaluation')
  args = ap.parse_args()

  from carrot.model_selector.validator import GENC_CAMERA_CONFIGS
  cameras = tuple(args.cameras) if args.cameras else GENC_CAMERA_CONFIGS
  args.out.mkdir(parents=True, exist_ok=True)

  if not (args.skip_compile and has_genc(args.out)):
    t0 = time.monotonic()
    compile_model(args.onnx, args.out, cameras, args.dev, sys.executable)
    print(f"compile total {time.monotonic() - t0:.0f}s", flush=True)
  meta = json.loads((args.out / GENC_METADATA_NAME).read_text())
  print(f"metadata: generation={meta['generation']} image_input={meta.get('image_input')} "
        + f"compile_dev={meta.get('compile_dev')} cameras={meta['cameras']}")

  # Import late: this activates the vendored tinygrad for the rest of the process.
  from carrot.model_selector.genc.runtime import ModelState
  from openpilot.selfdrive.modeld.constants import ModelConstants

  cam_w, cam_h = args.camera
  t0 = time.monotonic()
  model = ModelState(args.out, cam_w, cam_h)
  print(f"loaded ModelState({cam_w}x{cam_h}) on {model.device_spec} (model_device={model.model_device}) in {time.monotonic() - t0:.1f}s")
  print(f"  state_pairs={model.state_pairs}")
  t0 = time.monotonic()
  model.warmup()
  print(f"warmup {time.monotonic() - t0:.2f}s")

  # Keep the raw output vector of every frame: the parser normalises some slices in place,
  # so it has to be copied before `run()` parses it.
  raw_outputs: list[np.ndarray] = []
  slice_outputs = model.slice_outputs

  def tap(model_outputs, output_slices):
    raw_outputs.append(np.array(model_outputs, dtype=np.float32, copy=True))
    return slice_outputs(model_outputs, output_slices)

  model.slice_outputs = tap
  feeds: list[dict[str, np.ndarray]] = []

  rng = np.random.default_rng(0)
  # dst(model 512x256) -> src(camera) mapping: plain scaling keeps the sampled region inside the frame
  m_inv = np.array([[cam_w / 512, 0, 0], [0, cam_h / 256, 0], [0, 0, 1]], dtype=np.float32)
  failures: list[str] = []
  out: dict = {}
  prev_state = model.state_snapshot()
  engine_states: list[dict[str, np.ndarray]] = []
  ever_changed: set[str] = set()
  pulse_run = 1  # feed a lane-change desire on this run so the desire history queue has to change too
  for i in range(args.runs):
    frames = {k: SimpleNamespace(data=rng.integers(0, 256, model.frame_copy_size, dtype=np.uint8)) for k in model.vision_input_names}
    desire_pulse = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if i == pulse_run:
      desire_pulse[1] = 1.0
    inputs = {
      'desire_pulse': desire_pulse,
      'traffic_convention': np.array([1, 0], dtype=np.float32),
      'action_t': np.array([0.3, 0.5], dtype=np.float32),
    }
    t0 = time.monotonic()
    out = model.run(frames, dict.fromkeys(model.vision_input_names, m_inv), inputs, prepare_only=False)
    dt = time.monotonic() - t0
    # what the model was actually given this frame: the warp output and the packed host inputs
    feeds.append({model.image_input: model.input_queues[model.image_input].numpy().copy()}
                 | {name: value.copy() for name, value in model.npy.items() if name != 'tfm'})
    plan = out['plan']
    finite = all(np.all(np.isfinite(v)) for v in out.values() if isinstance(v, np.ndarray))
    state = model.state_snapshot()
    changed = {k: bool(np.any(state[k] != prev_state[k])) for k in state}
    prev_state = state
    engine_states.append(state)
    ever_changed.update(k for k, v in changed.items() if v)
    print(f"run {i}: {dt * 1000:.0f} ms plan{plan.shape} finite={finite} "
          + f"v0={plan[0, 0, 0]:.2f} state_changed={changed} keys={len(out)}")
    if tuple(plan.shape) != PLAN_SHAPE:
      failures.append(f"run {i}: plan shape {plan.shape} != {PLAN_SHAPE}")
    if not finite:
      failures.append(f"run {i}: non-finite output")
    # Image history (uint8) takes a new random frame every run. The float queues can lag: the
    # desire queue only differs from all-zero once the pulse has entered it, and Gen B pushes the
    # previous frame's feature, which is still zero on the first run. Those are checked after the loop.
    stuck = [k for k in changed if model.input_shapes[k][1] == np.uint8 and not changed[k]]
    if stuck:
      failures.append(f"run {i}: image history did not advance: {stuck}")
  for key in ('lane_lines', 'road_edges', 'lead', 'meta', 'pose', 'desire_state', 'plan_stds'):
    if key not in out:
      failures.append(f"missing parsed output {key}")
  if args.runs > pulse_run + 1:
    never = sorted(set(model.state_pairs) - ever_changed)
    if never:
      failures.append(f"state never changed in {args.runs} runs: {never}")

  if not args.no_reference:
    t0 = time.monotonic()
    ref_outputs, ref_states = reference_run(args.onnx, model, feeds)
    for i, (got, want) in enumerate(zip(raw_outputs, ref_outputs, strict=True)):
      diff = float(np.max(np.abs(got - want)))
      close = same_values(got, want)
      wrong_states = sorted(k for k in model.state_pairs if not same_values(engine_states[i][k], ref_states[i][k]))
      print(f"reference {i}: max|engine - onnx|={diff:.3g} (max|onnx|={float(np.max(np.abs(want))):.3g}) "
            + ("ok" if close else "MISMATCH") + f" states {'ok' if not wrong_states else 'MISMATCH ' + str(wrong_states)}")
      if not close:
        failures.append(f"run {i}: raw outputs differ from direct ONNX evaluation (max abs diff {diff:.3g})")
      if wrong_states:
        failures.append(f"run {i}: state left for the next frame differs from direct ONNX evaluation: {wrong_states}")
    print(f"reference total {time.monotonic() - t0:.0f}s")

  if failures:
    print("FAIL\n  " + "\n  ".join(failures))
    return 1
  print("PASS")
  return 0


if __name__ == '__main__':
  sys.exit(main())
