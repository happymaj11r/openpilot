#!/usr/bin/env python3
"""PC smoke check for the Gen C pipeline: compile a Gen C supercombo with the vendored
tinygrad, load it through `genc.runtime.ModelState`, and run a few synthetic frames.

    PYTHONPATH=<repo root> python3 carrot/model_selector/genc/pc_check.py \
        --onnx /path/driving_supercombo.onnx --out /path/out_dir [--camera 1928x1208] [--runs 3]

Exit status 0 only when every check passes. Uses DEV=CPU:LLVM unless --dev is given.
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
  args = ap.parse_args()

  from carrot.model_selector.validator import GENC_CAMERA_CONFIGS
  cameras = tuple(args.cameras) if args.cameras else GENC_CAMERA_CONFIGS
  args.out.mkdir(parents=True, exist_ok=True)

  if not (args.skip_compile and has_genc(args.out)):
    t0 = time.monotonic()
    compile_model(args.onnx, args.out, cameras, args.dev, sys.executable)
    print(f"compile total {time.monotonic() - t0:.0f}s", flush=True)
  meta = json.loads((args.out / GENC_METADATA_NAME).read_text())
  print(f"metadata: generation={meta['generation']} compile_dev={meta.get('compile_dev')} cameras={meta['cameras']}")

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

  rng = np.random.default_rng(0)
  # dst(model 512x256) -> src(camera) mapping: plain scaling keeps the sampled region inside the frame
  m_inv = np.array([[cam_w / 512, 0, 0], [0, cam_h / 256, 0], [0, 0, 1]], dtype=np.float32)
  failures: list[str] = []
  out: dict = {}
  prev_state = model.state_snapshot()
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
    plan = out['plan']
    finite = all(np.all(np.isfinite(v)) for v in out.values() if isinstance(v, np.ndarray))
    state = model.state_snapshot()
    changed = {k: bool(np.any(state[k] != prev_state[k])) for k in state}
    prev_state = state
    print(f"run {i}: {dt * 1000:.0f} ms plan{plan.shape} finite={finite} "
          + f"v0={plan[0, 0, 0]:.2f} state_changed={changed} keys={len(out)}")
    if tuple(plan.shape) != PLAN_SHAPE:
      failures.append(f"run {i}: plan shape {plan.shape} != {PLAN_SHAPE}")
    if not finite:
      failures.append(f"run {i}: non-finite output")
    # Image/feature history must shift every frame; the desire queue only differs from its
    # all-zero content once a pulse has entered it (an all-zero shift is a no-op).
    must_change = [k for k in changed if k != 'state_desire_q'] + (['state_desire_q'] if i == pulse_run else [])
    stuck = [k for k in must_change if not changed[k]]
    if stuck:
      failures.append(f"run {i}: state queue did not advance: {stuck}")
  for key in ('lane_lines', 'road_edges', 'lead', 'meta', 'pose', 'desire_state', 'plan_stds'):
    if key not in out:
      failures.append(f"missing parsed output {key}")

  if failures:
    print("FAIL\n  " + "\n  ".join(failures))
    return 1
  print("PASS")
  return 0


if __name__ == '__main__':
  sys.exit(main())
