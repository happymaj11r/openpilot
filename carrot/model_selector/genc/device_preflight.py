#!/usr/bin/env python3
"""On-device preflight for the Gen C engine — run once on a comma device BEFORE installing
the first Gen C model, so the vendored tinygrad generation is proven against this AGNOS
build (host compiler for the QCOM submission program, kgsl access, JIT replay):

    cd /data/openpilot && PYTHONPATH=/data/openpilot python3 carrot/model_selector/genc/device_preflight.py

Prints one PASS/FAIL line per check and exits non-zero on any FAIL. Safe to run while
offroad (opens the GPU briefly; nothing is installed or written).
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from carrot.model_selector.genc import TINYGRAD_DIR, TINYGRAD_PIN, activate_tinygrad

results: list[tuple[str, bool, str]] = []


def check(name: str):
  def wrap(fn):
    start = time.monotonic()
    try:
      detail = fn() or ""
      ok = True
    except Exception as e:  # report, keep going
      detail = f"{type(e).__name__}: {e}"
      ok = False
      traceback.print_exc()
    results.append((name, ok, f"{detail} ({time.monotonic() - start:.1f}s)"))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    return fn
  return wrap


def main() -> int:
  on_device = os.path.isfile('/AGNOS')
  device = 'QCOM' if on_device else 'CPU:LLVM'
  print(f"python {sys.version.split()[0]}  AGNOS={on_device}  vendored tinygrad {TINYGRAD_PIN[:12]} @ {TINYGRAD_DIR}")

  @check("host compiler (clang binary / libLLVM for the CPU helper runtime)")
  def _host_compiler():
    found = []
    if shutil.which('clang'):
      found.append(f"clang={shutil.which('clang')}")
    try:
      from tinygrad.runtime.autogen import llvm  # noqa: F401
      found.append("libLLVM=ok")
    except Exception as e:
      found.append(f"libLLVM=missing({type(e).__name__})")
    if not any(f.startswith('clang=') or f == 'libLLVM=ok' for f in found):
      raise RuntimeError("neither clang nor libLLVM available — QCOM runtime cannot build its submission program")
    return " ".join(found)

  activate_tinygrad()
  import tinygrad
  from tinygrad import Context, Device, Tensor, TinyJit
  assert Path(tinygrad.__file__).resolve().is_relative_to(TINYGRAD_DIR), f"system tinygrad leaked in: {tinygrad.__file__}"

  with Context(DEV=device):
    @check(f"open {device} + CPU helper device")
    def _open():
      d = Device[Device.DEFAULT]
      cpu = Device['CPU']
      return (f"DEFAULT={Device.DEFAULT} renderer={type(d.renderer).__name__} "
              + f"cpu_renderer={type(cpu.renderer).__name__} cpu_compiler={type(cpu.compiler).__name__}")

    @check(f"JIT kernel on {device} (compile + 2 replays)")
    def _jit():
      @TinyJit
      def f(x):
        return (x * 2 + 1).sum()
      vals = [f(Tensor.ones(256, 256)).item() for _ in range(3)]
      if any(abs(v - 256 * 256 * 3) > 1e-3 for v in vals):
        raise RuntimeError(f"wrong result {vals}")
      t = time.monotonic()
      f(Tensor.ones(256, 256)).item()
      return f"ok, replay {(time.monotonic() - t) * 1000:.1f} ms"

    @check("NPY -> device packed upload + device -> CPU readback (runtime copy path)")
    def _copies():
      import numpy as np
      host = np.arange(4096, dtype=np.uint8)
      t = Tensor(host, device='NPY').to(Device.DEFAULT).realize()
      back = t.to('CPU').numpy()
      if not np.array_equal(back, host):
        raise RuntimeError("roundtrip mismatch")
      return "roundtrip ok"

  failed = [n for n, ok, _ in results if not ok]
  print("\n" + ("PREFLIGHT FAIL: " + ", ".join(failed) if failed else "PREFLIGHT PASS"))
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main())
