#!/usr/bin/env python3
"""tinygrad compile3.py wrapper applying END(STORE) → STORE unwrap patch.

tinygrad 409bb0+는 큐 할당 부수효과를 ``END(STORE)`` 로 래핑할 수 있는데,
``tinygrad/engine/schedule.py::create_schedule`` 의 toposort 루프는 다음
assert 에서 이를 거부한다::

    AssertionError: END src[0] should be KERNEL, not Ops.STORE

commaai/openpilot ``op_model16_deep`` 브랜치(커밋 a2f8fc9a) 의
``_patch_tinygrad_schedule_end_store`` 와 동일한 의도이며, 다만 우리 트리에는
PR이 패치하는 ``tinygrad.schedule._split_after`` 가 존재하지 않으므로
``create_schedule`` 진입 직전에 ``AFTER`` 노드의 의존성 ``src[1:]`` 에서
``END(STORE)`` 패턴을 ``STORE`` 로 언랩하는 graph_rewrite 를 한 번 돌린다.

사용::

    python3 _compile3_patched.py <compile3.py> <onnx> <pkl>
"""
from __future__ import annotations

import runpy
import sys


def _apply_patch() -> None:
  from tinygrad.engine import schedule as _sched
  from tinygrad.uop.ops import Ops, UOp, graph_rewrite, PatternMatcher, UPat

  original_create_schedule = _sched.create_schedule

  def _unwrap_end_store(after: UOp):
    new_deps = []
    changed = False
    for s in after.src[1:]:
      if s.op is Ops.END and len(s.src) and s.src[0].op is Ops.STORE:
        new_deps.append(s.src[0])
        changed = True
      else:
        new_deps.append(s)
    if not changed:
      return None
    return after.replace(src=(after.src[0], *new_deps))

  pm = PatternMatcher([(UPat(Ops.AFTER, name='after'), _unwrap_end_store)])

  def create_schedule_patched(sched_sink: UOp) -> UOp:
    return original_create_schedule(graph_rewrite(sched_sink, pm, name="end_store_unwrap"))

  _sched.create_schedule = create_schedule_patched


def main() -> None:
  if len(sys.argv) < 2:
    print("usage: _compile3_patched.py <compile3.py> [args...]", file=sys.stderr)
    sys.exit(2)
  _apply_patch()
  script = sys.argv[1]
  sys.argv = [script] + sys.argv[2:]
  runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
  main()
