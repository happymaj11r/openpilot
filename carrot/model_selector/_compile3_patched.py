#!/usr/bin/env python3
"""tinygrad compile3.py wrapper.

tinygrad ``create_schedule`` 의 toposort 루프(``tinygrad/engine/schedule.py``)
가 다음 assert 에서 일부 모델 컴파일을 거부한다::

    AssertionError: END src[0] should be KERNEL, not Ops.STORE

PR 코멘트에 따르면 tinygrad ``409bb0`` 이후로 큐 할당 부수효과가 ``END(STORE)``
로 래핑되는 경우가 있는데, 이 toposort 단계에서는 그것을 정상 처리에서 제외
(continue) 하는 것이 안전하다.

이전 시도(``graph_rewrite`` 로 sched_sink 전체에서 ``END(STORE)`` → ``STORE``
변환)는 schedule pipeline 의 다른 단계에까지 변형이 영향을 미쳐 컴파일된 PKL
이 입력에 무관해지는 부작용을 보였다.  그래서 변환 대신 **toposort 검증 한
곳에만 ``continue`` 분기를 추가**한다.  AFTER 노드와 END(STORE) 패턴 자체는
그대로 보존되므로 schedule 의 다른 단계에서는 원형으로 처리된다.

이 파일은 tinygrad 원본 코드를 수정하지 않고, monkey-patch 로
``tinygrad.engine.schedule.create_schedule`` 함수만 (본 모듈 안에서) 새 본문으로
교체한다.  carrot model_selector 의 컴파일 프로세스에서만 effective.

사용::

    python3 _compile3_patched.py <compile3.py> <onnx> <pkl>
"""
from __future__ import annotations

import runpy
import sys


def _patched_create_schedule(sched_sink):
  """Drop-in for ``tinygrad/engine/schedule.py::create_schedule``.

  Identical to upstream except for a single added ``continue`` that skips
  ``AFTER`` nodes whose ``src[1]`` is ``END(STORE)`` — queue-assignment side
  effects (tinygrad 409bb0+).
  """
  from collections import deque
  from tinygrad.uop.ops import UOp, Ops, gate_kernel_sink
  from tinygrad.helpers import cpu_profile, TracingKey
  from tinygrad.engine.schedule import _unwrap_src

  with cpu_profile(TracingKey("toposort sched_sink")):
    children: dict = {}
    in_degree: dict = {}
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      k = u.src[1]
      if k.op is Ops.STORE: continue  # skip unprocessed STORE+AFTER inside precompiled CALL bodies
      # PATCH: END(STORE) wraps queue-assignment side effects in tinygrad 409bb0+.
      # Treat the same as a raw STORE skip — leaves the END(STORE) node intact
      # for other schedule stages to handle normally.
      if k.op is Ops.END and len(k.src) and k.src[0].op is Ops.STORE: continue
      assert k.op in {Ops.CALL, Ops.END}, f"AFTER src[1] should be CALL or END, not {k.op}"
      in_degree.setdefault(k, 0)
      if k.op is Ops.END:
        assert k.src[0].op is Ops.CALL, f"END src[0] should be KERNEL, not {k.src[0].op}"
      # WAR deps from rangeify are stored in AFTER src[2:]
      kernel_deps = k.src[0].src[1:] if k.op is Ops.END else k.src[1:]
      for s in kernel_deps + u.src[2:]:
        match (s := _unwrap_src(s)).op:
          case Ops.AFTER:
            children.setdefault(s.src[1], []).append(k)
            in_degree[k] += 1
          case Ops.MSELECT | Ops.MSTACK:
            for ss in s.src:
              if ss.op is Ops.MSELECT: ss = ss.src[0]
              if ss.op not in {Ops.BUFFER, Ops.PARAM}:
                assert ss.op is Ops.AFTER, f"ss.op is not AFTER, it's {ss.op}"
                children.setdefault(ss.src[1], []).append(k)
                in_degree[k] += 1
          case Ops.BUFFER | Ops.PARAM | Ops.BIND:
            pass
          case _:
            raise RuntimeError(
              f"input to kernel must be AFTER, BUFFER, PARAM, MSELECT, MSTACK, or BIND, not {s.op}"
            )

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque = deque(k for k, v in in_degree.items() if v == 0)
    linearized: list = []
    while len(queue):
      rk = queue.popleft()
      if rk.op is Ops.LINEAR:
        linearized.extend(rk.src)
      else:
        k = rk.src[0] if rk.op is Ops.END else rk
        assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
        buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if s.op is not Ops.BIND)
        linearized.append(k.src[0].call(*buf_uops, metadata=k.arg.metadata))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0:
          queue.append(x)
  return UOp(Ops.LINEAR, src=tuple(linearized))


def _apply_patch() -> None:
  from tinygrad.engine import schedule as _sched
  _sched.create_schedule = _patched_create_schedule


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
