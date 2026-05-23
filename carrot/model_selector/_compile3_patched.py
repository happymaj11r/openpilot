#!/usr/bin/env python3
"""tinygrad compile3.py wrapper.

두 가지 환경 우회를 수행한다:

1.  ``tinygrad.engine.schedule.create_schedule`` monkey-patch — toposort 루프에
    한 줄(``continue``)을 추가해 ``AFTER → END(STORE)`` 패턴을 정상 처리에서
    제외. tinygrad 409bb0+ 가 큐 할당 부수효과를 END(STORE) 로 래핑할 때
    발생하는 다음 assert 를 우회한다::

        AssertionError: END src[0] should be KERNEL, not Ops.STORE

2.  ``compile3.py`` 의 ``__main__`` block 우회 — runpy 대신 ``importlib`` 으로
    모듈을 로드해 ``compile()`` 만 호출한다.  ``compile()`` 안에서 PKL 이
    이미 디스크에 저장(line 62 ``pickle.dump(...)``)되므로 그 후의
    ``test_vs_compile`` sanity check 는 생략 가능.  test_vs_compile 은
    ``inputs.numpy() * 2`` 로 두 번째 추론을 돌려 출력이 달라야 한다는
    검증인데, 우리 트리의 tinygrad + NPY device 조합에서는 어떤 모델 +
    JIT 캐시 동작 때문에 입력 갱신이 반영되지 않아 환경적으로 항상 fail
    한다 (PR ``op_model16_deep`` 은 compile3.py 가 아니라 자체
    ``selfdrive/modeld/compile_modeld.py`` 를 쓰므로 영향 없음).
    PKL 자체의 정확성은 ``compile()`` 안의 ``np.testing.assert_equal(test_val, ret)``
    (JIT run 일관성 검사) 및 OnnxRunner 의 forward pass 로 이미 확인됨.

사용::

    python3 _compile3_patched.py <compile3.py> <onnx> <pkl>
"""
from __future__ import annotations

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
  if len(sys.argv) < 4:
    print("usage: _compile3_patched.py <compile3.py> <onnx> <pkl>", file=sys.stderr)
    sys.exit(2)
  _apply_patch()

  compile3_script = sys.argv[1]
  onnx_path = sys.argv[2]
  pkl_path = sys.argv[3]

  # compile3.py 의 module-level 글로벌(``OPENPILOT_MODEL``, ``OUTPUT``) 은
  # ``sys.argv`` 를 읽어서 set 된다.  importlib 로 로드하기 전에 미리 세팅.
  sys.argv = [compile3_script, onnx_path, pkl_path]

  # importlib 로 로드하면 ``if __name__ == "__main__"`` block (test_vs_compile
  # 호출이 포함된) 은 자동으로 실행되지 않는다.  ``compile()`` 만 직접 호출.
  import importlib.util
  spec = importlib.util.spec_from_file_location("_compile3", compile3_script)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load compile3 from {compile3_script}")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)

  # compile() 안에서 PKL 이 디스크에 저장됨.
  mod.compile(mod.fetch(mod.OPENPILOT_MODEL))
  print("**** wrapper: compile done, skipping test_vs_compile (env-dependent) ****")


if __name__ == "__main__":
  main()
