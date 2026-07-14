"""carrot 전용: PlotSchedGate(DebugPlot 스케줄러 안전 경계) 회귀 테스트.

UI 상시 SCHED_OTHER 설계(scheduler/affinity foundation)를 고정한다 (route 416:
FIFO53 UI가 core5의 plannerd/radard FIFO51을 굶겨 해제 사고, route 00000426:
core5 포화 재발, 교차 리뷰: core7 이전안은 dmonitoringmodeld FIFO5 기아
위험으로 기각):
- UI에 RT 승격 경로 재도입 금지 (FIFO 복구 기계 부재, gc.disable은 유지)
- 부트스트랩은 core0(offroad power-save의 core4~7 offline 대응), 목표는
  UI_CORE=5 단일 출처 — 비RT UI는 FIFO51을 선점하지 못하므로 core5 공유 안전,
  core7은 modeld+DM 전용 (재도입 금지)
- plot은 UI가 비RT임을 검증한 경우에만 허용(fail-closed), 강등은 affinity 불변
- 손상된 ShowPlotMode 값은 UI를 죽이지 않고 plot off로 처리
"""
import ast
import os
import sys
import types
from pathlib import Path

import pytest

import openpilot.selfdrive.ui.carrot_plot_sched as cps
from openpilot.selfdrive.ui.carrot_plot_sched import PLOT_MODE_MAX, PlotSchedGate, UI_CORE


def make_gate(demote_results=(True,)):
  """syscall 없이 전이 로직만 검증하는 게이트 (읽기/강등 스텁 — 복구 경로는 없다)."""
  g = object.__new__(PlotSchedGate)
  g._raw_mode = 0
  g._effective_mode = 0
  g._demoted = False
  g._failed_mode = None
  g._mode_to_set = 0
  g._params = types.SimpleNamespace()
  g._refresh_gate = types.SimpleNamespace(should_refresh=lambda now: True)
  g._read_mode = lambda: g._mode_to_set
  calls = {"demote": 0}

  def demote():
    calls["demote"] += 1
    return demote_results[min(calls["demote"] - 1, len(demote_results) - 1)]

  g._demote = demote
  return g, calls


class TestTransitions:
  def test_activate_demotes_exactly_once(self):
    g, calls = make_gate()
    g._mode_to_set = 1
    assert g.update() == 1
    for _ in range(50):  # 같은 모드 50프레임 — syscall 재호출 금지
      assert g.update() == 1
    assert calls["demote"] == 1

  def test_mode_change_while_active_no_redemote(self):
    g, calls = make_gate()
    g._mode_to_set = 1
    g.update()
    g._mode_to_set = 2
    assert g.update() == 2
    assert calls["demote"] == 1

  def test_deactivate_clears_latch_without_syscall(self):
    # 비활성 전이는 syscall 0회 — 복구할 RT 상태가 없다 (상시 SCHED_OTHER)
    g, calls = make_gate()
    g._mode_to_set = 1
    g.update()
    g._mode_to_set = 0
    assert g.update() == 0
    for _ in range(50):
      assert g.update() == 0
    assert calls["demote"] == 1  # 활성 전이의 검증 1회가 전부

  def test_reactivate_reverifies_nonrt(self):
    # off로 래치가 풀리면 다음 활성화가 비RT를 다시 검증해야 한다
    g, calls = make_gate()
    g._mode_to_set = 1
    g.update()
    g._mode_to_set = 0
    g.update()
    g._mode_to_set = 2
    assert g.update() == 2
    assert calls["demote"] == 2

  def test_demotion_failure_is_fail_closed_and_latched(self):
    g, calls = make_gate(demote_results=(False, True))
    g._mode_to_set = 1
    assert g.update() == 0  # 강등 실패 -> plot 비활성
    for _ in range(50):
      assert g.update() == 0
    assert calls["demote"] == 1  # 같은 모드로는 재시도 금지 (래치)

  def test_demotion_retried_after_mode_change(self):
    g, calls = make_gate(demote_results=(False, True))
    g._mode_to_set = 1
    g.update()
    g._mode_to_set = 3  # 모드 값이 바뀌면 재시도 허용
    assert g.update() == 3
    assert calls["demote"] == 2

  def test_active_to_invalid_mode_deactivates(self):
    # 1 -> -1 전이: _read_mode 정규화(0)로 plot이 반드시 꺼져야 한다 (syscall 없음)
    g, calls = make_gate()
    g._mode_to_set = 1
    g.update()
    g._read_mode = lambda: 0  # -1은 _read_mode에서 0으로 정규화됨
    assert g.update() == 0
    assert calls["demote"] == 1


class TestReadMode:
  def _gate_with_param(self, value=None, raises=None):
    g = object.__new__(PlotSchedGate)

    def get(key, block=False, return_default=False):
      assert key == "ShowPlotMode" and return_default
      if raises is not None:
        raise raises
      return value

    g._params = types.SimpleNamespace(get=get)
    return g

  def test_valid_range(self):
    for v in range(1, PLOT_MODE_MAX + 1):
      assert self._gate_with_param(v)._read_mode() == v
    assert self._gate_with_param(0)._read_mode() == 0

  def test_out_of_range_normalized_to_zero(self):
    for v in (-1, 9, 999, -999):
      assert self._gate_with_param(v)._read_mode() == 0

  def test_non_int_types_rejected(self):
    # Params.get은 손상된 값("abc" 등)을 default로 변환하지만, 방어적으로
    # int가 아닌 모든 타입(str/None/bool/float)을 0으로 처리해야 한다
    for v in ("abc", None, True, 3.5, b"1"):
      assert self._gate_with_param(v)._read_mode() == 0

  def test_overflow_like_large_int(self):
    assert self._gate_with_param(2**63)._read_mode() == 0

  def test_read_exception_fail_closed(self):
    assert self._gate_with_param(raises=RuntimeError("corrupt"))._read_mode() == 0


@pytest.mark.skipif(sys.platform != "linux", reason="sched_* API는 Linux 전용")
class TestDemoteContract:
  """_demote는 policy만 SCHED_OTHER로 내린다 — affinity를 건드리면 UI가 core7
  밖(planner/radar의 core5 등)으로 샐 수 있다."""

  def _run_demote(self, monkeypatch, *, policy_after):
    monkeypatch.setattr(cps, "PC", False)
    affinity_calls = []
    monkeypatch.setattr(os, "sched_setaffinity",
                        lambda pid, c: affinity_calls.append(set(c)))
    monkeypatch.setattr(cps, "drop_realtime", lambda: None)
    monkeypatch.setattr(os, "sched_getscheduler", lambda pid: policy_after)
    return PlotSchedGate._demote(), affinity_calls

  def test_demotion_verified_policy_other(self, monkeypatch):
    ok, _ = self._run_demote(monkeypatch, policy_after=os.SCHED_OTHER)
    assert ok is True

  def test_demotion_readback_mismatch_fails(self, monkeypatch):
    # drop 콜이 성공해도 실제 policy가 OTHER가 아니면 강등 실패 (fail-closed)
    ok, _ = self._run_demote(monkeypatch, policy_after=os.SCHED_FIFO)
    assert ok is False

  def test_demotion_never_touches_affinity(self, monkeypatch):
    ok, affinity_calls = self._run_demote(monkeypatch, policy_after=os.SCHED_OTHER)
    assert ok is True and affinity_calls == []


def _ui_py_tree():
  # ui.py는 import 부작용(gui_app 생성, 레이아웃→msgq)이 있어 AST로 고정한다
  return ast.parse((Path(cps.__file__).parent / "ui.py").read_text())


def _call_names(tree):
  names = []
  for n in ast.walk(tree):
    if isinstance(n, ast.Call):
      if isinstance(n.func, ast.Name):
        names.append(n.func.id)
      elif isinstance(n.func, ast.Attribute):
        names.append(n.func.attr)
  return names


class TestUiCoreSeparation:
  """UI 코어 배치 foundation: core0 부트스트랩(offroad power-save는 core4~7을
  offline — always_run UI가 offline 코어 affinity로 시작하면 크래시/재시작
  루프) 후 render loop가 UI_CORE=5로 re-affine. 비RT UI는 core5의 FIFO51을
  선점하지 못하므로 안전하고, core7은 modeld(FIFO54)+dmonitoringmodeld(FIFO5)
  전용이라 UI 재배치 금지."""

  def test_ui_core_is_five(self):
    assert UI_CORE == 5  # 명시적 고정 — 동적 참조 통과에 기대지 않는다

  def test_ui_py_bootstraps_core0_and_reaffines_cores_var(self):
    tree = _ui_py_tree()
    # 부트스트랩: set_core_affinity([0]) — 항상 online인 core0에서 시작
    aff_shapes = []
    for n in ast.walk(tree):
      if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "set_core_affinity"):
        continue
      a = n.args[0]
      if isinstance(a, ast.List) and len(a.elts) == 1 and isinstance(a.elts[0], ast.Constant):
        aff_shapes.append(("const_list", a.elts[0].value))
      elif (isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id == "list"
            and len(a.args) == 1 and isinstance(a.args[0], ast.Name)):
        aff_shapes.append(("list_of_var", a.args[0].id))
    assert ("const_list", 0) in aff_shapes      # core0 부트스트랩
    # re-affine이 cores 변수와 직접 연결 — cores만 바꾸면 대상이 함께 바뀐다
    # (cores = {UI_CORE,}이고 UI_CORE == 5이므로 결과적으로 {5}와의 관계가 고정)
    assert ("list_of_var", "cores") in aff_shapes

  def test_ui_py_cores_var_is_ui_core_single_source(self):
    tree = _ui_py_tree()
    sets = [s for s in ast.walk(tree) if isinstance(s, ast.Set)]
    # cores = {UI_CORE, } — 리터럴 하드코딩 대신 단일 출처 상수를 쓴다
    assert any(len(s.elts) == 1 and isinstance(s.elts[0], ast.Name)
               and s.elts[0].id == "UI_CORE" for s in sets)
    # 어떤 set 리터럴에도 코어 상수가 없어야 한다 — 특히 core7 재도입 금지
    # (core7은 modeld+DM 전용), core5도 리터럴 대신 UI_CORE 경유만 허용
    consts = {e.value for s in sets for e in s.elts if isinstance(e, ast.Constant)}
    assert 7 not in consts and not consts
    # UI_CORE는 carrot_plot_sched에서 임포트한다 (값 분기 금지)
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module and n.module.endswith("carrot_plot_sched")]
    assert any(alias.name == "UI_CORE" for n in imports for alias in n.names)

  def test_ui_py_gc_disable_retained(self):
    # RT 승격 제거 과정에서 gc.disable()까지 사라지면 안 된다 — GC pause가
    # 렌더 프레임 히치를 만든다 (기존 config_realtime_process가 하던 것)
    tree = _ui_py_tree()
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "disable" and isinstance(n.func.value, ast.Name)
               and n.func.value.id == "gc" for n in ast.walk(tree))

  def test_ui_py_reaffine_failure_swallowed_and_retryable(self):
    # affinity 실패(offroad에 목표 코어 offline 등)가 UI 프로세스를 죽이면 안
    # 된다 — try/except OSError로 삼키고, 호출부가 렌더 루프 안이라 다음
    # 프레임에 자연 재시도된다
    tree = _ui_py_tree()
    def contains_reaffine(node):
      return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "set_core_affinity"
                 and n.args and isinstance(n.args[0], ast.Call)
                 for n in ast.walk(node))
    guarded = [t for t in ast.walk(tree) if isinstance(t, ast.Try) and contains_reaffine(t)
               and any(isinstance(h.type, ast.Name) and h.type.id == "OSError"
                       for h in t.handlers if h.type is not None)]
    assert guarded, "re-affine은 try/except OSError 안에 있어야 한다"
    # 그리고 그 try는 루프(render loop) 안에 있어야 재시도가 성립한다
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
    assert any(any(t in ast.walk(loop) for t in guarded) for loop in loops)


class TestNoRtPromotion:
  """UI에 RT 승격 경로가 없어야 한다: FIFO53 UI는 core5에서 plannerd/radard
  (FIFO51)를 굶기고(route 416/00000426), core7에서는 dmonitoringmodeld(FIFO5)를
  굶긴다(교차 리뷰). offroad에 offline 코어로의 시작 승격은 startup 크래시를
  만든다. UI는 상시 SCHED_OTHER — 어느 소스에도 승격 프리미티브 재도입 금지."""

  def test_restore_machinery_removed(self):
    assert not hasattr(PlotSchedGate, "_restore")
    assert not hasattr(PlotSchedGate, "_rollback_to_other")

  def test_no_promotion_primitives_in_ui_py(self):
    tree = _ui_py_tree()
    names = _call_names(tree)
    assert "config_realtime_process" not in names  # 시작 FIFO 승격 금지 (재도입 금지)
    assert "sched_setscheduler" not in names
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "SCHED_FIFO" not in attrs
    # Priority(CTRL_HIGH/CTRL_LOW) 자체를 UI에서 쓰지 않는다
    idents = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "Priority" not in idents and "Priority" not in imported
    assert "CTRL_HIGH" not in attrs and "CTRL_LOW" not in attrs

  def test_no_promotion_primitives_in_plot_sched(self):
    src = (Path(cps.__file__).parent / "carrot_plot_sched.py").read_text()
    tree = ast.parse(src)
    assert "sched_setscheduler" not in _call_names(tree)  # drop_realtime 경유만 허용
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "SCHED_FIFO" not in attrs
    assert "CTRL_HIGH" not in attrs and "CTRL_LOW" not in attrs


class TestLegacyLogsRemoved:
  # 과거 스케줄러 로그 문자열이 남아 있으면 실기기 tmux/rlog 분석이 구 설계
  # (FIFO 복구/코어 표기)로 오판한다 — 소스 레벨에서 부재를 고정
  LEGACY = (
    "restored to SCHED_FIFO",       # FIFO 복구 로그 (복구 기계와 함께 제거됨)
    "SCHED_OTHER/core5",            # route 416 시절 강등 로그
    "SCHED_OTHER/core7",            # core7 이전안 로그
    "UI RT restore FAILED",         # 복구 실패 로그 계열
  )

  def test_no_legacy_log_strings_in_sources(self):
    for fname in ("carrot_plot_sched.py", "ui.py"):
      src = (Path(cps.__file__).parent / fname).read_text()
      for legacy in self.LEGACY:
        assert legacy not in src, f"{fname}: 과거 로그 문자열 잔존 — {legacy}"
