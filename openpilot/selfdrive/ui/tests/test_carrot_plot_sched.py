"""carrot 전용: PlotSchedGate(DebugPlot 스케줄러 안전 경계) 회귀 테스트.

UI 상시 SCHED_OTHER/core7 설계를 고정한다 (route 416: FIFO53 UI가 core5의
plannerd/radard FIFO51을 굶겨 해제 사고, route 00000426: core5 포화 재발,
교차 리뷰: core7의 FIFO53 UI는 dmonitoringmodeld FIFO5를 굶겨 DM 지연 가능):
- UI에 RT 승격 경로 재도입 금지 (FIFO 복구 기계 부재)
- 부트스트랩은 core0(offroad power-save의 core4~7 offline 대응), 목표는 {7}
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
  """route 00000426: DebugPlot OFF에서도 UI+radard+plannerd가 core5를 공유해
  94~107% 포화 → longitudinalPlan 16~18Hz → softDisabling 반복. UI는 core0
  부트스트랩(offroad power-save는 core4~7 offline — always_run UI가 offline
  core7 affinity로 시작하면 크래시/재시작 루프) 후 render loop가 core7로
  re-affine한다. core5 재도입 금지."""

  def test_ui_core_is_seven(self):
    assert UI_CORE == 7  # 명시적 고정 — 동적 참조 통과에 기대지 않는다

  def test_ui_py_bootstraps_core0_and_reaffines_core7(self):
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
    assert ("list_of_var", "cores") in aff_shapes

  def test_ui_py_cores_var_is_ui_core_single_source(self):
    tree = _ui_py_tree()
    sets = [s for s in ast.walk(tree) if isinstance(s, ast.Set)]
    # cores = {UI_CORE, } — 리터럴 하드코딩 대신 단일 출처 상수를 쓴다
    assert any(len(s.elts) == 1 and isinstance(s.elts[0], ast.Name)
               and s.elts[0].id == "UI_CORE" for s in sets)
    # 어떤 set 리터럴에도 core5 상수가 없어야 한다 (재도입 금지)
    consts = {e.value for s in sets for e in s.elts if isinstance(e, ast.Constant)}
    assert 5 not in consts
    # UI_CORE는 carrot_plot_sched에서 임포트한다 (값 분기 금지)
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module and n.module.endswith("carrot_plot_sched")]
    assert any(alias.name == "UI_CORE" for n in imports for alias in n.names)


class TestNoRtPromotion:
  """UI에 RT 승격 경로가 없어야 한다 (교차 리뷰 blocker): FIFO53 UI는 core7의
  dmonitoringmodeld(FIFO5)를 굶겨 운전자 감시를 지연시킬 수 있고, offroad
  core7 offline 상태의 시작 승격은 startup 크래시를 만든다. UI는 상시
  SCHED_OTHER — 어느 소스에도 승격 프리미티브 재도입 금지."""

  def test_restore_machinery_removed(self):
    assert not hasattr(PlotSchedGate, "_restore")
    assert not hasattr(PlotSchedGate, "_rollback_to_other")

  def test_no_promotion_primitives_in_ui_py(self):
    names = _call_names(_ui_py_tree())
    assert "config_realtime_process" not in names  # 시작 FIFO 승격 금지
    assert "sched_setscheduler" not in names

  def test_no_promotion_primitives_in_plot_sched(self):
    src = (Path(cps.__file__).parent / "carrot_plot_sched.py").read_text()
    tree = ast.parse(src)
    assert "sched_setscheduler" not in _call_names(tree)  # drop_realtime 경유만 허용
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "SCHED_FIFO" not in attrs and "CTRL_HIGH" not in attrs
