"""carrot 전용: PlotSchedGate(DebugPlot 스케줄러 안전 경계) 회귀 테스트.

route 416 주행 해제 사고의 수정을 고정한다: DebugPlot 활성 시 UI를 SCHED_OTHER로
강등(fail-closed), 비활성 시 FIFO 53 복구(부분 성공 금지 — 실패 시 반드시 롤백),
손상된 ShowPlotMode 값은 UI를 죽이지 않고 plot off로 처리.
"""
import os
import sys
import types

import pytest

import openpilot.selfdrive.ui.carrot_plot_sched as cps
from openpilot.common.realtime import Priority
from openpilot.selfdrive.ui.carrot_plot_sched import PLOT_MODE_MAX, PlotSchedGate, UI_CORE


def make_gate(demote_results=(True,), restore_results=(True,)):
  """syscall 없이 전이 로직만 검증하는 게이트 (읽기/강등/복구 스텁)."""
  g = object.__new__(PlotSchedGate)
  g._raw_mode = 0
  g._effective_mode = 0
  g._demoted = False
  g._failed_mode = None
  g._mode_to_set = 0
  g._params = types.SimpleNamespace()
  g._refresh_gate = types.SimpleNamespace(should_refresh=lambda now: True)
  g._read_mode = lambda: g._mode_to_set
  calls = {"demote": 0, "restore": 0}

  def demote():
    calls["demote"] += 1
    return demote_results[min(calls["demote"] - 1, len(demote_results) - 1)]

  def restore():
    calls["restore"] += 1
    return restore_results[min(calls["restore"] - 1, len(restore_results) - 1)]

  g._demote = demote
  g._restore = restore
  return g, calls


class TestTransitions:
  def test_activate_demotes_exactly_once(self):
    g, calls = make_gate()
    g._mode_to_set = 1
    assert g.update() == 1
    for _ in range(50):  # 같은 모드 50프레임 — syscall 재호출 금지
      assert g.update() == 1
    assert calls["demote"] == 1
    assert calls["restore"] == 0

  def test_mode_change_while_active_no_redemote(self):
    g, calls = make_gate()
    g._mode_to_set = 1
    g.update()
    g._mode_to_set = 2
    assert g.update() == 2
    assert calls["demote"] == 1

  def test_deactivate_restores_exactly_once(self):
    g, calls = make_gate()
    g._mode_to_set = 1
    g.update()
    g._mode_to_set = 0
    assert g.update() == 0
    for _ in range(50):
      assert g.update() == 0
    assert calls["restore"] == 1

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

  def test_restore_failure_no_exception_no_storm(self):
    g, calls = make_gate(restore_results=(False,))
    g._mode_to_set = 5
    g.update()
    g._mode_to_set = 0
    assert g.update() == 0  # 예외 없이 effective 0
    for _ in range(50):
      g.update()
    assert calls["restore"] == 1  # 실패 후 매 프레임 재시도 금지
    g._mode_to_set = 5
    assert g.update() == 5  # 이후 재활성화 정상 (demote 재호출)
    assert calls["demote"] == 2

  def test_active_to_invalid_mode_restores(self):
    # 1 -> -1 전이: _read_mode 정규화(0)로 FIFO 복구가 반드시 수행돼야 한다
    g, calls = make_gate()
    g._mode_to_set = 1
    g.update()
    g._read_mode = lambda: 0  # -1은 _read_mode에서 0으로 정규화됨
    assert g.update() == 0
    assert calls["restore"] == 1


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
class TestRestoreAtomicity:
  """_restore는 부분 성공을 남기면 안 된다 — 실패 시 최종 상태는 반드시 SCHED_OTHER."""

  def _run_restore(self, monkeypatch, *, affinity_raises=False, sched_raises=False,
                   verify_policy=None, verify_priority=None, verify_affinity=None,
                   drop_fails=0):
    calls = {"order": [], "rollback": 0}
    monkeypatch.setattr(cps, "PC", False)
    state = {"policy": os.SCHED_OTHER, "prio": 0, "affinity": {UI_CORE}}

    def fake_setaffinity(pid, cores):
      calls["order"].append("affinity")
      if affinity_raises:
        raise OSError("affinity failed")
      state["affinity"] = set(cores)

    def fake_setscheduler(pid, policy, param):
      calls["order"].append("sched")
      if sched_raises:
        raise OSError("sched failed")
      state["policy"], state["prio"] = policy, param.sched_priority

    def fake_drop():
      calls["rollback"] += 1
      if calls["rollback"] <= drop_fails:
        raise OSError("rollback refused")
      state["policy"], state["prio"] = os.SCHED_OTHER, 0

    def fake_getscheduler(pid):
      # 3중 검증에는 주입값(verify_policy)을, 롤백 readback에는 실제 상태를 돌려준다
      if verify_policy is not None and state["policy"] != os.SCHED_OTHER:
        return verify_policy
      return state["policy"]

    monkeypatch.setattr(os, "sched_setaffinity", fake_setaffinity)
    monkeypatch.setattr(os, "sched_setscheduler", fake_setscheduler)
    monkeypatch.setattr(os, "sched_getscheduler", fake_getscheduler)
    monkeypatch.setattr(os, "sched_getparam",
                        lambda pid: os.sched_param(state["prio"] if verify_priority is None else verify_priority))
    monkeypatch.setattr(os, "sched_getaffinity",
                        lambda pid: state["affinity"] if verify_affinity is None else set(verify_affinity))
    monkeypatch.setattr(cps, "drop_realtime", fake_drop)
    result = PlotSchedGate._restore()
    return result, calls, state

  def test_success_verifies_policy_priority_affinity(self, monkeypatch):
    result, calls, state = self._run_restore(monkeypatch)
    assert result is True
    assert calls["rollback"] == 0
    assert calls["order"] == ["affinity", "sched"]  # affinity 먼저, FIFO는 마지막
    assert state["policy"] == os.SCHED_FIFO
    assert state["prio"] == Priority.CTRL_HIGH

  def test_affinity_failure_never_promotes_to_fifo(self, monkeypatch):
    result, calls, state = self._run_restore(monkeypatch, affinity_raises=True)
    assert result is False
    assert "sched" not in calls["order"]  # affinity 실패 시 FIFO 승격 자체가 없어야 함
    assert calls["rollback"] == 1
    assert state["policy"] == os.SCHED_OTHER

  def test_sched_failure_rolls_back(self, monkeypatch):
    result, calls, state = self._run_restore(monkeypatch, sched_raises=True)
    assert result is False
    assert calls["rollback"] == 1
    assert state["policy"] == os.SCHED_OTHER

  def test_verification_mismatch_rolls_back(self, monkeypatch):
    # 설정 콜은 성공했지만 실제 policy가 FIFO가 아니면 롤백 후 False
    result, calls, state = self._run_restore(monkeypatch, verify_policy=os.SCHED_OTHER)
    assert result is False
    assert calls["rollback"] == 1
    assert state["policy"] == os.SCHED_OTHER

  def test_priority_mismatch_rolls_back(self, monkeypatch):
    result, calls, state = self._run_restore(monkeypatch, verify_priority=1)
    assert result is False
    assert calls["rollback"] >= 1
    assert state["policy"] == os.SCHED_OTHER

  def test_affinity_mismatch_rolls_back(self, monkeypatch):
    result, calls, state = self._run_restore(monkeypatch, verify_affinity={0, UI_CORE})
    assert result is False
    assert calls["rollback"] >= 1
    assert state["policy"] == os.SCHED_OTHER

  def test_rollback_retry_succeeds_on_second_attempt(self, monkeypatch):
    result, calls, state = self._run_restore(monkeypatch, verify_priority=1, drop_fails=1)
    assert result is False
    assert calls["rollback"] == 2  # 1차 실패 후 bounded 재시도
    assert state["policy"] == os.SCHED_OTHER

  def test_rollback_total_failure_returns_false_without_exception(self, monkeypatch):
    # 커널이 롤백 syscall까지 거부: 예외 전파 없이 False, FIFO 잔류(파이썬에서 강제 불가).
    # 이 상태에서도 update()는 plot을 잠그고 재활성화는 _demote 검증을 요구한다.
    result, calls, state = self._run_restore(monkeypatch, verify_priority=1, drop_fails=2)
    assert result is False
    assert calls["rollback"] == 2  # 재시도 상한 준수 (폭주 없음)
    assert state["policy"] == os.SCHED_FIFO  # 잔류를 숨기지 않고 그대로 노출


class TestRestoreFailureLogging:
  """update()의 복구 실패 로그가 실제 policy와 모순되지 않아야 한다."""

  def _gate_with_log(self, monkeypatch, policy):
    logs = {"error": [], "critical": [], "warning": []}
    monkeypatch.setattr(cps, "cloudlog", types.SimpleNamespace(
      error=logs["error"].append, critical=logs["critical"].append,
      warning=logs["warning"].append, exception=lambda m: None))
    g, _ = make_gate(restore_results=(False,))
    g._current_policy = lambda: policy
    g._mode_to_set = 1
    g.update()
    g._mode_to_set = 0
    g.update()
    return logs

  def test_rolled_back_logs_staying_other(self, monkeypatch):
    logs = self._gate_with_log(monkeypatch, os.SCHED_OTHER)
    assert any("staying SCHED_OTHER" in m for m in logs["error"])
    assert not logs["critical"]

  def test_rollback_failed_logs_actual_policy(self, monkeypatch):
    logs = self._gate_with_log(monkeypatch, os.SCHED_FIFO)
    assert any("rollback FAILED" in m and f"policy={os.SCHED_FIFO}" in m for m in logs["critical"])
    assert not any("staying SCHED_OTHER" in m for m in logs["error"])
