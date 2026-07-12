"""carrot 전용: SectionMetrics(wall/cpu 분리 계측) 회귀 테스트.

계측은 진단 도구일 뿐이므로 어떤 실패도 UI 렌더 루프로 전파되면 안 되고,
매 프레임 로그 없이 윈도 집계 한 줄만 남겨야 한다.
"""
import types

import openpilot.system.ui.lib.carrot_render_metrics as crm
from openpilot.system.ui.lib.carrot_render_metrics import SectionMetrics


def _capture_logs(monkeypatch):
  logs = []
  monkeypatch.setattr(crm, "cloudlog", types.SimpleNamespace(warning=logs.append))
  return logs


class TestSectionMetrics:
  def test_no_log_until_window_full(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=10)
    for _ in range(9):
      m.add(1.0, 0.5)
    assert logs == []  # 매 프레임 로그 금지 — 윈도가 차기 전에는 무출력

  def test_emits_once_per_window_and_resets(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=10)
    for _ in range(25):
      m.add(2.0, 1.0)
    assert len(logs) == 2  # 10개마다 1줄, 나머지 5개는 대기
    assert all("PLOTPERF test:" in line and "n=10" in line for line in logs)

  def test_negative_samples_clamped(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=2)
    m.add(-5.0, -1.0)  # 시계 이상 등 — 음수는 0으로
    m.add(3.0, 2.0)
    assert "max=3.00" in logs[0]
    assert "mean=1.50" in logs[0]  # (0+3)/2 — 음수가 통계를 왜곡하지 않는다

  def test_emit_failure_does_not_propagate(self, monkeypatch):
    def boom(_):
      raise RuntimeError("log backend down")
    monkeypatch.setattr(crm, "cloudlog", types.SimpleNamespace(warning=boom))
    m = SectionMetrics("test", window=1)
    m.add(1.0, 1.0)  # 예외가 렌더 루프로 전파되면 안 된다
    m.add(1.0, 1.0)

  def test_window_resets_after_emit_failure(self, monkeypatch):
    calls = {"n": 0}
    def flaky(_):
      calls["n"] += 1
      raise RuntimeError("down")
    monkeypatch.setattr(crm, "cloudlog", types.SimpleNamespace(warning=flaky))
    m = SectionMetrics("test", window=2)
    for _ in range(6):
      m.add(1.0, 1.0)
    assert calls["n"] == 3  # 실패해도 버퍼는 리셋되어 무한 누적되지 않는다

  def test_begin_end_produces_nonnegative_sample(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=1)
    tok = m.begin()
    m.end(tok)
    assert len(logs) == 1
    assert "wall mean=" in logs[0] and "cpu mean=" in logs[0]
