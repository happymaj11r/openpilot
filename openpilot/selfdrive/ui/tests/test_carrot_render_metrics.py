"""carrot 전용: SectionMetrics(wall/cpu 분리 계측) 회귀 테스트.

계측은 진단 도구일 뿐이므로 어떤 실패도 UI 렌더 루프로 전파되면 안 되고(no-throw),
매 프레임 로그 없이 윈도 집계 한 줄만 남기며, phase 경계가 한 줄에 섞이면 안 된다.
"""
import types

import openpilot.system.ui.lib.carrot_render_metrics as crm
from openpilot.system.ui.lib.carrot_render_metrics import SectionMetrics


def _capture_logs(monkeypatch):
  logs = []
  monkeypatch.setattr(crm, "cloudlog", types.SimpleNamespace(warning=logs.append))
  return logs


class TestAggregation:
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

  def test_window_timestamps_present(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=2)
    m.add(1.0, 1.0)
    m.add(1.0, 1.0)
    assert "t0=" in logs[0] and "t1=" in logs[0]  # 윈도 시작/종료 시각으로 단계 매핑

  def test_t0_t1_are_sample_boundaries_not_emit_time(self, monkeypatch):
    # 늦은 flush(예: 인코더 close 후 수십 초)가 t1을 오염시키면 단계 매핑이 불가능해진다
    logs = _capture_logs(monkeypatch)
    clock = {"mono": 10.0}
    monkeypatch.setattr(crm, "time", types.SimpleNamespace(
      perf_counter_ns=lambda: 0, thread_time_ns=lambda: 0,
      monotonic=lambda: clock["mono"]))
    m = SectionMetrics("test", window=100)
    m.add(1.0, 1.0)        # 첫 샘플: t0=10.0
    clock["mono"] = 12.5
    m.add(1.0, 1.0)        # 마지막 샘플: t1=12.5
    clock["mono"] = 70.0   # flush가 한참 뒤에 불림
    m.flush()
    assert "t0=10.0" in logs[0] and "t1=12.5" in logs[0]

  def test_end_uses_token_start_time(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    clock = {"mono": 5.0}
    monkeypatch.setattr(crm, "time", types.SimpleNamespace(
      perf_counter_ns=lambda: 0, thread_time_ns=lambda: 0,
      monotonic=lambda: clock["mono"]))
    m = SectionMetrics("test", window=100)
    tok = m.begin()        # 샘플 시작 5.0
    clock["mono"] = 6.0
    m.end(tok)             # 샘플 종료 6.0
    clock["mono"] = 99.0
    m.flush()
    assert "t0=5.0" in logs[0] and "t1=6.0" in logs[0]

  def test_negative_samples_clamped(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=2)
    m.add(-5.0, -1.0)  # 시계 이상 등 — 음수는 0으로
    m.add(3.0, 2.0)
    assert "max=3.00" in logs[0]
    assert "mean=1.50" in logs[0]  # (0+3)/2 — 음수가 통계를 왜곡하지 않는다

  def test_begin_end_produces_nonnegative_sample(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=1)
    tok = m.begin()
    m.end(tok)
    assert len(logs) == 1
    assert "wall mean=" in logs[0] and "cpu mean=" in logs[0]


class TestNoThrow:
  """계측의 어떤 실패도 호출자(렌더 루프)로 전파되면 안 된다."""

  def test_begin_clock_failure_returns_none(self, monkeypatch):
    def boom():
      raise OSError("clock down")
    monkeypatch.setattr(crm, "time", types.SimpleNamespace(
      perf_counter_ns=boom, thread_time_ns=boom, monotonic=lambda: 0.0))
    assert SectionMetrics.begin() is None

  def test_end_none_token_is_noop(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=1)
    m.end(None)  # begin 실패 시 토큰 — no-op이어야 한다
    assert logs == [] and len(m._wall) == 0

  def test_end_malformed_token_swallowed(self, monkeypatch):
    _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=10)
    m.end("garbage")
    m.end((1,))  # index 부족
    m.end((None, None))  # 연산 불가
    assert len(m._wall) == 0 == len(m._cpu)

  def test_add_conversion_failure_keeps_buffers_paired(self, monkeypatch):
    _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=10)
    m.add("abc", 1.0)      # wall 변환 실패
    m.add(1.0, object())   # cpu 변환 실패 — wall만 append되면 안 된다
    assert len(m._wall) == 0 == len(m._cpu)
    m.add(1.0, 1.0)
    assert len(m._wall) == 1 == len(m._cpu)

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


class TestPhaseAndFlush:
  def test_phase_change_flushes_partial_window(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=100)
    m.set_phase((1, False))
    for _ in range(7):
      m.add(1.0, 1.0)
    m.set_phase((1, True))  # 녹화 시작 — 이전 단계 7개가 섞이면 안 된다
    assert len(logs) == 1
    assert "phase=1/False" in logs[0] and "n=7" in logs[0]
    m.add(2.0, 2.0)
    assert len(m._wall) == 1  # 새 단계는 빈 버퍼에서 시작

  def test_same_phase_does_not_flush(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=100)
    m.set_phase((1, False))
    m.add(1.0, 1.0)
    m.set_phase((1, False))  # 동일 키 — flush 없음
    assert logs == [] and len(m._wall) == 1

  def test_phase_label_in_full_window_emit(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=2)
    m.set_phase((3, True))
    m.add(1.0, 1.0)
    m.add(1.0, 1.0)
    assert "phase=3/True" in logs[0]

  def test_recording_session_id_separates_rotation(self, monkeypatch):
    # 60초 회전: 같은 프레임에 stop→start라 recording bool은 True→True로 동일 —
    # 세션 ID가 증가해야 회전 전후 부분 윈도가 분리된다
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=100)
    m.set_phase((1, True, 1))
    for _ in range(4):
      m.add(1.0, 1.0)
    m.set_phase((1, True, 2))  # 회전 후 새 세션
    assert len(logs) == 1
    assert "phase=1/True/1" in logs[0] and "n=4" in logs[0]

  def test_flush_empty_is_noop(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=10)
    m.flush()
    assert logs == []

  def test_flush_partial_emits_and_resets(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=100)
    for _ in range(3):
      m.add(1.0, 1.0)
    m.flush()  # 녹화 stop 등 세션 경계
    assert len(logs) == 1 and "n=3" in logs[0]
    assert len(m._wall) == 0

  def test_set_phase_exception_swallowed(self, monkeypatch):
    _capture_logs(monkeypatch)
    m = SectionMetrics("test", window=10)
    class Weird:
      def __eq__(self, other):
        raise RuntimeError("cmp fail")
    m.set_phase(Weird())  # 비교 실패도 전파 금지
