"""carrot 전용: mici DebugPlot batch 렌더링 회귀 테스트.

route 416: per-segment draw_line(프레임당 ~2,691콜)이 UI를 상시 실행 상태로 만들어
plannerd/radard를 굶겼다. 이 테스트는 draw 콜 수가 batch 수준(시리즈당 1~stroke콜)으로
유지되고, 좌표·ring buffer(oldest->newest) 순서가 기존 구현과 동일함을 고정한다.
"""
import types

import pytest

import openpilot.selfdrive.ui.mici.onroad.debug_plot as dp
from openpilot.selfdrive.ui.mici.onroad.debug_plot import PLOT_MAX, DebugPlot


class FakeVec:
  __slots__ = ("x", "y")

  def __init__(self, x=0.0, y=0.0):
    self.x, self.y = x, y


class FakeRl:
  def __init__(self):
    self.spline_calls = []
    self.strip_calls = []
    self.line_calls = []
    self.text_calls = []

  def draw_spline_linear(self, pts, n, thick, color):
    self.spline_calls.append(([(p.x, p.y) for p in pts[:n]], thick))

  def draw_line_strip(self, pts, n, color):
    self.strip_calls.append([(p.x, p.y) for p in pts[:n]])

  def draw_line(self, x0, y0, x1, y1, color):
    self.line_calls.append((x0, y0, x1, y1))

  def draw_text(self, label, x, y, size, color):
    self.text_calls.append(label)

  def measure_text(self, label, size):
    return 50


def make_plot(monkeypatch, *, has_spline, has_strip, plot_index=137, plot_size=PLOT_MAX):
  fake = FakeRl()
  monkeypatch.setattr(dp, "rl", fake)
  monkeypatch.setattr(dp, "_HAS_SPLINE", has_spline)
  monkeypatch.setattr(dp, "_HAS_LINE_STRIP", has_strip)
  monkeypatch.setattr(dp, "_batch_warned", False)
  monkeypatch.setattr(dp, "_backend_logged", True)  # backend 로그는 전용 테스트에서만

  p = object.__new__(DebugPlot)
  p.plot_size = plot_size
  p.plot_index = plot_index  # ring wrap 상태
  p.plot_min, p.plot_max = -2.0, 4.0
  p.plot_x, p.plot_y = 10.0, 40.0
  p.plot_height, p.plot_dx = 300.0, 2.5
  p.plot_queue = [[float((i * 7 + s * 13) % 50) / 10.0 - 2.0 for i in range(PLOT_MAX)] for s in range(3)]
  p._pts = [FakeVec() for _ in range(PLOT_MAX)]
  p._xs = [0.0] * PLOT_MAX
  p._xs_key = None
  p._update_x_cache()
  return p, fake


def legacy_points(p, series_idx):
  """batch화 이전(per-segment) 구현과 동일한 수식으로 기대 좌표를 생성한다."""
  pr = p.plot_max - p.plot_min
  ratio = p.plot_height if pr < 1e-6 else (p.plot_height / pr)
  pts = []
  for i in range(p.plot_size):
    k_back = (p.plot_size - 1) - i  # oldest -> newest
    idx = (p.plot_index - k_back) % PLOT_MAX
    val = p.plot_queue[series_idx][idx]
    x = p.plot_x + i * p.plot_dx
    y = p.plot_y + p.plot_height - (val - p.plot_min) * ratio
    pts.append((x, y))
  return pts


RECT = types.SimpleNamespace(x=0.0, y=0.0, width=800.0, height=400.0)
COLOR = object()


class TestBatchDrawing:
  def test_spline_path_one_call_per_series(self, monkeypatch):
    p, fake = make_plot(monkeypatch, has_spline=True, has_strip=True)
    for s in range(3):
      p._draw_series(RECT, s, COLOR, stroke=3)
    assert len(fake.spline_calls) == 3  # 기존 ~2,691콜/frame -> 3콜
    assert not fake.strip_calls and not fake.line_calls
    assert all(thick == 3.0 for _, thick in fake.spline_calls)
    assert len(fake.text_calls) == 3  # 시리즈별 최신값 라벨 유지

  def test_spline_coordinates_match_legacy(self, monkeypatch):
    p, fake = make_plot(monkeypatch, has_spline=True, has_strip=True)
    for s in range(3):
      p._draw_series(RECT, s, COLOR, stroke=3)
    for s in range(3):
      assert fake.spline_calls[s][0] == legacy_points(p, s)

  def test_line_strip_fallback_offsets(self, monkeypatch):
    p, fake = make_plot(monkeypatch, has_spline=False, has_strip=True)
    p._draw_series(RECT, 0, COLOR, stroke=3)
    ref = legacy_points(p, 0)
    assert len(fake.strip_calls) == 3  # stroke 3 -> y offset -1/0/+1 세 번
    for j in range(3):
      assert fake.strip_calls[j] == [(x, y + (j - 1)) for x, y in ref]

  def test_legacy_fallback_same_call_count_and_warns_once(self, monkeypatch):
    warnings = []
    monkeypatch.setattr(dp, "cloudlog", types.SimpleNamespace(warning=warnings.append))
    p, fake = make_plot(monkeypatch, has_spline=False, has_strip=False)
    p._draw_series(RECT, 0, COLOR, stroke=3)
    p._draw_series(RECT, 1, COLOR, stroke=3)
    # 기존 구현과 동일: (PLOT_MAX-1) segment * stroke 3
    assert len(fake.line_calls) == (PLOT_MAX - 1) * 3 * 2
    assert len(warnings) == 1

  def test_ring_wrap_partial_fill_preserves_order(self, monkeypatch):
    p, fake = make_plot(monkeypatch, has_spline=True, has_strip=True, plot_index=3, plot_size=17)
    p._draw_series(RECT, 0, COLOR)
    assert fake.spline_calls[0][0] == legacy_points(p, 0)

  @pytest.mark.parametrize("size", [0, 1])
  def test_too_few_points_draws_nothing(self, monkeypatch, size):
    p, fake = make_plot(monkeypatch, has_spline=True, has_strip=True, plot_size=size)
    p._draw_series(RECT, 0, COLOR)
    assert not fake.spline_calls and not fake.strip_calls and not fake.line_calls

  def test_x_cache_rebuilds_only_on_layout_change(self, monkeypatch):
    p, _ = make_plot(monkeypatch, has_spline=True, has_strip=True)
    xs_before = list(p._xs)
    p._update_x_cache()  # 같은 레이아웃 -> 재계산 없음 (키 동일)
    assert p._xs == xs_before
    p.plot_dx = 3.0
    p._update_x_cache()
    assert p._xs[1] == p.plot_x + 3.0


class TestBackendLog:
  """실제 선택된 batch backend를 프로세스 수명당 정확히 1회만 로그해야 한다."""

  def _run(self, monkeypatch, has_spline, has_strip):
    p, _ = make_plot(monkeypatch, has_spline=has_spline, has_strip=has_strip)
    warnings = []
    monkeypatch.setattr(dp, "cloudlog", types.SimpleNamespace(warning=warnings.append))
    monkeypatch.setattr(dp, "_backend_logged", False)
    p._draw_series(RECT, 0, COLOR, stroke=3)
    p._draw_series(RECT, 1, COLOR, stroke=3)  # 두 번째 호출은 로그 없어야 함
    return [w for w in warnings if "PLOTDRAW: backend=" in w]

  def test_spline_logged_once(self, monkeypatch):
    logs = self._run(monkeypatch, True, True)
    assert len(logs) == 1
    assert "backend=spline" in logs[0]
    assert "points=300" in logs[0] and "series=3" in logs[0] and "stroke=3" in logs[0]
    assert "raylib=?" in logs[0]  # RAYLIB_VERSION 부재 시 getattr 폴백

  def test_line_strip_logged_once(self, monkeypatch):
    logs = self._run(monkeypatch, False, True)
    assert len(logs) == 1 and "backend=line_strip" in logs[0]

  def test_legacy_logged_once(self, monkeypatch):
    logs = self._run(monkeypatch, False, False)
    assert len(logs) == 1 and "backend=legacy" in logs[0]

  def test_legacy_backend_and_fallback_warning_each_once(self, monkeypatch):
    # legacy에서는 backend 로그와 per-segment 폴백 경고가 각각 정확히 1회
    p, _ = make_plot(monkeypatch, has_spline=False, has_strip=False)
    warnings = []
    monkeypatch.setattr(dp, "cloudlog", types.SimpleNamespace(warning=warnings.append))
    monkeypatch.setattr(dp, "_backend_logged", False)
    p._draw_series(RECT, 0, COLOR, stroke=3)
    p._draw_series(RECT, 1, COLOR, stroke=3)
    assert len([w for w in warnings if "backend=legacy" in w]) == 1
    assert len([w for w in warnings if "no batch line API" in w]) == 1

  def test_backend_log_failure_swallowed_and_not_retried(self, monkeypatch):
    # 진단 로그 장애가 첫 plot 프레임을 죽이면 안 되고, 다음 프레임 재시도도 없어야 한다
    p, fake = make_plot(monkeypatch, has_spline=True, has_strip=True)
    def boom(_):
      raise RuntimeError("log backend down")
    monkeypatch.setattr(dp, "cloudlog", types.SimpleNamespace(warning=boom))
    monkeypatch.setattr(dp, "_backend_logged", False)
    p._draw_series(RECT, 0, COLOR, stroke=3)  # 예외가 전파되면 테스트 실패
    assert dp._backend_logged is True
    assert len(fake.spline_calls) == 1  # 로그 실패와 무관하게 그리기는 수행

  def test_fallback_warning_failure_swallowed(self, monkeypatch):
    p, fake = make_plot(monkeypatch, has_spline=False, has_strip=False)
    def boom(_):
      raise RuntimeError("log backend down")
    monkeypatch.setattr(dp, "cloudlog", types.SimpleNamespace(warning=boom))
    p._draw_series(RECT, 0, COLOR, stroke=3)  # _batch_warned 경고 실패도 무전파
    assert len(fake.line_calls) == (PLOT_MAX - 1) * 3


class TestPlotOffFlush:
  """plot OFF(D→E) 전환 시 마지막 부분 윈도가 early return 전에 배출돼야 한다."""

  def test_mode_off_flushes_partial_window(self, monkeypatch):
    import openpilot.system.ui.lib.carrot_render_metrics as crm
    from openpilot.system.ui.lib.carrot_render_metrics import SectionMetrics
    logs = []
    monkeypatch.setattr(crm, "cloudlog", types.SimpleNamespace(warning=logs.append))
    monkeypatch.setattr(dp, "plot_sched_gate", types.SimpleNamespace(effective_mode=0))
    monkeypatch.setattr(dp, "gui_app", types.SimpleNamespace(recording_phase=lambda: (False, 0)))

    p = object.__new__(DebugPlot)
    p._plot_metrics = SectionMetrics("debugPlot", window=100)
    p._plot_metrics.set_phase((1, False, 0))
    p._plot_metrics.add(1.0, 1.0)  # plot 활성 중 남은 부분 샘플

    p._render(RECT)  # mode 0 — early return이지만 phase 전환 flush는 일어나야 한다
    assert len(logs) == 1
    assert "phase=1/False/0" in logs[0] and "n=1" in logs[0]

  def test_mode_off_repeated_render_no_further_logs(self, monkeypatch):
    import openpilot.system.ui.lib.carrot_render_metrics as crm
    from openpilot.system.ui.lib.carrot_render_metrics import SectionMetrics
    logs = []
    monkeypatch.setattr(crm, "cloudlog", types.SimpleNamespace(warning=logs.append))
    monkeypatch.setattr(dp, "plot_sched_gate", types.SimpleNamespace(effective_mode=0))
    monkeypatch.setattr(dp, "gui_app", types.SimpleNamespace(recording_phase=lambda: (False, 0)))

    p = object.__new__(DebugPlot)
    p._plot_metrics = SectionMetrics("debugPlot", window=100)
    for _ in range(5):
      p._render(RECT)  # 동일 phase(0, False, 0) 유지 — 추가 배출 없음
    assert logs == []
