"""carrot 전용: big-UI(hud_renderer) PlotRenderer batch 포팅 회귀 테스트.

route 41a: tizi는 BIG_UI라 mici batch가 실행되지 않고 legacy per-segment
draw_line_ex(최대 1,197콜/frame + Vector2 ~1,200개 할당)가 돌았다. 이 테스트는
big-UI가 공용 batch 헬퍼(carrot_plot_draw)로 시리즈당 1콜을 쓰고, 좌표/순서/라벨이
기존 구현과 동일함을 고정한다.

big-UI는 PlotRenderer.draw()가 uiRender/drawTimeMillis 측정 안에 중첩돼 있으므로,
draw() 경로에서는 계측이 절대 로그를 쓰지 않고(deferred) 바깥 구간 종료 후
emit_pending에서만 배출되는 불변식도 고정한다.
"""
import types

import openpilot.selfdrive.ui.carrot_plot_draw as cpd
import openpilot.selfdrive.ui.onroad.hud_renderer as hud
import openpilot.system.ui.lib.carrot_render_metrics as crm
from openpilot.selfdrive.ui.onroad.hud_renderer import PlotRenderer
from openpilot.system.ui.lib.carrot_render_metrics import SectionMetrics

PLOT_MAX = PlotRenderer.PLOT_MAX


class FakeVec:
  __slots__ = ("x", "y")

  def __init__(self, x=0.0, y=0.0):
    self.x, self.y = x, y


def make_renderer(monkeypatch, *, plot_index=157, plot_size=PLOT_MAX):
  calls = {"polyline": [], "backend": [], "text": []}
  monkeypatch.setattr(hud, "plot_draw", types.SimpleNamespace(
    log_backend_once=lambda *a: calls["backend"].append(a),
    draw_polyline=lambda pts, n, color, stroke=3: calls["polyline"].append(
      ([(pt.x, pt.y) for pt in pts[:n]], stroke)),
  ))
  monkeypatch.setattr(hud, "draw_text_ui_style", lambda text, *a, **k: calls["text"].append(text))

  p = object.__new__(PlotRenderer)
  p._plot_size = plot_size
  p._plot_index = plot_index
  p._plot_queue = [[float((i * 7 + s * 13) % 50) / 10.0 - 2.0 for i in range(PLOT_MAX)] for s in range(3)]
  p._plot_min, p._plot_max = -2.0, 4.0
  p._plot_height, p._plot_dx = 300.0, 2.0
  p._pts = [FakeVec() for _ in range(PLOT_MAX)]
  return p, calls


def legacy_points(p, index, x_base, y_base):
  """batch화 이전(per-segment draw_line_ex) 구현과 동일한 수식으로 기대 좌표 생성."""
  plot_range = p._plot_max - p._plot_min
  plot_ratio = p._plot_height if plot_range < 1.0 else (p._plot_height / plot_range)
  pts = []
  for i in range(p._plot_size):
    data = p._plot_queue[index][(p._plot_index - i + PLOT_MAX) % PLOT_MAX]
    plot_y = y_base + p._plot_height - (data - p._plot_min) * plot_ratio
    plot_x = x_base + (p._plot_size - i) * p._plot_dx
    pts.append((plot_x, plot_y))
  return pts


class TestHudPlotBatch:
  def test_one_polyline_call_per_series(self, monkeypatch):
    p, calls = make_renderer(monkeypatch)
    for s in range(3):
      p._draw_plotting(s, 350.0, 40.0, object(), None)
    assert len(calls["polyline"]) == 3  # 기존 최대 1,197콜/frame -> 3콜
    assert all(stroke == 3 for _, stroke in calls["polyline"])
    assert len(calls["text"]) == 3  # 시리즈별 최신값 라벨 유지

  def test_coordinates_match_legacy(self, monkeypatch):
    p, calls = make_renderer(monkeypatch)
    for s in range(3):
      p._draw_plotting(s, 350.0, 40.0, object(), None)
    for s in range(3):
      assert calls["polyline"][s][0] == legacy_points(p, s, 350.0, 40.0)

  def test_ring_wrap_partial_fill(self, monkeypatch):
    p, calls = make_renderer(monkeypatch, plot_index=3, plot_size=17)
    p._draw_plotting(0, 350.0, 40.0, object(), None)
    assert calls["polyline"][0][0] == legacy_points(p, 0, 350.0, 40.0)

  def test_latest_label_value_preserved(self, monkeypatch):
    p, calls = make_renderer(monkeypatch)
    p._draw_plotting(0, 350.0, 40.0, object(), None)
    expected = p._plot_queue[0][p._plot_index % PLOT_MAX]  # i=0 (최신) 값
    assert calls["text"][0] == f"{expected:.2f}"

  def test_single_point_draws_label_only(self, monkeypatch):
    # 기존 구현: 포인트 1개면 선은 없고 라벨만
    p, calls = make_renderer(monkeypatch, plot_size=1)
    p._draw_plotting(0, 350.0, 40.0, object(), None)
    polyline_pts = calls["polyline"][0][0] if calls["polyline"] else []
    assert len(polyline_pts) <= 1  # draw_polyline은 n<2에서 무동작 (헬퍼 계약)
    assert len(calls["text"]) == 1

  def test_empty_draws_nothing(self, monkeypatch):
    p, calls = make_renderer(monkeypatch, plot_size=0)
    p._draw_plotting(0, 350.0, 40.0, object(), None)
    assert not calls["polyline"] and not calls["text"]

  def test_backend_log_not_called_in_draw_path(self, monkeypatch):
    # PLOTDRAW cloudlog가 첫 plot 프레임의 drawTime/uiRender를 오염시키면 안 된다 —
    # draw 경로에서는 이월 표시만 하고 실제 로그는 emit_pending_metrics()에서
    p, calls = make_renderer(monkeypatch)
    p._backend_log_pending = False
    p._draw_plotting(0, 350.0, 40.0, object(), None)
    assert calls["backend"] == []
    assert p._backend_log_pending is True

  def test_backend_logged_on_emit_with_args(self, monkeypatch):
    p, calls = make_renderer(monkeypatch)
    p._backend_log_pending = False
    p._plot_metrics = SectionMetrics("debugPlot", window=100, deferred=True)
    p._draw_plotting(0, 350.0, 40.0, object(), None)
    p.emit_pending_metrics()
    assert calls["backend"] == [(PLOT_MAX, 3, 3)]  # 인자 보존 (points/series/stroke)

  def test_helper_end_to_end_spline(self, monkeypatch):
    # hud.plot_draw를 fake로 바꾸지 않고 실제 공용 헬퍼를 경유 — big-UI가 spline
    # backend에서 시리즈당 draw_spline_linear 1콜을 내는 실제 wiring을 고정한다
    spline_calls = []
    monkeypatch.setattr(cpd, "rl", types.SimpleNamespace(
      draw_spline_linear=lambda pts, n, thick, color: spline_calls.append((n, thick))))
    monkeypatch.setattr(cpd, "HAS_SPLINE", True)
    monkeypatch.setattr(cpd, "_backend_logged", True)  # cloudlog 억제
    monkeypatch.setattr(hud, "draw_text_ui_style", lambda *a, **k: None)

    p = object.__new__(PlotRenderer)
    p._plot_size, p._plot_index = 50, 7
    p._plot_queue = [[float(i % 5) for i in range(PLOT_MAX)] for _ in range(3)]
    p._plot_min, p._plot_max = -2.0, 4.0
    p._plot_height, p._plot_dx = 300.0, 2.0
    p._pts = [FakeVec() for _ in range(PLOT_MAX)]
    p._draw_plotting(0, 350.0, 40.0, object(), None)
    assert spline_calls == [(50, 3.0)]


class FakeGate:
  def __init__(self, state):
    self._state = state

  @property
  def effective_mode(self):
    return self._state["mode"]


def _capture_logs(monkeypatch):
  logs = []
  monkeypatch.setattr(crm, "cloudlog", types.SimpleNamespace(warning=logs.append))
  return logs


def make_wired(monkeypatch, *, mode=1, window=100, real_helper=False):
  """draw() 레벨 wiring 테스트용 — 전역 의존은 stub, 필드/계측은 실물.

  real_helper=True면 hud.plot_draw를 실제 공용 헬퍼로 두고(cpd.rl/cloudlog만 fake)
  PLOTDRAW 이월/1회 보장을 실제 로깅 경로로 검증할 수 있다."""
  state = {"mode": mode, "rec": (False, 0)}
  monkeypatch.setattr(hud, "plot_sched_gate", FakeGate(state))
  monkeypatch.setattr(hud, "gui_app", types.SimpleNamespace(recording_phase=lambda: state["rec"]))
  monkeypatch.setattr(hud, "ui_state", types.SimpleNamespace(
    sm=types.SimpleNamespace(alive={"carState": True, "longitudinalPlan": True})))
  if not real_helper:
    monkeypatch.setattr(hud, "plot_draw", types.SimpleNamespace(
      log_backend_once=lambda *a: None, draw_polyline=lambda *a, **k: None))
  monkeypatch.setattr(hud, "draw_text_ui_style", lambda *a, **k: None)

  p = object.__new__(PlotRenderer)
  p._show_plot_mode_prev = -1
  p._plot_size, p._plot_index = 0, 0
  p._plot_queue = [[0.0] * PLOT_MAX for _ in range(3)]
  p._plot_min, p._plot_max = 0.0, 0.0
  p._plot_x, p._plot_y = 350.0, 40.0
  p._plot_height, p._plot_dx = 300.0, 2.0
  p._pts = [FakeVec() for _ in range(PLOT_MAX)]
  p._backend_log_pending = False
  p._plot_metrics = SectionMetrics("debugPlot", window=window, deferred=True)
  p._make_plot_data = lambda sm, mode: ([0.1, 0.2, 0.3], "T")
  rect = types.SimpleNamespace(width=1400.0, x=0.0, y=0.0)
  return p, state, rect


class TestHudPlotWiring:
  """PlotRenderer.draw()의 phase/begin/end 배치와 nested-emit 오염 부재를 고정."""

  def test_draw_never_logs_inside_frame(self, monkeypatch):
    # 핵심 불변식(지피짱 BLOCK 해소): 윈도가 차는 프레임에도 draw() 안에서는
    # cloudlog가 절대 불리지 않는다 — drawTimeMillis/uiRender 오염 원천 차단
    logs = _capture_logs(monkeypatch)
    p, _, rect = make_wired(monkeypatch, window=2)
    for _ in range(5):
      p.draw(rect, None)  # 완성 윈도 2개가 생기는 횟수
    assert logs == []
    p.emit_pending_metrics()  # 바깥 구간 종료 후 배출
    assert len(logs) == 2 and all("PLOTPERF debugPlot:" in line for line in logs)

  def test_mode_off_moves_partial_to_pending_without_log(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    p, state, rect = make_wired(monkeypatch)
    for _ in range(3):
      p.draw(rect, None)
    state["mode"] = 0  # plot OFF (D→E 전환 프레임)
    p.draw(rect, None)  # early return이지만 set_phase가 partial을 pending으로
    assert logs == []  # 전환 프레임 draw 안에서도 로그 금지
    p.emit_pending_metrics()
    assert len(logs) == 1
    assert "phase=1/False/0" in logs[0] and "n=3" in logs[0]

  def test_mode_off_repeat_emits_nothing(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    p, state, rect = make_wired(monkeypatch, mode=0)
    for _ in range(5):
      p.draw(rect, None)  # 동일 phase(OFF) 반복 — 샘플도 pending도 없어야 한다
    p.emit_pending_metrics()
    assert logs == [] and len(p._plot_metrics._wall) == 0

  def test_recording_session_change_separates_windows(self, monkeypatch):
    logs = _capture_logs(monkeypatch)
    p, state, rect = make_wired(monkeypatch)
    state["rec"] = (True, 1)
    p.draw(rect, None)
    p.draw(rect, None)
    state["rec"] = (True, 2)  # 60초 회전 — recording bool은 True→True
    p.draw(rect, None)
    p.emit_pending_metrics()
    assert len(logs) == 1
    assert "phase=1/True/1" in logs[0] and "n=2" in logs[0]

  def test_narrow_rect_still_records_sample(self, monkeypatch):
    _capture_logs(monkeypatch)
    p, _, rect = make_wired(monkeypatch)
    rect.width = 800.0  # rect.width < 1200 경로에도 end가 있어 샘플이 기록된다
    p.draw(rect, None)
    assert len(p._plot_metrics._wall) == 1

  def test_make_plot_data_exception_drops_sample(self, monkeypatch):
    _capture_logs(monkeypatch)
    p, _, rect = make_wired(monkeypatch)
    def boom(sm, mode):
      raise RuntimeError("cereal down")
    p._make_plot_data = boom
    p.draw(rect, None)  # 예외 전파 없이 해당 프레임 샘플만 drop
    assert len(p._plot_metrics._wall) == 0

  def test_hud_renderer_delegates_emit(self, monkeypatch):
    hr = object.__new__(hud.HudRenderer)
    hr._plot_renderer = None
    hr.emit_pending_plot_metrics()  # plot 미생성 상태에서 no-op

    called = []
    hr._plot_renderer = types.SimpleNamespace(emit_pending_metrics=lambda: called.append(1))
    hr.emit_pending_plot_metrics()
    assert called == [1]

  def test_backend_log_deferred_and_once_real_helper(self, monkeypatch):
    # 실제 헬퍼 로깅을 살린 채(1회 가드 리셋) 첫 B 프레임 시나리오 재현:
    # draw() 안 cloudlog 0 → emit_pending_metrics()에서 PLOTDRAW 정확히 1회 →
    # 재 draw+emit에도 추가 PLOTDRAW 없음 (헬퍼 프로세스 가드)
    perf_logs = _capture_logs(monkeypatch)  # PLOTPERF (crm.cloudlog)
    helper_logs = []
    monkeypatch.setattr(cpd, "cloudlog", types.SimpleNamespace(warning=helper_logs.append))
    monkeypatch.setattr(cpd, "_backend_logged", False)
    monkeypatch.setattr(cpd, "HAS_SPLINE", True)
    monkeypatch.setattr(cpd, "rl", types.SimpleNamespace(
      draw_spline_linear=lambda pts, n, thick, color: None))

    p, state, rect = make_wired(monkeypatch, real_helper=True)
    p.draw(rect, None)  # 첫 plot 프레임
    assert helper_logs == [] and perf_logs == []  # draw 경로 어디서도 cloudlog 금지
    p.emit_pending_metrics()
    assert len(helper_logs) == 1 and helper_logs[0].startswith("PLOTDRAW: backend=spline")
    p.draw(rect, None)
    p.emit_pending_metrics()
    assert len(helper_logs) == 1  # 추가 PLOTDRAW 없음
