"""carrot 전용: big-UI(hud_renderer) PlotRenderer batch 포팅 회귀 테스트.

route 41a: tizi는 BIG_UI라 mici batch가 실행되지 않고 legacy per-segment
draw_line_ex(최대 1,197콜/frame + Vector2 ~1,200개 할당)가 돌았다. 이 테스트는
big-UI가 공용 batch 헬퍼(carrot_plot_draw)로 시리즈당 1콜을 쓰고, 좌표/순서/라벨이
기존 구현과 동일함을 고정한다.
"""
import types

import openpilot.selfdrive.ui.onroad.hud_renderer as hud
from openpilot.selfdrive.ui.onroad.hud_renderer import PlotRenderer

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

  def test_backend_logged_via_helper(self, monkeypatch):
    p, calls = make_renderer(monkeypatch)
    p._draw_plotting(0, 350.0, 40.0, object(), None)
    assert calls["backend"] == [(PLOT_MAX, 3, 3)]
