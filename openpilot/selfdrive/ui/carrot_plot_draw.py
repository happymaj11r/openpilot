"""carrot 전용: DebugPlot 폴리라인 batch 드로잉 헬퍼 (mici/big-UI 공용).

per-segment draw_line 계열은 프레임당 수천 Python→raylib 콜과 Vector2 할당으로
UI를 상시 실행 상태로 만든다 (route 416/41a — big-UI는 최대 1,197콜/frame).
draw_spline_linear(두께 지원, 시리즈당 1콜) → draw_line_strip(스트로크당 1콜) →
per-segment 순으로 폴백한다. 가용성은 로드 시 1회 판정하고, 실제 선택된 backend는
최초 사용 시 프로세스 수명당 1회만 로그(PLOTDRAW)한다.

mici(debug_plot.py)와 big-UI(hud_renderer.py PlotRenderer)가 같은 경로를 쓰도록
여기에만 폴백/로그 로직을 둔다 — 두 UI가 다시 갈라지지 않게 유지할 것.
"""
import pyray as rl

from openpilot.common.swaglog import cloudlog

HAS_SPLINE = hasattr(rl, "draw_spline_linear")
HAS_LINE_STRIP = hasattr(rl, "draw_line_strip")
_backend_logged = False
_fallback_warned = False


def backend_name() -> str:
  return "spline" if HAS_SPLINE else ("line_strip" if HAS_LINE_STRIP else "legacy")


def log_backend_once(points: int, series: int, stroke: int) -> None:
  """실제 선택된 backend를 프로세스 수명당 1회만 기록 — 진단 로그가 첫 plot
  프레임을 죽이면 안 되므로 no-throw."""
  global _backend_logged
  if _backend_logged:
    return
  _backend_logged = True
  try:
    ver = getattr(rl, "RAYLIB_VERSION", "?")
    cloudlog.warning(f"PLOTDRAW: backend={backend_name()} points={points} series={series} stroke={stroke} raylib={ver}")
  except Exception:
    pass


def draw_polyline(pts, n, color, stroke: int = 3) -> None:
  """pts[:n](rl.Vector2 재사용 버퍼)를 연결선으로 그린다.

  line_strip 폴백은 y offset을 누적 적용하므로 호출 후 pts의 y가 ±(stroke//2)
  이동할 수 있다 — 좌표를 다시 쓸 호출자는 미리 복사해 둘 것.
  """
  global _fallback_warned
  if n < 2:
    return
  if HAS_SPLINE:
    rl.draw_spline_linear(pts, n, float(max(1, stroke)), color)
  elif HAS_LINE_STRIP:
    offsets = range(-(stroke // 2), stroke // 2 + 1) if stroke > 1 else range(1)
    applied = 0
    for o in offsets:
      d = o - applied
      if d:
        for i in range(n):
          pts[i].y += d
        applied = o
      rl.draw_line_strip(pts, n, color)
  else:
    if not _fallback_warned:
      _fallback_warned = True
      try:
        cloudlog.warning("PLOTDRAW: no batch line API in pyray, falling back to per-segment draw_line")
      except Exception:
        pass
    for i in range(1, n):
      x0, y0 = pts[i - 1].x, pts[i - 1].y
      x1, y1 = pts[i].x, pts[i].y
      if stroke <= 1:
        rl.draw_line(int(x0), int(y0), int(x1), int(y1), color)
      else:
        for o in range(-(stroke // 2), stroke // 2 + 1):
          rl.draw_line(int(x0), int(y0) + o, int(x1), int(y1) + o, color)
