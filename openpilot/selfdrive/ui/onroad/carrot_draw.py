from typing import Optional

import numpy as np
import pyray as rl

from openpilot.system.ui.lib.shader_polygon import Gradient, ShaderState, _configure_shader_color, draw_polygon

_use_ffi = True


def draw_polygon_fast(origin_rect: rl.Rectangle, points: np.ndarray,
                      color: Optional[rl.Color] = None, gradient: Gradient | None = None) -> None:
  """upstream draw_polygon과 동일한 결과를 그리는 carrot 전용 고속 경로.

  트라이앵글 스트립을 float32 numpy 버퍼로 GPU에 직접 전달해서(Vector2[]와 메모리
  배치 동일) 포인트별 파이썬→C 변환을 없앤다. ffi 전달이 실패하는 환경이면
  upstream draw_polygon으로 영구 폴백한다. upstream shader_polygon의 셰이더
  상태를 그대로 재사용하므로 시각적 결과는 동일하다.
  """
  global _use_ffi
  if not _use_ffi:
    draw_polygon(origin_rect, points, color, gradient)
    return

  if len(points) < 3:
    return

  try:
    state = ShaderState.get_instance()
    state.initialize()

    pts = np.ascontiguousarray(points, dtype=np.float32)
    if pts.shape[0] % 2 != 0:
      pts = pts[:-1]
    half = pts.shape[0] // 2
    strip = np.empty((half * 2, 2), dtype=np.float32)
    strip[0::2] = pts[:half]
    strip[1::2] = pts[::-1][:half]

    _configure_shader_color(state, color, gradient, origin_rect)

    buf = rl.ffi.from_buffer(strip)
    rl.begin_shader_mode(state.shader)
    try:
      rl.draw_triangle_strip(rl.ffi.cast("Vector2 *", buf), strip.shape[0], rl.WHITE)
    finally:
      rl.end_shader_mode()
  except Exception:
    _use_ffi = False
    draw_polygon(origin_rect, points, color, gradient)
