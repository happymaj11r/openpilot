"""carrot 전용: wall vs thread-CPU 분리 렌더 계측 (route 418 진단).

SCHED_OTHER로 강등된 UI는 FIFO plannerd/radard에 선점되므로 drawTimeMillis(wall)
증가가 곧 렌더 비용 증가를 뜻하지 않는다. 구간별 wall/cpu를 함께 재서
'선점 대기'(wall 증가, cpu 유지)와 '실제 비용'(wall/cpu 동반 증가)을 구분한다.

매 프레임 로그는 금지 — 샘플만 쌓고 윈도(기본 200)가 차면 집계 한 줄을
cloudlog(PLOTPERF)로 남긴 뒤 리셋한다. 계측 실패가 UI를 죽여서는 안 된다.
"""
import time

from openpilot.common.swaglog import cloudlog


class SectionMetrics:
  """한 구간의 wall/cpu(ms) 샘플 카운터. begin() 토큰을 end()에 돌려주면 기록된다."""

  def __init__(self, name: str, window: int = 200):
    self._name = name
    self._window = max(1, window)
    self._wall: list[float] = []
    self._cpu: list[float] = []

  @staticmethod
  def begin() -> tuple[int, int]:
    return time.perf_counter_ns(), time.thread_time_ns()

  def end(self, token: tuple[int, int]) -> None:
    wall_ms = (time.perf_counter_ns() - token[0]) / 1e6
    cpu_ms = (time.thread_time_ns() - token[1]) / 1e6
    self.add(wall_ms, cpu_ms)

  def add(self, wall_ms: float, cpu_ms: float) -> None:
    self._wall.append(max(0.0, float(wall_ms)))
    self._cpu.append(max(0.0, float(cpu_ms)))
    if len(self._wall) >= self._window:
      self._emit()

  def _emit(self) -> None:
    try:
      n = len(self._wall)
      w = sorted(self._wall)
      c = sorted(self._cpu)
      i50, i95 = n // 2, min(n - 1, int(n * 0.95))
      wall_s = f"wall mean={sum(w) / n:.2f} p50={w[i50]:.2f} p95={w[i95]:.2f} max={w[-1]:.2f}"
      cpu_s = f"cpu mean={sum(c) / n:.2f} p50={c[i50]:.2f} p95={c[i95]:.2f} max={c[-1]:.2f}"
      cloudlog.warning(f"PLOTPERF {self._name}: n={n} {wall_s} | {cpu_s}")
    except Exception:
      pass  # 계측 로그 실패로 렌더 루프를 중단시키지 않는다
    finally:
      self._wall = []
      self._cpu = []
