"""carrot 전용: wall vs thread-CPU 분리 렌더 계측 (route 418 진단).

SCHED_OTHER로 강등된 UI는 FIFO plannerd/radard에 선점되므로 drawTimeMillis(wall)
증가가 곧 렌더 비용 증가를 뜻하지 않는다. 구간별 wall/cpu를 함께 재서
'선점/블로킹 대기'(wall 증가, cpu 유지)와 '실제 비용'(wall/cpu 동반 증가)을 구분한다.
(wall-cpu 차이에는 스케줄러 선점 외에 GPU 대기 등 모든 블로킹이 포함되므로
procLog와 함께 해석해야 한다.)

계측은 진단 부가 기능이므로 전 경로 no-throw — begin/end/add/set_phase/flush 어느
것도 렌더 루프로 예외를 전파하지 않는다. 매 프레임 로그는 금지: 샘플만 쌓고
윈도가 차면 집계 한 줄(PLOTPERF)을 cloudlog로 남긴 뒤 리셋한다. phase 키가 바뀌면
부분 윈도를 먼저 배출해 단계 경계(plot on/off, 녹화 on/off)가 한 줄에 섞이지 않는다.

측정 구간이 다른 측정 구간 안에 중첩되면(big-UI: PlotRenderer.draw가 uiRender/
drawTimeMillis 안) emit(sort/문자열/cloudlog) 비용이 바깥 지표에 주기적 spike로
찍힌다 — deferred=True로 만들면 윈도 완성/phase 전환 시 스냅샷을 pending으로
옮기기만 하고, 바깥 구간 종료 후 emit_pending()을 호출해 실제 로그를 배출한다.
"""
import time

from openpilot.common.swaglog import cloudlog


class SectionMetrics:
  """한 구간의 wall/cpu(ms) 샘플 카운터. begin() 토큰을 end()에 돌려주면 기록된다."""

  # deferred pending 상한 — emit_pending()이 매 프레임 불리는 정상 경로에서는 1을
  # 넘지 않지만, 호출이 끊긴 비정상 상황에서도 무한 성장하지 않게 오래된 것부터 버린다
  PENDING_MAX = 4

  def __init__(self, name: str, window: int = 200, deferred: bool = False):
    self._name = name
    self._window = max(1, window)
    self._deferred = deferred
    self._pending: list[tuple] = []  # (phase, wall, cpu, t0, t1) 완성 윈도 스냅샷
    self._wall: list[float] = []
    self._cpu: list[float] = []
    self._phase = None
    # 윈도 경계 시각: t0=첫 샘플 시작, t1=마지막 샘플 종료 (emit 시각이 아님 —
    # flush가 늦게 불려도(예: 인코더 close 후) 로그의 시간 경계는 정확해야 한다)
    self._t0: float | None = None
    self._t1: float | None = None

  @staticmethod
  def begin():
    try:
      return time.perf_counter_ns(), time.thread_time_ns(), time.monotonic()
    except Exception:
      return None  # end(None)은 no-op — 계측 실패가 렌더 루프를 못 죽인다

  def end(self, token) -> None:
    if token is None:
      return
    try:
      wall_ms = (time.perf_counter_ns() - token[0]) / 1e6
      cpu_ms = (time.thread_time_ns() - token[1]) / 1e6
      self.add(wall_ms, cpu_ms, t_start=token[2], t_end=time.monotonic())
    except Exception:
      pass

  def add(self, wall_ms, cpu_ms, t_start=None, t_end=None) -> None:
    try:
      # 두 값 변환을 모두 끝낸 뒤에 append — 한쪽만 실패해서 wall/cpu 버퍼 길이가
      # 어긋나는 일이 없게 한다 (paired append)
      wall = max(0.0, float(wall_ms))
      cpu = max(0.0, float(cpu_ms))
      now = time.monotonic()
      if self._t0 is None:
        self._t0 = t_start if t_start is not None else now
      self._t1 = t_end if t_end is not None else now
      self._wall.append(wall)
      self._cpu.append(cpu)
      if len(self._wall) >= self._window:
        self._close_window()
    except Exception:
      pass

  def set_phase(self, key) -> None:
    """단계 키(예: (plot_mode, recording))가 바뀌면 부분 윈도를 먼저 배출해
    서로 다른 단계의 샘플이 한 집계 줄에 섞이지 않게 한다."""
    try:
      if key != self._phase:
        if self._wall:
          self._close_window()
        self._phase = key
    except Exception:
      pass

  def flush(self) -> None:
    """세션 경계에서 부분 윈도(와 pending)를 즉시 배출한다 (예: 녹화 stop/60초 회전).
    로그를 실제로 쓰므로 중첩 측정 구간 안에서는 부르지 말 것."""
    try:
      self.emit_pending()
      if self._wall:
        self._emit()
    except Exception:
      self._wall = []
      self._cpu = []
      self._t0 = None
      self._t1 = None

  def emit_pending(self) -> None:
    """deferred로 쌓인 완성 윈도를 실제 로그로 배출한다 — 반드시 바깥 측정 구간
    종료 후(예: uiRender.end() 뒤)에 호출해야 emit 비용이 어느 지표에도 안 섞인다."""
    while self._pending:
      try:
        phase, wall, cpu, t0, t1 = self._pending.pop(0)
        self._emit_snapshot(phase, wall, cpu, t0, t1)
      except Exception:
        break  # no-throw + 진행 불가 시 무한루프 방지 (남은 항목은 다음 호출에서)

  def _close_window(self) -> None:
    """현재 버퍼를 윈도 하나로 마감한다 — deferred면 스냅샷을 pending으로 옮기기만
    하고(비용 O(1)), 아니면 즉시 emit한다."""
    if not self._deferred:
      self._emit()
      return
    try:
      if len(self._pending) >= self.PENDING_MAX:
        self._pending.pop(0)
      self._pending.append((self._phase, self._wall, self._cpu, self._t0, self._t1))
    except Exception:
      pass
    finally:
      self._wall = []
      self._cpu = []
      self._t0 = None
      self._t1 = None

  def _emit(self) -> None:
    try:
      self._emit_snapshot(self._phase, self._wall, self._cpu, self._t0, self._t1)
    finally:
      self._wall = []
      self._cpu = []
      self._t0 = None
      self._t1 = None

  def _emit_snapshot(self, phase, wall, cpu, t0, t1) -> None:
    try:
      n = len(wall)
      if n == 0:
        return
      w = sorted(wall)
      c = sorted(cpu)
      i50, i95 = n // 2, min(n - 1, int(n * 0.95))
      if isinstance(phase, tuple):
        phase = "/".join(str(x) for x in phase)
      phase_s = f" phase={phase}" if phase is not None else ""
      t0 = t0 if t0 is not None else 0.0
      t1 = t1 if t1 is not None else t0  # 마지막 샘플 종료 시각 (emit 시각 아님)
      wall_s = f"wall mean={sum(w) / n:.2f} p50={w[i50]:.2f} p95={w[i95]:.2f} max={w[-1]:.2f}"
      cpu_s = f"cpu mean={sum(c) / n:.2f} p50={c[i50]:.2f} p95={c[i95]:.2f} max={c[-1]:.2f}"
      cloudlog.warning(f"PLOTPERF {self._name}:{phase_s} n={n} t0={t0:.1f} t1={t1:.1f} {wall_s} | {cpu_s}")
    except Exception:
      pass  # 계측 로그 실패로 렌더 루프를 중단시키지 않는다
