"""carrot 전용: DebugPlot 활성 시 UI 메인 스레드를 planner 아래로 강등하는 안전 경계.

DebugPlot은 프레임당 수천 draw 콜로 UI를 상시 실행 상태로 만드는데, UI가
SCHED_FIFO 53으로 core5를 점유하면 같은 코어의 plannerd/radard(FIFO 51)가 굶어
longitudinalPlan/radarState 발행이 끊기고 commIssueAvgFreq → softDisable이
발생한다 (2026-07-11 route 416 실주행 해제 사고 — ScreenRecord ffmpeg 시작보다
1초 먼저 disengage가 났고, ffmpeg 종료 후에도 122초간 지속됨).

강등이 실제로 적용된 것을 검증한 경우에만 plot을 허용한다 (fail-closed).
plot을 그리는 쪽은 파라미터를 직접 읽지 말고 effective_mode를 읽어야 한다.
core5 affinity는 유지한다 — SCHED_OTHER면 같은 코어의 FIFO 51이 항상 선점한다.
"""
import os
import sys
import time

from openpilot.common.params import Params
from openpilot.common.realtime import Priority, drop_realtime
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.carrot_params_watch import ParamsRefreshGate
from openpilot.system.hardware import PC

UI_CORE = 5
PLOT_MODE_MIN, PLOT_MODE_MAX = 1, 8  # 공식 지원 plot 모드 범위


class PlotSchedGate:
  def __init__(self):
    self._params = Params()
    self._refresh_gate = ParamsRefreshGate()
    self._raw_mode = 0
    self._effective_mode = 0
    self._demoted = False
    self._failed_mode: int | None = None  # 강등 실패를 래치한 모드 값 (값이 바뀌면 재시도)

  @property
  def effective_mode(self) -> int:
    """강등이 검증된 경우에만 ShowPlotMode 값, 아니면 0."""
    return self._effective_mode

  @staticmethod
  def _demote() -> bool:
    """UI 메인 스레드를 SCHED_OTHER로 내리고 실제 적용을 검증한다 (affinity는 그대로)."""
    if sys.platform != "linux" or PC:
      return True  # RT 스케줄링이 없는 환경은 강등 자체가 불필요
    try:
      drop_realtime()
      return os.sched_getscheduler(0) == os.SCHED_OTHER
    except OSError:
      return False

  @staticmethod
  def _restore() -> bool:
    """FIFO 53 + core5 복구. 어느 단계든 실패하면 SCHED_OTHER로 롤백해
    '복구 실패 = 확실히 비RT 상태'를 보장한다 (부분 성공으로 FIFO만 남는 것 금지)."""
    if sys.platform != "linux" or PC:
      return True
    try:
      # 아직 SCHED_OTHER인 상태에서 affinity 먼저 — FIFO 승격은 마지막 단계로 미뤄서
      # affinity 실패가 'FIFO인데 False 반환' 같은 부분 성공을 만들지 못하게 한다
      os.sched_setaffinity(0, {UI_CORE})
      os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(Priority.CTRL_HIGH))
      ok = (os.sched_getscheduler(0) == os.SCHED_FIFO
            and os.sched_getparam(0).sched_priority == Priority.CTRL_HIGH
            and os.sched_getaffinity(0) == {UI_CORE})
      if not ok:
        drop_realtime()
      return ok
    except OSError:
      try:
        drop_realtime()  # 부분 성공했을 가능성까지 롤백
      except OSError:
        cloudlog.exception("PLOTSCHED: failed to roll back UI to SCHED_OTHER")
      return False

  @staticmethod
  def _current_policy() -> int:
    if sys.platform != "linux":
      return -1
    try:
      return os.sched_getscheduler(0)
    except OSError:
      return -1

  def _read_mode(self) -> int:
    """ShowPlotMode를 fail-closed로 읽는다. get_int()는 C++ std::stoi가 except+ 없이
    선언되어 있어 손상된 값(비정수/overflow)이 UI 프로세스 종료로 전파될 수 있으므로
    금지 — get()은 파이썬 변환 단계에서 ValueError를 잡아 default로 처리한다."""
    try:
      value = self._params.get("ShowPlotMode", return_default=True)
    except Exception:
      cloudlog.exception("PLOTSCHED: failed to read ShowPlotMode")
      return 0
    if type(value) is not int:  # bool은 int subclass이므로 정확한 int만 허용
      return 0
    # 범위 밖(음수/9+)은 0으로 정규화 — 1→-1 같은 전이에서도 FIFO 복구가 수행된다
    return value if PLOT_MODE_MIN <= value <= PLOT_MODE_MAX else 0

  def update(self) -> int:
    """ui.py 렌더 루프에서 매 프레임 호출. 파라미터는 실제로 바뀐 경우에만 재읽기,
    스케줄러 syscall은 plot 활성/비활성 전이 시에만 1회 수행한다."""
    if self._refresh_gate.should_refresh(time.monotonic()):
      self._raw_mode = self._read_mode()

    mode = self._raw_mode
    if self._failed_mode is not None and mode != self._failed_mode:
      self._failed_mode = None  # 사용자가 모드를 바꾸면 강등 재시도 허용

    if mode > 0 and not self._demoted and self._failed_mode is None:
      if self._demote():
        self._demoted = True
        cloudlog.warning("PLOTSCHED: DebugPlot active, UI demoted to SCHED_OTHER/core5")
      else:
        # 강등 안 된 FIFO 53 UI로 plot을 그리면 planner가 굶으므로 plot을 켜지 않는다
        self._failed_mode = mode
        cloudlog.error("PLOTSCHED: UI demotion FAILED, DebugPlot disabled (fail-closed)")
    elif mode == 0 and self._demoted:
      if self._restore():
        cloudlog.warning("PLOTSCHED: DebugPlot inactive, UI restored to SCHED_FIFO 53")
      else:
        # 복구 실패는 안전 문제가 아니다(planner가 계속 UI를 선점 가능) — UI를 죽이지
        # 않고 SCHED_OTHER로 유지(_restore가 롤백을 보장). 다음 전이에서 자연 재시도.
        cloudlog.error(f"PLOTSCHED: UI RT restore FAILED (policy now {self._current_policy()}), staying SCHED_OTHER")
      self._demoted = False

    self._effective_mode = mode if (mode > 0 and self._demoted) else 0
    return self._effective_mode


plot_sched_gate = PlotSchedGate()
