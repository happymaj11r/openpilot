"""carrot 전용: DebugPlot 활성 시 UI가 실제로 비RT(SCHED_OTHER)임을 검증하는 안전 경계.

UI는 상시 SCHED_OTHER로 운용한다 — 비RT UI는 어떤 RT도 선점하지 못하므로
core5의 plannerd/radard(FIFO51)와 코어를 공유해도 기아 구조가 성립하지 않는다.
FIFO53 UI의 사고 이력이 이 설계의 근거다: core5에서는 FIFO51을 선점해
longitudinalPlan/radarState 발행이 끊기고 commIssueAvgFreq → softDisable
(2026-07-11 route 416 실주행 해제 사고, 2026-07-14 route 00000426 core5 포화
재발), core7로 옮기는 안은 DM 활성 구성의 dmonitoringmodeld(FIFO5)를 굶겨
운전자 감시가 지연될 수 있어 기각됐다 (교차 리뷰). core7은 modeld(FIFO54)+
dmonitoringmodeld(FIFO5) 전용으로 비워 둔다. 그래서 이 모듈에는 FIFO 승격/
복구 경로 자체가 없다 — 재도입 금지.

DebugPlot은 프레임당 수천 draw 콜로 UI를 상시 실행 상태로 만들므로, plot은
UI가 비RT임을 검증(어긋나 있으면 강등)한 경우에만 허용한다 (fail-closed).
plot을 그리는 쪽은 파라미터를 직접 읽지 말고 effective_mode를 읽어야 한다.
affinity는 건드리지 않는다 — core 배치는 ui.py의 부트스트랩/re-affine 소관.
"""
import os
import sys
import time

from openpilot.common.params import Params
from openpilot.common.realtime import drop_realtime
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.carrot_params_watch import ParamsRefreshGate
from openpilot.system.hardware import PC

# UI 목표 코어 (ui.py의 core0 부트스트랩 후 re-affine 대상) — 단일 출처.
# 비RT UI는 FIFO51을 선점하지 못하므로 core5 공유가 안전하고, core7은
# modeld(FIFO54)+dmonitoringmodeld(FIFO5) 전용으로 남긴다 (재배치 금지)
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
    """UI 메인 스레드가 비RT(SCHED_OTHER)임을 보장하고 readback으로 검증한다.
    상시 SCHED_OTHER 설계에서 정상이라면 이미 OTHER라 no-op 검증만 통과한다 —
    어떤 경로로든 RT로 어긋나 있으면 여기서 내린다 (affinity는 건드리지 않는다)."""
    if sys.platform != "linux" or PC:
      return True  # RT 스케줄링이 없는 환경은 강등 자체가 불필요
    try:
      drop_realtime()
      return os.sched_getscheduler(0) == os.SCHED_OTHER
    except OSError:
      return False

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
    스케줄러 syscall(비RT 검증/강등)은 plot 활성 전이 시에만 1회 수행한다 —
    비활성 전이는 syscall 없음 (복구할 RT 상태가 없다)."""
    if self._refresh_gate.should_refresh(time.monotonic()):
      self._raw_mode = self._read_mode()

    mode = self._raw_mode
    if self._failed_mode is not None and mode != self._failed_mode:
      self._failed_mode = None  # 사용자가 모드를 바꾸면 강등 재시도 허용

    if mode > 0 and not self._demoted and self._failed_mode is None:
      if self._demote():
        self._demoted = True
        cloudlog.warning("PLOTSCHED: DebugPlot active, UI non-RT verified (SCHED_OTHER)")
      else:
        # UI가 비RT임을 검증하지 못하면 plot을 켜지 않는다 (fail-closed) —
        # RT UI로 plot을 그리면 같은 코어의 plannerd/radard(FIFO51)가 굶는다 (route 416)
        self._failed_mode = mode
        cloudlog.error("PLOTSCHED: UI demotion FAILED, DebugPlot disabled (fail-closed)")
    elif mode == 0 and self._demoted:
      # 상시 SCHED_OTHER 설계 — 복구할 RT 상태가 없다 (FIFO 승격 재도입 금지:
      # FIFO53 UI는 core5의 plannerd/radard FIFO51을 굶긴다 — route 416/
      # 00000426). 다음 활성화가 재검증하도록 래치만 푼다
      self._demoted = False
      cloudlog.warning("PLOTSCHED: DebugPlot inactive")

    self._effective_mode = mode if (mode > 0 and self._demoted) else 0
    return self._effective_mode


plot_sched_gate = PlotSchedGate()
