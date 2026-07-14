#!/usr/bin/env python3
import os

from openpilot.system.hardware import TICI
from openpilot.common.realtime import Priority, config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.carrot_plot_sched import plot_sched_gate
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()


def main():
  cores = {7, }
  # UI는 planner/radar(FIFO51/core5)와 코어를 분리한다 — FIFO53 UI가 core5를
  # 공유하면 core5가 포화되어(UI ~54% + radard ~29% + plannerd ~15%)
  # longitudinalPlan이 16~18Hz로 떨어지고 commIssue → softDisable이 반복된다
  # (route 00000426, DebugPlot/ScreenRecord OFF 상태에서 실측). core7에서는
  # modeld(FIFO54)가 UI(FIFO53)보다 높아 포화 시 항상 UI를 선점하므로 모델
  # 추론은 굶지 않는다. DebugPlot 활성 시에는 plot_sched_gate가 UI를
  # SCHED_OTHER로 추가 강등한다.
  config_realtime_process(7, Priority.CTRL_HIGH)

  gui_app.init_window("UI")
  if BIG_UI:
    MainLayout()
  else:
    MiciMainLayout()

  for should_render in gui_app.render():
    ui_state.update()
    # DebugPlot 활성 시 UI를 SCHED_OTHER로 강등하는 안전 경계 (route 416 해제
    # 사고 기원 — 현재는 core7의 modeld 추가 보호 + plot RT 부하 제거 역할)
    plot_sched_gate.update()
    if should_render:
      # reaffine after power save offlines our core
      if TICI and os.sched_getaffinity(0) != cores:
        try:
          set_core_affinity(list(cores))
        except OSError:
          pass


if __name__ == "__main__":
  main()
