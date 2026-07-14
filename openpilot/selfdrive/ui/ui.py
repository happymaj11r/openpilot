#!/usr/bin/env python3
import gc
import os

from openpilot.system.hardware import TICI
from openpilot.common.realtime import set_core_affinity
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.carrot_plot_sched import UI_CORE, plot_sched_gate
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()


def main():
  cores = {UI_CORE, }  # 단일 출처: carrot_plot_sched.UI_CORE (= 7)
  # UI는 상시 SCHED_OTHER — RT 승격이 없어야 core7의 modeld(FIFO54)는 물론
  # dmonitoringmodeld(FIFO5)도 UI를 선점할 수 있다. FIFO53 UI는 core5에서는
  # plannerd/radard(FIFO51)를 굶겨 softDisable을(route 00000426: core5 포화로
  # longitudinalPlan 16~18Hz), core7에서는 DM 활성 구성의 dmonitoringmodeld를
  # 굶겨 운전자 감시 지연을 만들 수 있다. GC만 끈다 (GC pause가 프레임 히치를
  # 만들지 않게 — 기존 config_realtime_process가 하던 것과 동일).
  gc.disable()
  # TICI offroad power-save는 big core4~7을 offline한다 — UI는 always_run이라
  # 항상 online인 core0에서 부트스트랩하고, onroad에서 core7이 online되면
  # 아래 render loop가 best-effort로 re-affine한다.
  set_core_affinity([0])

  gui_app.init_window("UI")
  if BIG_UI:
    MainLayout()
  else:
    MiciMainLayout()

  for should_render in gui_app.render():
    ui_state.update()
    # DebugPlot 활성 시 UI가 실제로 비RT임을 검증하는 fail-closed 게이트
    # (route 416 해제 사고 기원 — UI는 상시 SCHED_OTHER, FIFO 복구 없음)
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
