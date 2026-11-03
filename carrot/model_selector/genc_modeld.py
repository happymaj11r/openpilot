#!/usr/bin/env python3
"""Gen C driving-model engine for the carrot model selector.

carrot-ms `openpilot/selfdrive/modeld/modeld.py::main()` with the Gen A ModelState
swapped for `genc.runtime.ModelState` (comma master state-queue supercombo running on
the vendored tinygrad under genc/).

Kept from carrot modeld: DesireHelper/indicator semantics, MapInput, dynamic lateral
smoothing, desiredVelocity/vEgoStopping, RuntimeDiagnostics, CameraGapTrace, SCHED
policy. Dropped: eGPU (USB GPU) load/fallback/recovery, Jetlink wrapper, signal
shadow, precompiled big models — the Gen C engine runs on the integrated GPU only,
and a load/runtime failure simply restarts the process (modeld_runner's crash-loop
breaker quarantines the model after repeated early crashes).

`get_lat_smooth_seconds_dynamic` / `get_action_from_model` are copies of the carrot
modeld.py functions: that module imports the system tinygrad at import time, so it
cannot be imported here. check_contracts.py (modeld-mirror) flags upstream edits to
modeld.py so they can be ported here as well.
"""
import os
import time

import numpy as np

# Must precede every other import: activates the vendored tinygrad for this process.
from .genc.runtime import ModelState

import openpilot.cereal.messaging as messaging
from openpilot.cereal import car, log
from openpilot.cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.runtime_diagnostics import RuntimeDiagnostics
from openpilot.common.params import Params
from openpilot.common.stopping_params import get_stopping_speed
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.desire_lib.boundary_observer import BoundaryObserver
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value, get_curvature_from_plan
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_driving_model_data, fill_pose_msg, PublishState
from openpilot.selfdrive.modeld.camera_gap import CameraGapTrace
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.helpers import select_vision_streams
from openpilot.selfdrive.navigation.map_client import MapInput

from .config import MODELS_DIR as CUSTOM_MODELS_DIR

PROCESS_NAME = "carrot.model_selector.genc_modeld"
INFERENCE_BACKEND = "genc"
SIMULATION = os.getenv('SIMULATION') == '1'

LAT_SMOOTH_SECONDS = 0.0
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3


def get_lat_smooth_seconds_dynamic(model_output: dict[str, np.ndarray],
                                   base_lat_smooth_seconds: float) -> tuple[float, float, float]:
  if base_lat_smooth_seconds <= 0.0:
    return 0.0, 0.0, 0.0

  try:
    y_std_1s = float(model_output['plan_stds'][0, 10, Plan.POSITION, 1])
  except Exception:
    y_std_1s = 0.0

  extra_smooth_seconds = float(np.interp(y_std_1s, [0.15, 0.25], [0.0, base_lat_smooth_seconds * 2]))

  dynamic_lat_smooth_seconds = float(np.clip(base_lat_smooth_seconds + extra_smooth_seconds, 0.0, 0.60))

  return dynamic_lat_smooth_seconds, y_std_1s, extra_smooth_seconds


def get_action_from_model(model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float, lat_smooth_seconds: float, vEgoStopping: float) -> log.ModelDataV2.Action:
  plan = model_output['plan'][0]
  desired_accel, should_stop, _, desired_velocity_now = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                                   plan[:,Plan.ACCELERATION][:,0],
                                                   ModelConstants.T_IDXS,
                                                   action_t=long_action_t,
                                                   vEgoStopping=vEgoStopping)
  if 'action' not in model_output:
    desired_curvature = get_curvature_from_plan(plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
                                                plan[:,Plan.ORIENTATION_RATE][:,2],
                                                ModelConstants.T_IDXS,
                                                v_ego,
                                                lat_action_t)
  else:
    desired_accel = model_output['action'][0,1]
    desired_curvature = model_output['action'][0,0] / (max(2.5, v_ego))**2
  desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)
  desired_velocity_now = smooth_value(desired_velocity_now, prev_action.desiredVelocity, LONG_SMOOTH_SECONDS)

  if v_ego > MIN_LAT_CONTROL_SPEED:
    desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, lat_smooth_seconds)
  else:
    desired_curvature = prev_action.desiredCurvature

  return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                desiredAcceleration=float(desired_accel),
                                shouldStop=bool(should_stop),
                                desiredVelocity=float(desired_velocity_now))


def main(demo=False):
  cloudlog.warning("genc modeld init")

  params = Params()
  use_wide_camera = bool(params.get("UseWideCamera", return_default=True))

  config_realtime_process(7, 54)

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    vipc_client_main_stream, use_extra_client = select_vision_streams(
      available_streams,
      VisionStreamType.VISION_STREAM_ROAD,
      VisionStreamType.VISION_STREAM_WIDE_ROAD,
      use_wide_camera,
    )
    if vipc_client_main_stream is not None:
      break
    time.sleep(.1)

  main_wide_camera = vipc_client_main_stream == VisionStreamType.VISION_STREAM_WIDE_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  cloudlog.warning(f"vision stream set up, use_wide_camera: {use_wide_camera}, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  st = time.monotonic()
  cloudlog.warning(f"loading Gen C model from {CUSTOM_MODELS_DIR}")
  model = ModelState(CUSTOM_MODELS_DIR, vipc_client_main.width, vipc_client_main.height)
  model.warmup()
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s (device {model.device_spec}, "
                   + f"state queues {list(model.state_pairs)}), genc modeld starting")

  # messaging
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "selfdriveState",
                  "liveDelay", "carrotMan", "radarState", "navInstructionCarrot", "navRoute"])

  publish_state = PublishState()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False

  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("genc modeld got CarParams: %s", CP.brand)

  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  from opendbc.car.tesla.values import TeslaFlags
  DH = DesireHelper(tesla_auto_signals=CP.brand == "tesla" and bool(CP.flags & TeslaFlags.AUTO_TURN_SIGNALS),
                    boundary_indicator_supported=CP.carFingerprint == 'HYUNDAI_IONIQ_5_PE')
  boundary_observer = BoundaryObserver(cloudlog.event) if DH.boundary_indicator_supported else None
  nav_input = MapInput()

  frame = 0
  custom_lat_delay = 0.0
  lat_smooth_seconds = LAT_SMOOTH_SECONDS
  vEgoStopping = get_stopping_speed(params)
  camera_yaw_trim_deg = params.get_float("CameraYawTrimDeg") * 0.01
  lat_delay_dynamic = lat_smooth_seconds
  diagnostics = RuntimeDiagnostics('modeld', cloudlog.event)
  camera_gap = CameraGapTrace(cloudlog.event)
  while True:
    loop_start, cpu_start = time.monotonic(), time.thread_time()
    frame += 1
    if frame % 100 == 0:
      custom_lat_delay = params.get_float("SteerActuatorDelay") * 0.01
      lat_smooth_seconds = params.get_float("LatSmoothSec") * 0.01
      long_delay = params.get_float("LongActuatorDelay")*0.01
      vEgoStopping = get_stopping_speed(params)
      camera_yaw_trim_deg = params.get_float("CameraYawTrimDeg") * 0.01

    frames = camera_gap.receive(vipc_client_main, vipc_client_extra if use_extra_client else None)
    if frames is None:
      cloudlog.debug("camera pair unavailable or out of sync")
      continue
    buf_main, meta_main, buf_extra, meta_extra = frames

    camera_ready = time.monotonic()
    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)

      calib_done = sm["liveCalibration"].calStatus == log.LiveCalibrationData.Status.calibrated
      applied_yaw_trim_deg = camera_yaw_trim_deg if calib_done else 0.0

      if applied_yaw_trim_deg != 0.0:
        device_from_calib_euler[2] -= np.radians(applied_yaw_trim_deg)

      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      has_wide_camera = use_extra_client or main_wide_camera
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if has_wide_camera else dc.fcam.intrinsics, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    if vipc_dropped_frames > 0:
      # Gen C keeps its frame/feature history inside the ONNX: there is no separate
      # queue to catch up, so the frame is run and published like comma master does.
      cloudlog.error(f"camera dropped {vipc_dropped_frames} frames")

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
    frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
    action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
    lat_action_t = lat_delay_dynamic + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay
    inputs: dict[str, np.ndarray] = {
      'desire_pulse': vec_desire,
      'traffic_convention': traffic_convention,
      'action_t': np.array([lat_action_t, long_action_t], dtype=np.float32),
    }

    inputs.update(nav_input.update(model, sm, v_ego))

    inference_started = time.monotonic()
    mt1 = time.perf_counter()
    camera_age_at_run_ms = (inference_started - meta_main.timestamp_eof * 1e-9) * 1000
    inference_cpu_start = time.thread_time()
    model_output = model.run(bufs, transforms, inputs)
    mt2 = time.perf_counter()
    inference_cpu_ms = (time.thread_time() - inference_cpu_start) * 1000
    model_execution_time = mt2 - mt1

    modelv2_send = messaging.new_message('modelV2')
    drivingdata_send = messaging.new_message('drivingModelData')
    posenet_send = messaging.new_message('cameraOdometry')

    lat_smooth_seconds_dynamic, y_std_1s, lat_smooth_extra = get_lat_smooth_seconds_dynamic(
        model_output,
        lat_smooth_seconds,
      )
    if custom_lat_delay > 0.0:
      lat_delay_dynamic = custom_lat_delay + lat_smooth_seconds_dynamic
    else:
      lat_delay_dynamic = sm["liveDelay"].lateralDelay + lat_smooth_seconds_dynamic

    action = get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego, lat_smooth_seconds_dynamic, vEgoStopping)
    prev_action = action
    fill_model_msg(modelv2_send, model_output, action,
                   publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                   frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

    desire_state = modelv2_send.modelV2.meta.desireState
    l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
    r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
    lane_change_prob = l_lane_change_prob + r_lane_change_prob
    indicator_now = time.monotonic()
    indicator_valid = modelv2_send.valid and all(
      sm.valid[k] and sm.alive[k] and 0 <= indicator_now-sm.recv_time[k] < .35
      for k in ('carState', 'carControl'))
    DH.update(sm['carState'], modelv2_send.modelV2, sm['carControl'].latActive, lane_change_prob, sm['carrotMan'], sm['radarState'],
              indicator_valid=indicator_valid, indicator_curvature=sm['carControl'].actuators.curvature)
    modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
    modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
    modelv2_send.modelV2.meta.desireLog = DH.desireLog #carrot
    modelv2_send.modelV2.meta.laneChangeBlinkerHold = DH.blinker_hold_direction
    modelv2_send.modelV2.meta.leverTurn = DH.lever_turn
    modelv2_send.modelV2.meta.autoTurnSignal = DH.auto_turn_signal
    modelv2_send.modelV2.meta.autoTurnSignalLaneChange = DH.auto_turn_signal_lane_change
    modelv2_send.modelV2.meta.autoTurnSignalDirection = DH.auto_signal_direction
    modelv2_send.modelV2.meta.autoTurnSignalRequestId = DH.auto_signal_request_id
    modelv2_send.modelV2.meta.manualLaneIndicator = DH.boundary_indicator.request_direction
    modelv2_send.modelV2.meta.manualLaneIndicatorMonoTime = int(DH.boundary_indicator.request_time * 1e9)
    observation = DH.model_indicator.active
    modelv2_send.modelV2.meta.observedIndicator = observation.direction if observation else 0
    modelv2_send.modelV2.meta.observedIndicatorMonoTime = int(DH.model_indicator.request_time * 1e9)
    modelv2_send.modelV2.meta.observedIndicatorSource = DH.model_indicator.reason
    modelv2_send.modelV2.meta.indicatorIsolationMask = DH.boundary_indicator.blocked
    modelv2_send.modelV2.meta.indicatorJournalMonoTime = int(DH.indicator_journal_time * 1e9)
    preparation = DH.model_indicator.candidate
    if preparation and not (DH.model_path_indicator_enabled if preparation[1] == 'model_path' else DH.model_desire_indicator_enabled):
      preparation = None
    if DH.boundary_indicator_enabled and DH.boundary_indicator.episode:
      preparation = (DH.boundary_indicator.episode['direction'], 'manual')
    modelv2_send.modelV2.meta.indicatorPrepareDirection = preparation[0] if preparation and not observation else 0
    modelv2_send.modelV2.meta.indicatorPrepareSource = preparation[1] if preparation and not observation else ''
    if boundary_observer is not None:
      # Read-only experiment: independent of navigation freshness, no request/desire assignment.
      boundary_observer.observe(indicator_now, sm['carState'], modelv2_send.modelV2,
        enabled=DH.boundary_indicator_enabled,
        valid=modelv2_send.valid and sm.valid['carState'] and sm.alive['carState'] and
              0 <= indicator_now-sm.recv_time['carState'] < .35,
        busy=DH.lane_change_state in (log.LaneChangeState.laneChangeStarting, log.LaneChangeState.laneChangeFinishing),
        legacy_direction=DH.boundary_indicator.request_direction)

    drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
    drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

    modelv2_send.modelV2.meta.laneWidthLeft = float(DH.left.lane_width)
    modelv2_send.modelV2.meta.laneWidthRight = float(DH.right.lane_width)
    modelv2_send.modelV2.meta.distanceToRoadEdgeLeft = float(DH.left.dist_to_edge)
    modelv2_send.modelV2.meta.distanceToRoadEdgeRight = float(DH.right.dist_to_edge)
    modelv2_send.modelV2.meta.desire = DH.desire
    modelv2_send.modelV2.meta.laneChangeProb = DH.lane_change_ll_prob
    # Retain the wire field for older log/replay readers; this limiter is retired.
    modelv2_send.modelV2.meta.modelTurnSpeed = 200.0
    modelv2_send.modelV2.meta.laneChangeAvailableLeft = DH.lane_change_available_left
    modelv2_send.modelV2.meta.laneChangeAvailableRight = DH.lane_change_available_right
    mt3 = time.perf_counter()
    drivingdata_send.drivingModelData.modelExecutionTime = mt3 - mt1

    fill_driving_model_data(drivingdata_send, modelv2_send)
    # Slow PC inference can exceed locationd's rewind window. The simulator
    # pose represents current simulated motion, so timestamp it at publish.
    pose_timestamp_eof = time.monotonic_ns() if SIMULATION else meta_main.timestamp_eof
    fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, pose_timestamp_eof, live_calib_seen)
    pm.send('modelV2', modelv2_send)
    pm.send('drivingModelData', drivingdata_send)
    pm.send('cameraOdometry', posenet_send)
    last_vipc_frame_id = meta_main.frame_id

    camera_gap.record(meta_main.frame_id, vipc_dropped_frames, INFERENCE_BACKEND,
                      camera_wait=(camera_ready - loop_start) * 1000,
                      before_inference=(inference_started - camera_ready) * 1000,
                      inference=model_execution_time * 1000,
                      postprocess=(time.perf_counter() - mt2) * 1000,
                      loop=(time.monotonic() - loop_start) * 1000)
    diagnostics.record(
      context={'backend': INFERENCE_BACKEND, 'usbgpu': False, 'frame_id': meta_main.frame_id},
      camera_wait_ms=(camera_ready - loop_start) * 1000,
      camera_age_at_run_ms=camera_age_at_run_ms,
      inference_ms=model_execution_time * 1000, inference_thread_cpu_ms=inference_cpu_ms,
      postprocess_ms=(time.perf_counter() - mt2) * 1000,
      loop_ms=(time.monotonic() - loop_start) * 1000, thread_cpu_ms=(time.thread_time() - cpu_start) * 1000,
      dropped_frames=vipc_dropped_frames, published=1,
    )


if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
