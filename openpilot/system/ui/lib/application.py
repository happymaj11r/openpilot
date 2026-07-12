import atexit
import cffi
import math
import os
import queue
import time
import signal
import sys
import struct
import pyray as rl
import threading
import platform
import subprocess
from contextlib import contextmanager
from collections.abc import Callable
from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple
from importlib.resources import as_file
from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.carrot_render_metrics import SectionMetrics
from openpilot.system.hardware import HARDWARE, PC
from openpilot.system.ui.lib.multilang import multilang
from openpilot.common.realtime import Ratekeeper
import datetime

#_DEFAULT_FPS = int(os.getenv("FPS", {'tizi': 20}.get(HARDWARE.get_device_type(), 60)))
_DEFAULT_FPS = 20 
FPS_LOG_INTERVAL = 5  # Seconds between logging FPS drops
FPS_DROP_THRESHOLD = 0.9  # FPS drop threshold for triggering a warning
FPS_CRITICAL_THRESHOLD = 0.5  # Critical threshold for triggering strict actions
MOUSE_THREAD_RATE = 140  # touch controller runs at 140Hz
MAX_TOUCH_SLOTS = 2
TOUCH_HISTORY_TIMEOUT = 3.0  # Seconds before touch points fade out

TOUCH_EVENT_DEVICE = "/dev/input/by-path/platform-894000.i2c-event"
EVENT_FORMAT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
EV_SYN, EV_KEY, EV_ABS = 0x00, 0x01, 0x03
SYN_REPORT = 0x00
BTN_TOUCH = 0x14a
ABS_MT_SLOT = 0x2f
ABS_MT_TRACKING_ID = 0x39

BIG_UI = os.getenv("BIG", "0") == "1"
ENABLE_VSYNC = os.getenv("ENABLE_VSYNC", "0") == "1"
SHOW_FPS = os.getenv("SHOW_FPS") == "1"
SHOW_TOUCHES = os.getenv("SHOW_TOUCHES") == "1"
STRICT_MODE = os.getenv("STRICT_MODE") == "1"
SCALE = float(os.getenv("SCALE", "1.0"))
GRID_SIZE = int(os.getenv("GRID", "0"))
PROFILE_RENDER = int(os.getenv("PROFILE_RENDER", "0"))
PROFILE_STATS = int(os.getenv("PROFILE_STATS", "100"))  # Number of functions to show in profile output
RECORD = os.getenv("RECORD") == "1"
RECORD_OUTPUT = str(Path(os.getenv("RECORD_OUTPUT", "output")).with_suffix(".mp4"))
RECORD_QUALITY = int(os.getenv("RECORD_QUALITY", "23"))  # Dynamic bitrate quality level (CRF); 0 is lossless (bigger size), max is 51, default is 23 for x264
RECORD_BITRATE = os.getenv("RECORD_BITRATE", "")  # Target bitrate e.g. "2000k" (overrides RECORD_QUALITY when set)
RECORD_SPEED = int(os.getenv("RECORD_SPEED", "1"))  # Speed multiplier
OFFSCREEN = os.getenv("OFFSCREEN") == "1"  # Disable FPS limiting for fast offline rendering

GL_VERSION = """
#version 300 es
precision highp float;
"""
if platform.system() == "Darwin":
  GL_VERSION = """
    #version 330 core
  """

BURN_IN_MODE = "BURN_IN" in os.environ
BURN_IN_VERTEX_SHADER = GL_VERSION + """
in vec3 vertexPosition;
in vec2 vertexTexCoord;
uniform mat4 mvp;
out vec2 fragTexCoord;
void main() {
  fragTexCoord = vertexTexCoord;
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""
BURN_IN_FRAGMENT_SHADER = GL_VERSION + """
in vec2 fragTexCoord;
uniform sampler2D texture0;
out vec4 fragColor;
void main() {
  vec4 sampled = texture(texture0, fragTexCoord);
  float intensity = sampled.b;
  // Map blue intensity to green -> yellow -> red to highlight burn-in risk.
  vec3 start = vec3(0.0, 1.0, 0.0);
  vec3 middle = vec3(1.0, 1.0, 0.0);
  vec3 end = vec3(1.0, 0.0, 0.0);
  vec3 gradient = mix(start, middle, clamp(intensity * 2.0, 0.0, 1.0));
  gradient = mix(gradient, end, clamp((intensity - 0.5) * 2.0, 0.0, 1.0));
  fragColor = vec4(gradient, sampled.a);
}
"""

DEFAULT_TEXT_SIZE = 60
DEFAULT_TEXT_COLOR = rl.Color(255, 255, 255, int(255 * 0.9))

# Qt draws fonts accounting for ascent/descent differently, so compensate to match old styles
# The real scales for the fonts below range from 1.212 to 1.266
FONT_SCALE = 1.242 if BIG_UI else 1.16

ASSETS_DIR = Path(BASEDIR) / "openpilot" / "selfdrive" / "assets"
FONT_DIR = ASSETS_DIR.joinpath("fonts")
FONT_SOURCE_EXTS = (".ttf", ".otf")


class FontWeight(StrEnum):
  NORMAL = "Inter-Regular.fnt" if BIG_UI else "Inter-Medium.fnt"
  MEDIUM = "Inter-Medium.fnt"
  BOLD = "Inter-Bold.fnt"
  SEMI_BOLD = "Inter-SemiBold.fnt"
  PRETENDARD = "Pretendard-SemiBold.fnt"
  UNIFONT = "unifont.fnt"

  # Small UI fonts
  DISPLAY_REGULAR = "Inter-Regular.fnt"
  ROMAN = "Inter-Regular.fnt"
  #DISPLAY = "Inter-Bold.fnt"
  DISPLAY = "KaiGenGothicKR-Bold.fnt"


def font_fallback(font: rl.Font) -> rl.Font:
  """Fall back to unifont for languages that require it."""
  if multilang.requires_unifont():
    return gui_app.font(FontWeight.DISPLAY)
  return font


class MousePos(NamedTuple):
  x: float
  y: float


class MousePosWithTime(NamedTuple):
  x: float
  y: float
  t: float


class MouseEvent(NamedTuple):
  pos: MousePos
  slot: int
  left_pressed: bool
  left_released: bool
  left_down: bool
  t: float


class MouseState:
  def __init__(self, scale: float = 1.0):
    self._scale = scale
    self._events: deque[MouseEvent] = deque(maxlen=MOUSE_THREAD_RATE)  # bound event list
    self._prev_mouse_event: list[MouseEvent | None] = [None] * MAX_TOUCH_SLOTS

    self._slot_active: list[bool] = [False] * MAX_TOUCH_SLOTS
    self._cur_slot = 0
    self._saw_mt = False

    self._rk = Ratekeeper(MOUSE_THREAD_RATE, print_delay_threshold=None)
    self._lock = threading.Lock()
    self._exit_event = threading.Event()
    self._thread = None

  def get_events(self) -> list[MouseEvent]:
    with self._lock:
      events = list(self._events)
      self._events.clear()
    return events

  def start(self):
    self._exit_event.clear()
    if self._thread is None or not self._thread.is_alive():
      self._thread = threading.Thread(target=self._run_thread, daemon=True)
      self._thread.start()

  def stop(self):
    self._exit_event.set()
    if self._thread is not None and self._thread.is_alive():
      self._thread.join()

  def _run_thread(self):
    touch_fd = self._open_touch_device()
    try:
      while not self._exit_event.is_set():
        rl.poll_input_events()
        if touch_fd is not None:
          self._read_touch_events(touch_fd)
        else:
          self._handle_mouse_event()
        self._rk.keep_time()
    finally:
      if touch_fd is not None:
        os.close(touch_fd)

  def _open_touch_device(self) -> int | None:
    if PC:
      return None
    try:
      return os.open(TOUCH_EVENT_DEVICE, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
      cloudlog.warning(f"mouse: using raylib touch, can't open {TOUCH_EVENT_DEVICE}: {e}")
      return None

  def _read_touch_events(self, fd: int) -> None:
    try:
      data = os.read(fd, EVENT_SIZE * 64)
    except (BlockingIOError, OSError):
      return

    for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
      _sec, _usec, etype, code, value = struct.unpack(EVENT_FORMAT, data[off:off + EVENT_SIZE])
      if etype == EV_ABS:
        if code == ABS_MT_SLOT:
          self._cur_slot = value
        elif code == ABS_MT_TRACKING_ID:
          self._saw_mt = True
          if 0 <= self._cur_slot < MAX_TOUCH_SLOTS:
            self._slot_active[self._cur_slot] = value != -1
      elif etype == EV_KEY and code == BTN_TOUCH and not self._saw_mt:
        self._slot_active[0] = value != 0
      elif etype == EV_SYN and code == SYN_REPORT:
        for slot in range(MAX_TOUCH_SLOTS):
          self._append_event(slot, self._slot_active[slot])

  def _handle_mouse_event(self):
    for slot in range(MAX_TOUCH_SLOTS):
      self._append_event(slot, rl.is_mouse_button_down(slot))

  def _append_event(self, slot: int, down: bool) -> None:
    prev = self._prev_mouse_event[slot]
    prev_down = prev.left_down if prev is not None else False
    pressed = down and not prev_down
    released = prev_down and not down

    if down:
      mouse_pos = rl.get_touch_position(slot)
      x = mouse_pos.x / self._scale if self._scale != 1.0 else mouse_pos.x
      y = mouse_pos.y / self._scale if self._scale != 1.0 else mouse_pos.y
      pos = MousePos(x, y)
    else:
      pos = prev.pos if prev is not None else MousePos(0.0, 0.0)

    ev = MouseEvent(pos, slot, pressed, released, down, time.monotonic())

    if prev is None or ev[:-1] != prev[:-1]:
      with self._lock:
        self._events.append(ev)
      self._prev_mouse_event[slot] = ev


class GuiApplication:
  def __init__(self, width: int | None = None, height: int | None = None):
    self._set_log_callback()

    self._fonts: dict[FontWeight, rl.Font] = {}
    self._width = width if width is not None else GuiApplication._default_width()
    self._height = height if height is not None else GuiApplication._default_height()

    if PC and os.getenv("SCALE") is None:
      self._scale = self._calculate_auto_scale()
    else:
      self._scale = SCALE

    # Scale, then ensure dimensions are even
    self._scaled_width = int(self._width * self._scale)
    self._scaled_height = int(self._height * self._scale)
    self._scaled_width += self._scaled_width % 2
    self._scaled_height += self._scaled_height % 2

    self._render_texture: rl.RenderTexture | None = None
    self._burn_in_shader: rl.Shader | None = None
    self._ffmpeg_proc: subprocess.Popen | None = None
    self._ffmpeg_queue: queue.Queue | None = None
    self._ffmpeg_thread: threading.Thread | None = None
    self._ffmpeg_stop_event: threading.Event | None = None
    self._textures: dict[str, rl.Texture] = {}
    self._target_fps: int = _DEFAULT_FPS
    self._last_fps_log_time: float = time.monotonic()
    self._frame = 0
    self._window_close_requested = False
    self._nav_stack: list[object] = []
    self._nav_stack_ticks: list[Callable[[], None]] = []
    self._nav_stack_widgets_to_render = 1 if self.big_ui() else 2

    self._mouse = MouseState(self._scale)
    self._mouse_events: list[MouseEvent] = []
    self._last_mouse_event: MouseEvent = MouseEvent(MousePos(0, 0), 0, False, False, False, 0.0)

    self._should_render = True

    # Debug variables
    self._mouse_history: deque[MousePosWithTime] = deque(maxlen=MOUSE_THREAD_RATE)
    self._show_touches = SHOW_TOUCHES
    self._show_fps = SHOW_FPS
    self._grid_size = GRID_SIZE
    self._profile_render_frames = PROFILE_RENDER
    self._render_profiler = None
    self._render_profile_start_time = None

    self._record_enabled = False
    self._record_dir = Path("/data/media/0/videos")
    self._record_max_sec = 60
    self._record_t0 = 0.0
    self._record_every_n = 3
    self._record_frame_idx = 0
    self._record_fail_count = 0   # 인코더 연속 실패 횟수 (정상 녹화 5초 유지 시 리셋)
    self._record_fail_t = 0.0
    # 라이터 스레드의 비정상 종료 신호 — 인코더 프로세스가 살아 있어도(예: kill 실패,
    # stdin write 예외) 렌더 루프가 실패를 감지할 수 있게 한다. 세션마다 새 객체로 교체.
    self._record_failure_event = threading.Event()
    # 캡처(동기 GPU readback) 구간의 wall/cpu 분리 계측 (route 418 진단).
    # 캡처는 3프레임당 1회뿐이라 window 50 ≈ 13fps 기준 11.6초 — 30초 진단
    # 구간에서도 집계가 나온다 (200이면 46초가 필요해 로그 0줄 가능)
    self._capture_metrics = SectionMetrics("screenCapture", window=50)

  def _new_record_path(self) -> Path:
    self._record_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".mp4"
    return self._record_dir / name
  
  def start_recording(self):
    if self._record_enabled:
      return

    # 연속 실패 시 재시도 폭주 차단: MainLayout이 캐시된 ScreenRecord=True로 매 프레임
    # 재시작을 시도하므로, 짧은 간격 3연속 실패면 60초간 시작을 거부한다.
    # is_recording()이 False로 유지되면 MainLayout이 ScreenRecord 파라미터를 꺼서
    # 사용자에게도 실패가 드러난다.
    if self._record_fail_count >= 3 and (time.monotonic() - self._record_fail_t) < 60.0:
      return

    self._ensure_render_texture_for_recording()
    if not self._render_texture:
      return

    out_path = self._new_record_path()
    if not self._init_ffmpeg(out_path):
      # 강등 런처 없이 ffmpeg를 띄우면 UI의 RT 스케줄링을 상속해 주행 프로세스를
      # 굶기므로(2026-07-10 인게이지 해제 사고) 녹화를 시작하지 않는다 (fail-closed)
      print("[REC] failed to start encoder, recording aborted")
      return

    self._record_enabled = True
    self._record_t0 = time.monotonic()
    print(f"[REC] start -> {out_path}")

  def stop_recording(self):
    if not self._record_enabled:
      return
    self._record_enabled = False
    self.close_ffmpeg()  # application.py에 이미 있는 close_ffmpeg 그대로 사용
    # 세션 경계에서 부분 윈도 배출 — 다음 녹화(60초 회전 포함)와 집계가 섞이지 않게
    self._capture_metrics.flush()
    print("[REC] stop")

  def toggle_recording(self):
    if self._record_enabled:
      self.stop_recording()
    else:
      self.start_recording()

  def is_recording(self) -> bool:
    return self._record_enabled

  def _ensure_render_texture_for_recording(self):
    if self._render_texture is None:
      self._render_texture = rl.load_render_texture(self._width, self._height)
      rl.set_texture_filter(self._render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

  @staticmethod
  def _demote_record_sched() -> bool:
    """호출한 스레드를 일반 스케줄러·비RT 코어로 강등하고 실제 적용 여부를 돌려준다.

    UI는 core5 + SCHED_FIFO 53(RT)이라 스레드가 이를 그대로 상속하는데, 녹화 파이프라인이
    FIFO 53으로 core5를 점유하면 같은 코어의 plannerd/radard(FIFO 51)가 굶어
    radarState/longitudinalPlan 발행이 끊기고 commIssue → soft disable이 발생한다.
    비Linux는 RT 상속 자체가 없으므로 항상 성공으로 본다.
    """
    if sys.platform != "linux":
      return True
    try:
      os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
      os.sched_setaffinity(0, {0, 1, 2, 3})
      return (os.sched_getscheduler(0) == os.SCHED_OTHER
              and not (os.sched_getaffinity(0) & {4, 5}))
    except OSError:
      return False

  @staticmethod
  def _kill_record_encoder(proc: subprocess.Popen | None) -> None:
    if proc is None:
      return
    try:
      proc.kill()
    except OSError as e:
      # kill이 실패해도 라이터 스레드 종료 시 failure_event가 설정되어 렌더 루프가
      # 실패를 감지하므로 (fail-closed), 여기서는 기록만 남긴다
      print(f"[REC] encoder kill FAILED: {e!r}")

  def _record_writer_thread(self):
    # 스레드도 UI의 RT 스케줄링을 상속하므로 시작하자마자 강등 — 실패하면 FIFO 53으로
    # 프레임을 쓰게 되므로 인코더까지 함께 중단한다 (fail-closed)
    proc = self._ffmpeg_proc
    stop_event = self._ffmpeg_stop_event
    failure_event = self._record_failure_event
    try:
      if not self._demote_record_sched():
        print("[REC] writer demotion FAILED, killing encoder to protect driving processes")
        self._kill_record_encoder(proc)
        return

      # ffmpeg 강등 검증 (fail-open 방지): chrt/taskset이 exec 체인에서 적용될 때까지
      # 최대 1초간 재확인한다 — 시스템 부하가 높으면(60초 회전 직후 등) 100ms로는 부족해
      # 위양성 실패가 났었음. 검증 전의 자식은 exec 체인 진행 중이라 CPU를 태우지 않으므로
      # 기다려도 안전하고, 이 스레드는 이미 비RT라 sleep이 렌더 루프를 막지 않는다.
      if proc is not None and sys.platform == "linux":
        try:
          policy, affinity = -1, set()
          deadline = time.monotonic() + 1.0
          while True:
            if proc.poll() is not None:
              return  # 죽은 인코더 — finally가 실패로 표시하고 렌더 루프가 녹화를 멈춘다
            policy = os.sched_getscheduler(proc.pid)
            affinity = os.sched_getaffinity(proc.pid)
            if policy == os.SCHED_OTHER and not (affinity & {4, 5}):
              break
            if time.monotonic() >= deadline:
              print(f"[REC] ffmpeg demotion verification FAILED (policy={policy}, affinity={sorted(affinity)}), killing encoder")
              self._kill_record_encoder(proc)
              return
            time.sleep(0.05)
        except OSError as e:
          # 살아 있는 인코더를 검증할 수 없으면 안전하지 않은 것으로 간주하고 중단한다.
          # 이미 죽은 경우(ESRCH 등)도 finally의 실패 신호로 렌더 루프가 녹화를 멈춘다.
          if proc.poll() is None:
            print(f"[REC] ffmpeg verification ERROR: {e!r}; killing encoder")
            self._kill_record_encoder(proc)
          return

      self._ffmpeg_writer_thread()
    finally:
      # 정지 요청 없이 라이터가 끝났다 = 강등/검증 실패, stdin write 예외 등 비정상 종료.
      # 인코더 프로세스가 살아 있어도(kill 실패 포함) 렌더 루프가 poll() 대신 이 신호로
      # 실패를 감지해 녹화를 멈춘다 (fail-closed). 예상 밖 예외로 죽는 경로까지 포괄한다.
      if stop_event is None or not stop_event.is_set():
        failure_event.set()

  def _init_ffmpeg(self, out_path: Path) -> bool:
    self.close_ffmpeg()
    # 세션마다 새 Event — 이전 세션 라이터의 늦은 실패 신호가 새 세션을 오염시키지 않게 한다
    self._record_failure_event = threading.Event()

    # 내부 튜닝(원하면 여기만 조절)
    record_quality = 23          # CRF
    record_bitrate = ""          # e.g. "2000k" (원하면 사용)
    record_speed = 1             # 배속(출력 fps = 입력 fps * speed)
    preset = "ultrafast"

    fps = self._target_fps if self._target_fps > 0 else _DEFAULT_FPS
    output_fps = fps * record_speed

    ffmpeg_args = [
      "ffmpeg",
      "-v", "warning",
      "-nostats",
      "-f", "rawvideo",
      "-pix_fmt", "rgba",
      "-s", f"{self._width}x{self._height}",
      "-r", str(fps),
      "-i", "pipe:0",
      "-vf", "vflip,format=yuv420p",
      "-r", str(output_fps),
      "-c:v", "libx264",
      "-preset", preset,
      "-crf", str(record_quality),
      # 입력이 ~6.7fps(20fps의 1/3)뿐이라 스레드 2개로 충분 — 무제한이면 x264가
      # core0~3을 포화시켜 soundd 등 일반 프로세스가 밀린다 (녹화 중 lagging 관측)
      "-threads", "2",
    ]

    if record_bitrate:
      ffmpeg_args += ["-b:v", record_bitrate, "-maxrate", record_bitrate, "-bufsize", record_bitrate]

    ffmpeg_args += [
      "-y",
      "-f", "mp4",
      str(out_path),
    ]

    if sys.platform == "linux":
      # 자식 프로세스가 UI의 SCHED_FIFO 53/core5를 상속하지 않도록 검증된 런처로 강등해 실행.
      # preexec_fn은 멀티스레드 프로세스에서 fork 후 데드락 위험이 있어 쓰지 않는다.
      # nice 10: 인코딩은 최하 우선 — 같은 코어의 soundd 등 일반 프로세스에 양보
      ffmpeg_args = ["chrt", "--other", "0", "taskset", "--cpu-list", "0-3", "nice", "-n", "10", *ffmpeg_args]

    try:
      self._ffmpeg_proc = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE)
    except OSError as e:
      # chrt/taskset 부재 등 — RT 상속 상태로 녹화하느니 시작하지 않는다
      print(f"[REC] failed to launch encoder: {e!r}")
      self._ffmpeg_proc = None
      return False
    self._ffmpeg_queue = queue.Queue(maxsize=8) # 60 -> 8, 메모리 사용량 줄이기 위해 버퍼 크기 감소
    self._ffmpeg_stop_event = threading.Event()
    self._ffmpeg_thread = threading.Thread(target=self._record_writer_thread, daemon=True)
    self._ffmpeg_thread.start()
    return True

  def close_ffmpeg(self):
    if self._ffmpeg_thread is not None:
      self._ffmpeg_stop_event.set()
      try:
        self._ffmpeg_queue.put(None, timeout=1.0)
      except Exception:
        pass
      self._ffmpeg_thread.join(timeout=30)

    if self._ffmpeg_proc is not None:
      try:
        if self._ffmpeg_proc.stdin:
          try:
            self._ffmpeg_proc.stdin.flush()
          except Exception:
            pass
          self._ffmpeg_proc.stdin.close()
        try:
          self._ffmpeg_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
          self._ffmpeg_proc.terminate()
          self._ffmpeg_proc.wait()
      except Exception:
        try:
          self._ffmpeg_proc.kill()
        except Exception:
          pass

    self._ffmpeg_proc = None
    self._ffmpeg_queue = None
    self._ffmpeg_thread = None
    self._ffmpeg_stop_event = None  

  @property
  def frame(self):
    return self._frame

  def set_show_touches(self, show: bool):
    self._show_touches = show

  def set_show_fps(self, show: bool):
    self._show_fps = show

  @property
  def show_touches(self) -> bool:
    return self._show_touches

  @property
  def target_fps(self):
    return self._target_fps

  def request_close(self):
    self._window_close_requested = True

  def init_window(self, title: str, fps: int = _DEFAULT_FPS):
    with self._startup_profile_context():
      def _close(sig, frame):
        self.close()
        sys.exit(0)
      signal.signal(signal.SIGINT, _close)
      atexit.register(self.close)

      flags = rl.ConfigFlags.FLAG_MSAA_4X_HINT
      if ENABLE_VSYNC:
        flags |= rl.ConfigFlags.FLAG_VSYNC_HINT
      rl.set_config_flags(flags)

      rl.init_window(self._scaled_width, self._scaled_height, title)

      needs_render_texture = self._scale != 1.0 or BURN_IN_MODE or RECORD
      if self._scale != 1.0:
        rl.set_mouse_scale(1 / self._scale, 1 / self._scale)
      if needs_render_texture:
        self._render_texture = rl.load_render_texture(self._scaled_width, self._scaled_height)
        rl.set_texture_filter(self._render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

      if RECORD:
        output_fps = fps * RECORD_SPEED
        ffmpeg_args = [
          'ffmpeg',
          '-v', 'warning',          # Reduce ffmpeg log spam
          '-nostats',               # Suppress encoding progress
          '-f', 'rawvideo',         # Input format
          '-pix_fmt', 'rgba',       # Input pixel format
          '-s', f'{self._scaled_width}x{self._scaled_height}',  # Input resolution
          '-r', str(fps),           # Input frame rate
          '-i', 'pipe:0',           # Input from stdin
          '-vf', 'vflip,format=yuv420p',  # Flip vertically and convert to yuv420p
          '-r', str(output_fps),    # Output frame rate (for speed multiplier)
          '-c:v', 'libx264',
          '-preset', 'veryfast',
          '-crf', str(RECORD_QUALITY)
        ]
        if RECORD_BITRATE:
          # NOTE: custom bitrate overrides crf setting
          ffmpeg_args += ['-b:v', RECORD_BITRATE, '-maxrate', RECORD_BITRATE, '-bufsize', RECORD_BITRATE]
        ffmpeg_args += [
          '-y',                     # Overwrite existing file
          '-f', 'mp4',              # Output format
          RECORD_OUTPUT,            # Output file path
        ]
        self._ffmpeg_proc = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE)
        self._ffmpeg_queue = queue.Queue(maxsize=60)  # Buffer up to 60 frames
        self._ffmpeg_stop_event = threading.Event()
        self._ffmpeg_thread = threading.Thread(target=self._ffmpeg_writer_thread, daemon=True)
        self._ffmpeg_thread.start()

      # OFFSCREEN disables FPS limiting for fast offline rendering (e.g. clips)
      rl.set_target_fps(0 if OFFSCREEN else fps)

      self._target_fps = fps
      self._set_styles()
      self._load_fonts()
      self._patch_text_functions()
      self._patch_scissor_mode()
      if BURN_IN_MODE and self._burn_in_shader is None:
        self._burn_in_shader = rl.load_shader_from_memory(BURN_IN_VERTEX_SHADER, BURN_IN_FRAGMENT_SHADER)

      if not PC:
        self._mouse.start()

  @contextmanager
  def _startup_profile_context(self):
    if "PROFILE_STARTUP" not in os.environ:
      yield
      return

    import cProfile
    import io
    import pstats

    profiler = cProfile.Profile()
    start_time = time.monotonic()
    profiler.enable()

    # do the init
    yield

    profiler.disable()
    elapsed_ms = (time.monotonic() - start_time) * 1e3

    stats_stream = io.StringIO()
    pstats.Stats(profiler, stream=stats_stream).sort_stats("cumtime").print_stats(25)
    print("\n=== Startup profile ===")
    print(stats_stream.getvalue().rstrip())

    green = "\033[92m"
    reset = "\033[0m"
    print(f"{green}UI window ready in {elapsed_ms:.1f} ms{reset}")
    sys.exit(0)

  def _ffmpeg_writer_thread(self):
    """Background thread that writes frames to ffmpeg."""
    while True:
      try:
        data = self._ffmpeg_queue.get(timeout=1.0)
        if data is None:  # Sentinel to stop
          break
        self._ffmpeg_proc.stdin.write(data)
      except queue.Empty:
        if self._ffmpeg_stop_event.is_set():
          break
        continue
      except Exception:
        break

  def push_widget(self, widget: object):
    if widget in self._nav_stack:
      cloudlog.warning("Widget already in stack, cannot push again!")
      return

    # disable previous widget to prevent input processing
    if len(self._nav_stack) > 0:
      prev_widget = self._nav_stack[-1]
      # TODO: change these to touch_valid
      prev_widget.set_enabled(False)

    self._nav_stack.append(widget)
    widget.show_event()
    widget.set_enabled(True)

  def pop_widget(self, idx: int | None = None):
    # Pops widget instantly without animation
    if len(self._nav_stack) < 2:
      cloudlog.warning("At least one widget should remain on the stack, ignoring pop!")
      return

    idx_to_pop = len(self._nav_stack) - 1 if idx is None else idx
    if idx_to_pop <= 0 or idx_to_pop >= len(self._nav_stack):
      cloudlog.warning(f"Invalid index {idx_to_pop} to pop, ignoring!")
      return

    # only re-enable previous widget if popping top widget
    if idx_to_pop == len(self._nav_stack) - 1:
      prev_widget = self._nav_stack[idx_to_pop - 1]
      prev_widget.set_enabled(True)

    widget = self._nav_stack.pop(idx_to_pop)
    widget.hide_event()

  def pop_widgets_to(self, widget: object, callback: Callable[[], None] | None = None, instant: bool = False):
    # Pops middle widgets instantly without animation then dismisses top, animated out if NavWidget
    if widget not in self._nav_stack:
      cloudlog.warning("Widget not in stack, cannot pop to it!")
      return

    # Nothing to pop, ensure we still run callback
    top_widget = self._nav_stack[-1]
    if top_widget == widget:
      if callback:
        callback()
      return

    # instantly pop widgets in between, then dismiss top widget for animation
    while len(self._nav_stack) > 1 and self._nav_stack[-2] != widget:
      self.pop_widget(len(self._nav_stack) - 2)

    if not instant:
      top_widget.dismiss(callback)
    else:
      self.pop_widget()

  def get_active_widget(self):
    if len(self._nav_stack) > 0:
      return self._nav_stack[-1]
    return None

  def widget_in_stack(self, widget: object) -> bool:
    return widget in self._nav_stack

  def add_nav_stack_tick(self, tick_function: Callable[[], None]):
    if tick_function not in self._nav_stack_ticks:
      self._nav_stack_ticks.append(tick_function)

  def remove_nav_stack_tick(self, tick_function: Callable[[], None]):
    if tick_function in self._nav_stack_ticks:
      self._nav_stack_ticks.remove(tick_function)

  def set_should_render(self, should_render: bool):
    self._should_render = should_render

  def texture(self, asset_path: str, width: int | None = None, height: int | None = None,
              alpha_premultiply=False, keep_aspect_ratio=True, flip_x: bool = False) -> rl.Texture:
    if width is not None:
      width = round(width)
    if height is not None:
      height = round(height)

    cache_key = f"{asset_path}_{width}_{height}_{alpha_premultiply}_{keep_aspect_ratio}_{flip_x}"
    if cache_key in self._textures:
      return self._textures[cache_key]

    with as_file(ASSETS_DIR.joinpath(asset_path)) as fspath:
      image_obj = self._load_image_from_path(fspath.as_posix(), width, height, alpha_premultiply, keep_aspect_ratio, flip_x)
      texture_obj = self._load_texture_from_image(image_obj)

    # Set logical size so widget layout math stays at 1x coordinates
    if self._scale != 1.0 and width is not None and height is not None:
      texture_obj.width = width
      texture_obj.height = height

    self._textures[cache_key] = texture_obj
    return texture_obj

  def _load_image_from_path(self, image_path: str, width: int | None = None, height: int | None = None,
                            alpha_premultiply: bool = False, keep_aspect_ratio: bool = True, flip_x: bool = False) -> rl.Image:
    """Load and resize an image, storing it for later automatic unloading."""
    image = rl.load_image(image_path)

    if alpha_premultiply:
      rl.image_alpha_premultiply(image)

    # Scale up load size for sharper rendering, capped at source resolution
    if self._scale != 1.0 and width is not None and height is not None:
      width = min(int(width * self._scale), image.width)
      height = min(int(height * self._scale), image.height)

    if width is not None and height is not None:
      same_dimensions = image.width == width and image.height == height

      # Resize with aspect ratio preservation if requested
      if not same_dimensions:
        if keep_aspect_ratio:
          orig_width = image.width
          orig_height = image.height

          scale_width = width / orig_width
          scale_height = height / orig_height

          # Calculate new dimensions
          scale = min(scale_width, scale_height)
          new_width = int(orig_width * scale)
          new_height = int(orig_height * scale)

          rl.image_resize(image, new_width, new_height)
        else:
          rl.image_resize(image, width, height)
    else:
      assert keep_aspect_ratio, "Cannot resize without specifying width and height"

    if flip_x:
      rl.image_flip_horizontal(image)

    return image

  def _load_texture_from_image(self, image: rl.Image) -> rl.Texture:
    """Send image to GPU and unload original image."""
    texture = rl.load_texture_from_image(image)
    # Set texture filtering to smooth the result
    rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    # prevent artifacts from wrapping coordinates
    rl.set_texture_wrap(texture, rl.TextureWrap.TEXTURE_WRAP_CLAMP)

    rl.unload_image(image)
    return texture

  def close_ffmpeg(self):
    th = self._ffmpeg_thread
    q = self._ffmpeg_queue
    ev = self._ffmpeg_stop_event
    proc = self._ffmpeg_proc

    # 먼저 참조 끊기(재진입/중복 호출 방지)
    self._ffmpeg_thread = None
    self._ffmpeg_queue = None
    self._ffmpeg_stop_event = None
    self._ffmpeg_proc = None

    # thread stop
    try:
      if th is not None and ev is not None:
        ev.set()
      if th is not None and q is not None:
        try:
          q.put_nowait(None)
        except Exception:
          pass
        th.join(timeout=30)
    except Exception:
      pass

    # proc stop
    if proc is not None:
      try:
        stdin = proc.stdin
        if stdin is not None:
          try:
            # 이미 닫혔으면 flush 금지
            if not getattr(stdin, "closed", False):
              try:
                stdin.flush()
              except Exception:
                pass
              try:
                stdin.close()
              except Exception:
                pass
          except Exception:
            pass

        try:
          proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
          try:
            proc.terminate()
          except Exception:
            pass
          try:
            proc.wait(timeout=5)
          except Exception:
            pass
      except Exception:
        try:
          proc.kill()
        except Exception:
          pass

  def close(self):
    if not rl.is_window_ready():
      return

    for texture in self._textures.values():
      rl.unload_texture(texture)
    self._textures = {}

    for font in self._fonts.values():
      rl.unload_font(font)
    self._fonts = {}

    if self._render_texture is not None:
      rl.unload_render_texture(self._render_texture)
      self._render_texture = None

    if self._burn_in_shader:
      rl.unload_shader(self._burn_in_shader)
      self._burn_in_shader = None

    if not PC:
      self._mouse.stop()

    self.close_ffmpeg()

    self.stop_recording()
    self.close_ffmpeg()
    rl.close_window()

  @property
  def mouse_events(self) -> list[MouseEvent]:
    return self._mouse_events

  @property
  def last_mouse_event(self) -> MouseEvent:
    return self._last_mouse_event

  def render(self):
    try:
      if self._profile_render_frames > 0:
        import cProfile
        self._render_profiler = cProfile.Profile()
        self._render_profile_start_time = time.monotonic()
        self._render_profiler.enable()

      while not (self._window_close_requested or rl.window_should_close()):
        if PC:
          # Thread is not used on PC, need to manually add mouse events
          self._mouse._handle_mouse_event()

        # Store all mouse events for the current frame
        self._mouse_events = self._mouse.get_events()
        if len(self._mouse_events) > 0:
          self._last_mouse_event = self._mouse_events[-1]

        # Skip rendering when screen is off
        if not self._should_render:
          if PC:
            rl.poll_input_events()
          time.sleep(1 / self._target_fps)
          yield False
          continue

        if self._render_texture:
          rl.begin_texture_mode(self._render_texture)
          rl.clear_background(rl.BLACK)
        else:
          rl.begin_drawing()
          rl.clear_background(rl.BLACK)

        if self._scale != 1.0:
          rl.rl_push_matrix()
          rl.rl_scalef(self._scale, self._scale, 1.0)

        # Allow a Widget to still run a function regardless of the stack depth
        for tick in self._nav_stack_ticks:
          tick()

        # Only render top widgets
        for widget in self._nav_stack[-self._nav_stack_widgets_to_render:]:
          widget.render(rl.Rectangle(0, 0, self.width, self.height))

        yield True

        if self._scale != 1.0:
          rl.rl_pop_matrix()

        if self._render_texture:
          rl.end_texture_mode()
          rl.begin_drawing()
          rl.clear_background(rl.BLACK)
          src_rect = rl.Rectangle(0, 0, float(self._scaled_width), -float(self._scaled_height))
          dst_rect = rl.Rectangle(0, 0, float(self._scaled_width), float(self._scaled_height))
          texture = self._render_texture.texture
          if texture:
            if BURN_IN_MODE and self._burn_in_shader:
              rl.begin_shader_mode(self._burn_in_shader)
              rl.draw_texture_pro(texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
              rl.end_shader_mode()
            else:
              rl.draw_texture_pro(texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)

        if self._show_fps:
          rl.draw_fps(10, 10)

        if self._show_touches:
          self._draw_touch_points()

        if self._grid_size > 0:
          self._draw_grid()

        rl.end_drawing()

        if RECORD or self._record_enabled:
          self._record_frame_idx += 1
          if self._record_frame_idx % self._record_every_n == 0:
            # 동기식 GPU readback: rlReadTexturePixels가 캡처마다 임시 FBO 생성/해제
            # (rlog의 "FBO: Unloaded framebuffer" 반복) + ~9MB 이미지 할당 + bytes 복사.
            # 개선 방향 선택을 위해 wall/cpu를 분리 계측한다 (carrot 녹화 경로만)
            cap_tok = SectionMetrics.begin() if self._record_enabled else None
            image = rl.load_image_from_texture(self._render_texture.texture)
            data_size = image.width * image.height * 4
            data = bytes(rl.ffi.buffer(image.data, data_size))
            rl.unload_image(image)
            if cap_tok is not None:
              # readback+복사+해제까지만 계측 — 큐 비용은 캡처 비용이 아니므로 제외
              self._capture_metrics.end(cap_tok)
            try:
              self._ffmpeg_queue.put_nowait(data)  # Async write via background thread
            except queue.Full:
              pass
            
          if self._record_enabled:
            if (self._record_failure_event.is_set()
                or self._ffmpeg_proc is None or self._ffmpeg_proc.poll() is not None):
              # 인코더 사망(강등 검증 실패로 kill된 경우 포함) 또는 라이터 스레드 비정상
              # 종료(kill 실패로 프로세스만 살아있는 경우, stdin write 예외 포함) —
              # 실제로는 기록되지 않는데 녹화 중인 것처럼 보이지 않도록 즉시 녹화를 멈춘다
              self._record_fail_count += 1
              self._record_fail_t = time.monotonic()
              giving_up = " (repeated failures, giving up)" if self._record_fail_count >= 3 else ""
              why = "recording failure flagged" if self._record_failure_event.is_set() else "encoder not running"
              print(f"[REC] {why}, stopping recording{giving_up}")
              self.stop_recording()
            elif (time.monotonic() - self._record_t0) >= self._record_max_sec:
              self.stop_recording()
              self.start_recording()
            elif (self._record_fail_count and (time.monotonic() - self._record_t0) > 5.0
                  and self._ffmpeg_thread is not None and self._ffmpeg_thread.is_alive()
                  and not self._record_failure_event.is_set()):
              # 5초 이상 정상 녹화(인코더 프로세스·라이터 스레드 모두 생존) = 직전 실패는
              # 일시적이었던 것 — 카운터 리셋. 마지막 Event 재확인은 is_alive() 평가 중
              # 라이터가 finally에서 실패를 알리는 race에서 카운터가 리셋되는 것을 막는다
              self._record_fail_count = 0

        self._monitor_fps()
        self._frame += 1

        if self._profile_render_frames > 0 and self._frame >= self._profile_render_frames:
          self._output_render_profile()
    except KeyboardInterrupt:
      pass

  def font(self, font_weight: FontWeight = FontWeight.NORMAL) -> rl.Font:
    return self._fonts[font_weight]

  @property
  def width(self):
    return self._width

  @property
  def height(self):
    return self._height

  def _load_fonts(self):
    for fw in FontWeight:
      fnt_path = FONT_DIR / fw
      font_path = self._resolve_font_path(fnt_path)
      if font_path != fnt_path:
        cloudlog.warning(f"Font atlas missing, loading source font instead: {font_path}")

      font = self._load_font_path(font_path, fw)
      if fw != FontWeight.UNIFONT and self._font_texture_valid(font):
        rl.gen_texture_mipmaps(font.texture)
        rl.set_texture_filter(font.texture, rl.TextureFilter.TEXTURE_FILTER_TRILINEAR)

      self._fonts[fw] = font

    rl.gui_set_font(self._fonts[FontWeight.NORMAL])

  def _resolve_font_path(self, fnt_path: Path) -> Path:
    if fnt_path.exists():
      return fnt_path

    stem = fnt_path.with_suffix("")
    for ext in FONT_SOURCE_EXTS:
      source_path = stem.with_suffix(ext)
      if source_path.exists():
        return source_path
    return fnt_path

  def _load_font_path(self, font_path: Path, font_weight: FontWeight) -> rl.Font:
    if font_path.suffix.lower() == ".fnt":
      return rl.load_font(font_path.as_posix())

    try:
      font_size = 16 if font_weight == FontWeight.UNIFONT else 48 if font_weight == FontWeight.DISPLAY else 200
      codepoints = self._font_codepoints(font_weight)
      cp_buffer = rl.ffi.new("int[]", codepoints)
      cp_ptr = rl.ffi.cast("int *", cp_buffer)
      return rl.load_font_ex(font_path.as_posix(), font_size, cp_ptr, len(codepoints))
    except Exception:
      cloudlog.exception(f"Failed to load source font with codepoints: {font_path}")
      return rl.load_font(font_path.as_posix())

  def _font_codepoints(self, font_weight: FontWeight) -> list[int]:
    codepoints = set(range(32, 127))
    if font_weight in (FontWeight.DISPLAY, FontWeight.UNIFONT):
      codepoints.update(range(0xAC00, 0xD7A4))
    return sorted(codepoints)

  @staticmethod
  def _font_texture_valid(font: rl.Font) -> bool:
    try:
      return int(font.texture.id) > 0
    except Exception:
      return False

  def _set_styles(self):
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BORDER_WIDTH, 0)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, DEFAULT_TEXT_SIZE)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.BACKGROUND_COLOR, rl.color_to_int(rl.BLACK))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(DEFAULT_TEXT_COLOR))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BASE_COLOR_NORMAL, rl.color_to_int(rl.Color(50, 50, 50, 255)))

  def _patch_text_functions(self):
    # Wrap pyray text APIs to apply a global text size scale so our px sizes match Qt
    if not hasattr(rl, "_orig_draw_text_ex"):
      rl._orig_draw_text_ex = rl.draw_text_ex

    def _draw_text_ex_scaled(font, text, position, font_size, spacing, tint):
      font = font_fallback(font)
      return rl._orig_draw_text_ex(font, text, position, font_size * FONT_SCALE, spacing, tint)

    rl.draw_text_ex = _draw_text_ex_scaled

  def _patch_scissor_mode(self):
    if self._scale == 1.0:
      return

    if not hasattr(rl, "_orig_begin_scissor_mode"):
      rl._orig_begin_scissor_mode = rl.begin_scissor_mode

    def _begin_scissor_mode_scaled(x, y, width, height):
      return rl._orig_begin_scissor_mode(
        int(x * self._scale), int(y * self._scale),
        int(math.ceil(width * self._scale)), int(math.ceil(height * self._scale)))

    rl.begin_scissor_mode = _begin_scissor_mode_scaled

  def _set_log_callback(self):
    ffi_libc = cffi.FFI()
    ffi_libc.cdef("""
      int vasprintf(char **strp, const char *fmt, void *ap);
      void free(void *ptr);
    """)
    libc = ffi_libc.dlopen(None)

    @rl.ffi.callback("void(int, char *, void *)")
    def trace_log_callback(log_level, text, args):
      try:
        text_addr = int(rl.ffi.cast("uintptr_t", text))
        args_addr = int(rl.ffi.cast("uintptr_t", args))
        text_libc = ffi_libc.cast("char *", text_addr)
        args_libc = ffi_libc.cast("void *", args_addr)

        out = ffi_libc.new("char **")
        if libc.vasprintf(out, text_libc, args_libc) >= 0 and out[0] != ffi_libc.NULL:
          text_str = ffi_libc.string(out[0]).decode("utf-8", "replace")
          libc.free(out[0])
        else:
          text_str = rl.ffi.string(text).decode("utf-8", "replace")
      except Exception as e:
        text_str = f"[Log decode error: {e}]"

      if log_level == rl.TraceLogLevel.LOG_ERROR:
        cloudlog.error(f"raylib: {text_str}")
      elif log_level == rl.TraceLogLevel.LOG_WARNING:
        cloudlog.warning(f"raylib: {text_str}")
      elif log_level == rl.TraceLogLevel.LOG_INFO:
        cloudlog.info(f"raylib: {text_str}")
      elif log_level == rl.TraceLogLevel.LOG_DEBUG:
        cloudlog.debug(f"raylib: {text_str}")
      else:
        cloudlog.error(f"raylib: Unknown level {log_level}: {text_str}")

    # ensure we get all the logs forwarded to us
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_DEBUG)

    # Store callback reference
    self._trace_log_callback = trace_log_callback
    rl.set_trace_log_callback(self._trace_log_callback)

  def _monitor_fps(self):
    fps = rl.get_fps()

    # Log FPS drop below threshold at regular intervals
    if fps < self._target_fps * FPS_DROP_THRESHOLD:
      current_time = time.monotonic()
      if current_time - self._last_fps_log_time >= FPS_LOG_INTERVAL:
        cloudlog.warning(f"FPS dropped below {self._target_fps}: {fps}")
        self._last_fps_log_time = current_time

    # Strict mode: terminate UI if FPS drops too much
    if STRICT_MODE and fps < self._target_fps * FPS_CRITICAL_THRESHOLD:
      cloudlog.error(f"FPS dropped critically below {fps}. Shutting down UI.")
      self.close_ffmpeg()
      os._exit(1)

  def _draw_touch_points(self):
    current_time = time.monotonic()

    for mouse_event in self._mouse_events:
      if mouse_event.left_pressed:
        self._mouse_history.clear()
      self._mouse_history.append(MousePosWithTime(mouse_event.pos.x * self._scale, mouse_event.pos.y * self._scale, current_time))

    # Remove old touch points that exceed the timeout
    while self._mouse_history and (current_time - self._mouse_history[0].t) > TOUCH_HISTORY_TIMEOUT:
      self._mouse_history.popleft()

    if self._mouse_history:
      mouse_pos = self._mouse_history[-1]
      rl.draw_circle(int(mouse_pos.x), int(mouse_pos.y), 15, rl.RED)
      for idx, mouse_pos in enumerate(self._mouse_history):
        perc = idx / len(self._mouse_history)
        color = rl.Color(min(int(255 * (1.5 - perc)), 255), int(min(255 * (perc + 0.5), 255)), 50, 255)
        rl.draw_circle(int(mouse_pos.x), int(mouse_pos.y), 5, color)

  def _draw_grid(self):
    grid_color = rl.Color(60, 60, 60, 255)
    # Draw vertical lines
    x = 0
    while x <= self._scaled_width:
      rl.draw_line(x, 0, x, self._scaled_height, grid_color)
      x += self._grid_size
    # Draw horizontal lines
    y = 0
    while y <= self._scaled_height:
      rl.draw_line(0, y, self._scaled_width, y, grid_color)
      y += self._grid_size

  def _output_render_profile(self):
    import io
    import pstats

    self._render_profiler.disable()
    elapsed_ms = (time.monotonic() - self._render_profile_start_time) * 1e3
    avg_frame_time = elapsed_ms / self._frame if self._frame > 0 else 0

    stats_stream = io.StringIO()
    pstats.Stats(self._render_profiler, stream=stats_stream).sort_stats("cumtime").print_stats(PROFILE_STATS)
    print("\n=== Render loop profile ===")
    print(stats_stream.getvalue().rstrip())

    green = "\033[92m"
    reset = "\033[0m"
    print(f"\n{green}Rendered {self._frame} frames in {elapsed_ms:.1f} ms{reset}")
    print(f"{green}Average frame time: {avg_frame_time:.2f} ms ({1000/avg_frame_time:.1f} FPS){reset}")
    sys.exit(0)

  def _calculate_auto_scale(self) -> float:
     # Create temporary window to query monitor info
    rl.init_window(1, 1, "")
    w, h = rl.get_monitor_width(0), rl.get_monitor_height(0)
    rl.close_window()

    if w == 0 or h == 0 or (w >= self._width and h >= self._height):
      return 1.0

    # Apply 0.95 factor for window decorations/taskbar margin
    return max(0.3, min(w / self._width, h / self._height) * 0.95)

  @staticmethod
  def _default_width() -> int:
    return 2160 if GuiApplication.big_ui() else 536

  @staticmethod
  def _default_height() -> int:
    return 1080 if GuiApplication.big_ui() else 240

  @staticmethod
  def big_ui() -> bool:
    return HARDWARE.get_device_type() in ('tici', 'tizi') or BIG_UI


gui_app = GuiApplication()
