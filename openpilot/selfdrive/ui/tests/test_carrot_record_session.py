"""carrot 전용: 녹화 pooled buffer + 비동기 인코더 정리 상태기계 회귀 테스트.

route 41d 후속(pooled buffer ①안 + rotation/stop stall 제거)의 리뷰 확정 수정
목록을 고정한다: 세대별 풀 오염 차단, 캡처 no-throw 회수, teardown kill 체인,
cleanup 강등 fail-closed, cleanup start 실패 동기 fallback, 앱 종료 reap,
deferred start의 ScreenRecord 상태 계약.
"""
import queue
import subprocess
import threading
import types

import openpilot.system.ui.lib.application as app_mod
from openpilot.system.ui.lib.application import GuiApplication, _RecordBufPool


def make_app(**over):
  app = object.__new__(GuiApplication)
  app._record_buf_count = 3
  app._record_buf_pool = None
  app._record_cleanups = []
  app._record_start_deferred_logged = False
  app._record_start_pending = False
  app._record_enabled = False
  app._record_fail_count = 0
  app._record_fail_t = 0.0
  app._ffmpeg_thread = None
  app._ffmpeg_queue = None
  app._ffmpeg_stop_event = None
  app._ffmpeg_proc = None
  for k, v in over.items():
    setattr(app, k, v)
  return app


class FakeProc:
  def __init__(self, wait_script=()):
    # wait_script: proc.wait 호출마다 소비 — "ok" | "timeout" | Exception 인스턴스
    self.wait_calls = []
    self.kill_calls = 0
    self.terminate_calls = 0
    self._script = list(wait_script)
    self.stdin = None

  def wait(self, timeout=None):
    self.wait_calls.append(timeout)
    action = self._script.pop(0) if self._script else "ok"
    if action == "timeout":
      raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout)
    if isinstance(action, Exception):
      raise action
    return 0

  def terminate(self):
    self.terminate_calls += 1

  def kill(self):
    self.kill_calls += 1


class TestRecordBufPool:
  def test_take_until_exhausted_then_none(self):
    pool = _RecordBufPool(16, 2)
    assert len(pool.take()) == 16 and len(pool.take()) == 16
    assert pool.take() is None  # 고갈 = 프레임 드랍 (무제한 버퍼링 금지)

  def test_put_rejects_wrong_size(self):
    # 세대 교체 후 이전 세대 writer의 늦은 반환이 새 풀을 오염시키면 안 된다
    pool = _RecordBufPool(16, 1)
    pool.take()
    pool.put(bytearray(8))
    assert pool.take() is None  # 크기 불일치 버퍼는 받지 않는다
    pool.put(bytearray(16))
    assert len(pool.take()) == 16

  def test_put_none_and_overflow_ignored(self):
    pool = _RecordBufPool(16, 1)
    pool.put(None)
    pool.put(bytearray(16))  # 상한 초과 — 폐기
    assert len(pool.take()) == 16 and pool.take() is None


class TestEnsurePool:
  def test_same_size_reused_without_refill(self):
    app = make_app()
    app._ensure_record_buf_pool(16)
    first = app._record_buf_pool
    lost = first.take()  # writer 급사로 소실됐다고 가정
    assert lost is not None
    app._ensure_record_buf_pool(16)  # 같은 크기 — 재사용, qsize 기반 보충 금지
    assert app._record_buf_pool is first
    assert first._q.qsize() == app._record_buf_count - 1

  def test_size_change_creates_new_generation(self):
    app = make_app()
    app._ensure_record_buf_pool(16)
    old = app._record_buf_pool
    app._ensure_record_buf_pool(32)
    assert app._record_buf_pool is not old
    assert app._record_buf_pool.buf_size == 32


class TestWriterLoop:
  def test_returns_buffers_to_own_pool(self):
    pool = _RecordBufPool(4, 2)
    b1, b2 = pool.take(), pool.take()
    wq = queue.Queue()
    writes = []
    proc = types.SimpleNamespace(stdin=types.SimpleNamespace(write=lambda d: writes.append(bytes(d))))
    for item in (b1, b2, None):
      wq.put(item)
    GuiApplication._record_writer_loop(proc, wq, threading.Event(), pool)
    assert len(writes) == 2 and pool._q.qsize() == 2

  def test_write_failure_recovers_all_buffers(self):
    pool = _RecordBufPool(4, 2)
    b1, b2 = pool.take(), pool.take()
    wq = queue.Queue()
    wq.put(b1)
    wq.put(b2)  # 첫 write 실패로 종료 후 잔여 — finally 드레인이 회수
    def boom(_):
      raise BrokenPipeError("pipe closed")
    proc = types.SimpleNamespace(stdin=types.SimpleNamespace(write=boom))
    GuiApplication._record_writer_loop(proc, wq, threading.Event(), pool)
    assert pool._q.qsize() == 2 and wq.empty()

  def test_old_generation_return_does_not_pollute_new_pool(self):
    old_pool = _RecordBufPool(4, 1)
    buf = old_pool.take()
    wq = queue.Queue()
    wq.put(buf)
    wq.put(None)
    proc = types.SimpleNamespace(stdin=types.SimpleNamespace(write=lambda d: None))
    # writer는 자기 세대(old_pool) 지역 참조로만 반환한다 — 새 풀은 무관
    GuiApplication._record_writer_loop(proc, wq, threading.Event(), old_pool)
    assert old_pool._q.qsize() == 1


class TestCaptureNoThrow:
  def _wire(self, monkeypatch, app, *, w=2, h=2, readback_raises=False):
    unloads = []
    def load(_tex):
      if readback_raises:
        raise RuntimeError("GL context lost")
      return types.SimpleNamespace(width=w, height=h, data=bytes(w * h * 4))
    monkeypatch.setattr(app_mod, "rl", types.SimpleNamespace(
      load_image_from_texture=load,
      unload_image=lambda img: unloads.append(img),
      ffi=types.SimpleNamespace(memmove=lambda dest, src, n: dest.__setitem__(slice(0, n), src[:n])),
    ))
    app._render_texture = types.SimpleNamespace(texture=object())
    app._capture_metrics = types.SimpleNamespace(end=lambda tok: None, flush=lambda: None)
    app._ffmpeg_queue = queue.Queue(maxsize=8)
    return unloads

  def test_success_moves_ownership_to_writer(self, monkeypatch):
    app = make_app()
    app._ensure_record_buf_pool(16)
    unloads = self._wire(monkeypatch, app)
    app._capture_record_frame()
    assert app._ffmpeg_queue.qsize() == 1 and len(unloads) == 1
    assert app._record_buf_pool._q.qsize() == app._record_buf_count - 1

  def test_readback_exception_swallowed_and_buffer_recovered(self, monkeypatch):
    app = make_app()
    app._ensure_record_buf_pool(16)
    self._wire(monkeypatch, app, readback_raises=True)
    app._capture_record_frame()  # 예외가 렌더 루프로 전파되면 테스트 실패
    assert app._record_buf_pool._q.qsize() == app._record_buf_count

  def test_size_mismatch_drops_and_recovers(self, monkeypatch):
    app = make_app()
    app._ensure_record_buf_pool(64)  # 텍스처는 2x2x4=16 — 불일치
    unloads = self._wire(monkeypatch, app)
    app._capture_record_frame()
    assert app._ffmpeg_queue.empty() and len(unloads) == 1
    assert app._record_buf_pool._q.qsize() == app._record_buf_count

  def test_queue_full_recovers_buffer(self, monkeypatch):
    app = make_app()
    app._ensure_record_buf_pool(16)
    self._wire(monkeypatch, app)
    app._ffmpeg_queue = queue.Queue(maxsize=1)
    app._ffmpeg_queue.put_nowait(b"x")
    app._capture_record_frame()
    assert app._record_buf_pool._q.qsize() == app._record_buf_count


class TestTeardownKillChain:
  def test_terminate_timeout_falls_through_to_kill_and_reap(self):
    proc = FakeProc(wait_script=("timeout", "timeout", "ok"))
    GuiApplication._record_teardown(None, proc)
    assert proc.terminate_calls == 1 and proc.kill_calls == 1
    assert len(proc.wait_calls) == 3  # wait(10) → wait(5) → kill 후 reap wait(5)

  def test_generic_wait_error_kills_and_reaps(self):
    proc = FakeProc(wait_script=(OSError("who knows"), "ok"))
    GuiApplication._record_teardown(None, proc)
    assert proc.kill_calls == 1 and len(proc.wait_calls) == 2  # 오류 → kill → reap

  def test_normal_exit_no_kill(self):
    proc = FakeProc()
    GuiApplication._record_teardown(None, proc)
    assert proc.kill_calls == 0 and proc.terminate_calls == 0


class TestCleanupWorker:
  def test_demotion_failure_kills_immediately(self, monkeypatch):
    app = make_app()
    app._demote_record_sched = lambda: False
    torn = []
    app._record_teardown = lambda th, proc: torn.append(1)
    proc = FakeProc()
    app._record_cleanup_worker(None, proc)
    # RT 상속 상태로 blocking 정리 금지 — 즉시 kill + reap, teardown 안 탄다
    assert proc.kill_calls == 1 and len(proc.wait_calls) == 1 and torn == []

  def test_demotion_ok_runs_teardown(self):
    app = make_app()
    app._demote_record_sched = lambda: True
    torn = []
    app._record_teardown = lambda th, proc: torn.append((th, proc))
    app._record_cleanup_worker("th", "proc")
    assert torn == [("th", "proc")]


class TestAsyncClose:
  def test_thread_start_failure_falls_back_to_sync(self, monkeypatch):
    app = make_app()
    torn = []
    app._record_teardown = lambda th, proc: torn.append((th, proc))
    class DeadThread:
      def __init__(self, *a, **k):
        pass
      def start(self):
        raise RuntimeError("cannot spawn")
    monkeypatch.setattr(app_mod.threading, "Thread", DeadThread)
    ev = threading.Event()
    app._ffmpeg_thread, app._ffmpeg_queue = object(), queue.Queue()
    app._ffmpeg_stop_event, app._ffmpeg_proc = ev, "proc"
    app._close_ffmpeg_async()  # 예외 무전파 + 동기 정리로 대체
    assert torn and app._record_cleanups == []  # 실패한 스레드는 등록되지 않는다
    assert ev.is_set()  # 정지 신호는 이미 나감

  def test_success_registers_thread_and_proc(self):
    app = make_app()
    done = threading.Event()
    app._record_teardown = lambda th, proc: done.set()
    app._demote_record_sched = lambda: True
    ev = threading.Event()
    app._ffmpeg_thread, app._ffmpeg_queue = object(), queue.Queue()
    app._ffmpeg_stop_event, app._ffmpeg_proc = ev, "proc"
    app._close_ffmpeg_async()
    assert app._ffmpeg_proc is None and ev.is_set()  # 참조 즉시 분리 + 신호 즉시
    assert len(app._record_cleanups) == 1 and app._record_cleanups[0][1] == "proc"
    assert done.wait(timeout=5)


class TestJoinCleanups:
  def test_leftover_encoders_killed_and_reaped(self):
    app = make_app()
    proc = FakeProc()
    stuck = types.SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: True)
    app._record_cleanups = [(stuck, proc)]
    app._join_record_cleanups(timeout=0.1)
    assert proc.kill_calls == 1 and len(proc.wait_calls) == 1  # 살아있는 ffmpeg 금지


class TestDeferredStartContract:
  def test_backlog_defers_with_pending_and_single_log(self, monkeypatch, capsys):
    app = make_app()
    alive = types.SimpleNamespace(is_alive=lambda: True)
    app._record_cleanups = [(alive, None), (alive, None)]
    ensured = []
    app._ensure_render_texture_for_recording = lambda: ensured.append(1)
    app.start_recording()
    app.start_recording()
    assert app.is_record_start_pending() and ensured == []
    assert capsys.readouterr().out.count("deferring start") == 1

  def test_cooldown_failure_is_not_pending(self):
    import time as _time
    app = make_app()
    app._record_start_pending = True
    app._record_fail_count = 3
    app._record_fail_t = _time.monotonic()
    app.start_recording()
    assert not app.is_record_start_pending()  # 진짜 실패 — 파라미터가 꺼져야 함

  def test_stop_clears_pending(self):
    app = make_app()
    app._record_start_pending = True
    app.stop_recording()
    assert not app.is_record_start_pending()


class TestMainLayoutSync:
  def test_pending_does_not_clear_param(self, monkeypatch):
    import openpilot.selfdrive.ui.mici.layouts.main as mici_main
    puts = []
    monkeypatch.setattr(mici_main, "gui_app", types.SimpleNamespace(
      is_recording=lambda: False, is_record_start_pending=lambda: True))
    monkeypatch.setattr(mici_main, "ui_state", types.SimpleNamespace(
      params=types.SimpleNamespace(put_bool_nonblocking=lambda k, v: puts.append((k, v)))))
    assert mici_main.MiciMainLayout._sync_screen_record_state(True) is False
    assert puts == []  # 보류 중에는 ScreenRecord를 끄지 않는다 (재시도 유지)

  def test_not_pending_mismatch_clears_param(self, monkeypatch):
    import openpilot.selfdrive.ui.mici.layouts.main as mici_main
    puts = []
    monkeypatch.setattr(mici_main, "gui_app", types.SimpleNamespace(
      is_recording=lambda: False, is_record_start_pending=lambda: False))
    monkeypatch.setattr(mici_main, "ui_state", types.SimpleNamespace(
      params=types.SimpleNamespace(put_bool_nonblocking=lambda k, v: puts.append((k, v)))))
    mici_main.MiciMainLayout._sync_screen_record_state(True)
    assert puts == [("ScreenRecord", False)]  # 실패는 기존대로 사용자에게 드러난다
