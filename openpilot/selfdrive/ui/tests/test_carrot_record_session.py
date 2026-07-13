"""carrot 전용: 녹화 pooled buffer + 비동기 인코더 정리 상태기계 회귀 테스트.

route 41d 후속(pooled buffer ①안 + rotation/stop stall 제거)의 리뷰 확정 수정
목록을 고정한다: 세대별 풀 오염 차단, 캡처 no-throw 회수, teardown kill 체인,
cleanup 강등 fail-closed, cleanup start 실패 동기 fallback, 앱 종료 reap,
deferred start의 ScreenRecord 상태 계약.
"""
import ast
import inspect
import os
import queue
import subprocess
import threading
import types

import openpilot.system.ui.lib.application as app_mod
from openpilot.system.ui.lib.application import GuiApplication, _RecordBufPool, _record_stdin_write_all


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
  app._record_session_id = 0
  app._record_t0 = 0.0
  app._record_drop_counts = dict.fromkeys(
    ("pool_empty", "queue_full", "size_mismatch", "capture_error", "stop_gate", "gate_busy"), 0)
  app._ffmpeg_thread = None
  app._ffmpeg_enqueue_lock = None
  app._ffmpeg_queue = None
  app._ffmpeg_stop_event = None
  app._ffmpeg_proc = None
  for k, v in over.items():
    setattr(app, k, v)
  return app


class FakeProc:
  def __init__(self, wait_script=(), alive=False, dies_on_kill=True):
    # wait_script: proc.wait 호출마다 소비 — "ok" | "timeout" | Exception 인스턴스
    self.wait_calls = []
    self.kill_calls = 0
    self.terminate_calls = 0
    self._script = list(wait_script)
    self._alive = alive
    self._dies_on_kill = dies_on_kill
    self.stdin = None

  def poll(self):
    return None if self._alive else 0

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
    if self._dies_on_kill:
      self._alive = False


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
    proc = types.SimpleNamespace(stdin=types.SimpleNamespace(write=lambda d: (writes.append(bytes(d)), len(d))[1]))
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
    proc = types.SimpleNamespace(stdin=types.SimpleNamespace(write=len))
    # writer는 자기 세대(old_pool) 지역 참조로만 반환한다 — 새 풀은 무관
    GuiApplication._record_writer_loop(proc, wq, threading.Event(), old_pool)
    assert old_pool._q.qsize() == 1


class TestStdinWriteAll:
  # raw(unbuffered) stdin은 write가 partial일 수 있다 — write-all loop 계약 고정
  def test_partial_writes_deliver_full_frame(self):
    chunks = []
    def partial_write(mv):
      n = min(3, len(mv))  # 파이프가 3바이트씩만 받는 상황
      chunks.append(bytes(mv[:n]))
      return n
    stdin = types.SimpleNamespace(write=partial_write)
    data = bytearray(b"0123456789")
    _record_stdin_write_all(stdin, data)
    assert b"".join(chunks) == bytes(data)

  def test_no_progress_raises(self):
    # 진행 불가(0/None)를 성공으로 오인해 무한 루프하면 안 된다 — 예외로 승격해
    # writer가 비정상 종료(failure_event) 경로를 타게 한다
    for stall in (0, None):
      stdin = types.SimpleNamespace(write=lambda mv, _s=stall: _s)
      try:
        _record_stdin_write_all(stdin, bytearray(b"xx"))
        raise AssertionError("no-progress write must raise")
      except OSError:
        pass

  def test_write_exception_propagates(self):
    def boom(_mv):
      raise BrokenPipeError("pipe closed")
    try:
      _record_stdin_write_all(types.SimpleNamespace(write=boom), bytearray(b"xx"))
      raise AssertionError("must propagate")
    except BrokenPipeError:
      pass

  def test_out_of_contract_count_raises(self):
    # 음수/초과 보고는 실제 FileIO 계약 밖 — 성공으로 오인해 memoryview를 잘못
    # 소비(음수 슬라이스/조기 완료 판정)하면 안 된다
    for bogus in (-1, 3):
      stdin = types.SimpleNamespace(write=lambda mv, _n=bogus: _n)
      try:
        _record_stdin_write_all(stdin, bytearray(b"xx"))  # len 2 < 3
        raise AssertionError("invalid count must raise")
      except OSError:
        pass

  def test_real_raw_pipe_delivers_all_bytes(self):
    # 실제 FileIO/파이프 통합 검증 — 파이프 용량(통상 64KiB)보다 큰 프레임은
    # partial write가 실제로 발생하므로 write-all 루프가 실전 계약으로 검증된다
    r_fd, w_fd = os.pipe()
    received = bytearray()
    def drain():
      with open(r_fd, "rb", buffering=0) as r:
        while True:
          b = r.read(65536)
          if not b:
            break
          received.extend(b)
    reader = threading.Thread(target=drain)
    reader.start()
    data = bytes(range(256)) * 4096  # 1 MiB
    try:
      with open(w_fd, "wb", buffering=0) as w:  # raw FileIO — 인코더 stdin과 동일 계층
        _record_stdin_write_all(w, data)
    finally:
      reader.join(timeout=30)
    assert not reader.is_alive() and bytes(received) == data


class TestUpstreamRecordWriter:
  # upstream RECORD(개발용) writer도 raw stdin 전제를 공유한다 — 공유 teardown의
  # raw close 계약이 모든 호출부에서 성립해야 하므로 write도 전량 쓰기여야 하고,
  # 실패 시에는 기존 의미(첫 실패에서 곧장 종료)를 유지해야 한다

  def _writer_app(self, stdin):
    q = queue.Queue()
    proc = types.SimpleNamespace(stdin=stdin)
    app = make_app(_ffmpeg_queue=q, _ffmpeg_proc=proc, _ffmpeg_stop_event=threading.Event())
    return app, q

  def test_partial_writes_deliver_full_frame(self):
    chunks = []
    def partial_write(mv):
      n = min(7, len(mv))  # 파이프가 7바이트씩만 받는 상황
      chunks.append(bytes(mv[:n]))
      return n
    app, q = self._writer_app(types.SimpleNamespace(write=partial_write))
    frame = bytes(range(64))
    q.put(frame)
    q.put(None)  # 정지 sentinel — 정상 종료로 반환해야 한다
    app._ffmpeg_writer_thread()
    assert b"".join(chunks) == frame

  def test_no_progress_stops_writer(self):
    # 진행 불가(0/None)는 write-all이 예외로 승격 → 기존 except-break 의미대로
    # 첫 실패에서 writer가 끝난다 (무한 재시도/이후 프레임 계속 쓰기 금지)
    for stall in (0, None):
      calls = []
      stdin = types.SimpleNamespace(write=lambda mv, _s=stall, _c=calls: _c.append(1) or _s)
      app, q = self._writer_app(stdin)
      q.put(b"first")
      q.put(b"second")
      app._ffmpeg_writer_thread()  # 예외가 밖으로 새면 테스트 실패
      assert len(calls) == 1 and q.qsize() == 1

  def test_write_exception_stops_writer(self):
    def boom(_mv):
      raise BrokenPipeError("pipe closed")
    app, q = self._writer_app(types.SimpleNamespace(write=boom))
    q.put(b"first")
    q.put(b"second")
    app._ffmpeg_writer_thread()  # 기존 except-break 의미 보존 — 예외 없이 반환
    assert q.qsize() == 1


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
    app._ffmpeg_enqueue_lock = threading.Lock()
    app._ffmpeg_stop_event = threading.Event()
    app._record_failure_event = threading.Event()
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

  def test_normal_exit_closes_raw_stdin_once(self):
    # raw stdin은 flush 없이 close(2) 한 번으로 EOF 전달 — 블록 지점이 없다
    closed = []
    proc = FakeProc()
    proc.stdin = types.SimpleNamespace(closed=False, close=lambda: closed.append(1))
    GuiApplication._record_teardown(None, proc)
    assert closed == [1] and proc.kill_calls == 0

  def test_stdin_close_error_falls_through_to_kill_chain(self):
    # EOF 전달 실패 — 인코더가 스스로 끝나길 기대할 수 없으니 kill 체인으로
    def bad_close():
      raise OSError("close failed")
    proc = FakeProc(alive=True, dies_on_kill=True)
    proc.stdin = types.SimpleNamespace(closed=False, close=bad_close)
    resolved = GuiApplication._record_teardown(None, proc)
    assert proc.kill_calls == 1 and resolved is True


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
    app._record_teardown = lambda th, proc: bool(torn.append((th, proc))) or True
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


class TestWriterLifecycle:
  """재재리뷰 확정 경로: pre-loop 회수, 세션 고정, start 실패 rollback."""

  def test_preloop_failure_recovers_queued_buffers(self):
    # 강등 실패로 loop 진입 전 종료해도, 검증하는 동안 캡처가 큐에 넣은 버퍼를
    # 래퍼 finally가 회수한다 — 안 거두면 실패 한 번에 풀이 전량 소실
    app = make_app()
    app._demote_record_sched = lambda: False
    pool = _RecordBufPool(4, 2)
    wq = queue.Queue()
    wq.put(pool.take())
    wq.put(pool.take())
    fev = threading.Event()
    proc = FakeProc(alive=True)
    app._record_writer_thread(proc, wq, threading.Event(), fev, pool, threading.Lock())
    assert pool._q.qsize() == 2 and wq.empty()
    assert fev.is_set() and proc.kill_calls == 1  # fail-closed 의미 유지

  def test_writer_uses_only_bound_session(self, monkeypatch):
    # Thread.start() 직후 stop/회전이 self._ffmpeg_*를 끊고 새 세션으로 바꿔도,
    # 늦게 실행된 writer는 args로 고정된 자기 세션만 본다
    monkeypatch.setattr(app_mod, "sys", types.SimpleNamespace(platform="test"))  # ffmpeg 검증 스킵
    app = make_app()
    app._demote_record_sched = lambda: True
    new_pool = _RecordBufPool(8, 1)  # 새 세션 (다른 크기 세대)
    app._record_buf_pool = new_pool
    app._ffmpeg_queue = queue.Queue()  # 새 세션 큐 — writer가 건드리면 안 된다
    old_pool = _RecordBufPool(4, 1)
    old_q = queue.Queue()
    writes = []
    old_proc = types.SimpleNamespace(
      stdin=types.SimpleNamespace(write=lambda d: (writes.append(bytes(d)), len(d))[1]), poll=lambda: None)
    old_q.put(old_pool.take())
    old_q.put(None)
    old_fev = threading.Event()
    app._record_writer_thread(old_proc, old_q, threading.Event(), old_fev, old_pool, threading.Lock())
    assert len(writes) == 1 and old_pool._q.qsize() == 1  # 자기 세션 풀로만 반환
    assert new_pool._q.qsize() == 1 and app._ffmpeg_queue.empty()  # 새 세션 불가침

  def test_writer_thread_start_failure_rolls_back(self, monkeypatch):
    from pathlib import Path
    app = make_app()
    app._width, app._height, app._target_fps = 4, 4, 20
    app._record_failure_event = threading.Event()
    fp = FakeProc(alive=True)
    monkeypatch.setattr(app_mod.subprocess, "Popen", lambda *a, **k: fp)
    class DeadThread:
      def __init__(self, *a, **k):
        pass
      def start(self):
        raise RuntimeError("no threads")
    monkeypatch.setattr(app_mod.threading, "Thread", DeadThread)
    ok = app._init_ffmpeg(Path("/tmp/x.mp4"))  # 예외 무전파 + False (정상 실패 경로 합류)
    assert ok is False
    assert fp.kill_calls == 1 and fp.wait_calls  # 인코더 rollback: kill + reap
    assert (app._ffmpeg_proc is None and app._ffmpeg_queue is None
            and app._ffmpeg_thread is None and app._ffmpeg_stop_event is None
            and app._ffmpeg_enqueue_lock is None)


class TestTeardownWriterAlive:
  def test_skips_stdin_and_kills_when_writer_stuck(self):
    # writer가 join(30s) 후에도 살아 있으면(stdin.write 블록) stdin flush/close를
    # 동시에 하지 않는다 — 곧장 kill로 파이프를 끊는다 (bounded)
    stuck_writer = types.SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: True)
    touched = []
    proc = FakeProc(alive=True)
    proc.stdin = types.SimpleNamespace(
      closed=False, flush=lambda: touched.append("flush"), close=lambda: touched.append("close"))
    GuiApplication._record_teardown(stuck_writer, proc)
    assert touched == []  # 경합 금지
    assert proc.kill_calls == 1 and len(proc.wait_calls) == 1  # kill + reap


class TestPruneCleanups:
  def test_dead_thread_live_proc_kept_and_rekilled(self):
    # 스레드가 끝나도 인코더가 살아 있으면 백로그로 유지 + 지날 때마다 kill 재시도
    app = make_app()
    dead = types.SimpleNamespace(is_alive=lambda: False)
    stubborn = FakeProc(alive=True, dies_on_kill=False)
    app._record_cleanups = [[dead, stubborn, 0.0]]
    app._prune_record_cleanups()
    assert len(app._record_cleanups) == 1 and stubborn.kill_calls == 1

  def test_dead_thread_proc_dies_on_kill_resolved(self):
    app = make_app()
    dead = types.SimpleNamespace(is_alive=lambda: False)
    proc = FakeProc(alive=True, dies_on_kill=True)
    app._record_cleanups = [[dead, proc, 0.0]]
    app._prune_record_cleanups()
    assert app._record_cleanups == [] and proc.kill_calls == 1  # kill로 해소

  def test_dead_thread_dead_proc_removed(self):
    app = make_app()
    dead = types.SimpleNamespace(is_alive=lambda: False)
    app._record_cleanups = [[dead, FakeProc(alive=False), 0.0]]
    app._prune_record_cleanups()
    assert app._record_cleanups == []

  def test_alive_thread_kept_without_kill(self):
    app = make_app()
    alive = types.SimpleNamespace(is_alive=lambda: True)
    proc = FakeProc(alive=True)
    app._record_cleanups = [[alive, proc, 0.0]]
    app._prune_record_cleanups()
    assert len(app._record_cleanups) == 1 and proc.kill_calls == 0  # 진행 중 정리는 존중


class TestJoinCleanups:
  def test_leftover_encoders_killed_and_reaped(self):
    app = make_app()
    proc = FakeProc(alive=True)
    stuck = types.SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: True)
    app._record_cleanups = [[stuck, proc, 0.0]]
    app._join_record_cleanups(timeout=0.1)
    assert proc.kill_calls == 1 and len(proc.wait_calls) == 1  # 살아있는 ffmpeg 금지

  def test_dead_thread_live_proc_also_killed_on_exit(self):
    # 스레드는 끝났지만 인코더가 잔존한 항목도 앱 종료 시 kill/reap 대상
    app = make_app()
    proc = FakeProc(alive=True)
    dead = types.SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: False)
    app._record_cleanups = [[dead, proc, 0.0]]
    app._join_record_cleanups(timeout=0.1)
    assert proc.kill_calls == 1 and app._record_cleanups == []


class TestDeferredStartContract:
  def test_backlog_defers_with_pending_and_single_log(self, monkeypatch, capsys):
    app = make_app()
    alive = types.SimpleNamespace(is_alive=lambda: True)
    app._record_cleanups = [[alive, None, 0.0], [alive, None, 0.0]]
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

  def test_backlog_resolution_resumes_start(self):
    # 백로그가 해소되면(스레드 종료+인코더 사망) pending이 풀리고 시작 로직이 진행된다
    app = make_app()
    dead = types.SimpleNamespace(is_alive=lambda: False)
    app._record_cleanups = [[dead, FakeProc(alive=False), 0.0], [dead, FakeProc(alive=False), 0.0]]
    app._record_start_pending = True
    ensured = []
    app._ensure_render_texture_for_recording = lambda: ensured.append(1)
    app._render_texture = None  # 시작 로직 진입 확인 후 여기서 중단
    app.start_recording()
    assert not app.is_record_start_pending() and ensured == [1]


class TestDropCounters:
  def _wire(self, monkeypatch, app, *, w=2, h=2, readback_raises=False):
    def load(_tex):
      if readback_raises:
        raise RuntimeError("GL context lost")
      return types.SimpleNamespace(width=w, height=h, data=bytes(w * h * 4))
    monkeypatch.setattr(app_mod, "rl", types.SimpleNamespace(
      load_image_from_texture=load, unload_image=lambda img: None,
      ffi=types.SimpleNamespace(memmove=lambda dest, src, n: dest.__setitem__(slice(0, n), src[:n])),
    ))
    app._render_texture = types.SimpleNamespace(texture=object())
    app._capture_metrics = types.SimpleNamespace(end=lambda tok: None, flush=lambda: None)
    app._ffmpeg_queue = queue.Queue(maxsize=8)
    app._ffmpeg_enqueue_lock = threading.Lock()
    app._ffmpeg_stop_event = threading.Event()
    app._record_failure_event = threading.Event()

  def test_each_drop_path_counted(self, monkeypatch):
    app = make_app()
    app._ensure_record_buf_pool(16)
    self._wire(monkeypatch, app)
    app._record_buf_pool = _RecordBufPool(16, 1)
    app._record_buf_pool.take()  # 고갈
    app._capture_record_frame()
    assert app._record_drop_counts["pool_empty"] == 1

    app._ensure_record_buf_pool(16)  # 같은 크기 — 재사용이므로 새로 생성
    app._record_buf_pool = _RecordBufPool(16, 3)
    app._ffmpeg_queue = queue.Queue(maxsize=1)
    app._ffmpeg_queue.put_nowait(b"x")  # 큐 가득
    app._capture_record_frame()
    assert app._record_drop_counts["queue_full"] == 1

    app._ffmpeg_queue = queue.Queue(maxsize=8)
    app._record_buf_pool = _RecordBufPool(64, 1)  # 텍스처(16B)와 불일치
    app._capture_record_frame()
    assert app._record_drop_counts["size_mismatch"] == 1

    app._record_buf_pool = _RecordBufPool(16, 1)
    self._wire(monkeypatch, app, readback_raises=True)
    app._ffmpeg_queue = queue.Queue(maxsize=8)
    app._capture_record_frame()
    assert app._record_drop_counts["capture_error"] == 1

  def test_stop_emits_summary_even_when_zero(self, capsys):
    # '로그 없음'과 '드랍 0'을 구분할 수 있어야 한다 — 0이어도 한 줄 배출
    app = make_app()
    app._record_enabled = True
    app._capture_metrics = types.SimpleNamespace(flush=lambda: None)
    app._close_ffmpeg_async = lambda: None
    app.stop_recording()
    out = capsys.readouterr().out
    assert "session drops: pool_empty=0 queue_full=0 size_mismatch=0 capture_error=0 stop_gate=0 gate_busy=0" in out

  def test_start_resets_counters_per_session(self):
    app = make_app()
    app._record_drop_counts["pool_empty"] = 7
    app._render_texture = types.SimpleNamespace(texture=types.SimpleNamespace(width=2, height=2))
    app._ensure_render_texture_for_recording = lambda: None
    app._new_record_path = lambda: "/tmp/x.mp4"
    app._init_ffmpeg = lambda path: True
    app.start_recording()
    assert app._record_enabled and all(v == 0 for v in app._record_drop_counts.values())


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


class TestEnqueueGate:
  """writer terminal drain과 UI의 늦은 enqueue 사이 race — 소유권 소실 금지."""

  def _wire_capture(self, monkeypatch, app):
    monkeypatch.setattr(app_mod, "rl", types.SimpleNamespace(
      load_image_from_texture=lambda tex: types.SimpleNamespace(width=2, height=2, data=bytes(16)),
      unload_image=lambda img: None,
      ffi=types.SimpleNamespace(memmove=lambda dest, src, n: dest.__setitem__(slice(0, n), src[:n])),
    ))
    app._render_texture = types.SimpleNamespace(texture=object())
    app._capture_metrics = types.SimpleNamespace(end=lambda tok: None, flush=lambda: None)
    app._ffmpeg_queue = queue.Queue(maxsize=8)
    app._ffmpeg_enqueue_lock = threading.Lock()
    app._ffmpeg_stop_event = threading.Event()
    app._record_failure_event = threading.Event()

  def test_late_enqueue_after_failure_is_gated(self, monkeypatch):
    # writer-first: failure(+stop) set + drain 완료 후의 캡처는 큐에 넣지 않고
    # 반려 — 소유자 없는 버퍼가 생기지 않고 stop_gate로 관측된다
    app = make_app()
    app._ensure_record_buf_pool(16)
    self._wire_capture(monkeypatch, app)
    app._record_failure_event.set()
    app._ffmpeg_stop_event.set()
    app._capture_record_frame()
    assert app._ffmpeg_queue.empty()
    assert app._record_buf_pool._q.qsize() == app._record_buf_count  # buf 회수
    assert app._record_drop_counts["stop_gate"] == 1

  def test_producer_first_buffer_recovered_by_terminal_drain(self, monkeypatch):
    # producer-first: gate 통과 시점에 Event 미설정이라 put됐다면, 이후 writer의
    # terminal drain(락 아래)이 그 버퍼를 회수한다
    app = make_app()
    app._ensure_record_buf_pool(16)
    self._wire_capture(monkeypatch, app)
    app._capture_record_frame()  # 정상 enqueue
    assert app._ffmpeg_queue.qsize() == 1
    app._demote_record_sched = lambda: False  # writer 즉시 비정상 종료 경로
    fev = app._record_failure_event
    app._record_writer_thread(FakeProc(alive=True), app._ffmpeg_queue, app._ffmpeg_stop_event,
                              fev, app._record_buf_pool, app._ffmpeg_enqueue_lock)
    assert app._record_buf_pool._q.qsize() == app._record_buf_count  # drain이 회수
    assert fev.is_set() and app._ffmpeg_stop_event.is_set()  # failure 시 stop도 set

  def test_detached_session_gates_capture(self, monkeypatch):
    # detach 후(락 None) 캡처는 반려 — 세션 없는 큐 접근 자체가 없다
    app = make_app()
    app._ensure_record_buf_pool(16)
    self._wire_capture(monkeypatch, app)
    app._ffmpeg_enqueue_lock = None
    app._capture_record_frame()
    assert app._record_drop_counts["stop_gate"] == 1
    assert app._record_buf_pool._q.qsize() == app._record_buf_count


class TestCleanupConstructorFailure:
  def test_constructor_failure_falls_back_to_sync(self, monkeypatch):
    # Thread(...) 생성자 자체의 예외도 rollback 범위 — 정리 주체가 사라지면 안 된다
    app = make_app()
    torn = []
    app._record_teardown = lambda th, proc: bool(torn.append((th, proc))) or True
    class BadCtor:
      def __init__(self, *a, **k):
        raise RuntimeError("ctor fail")
    monkeypatch.setattr(app_mod.threading, "Thread", BadCtor)
    ev = threading.Event()
    app._ffmpeg_thread, app._ffmpeg_queue = object(), queue.Queue()
    app._ffmpeg_stop_event, app._ffmpeg_proc = ev, "proc"
    app._close_ffmpeg_async()  # 예외 무전파
    assert torn == [(app._ffmpeg_thread, "proc")] or torn  # 동기 fallback 실행
    assert ev.is_set() and app._record_cleanups == []

  def test_unresolved_sync_fallback_is_tracked(self, monkeypatch):
    # 동기 fallback조차 인코더를 못 죽였으면(unresolved) 추적 목록에 남아
    # prune/앱 종료가 kill을 재시도한다
    app = make_app()
    app._record_teardown = lambda th, proc: False  # 미해소
    class BadCtor:
      def __init__(self, *a, **k):
        raise RuntimeError("ctor fail")
    monkeypatch.setattr(app_mod.threading, "Thread", BadCtor)
    stubborn = FakeProc(alive=True, dies_on_kill=False)
    app._ffmpeg_thread, app._ffmpeg_queue = object(), queue.Queue()
    app._ffmpeg_stop_event, app._ffmpeg_proc = threading.Event(), stubborn
    app._close_ffmpeg_async()
    assert len(app._record_cleanups) == 1 and app._record_cleanups[0][1] is stubborn
    assert not app._record_cleanups[0][0].is_alive()  # 마커 — prune이 proc 기준으로 처리


class TestKillRetryRateLimit:
  def test_prune_rekill_is_rate_limited(self, monkeypatch):
    # prune은 매 프레임 불릴 수 있다 — 잔존 인코더 kill이 프레임 속도로 폭주하면 안 된다
    app = make_app()
    dead = types.SimpleNamespace(is_alive=lambda: False)
    stubborn = FakeProc(alive=True, dies_on_kill=False)
    app._record_cleanups = [[dead, stubborn, 0.0]]
    now = {"t": 1000.0}
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: now["t"])
    app._prune_record_cleanups()
    app._prune_record_cleanups()  # 같은 시각 — backoff로 재kill 없음
    assert stubborn.kill_calls == 1
    now["t"] += GuiApplication.RECORD_KILL_RETRY_INTERVAL + 0.1
    app._prune_record_cleanups()
    assert stubborn.kill_calls == 2  # 간격 경과 후 재시도


class TestEncoderClosedTruth:
  def test_no_false_closed_log(self, capsys):
    # 'encoder closed'는 실제 종료 확인 후에만 — 거짓 로그는 진단을 오도한다
    app = make_app()
    app._demote_record_sched = lambda: True
    app._record_teardown = lambda th, proc: False  # 미해소
    app._record_cleanup_worker(None, FakeProc(alive=True, dies_on_kill=False))
    out = capsys.readouterr().out
    assert "encoder closed" not in out and "cleanup incomplete" in out

  def test_closed_logged_when_confirmed(self, capsys):
    app = make_app()
    app._demote_record_sched = lambda: True
    app._record_teardown = lambda th, proc: True
    app._record_cleanup_worker(None, FakeProc(alive=False))
    assert "encoder closed" in capsys.readouterr().out

  def test_demotion_failure_path_also_truthful(self, capsys):
    app = make_app()
    app._demote_record_sched = lambda: False
    app._record_cleanup_worker(None, FakeProc(alive=True, dies_on_kill=False))  # kill 불응
    out = capsys.readouterr().out
    assert "encoder closed" not in out and "cleanup incomplete" in out


class TestAppCloseLiveness:
  def test_still_alive_after_kill_is_reported_and_kept(self, capsys):
    # 앱 종료 kill 후에도 살아 있으면 거짓 '정리 완료'를 남기지 않는다
    app = make_app()
    stubborn = FakeProc(alive=True, dies_on_kill=False)
    dead = types.SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: False)
    app._record_cleanups = [[dead, stubborn, 0.0]]
    app._join_record_cleanups(timeout=0.1)
    assert "STILL ALIVE" in capsys.readouterr().out
    assert len(app._record_cleanups) == 1  # 추적 유지


class TestFailCountPolicy:
  def test_init_failure_counts_toward_cooldown(self):
    # launch/writer-start 실패도 연속 실패 쿨다운 정책에 포함 (문서-동작 일치)
    app = make_app()
    app._render_texture = types.SimpleNamespace(texture=types.SimpleNamespace(width=2, height=2))
    app._ensure_render_texture_for_recording = lambda: None
    app._new_record_path = lambda: "/tmp/x.mp4"
    app._init_ffmpeg = lambda path: False
    app.start_recording()
    assert app._record_fail_count == 1 and app._record_fail_t > 0
    assert not app._record_enabled


class TestOrphanTracking:
  """kill에 불응한 인코더가 어떤 동기 경로에서도 추적 밖으로 빠지면 안 된다."""

  def test_writer_start_rollback_tracks_stubborn_proc(self, monkeypatch, capsys):
    from pathlib import Path
    app = make_app()
    app._width, app._height, app._target_fps = 4, 4, 20
    app._record_failure_event = threading.Event()
    stubborn = FakeProc(alive=True, dies_on_kill=False)  # kill 불응
    monkeypatch.setattr(app_mod.subprocess, "Popen", lambda *a, **k: stubborn)
    class DeadThread:
      def __init__(self, *a, **k):
        pass
      def start(self):
        raise RuntimeError("no threads")
    monkeypatch.setattr(app_mod.threading, "Thread", DeadThread)
    assert app._init_ffmpeg(Path("/tmp/x.mp4")) is False
    # orphan 금지: 백로그에 등록되어 prune 재kill/앱 종료 회수 대상이 된다
    assert len(app._record_cleanups) == 1 and app._record_cleanups[0][1] is stubborn
    assert "cleanup incomplete" in capsys.readouterr().out

  def test_sync_close_tracks_unresolved_teardown(self, capsys):
    app = make_app()
    app._record_teardown = lambda th, proc: False  # 미해소
    stubborn = FakeProc(alive=True, dies_on_kill=False)
    app._ffmpeg_thread, app._ffmpeg_queue = object(), queue.Queue()
    app._ffmpeg_stop_event, app._ffmpeg_proc = threading.Event(), stubborn
    assert app.close_ffmpeg() is False
    assert len(app._record_cleanups) == 1 and app._record_cleanups[0][1] is stubborn
    out = capsys.readouterr().out
    assert "cleanup incomplete" in out and "encoder closed" not in out
    # 이후 앱 종료 join이 이 proc을 회수 시도하고 잔존을 보고한다
    app._join_record_cleanups(timeout=0.1)
    assert "STILL ALIVE" in capsys.readouterr().out

  def test_sync_close_success_logs_closed_once(self, capsys):
    app = make_app()
    app._record_teardown = lambda th, proc: True
    app._ffmpeg_thread, app._ffmpeg_queue = object(), queue.Queue()
    app._ffmpeg_stop_event, app._ffmpeg_proc = threading.Event(), FakeProc(alive=False)
    assert app.close_ffmpeg() is True
    assert capsys.readouterr().out.count("encoder closed") == 1
    assert app._record_cleanups == []

  def test_no_duplicate_tracking_for_same_proc(self):
    app = make_app()
    stubborn = FakeProc(alive=True, dies_on_kill=False)
    assert app._track_unresolved_record_proc(stubborn) is False
    assert app._track_unresolved_record_proc(stubborn) is False
    assert len(app._record_cleanups) == 1  # 같은 proc 이중 등록 금지

  def test_init_refuses_when_previous_close_unresolved(self):
    # 시작 전 safety close가 미해소면 새 인코더를 띄우지 않는다 (정책 명시)
    app = make_app()
    app._record_teardown = lambda th, proc: False
    app._ffmpeg_thread, app._ffmpeg_queue = object(), queue.Queue()
    app._ffmpeg_stop_event, app._ffmpeg_proc = threading.Event(), FakeProc(alive=True, dies_on_kill=False)
    app._width, app._height, app._target_fps = 4, 4, 20
    assert app._init_ffmpeg("/tmp/x.mp4") is False  # Popen 이전에 중단


class TestGateNonBlocking:
  def test_lock_busy_drops_without_waiting(self, monkeypatch):
    # writer가 락을 쥔 채 선점당해도 UI는 기다리지 않는다 — try-lock 실패 = 드랍
    app = make_app()
    app._ensure_record_buf_pool(16)
    monkeypatch.setattr(app_mod, "rl", types.SimpleNamespace(
      load_image_from_texture=lambda tex: types.SimpleNamespace(width=2, height=2, data=bytes(16)),
      unload_image=lambda img: None,
      ffi=types.SimpleNamespace(memmove=lambda dest, src, n: dest.__setitem__(slice(0, n), src[:n])),
    ))
    app._render_texture = types.SimpleNamespace(texture=object())
    app._capture_metrics = types.SimpleNamespace(end=lambda tok: None, flush=lambda: None)
    app._ffmpeg_queue = queue.Queue(maxsize=8)
    app._ffmpeg_enqueue_lock = threading.Lock()
    app._ffmpeg_stop_event = threading.Event()
    app._record_failure_event = threading.Event()
    app._ffmpeg_enqueue_lock.acquire()  # writer가 drain 중인 상황 재현
    try:
      app._capture_record_frame()  # 즉시 반환해야 한다 (블록 금지)
    finally:
      app._ffmpeg_enqueue_lock.release()
    assert app._record_drop_counts["gate_busy"] == 1
    assert app._ffmpeg_queue.empty()
    assert app._record_buf_pool._q.qsize() == app._record_buf_count  # buf 회수


class TestRawStdinNoNestedThread:
  # buffered stdin의 bounded-close helper(중첩 closer 스레드)는 제거됐다 — closer가
  # UI FIFO53/core5를 상속하거나 추적 밖에서 누적될 여지 자체가 없어야 한다
  def test_bounded_close_helper_removed(self):
    assert not hasattr(GuiApplication, "_close_encoder_stdin_bounded")

  def test_teardown_spawns_no_thread(self, monkeypatch):
    # teardown 경로 전체가 스레드를 만들지 않는다 — Thread 생성을 봉쇄해도 완주
    def no_threads(*a, **k):
      raise AssertionError("teardown must not spawn threads")
    monkeypatch.setattr(app_mod.threading, "Thread", no_threads)
    closed = []
    proc = FakeProc()
    proc.stdin = types.SimpleNamespace(closed=False, close=lambda: closed.append(1))
    resolved = GuiApplication._record_teardown(None, proc)
    assert resolved is True and closed == [1]

  def test_init_ffmpeg_launches_unbuffered_stdin(self, monkeypatch):
    # raw close 전제(bufsize=0)를 고정 — buffered로 돌아가면 close가 flush를 물고
    # 무한 블록할 수 있다
    from pathlib import Path
    app = make_app()
    app._width, app._height, app._target_fps = 4, 4, 20
    captured = {}
    def fake_popen(args, **kwargs):
      captured.update(kwargs)
      return FakeProc(alive=True)
    monkeypatch.setattr(app_mod.subprocess, "Popen", fake_popen)
    started = []
    class OkThread:
      def __init__(self, *a, **k):
        pass
      def start(self):
        started.append(1)
    monkeypatch.setattr(app_mod.threading, "Thread", OkThread)
    assert app._init_ffmpeg(Path("/tmp/x.mp4")) is True
    assert captured.get("bufsize") == 0 and captured.get("stdin") is subprocess.PIPE
    assert started == [1]

  def test_every_module_popen_is_unbuffered(self):
    # 공유 teardown은 stdin을 raw로 가정하고 close(2) 한 번으로 닫는다 — 이 전제는
    # 모듈의 모든 인코더 Popen(carrot _init_ffmpeg + upstream RECORD init_window)이
    # bufsize=0일 때만 성립한다. buffered stdin 재도입을 소스 레벨에서 봉쇄한다.
    tree = ast.parse(inspect.getsource(app_mod))
    popen_calls = [
      node for node in ast.walk(tree)
      if isinstance(node, ast.Call)
      and ((isinstance(node.func, ast.Attribute) and node.func.attr == "Popen")
           or (isinstance(node.func, ast.Name) and node.func.id == "Popen"))
    ]
    assert len(popen_calls) >= 2, "carrot과 upstream RECORD의 Popen 호출부가 모두 있어야 한다"
    for call in popen_calls:
      kws = {k.arg: k.value for k in call.keywords}
      buf = kws.get("bufsize")
      assert isinstance(buf, ast.Constant) and buf.value == 0, \
        f"line {call.lineno}: Popen에 bufsize=0 누락 — 공유 teardown의 raw close 전제 위반"


class TestCloseWindowIndependence:
  def test_recorder_cleanup_runs_when_window_not_ready(self, monkeypatch):
    # recorder lifecycle은 raylib window와 독립 — window가 없어도 인코더는 회수한다
    app = make_app()
    calls = []
    app.close_ffmpeg = lambda: calls.append("close_ffmpeg") or True
    app.stop_recording = lambda: calls.append("stop_recording")
    app._join_record_cleanups = lambda timeout: calls.append("join")
    monkeypatch.setattr(app_mod, "rl", types.SimpleNamespace(is_window_ready=lambda: False))
    app.close()  # window 미준비 — 녹화 정리는 수행, 텍스처 정리는 스킵
    assert calls == ["close_ffmpeg", "stop_recording", "close_ffmpeg", "join"]
