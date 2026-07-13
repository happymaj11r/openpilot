"""carrot 전용: 녹화 캡처의 비동기 GPU readback (더블 버퍼 PBO).

동기 readback(rl.load_image_from_texture → glGetTexImage/glReadPixels)은 GPU가
해당 프레임 렌더를 끝낼 때까지 CPU가 서서 기다린다 — route 0000041f 실측에서
screenCapture wall p95 25.1ms / max 74.0ms(기준 20/40ms), C 구간 UI FPS
15.7(기준 17)의 주원인. PBO(GL_PIXEL_PACK_BUFFER)를 바인드한 glReadPixels는
GPU→PBO DMA 전송만 시작하고 즉시 반환하며, 한 캡처 간격(~150ms, 20fps에서
3프레임당 1캡처) 뒤의 다음 캡처에서 map하면 전송이 이미 끝나 있어 stall 없이
memcpy 한 번으로 프레임을 얻는다 — 산출 영상은 1캡처만큼 지연된 프레임이지만
균일 시프트라 화면 연속성에는 영향이 없다.

GL 함수는 pyray가 노출하지 않으므로 cffi dlopen으로 직접 얻는다(application.py
의 libc dlopen과 같은 관행). READ framebuffer 바인딩만 rlgl(rl_enable_
framebuffer)로 호출자가 수행한다. 실패 계약: 어떤 GL 오류/미지원(ES2 컨텍스트
등)이든 ok=False로 수렴하고 예외를 던지지 않는다 — 호출자는 기존 동기
readback으로 영구 폴백한다 (fail-closed, 같은 프레임에서 재시도)."""
import cffi

GL_PIXEL_PACK_BUFFER = 0x88EB
GL_STREAM_READ = 0x88E1
GL_RGBA = 0x1908
GL_UNSIGNED_BYTE = 0x1401
GL_MAP_READ_BIT = 0x0001
GL_NO_ERROR = 0

_GL_CDEF = """
typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef int GLsizei;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef ssize_t GLsizeiptr;
typedef ssize_t GLintptr;
void glGenBuffers(GLsizei n, GLuint *buffers);
void glDeleteBuffers(GLsizei n, const GLuint *buffers);
void glBindBuffer(GLenum target, GLuint buffer);
void glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
void glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height,
                  GLenum format, GLenum type, void *pixels);
void *glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length,
                       GLbitfield access);
GLboolean glUnmapBuffer(GLenum target);
GLenum glGetError(void);
"""

# 기기(agnos)는 GLES, PC는 데스크톱 GL — 둘 다 core 심볼을 export하므로
# 이름 순회로 충분하다. 전부 실패하면 동기 폴백 (기존 동작과 동일)
_GL_LIB_NAMES = ("libGLESv2.so.2", "libGLESv2.so", "libGL.so.1", "libGL.so")

_ffi = None
_gl = None
_load_attempted = False


def _load_gl():
  """GL 라이브러리 1회 dlopen — 실패하면 (ffi, None). 결과는 프로세스 캐시."""
  global _ffi, _gl, _load_attempted
  if _load_attempted:
    return _ffi, _gl
  _load_attempted = True
  try:
    ffi = cffi.FFI()
    ffi.cdef(_GL_CDEF)
  except Exception:
    return None, None
  _ffi = ffi
  for name in _GL_LIB_NAMES:
    try:
      _gl = ffi.dlopen(name)
      break
    except OSError:
      continue
  return _ffi, _gl


class PboReadback:
  """더블 버퍼 PBO 비동기 readback 상태기계 — GL 컨텍스트 스레드(렌더 루프) 전용.

  issue()는 현재 READ framebuffer 픽셀을 PBO로 DMA 시작만 하고 반환한다
  (framebuffer 바인딩은 호출자가 rlgl로 수행). retrieve_into()는 직전 issue의
  PBO를 map해 dst로 복사한다 — 대기 프레임이 없으면 False(파이프라인 프라이밍,
  드랍이 아니다). 전 메서드 no-throw: GL 오류는 ok=False로 수렴하고 이후 호출은
  no-op — 호출자가 동기 경로로 폴백한다."""

  def __init__(self, ffi, gl, data_size: int):
    self._ffi = ffi
    self._gl = gl
    self.data_size = data_size
    self.ok = False
    self._pending: int | None = None  # 직전 issue가 쓴 PBO 인덱스 (map 대상)
    self._next = 0                    # 다음 issue가 쓸 PBO 인덱스 (교대)
    self._ids = None
    try:
      self._drain_errors()
      ids = ffi.new("GLuint[2]")
      gl.glGenBuffers(2, ids)
      self._ids = ids
      for i in (0, 1):
        gl.glBindBuffer(GL_PIXEL_PACK_BUFFER, ids[i])
        gl.glBufferData(GL_PIXEL_PACK_BUFFER, data_size, ffi.NULL, GL_STREAM_READ)
      gl.glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
      # 할당까지의 오류를 여기서 확정 — PBO가 없는 ES2 컨텍스트는 PACK_BUFFER
      # 바인딩부터 INVALID_ENUM이 나므로 생성 시점에 폴백이 결정된다
      self.ok = gl.glGetError() == GL_NO_ERROR and ids[0] != 0 and ids[1] != 0
      if not self.ok:
        self.release()
    except Exception:
      self.ok = False

  def _drain_errors(self) -> None:
    # raylib이 남겨 둔 이전 GL 오류가 우리 검증을 오염시키지 않게 비운다 (bounded)
    for _ in range(8):
      if self._gl.glGetError() == GL_NO_ERROR:
        break

  def issue(self, width: int, height: int) -> bool:
    """현재 READ framebuffer → PBO 비동기 전송 시작 (논블로킹)."""
    if not self.ok:
      return False
    try:
      gl = self._gl
      gl.glBindBuffer(GL_PIXEL_PACK_BUFFER, self._ids[self._next])
      # PBO 바인드 상태의 pixels 인자는 포인터가 아니라 PBO 안 오프셋(0)이다
      gl.glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                      self._ffi.cast("void *", 0))
      gl.glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
      if gl.glGetError() != GL_NO_ERROR:
        self.ok = False
        return False
      self._pending = self._next
      self._next = 1 - self._next
      return True
    except Exception:
      self.ok = False
      return False

  def retrieve_into(self, dst) -> bool:
    """직전 issue 프레임을 dst(재사용 bytearray)로 복사. 한 캡처 간격 전에 시작한
    DMA라 map이 GPU를 기다리지 않는다."""
    if not self.ok or self._pending is None:
      return False
    try:
      gl = self._gl
      gl.glBindBuffer(GL_PIXEL_PACK_BUFFER, self._ids[self._pending])
      ptr = gl.glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, self.data_size, GL_MAP_READ_BIT)
      got = ptr != self._ffi.NULL
      if got:
        self._ffi.memmove(dst, ptr, self.data_size)
        gl.glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
      gl.glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
      self._pending = None
      if not got or gl.glGetError() != GL_NO_ERROR:
        self.ok = False
        return False
      return True
    except Exception:
      self.ok = False
      return False

  def reset(self) -> None:
    """세션 시작 시 이전 세션의 잔여 프레임을 버린다 — 오래된 화면이 새 파일의
    첫 프레임으로 들어가지 않게 한다 (GL 호출 없음, 어느 스레드든 안전)."""
    self._pending = None

  def release(self) -> None:
    """PBO 반납 (GL 컨텍스트 스레드에서만 — 크기 세대 교체/폴백 시). 실패해도
    조용히 — 프로세스 종료가 최종 회수자다."""
    self.ok = False
    ids, self._ids = self._ids, None
    if ids is None:
      return
    try:
      self._gl.glDeleteBuffers(2, ids)
    except Exception:
      pass


def create(data_size: int):
  """PBO readback 생성 — GL 미지원/오류 환경이면 None (호출자는 동기 폴백)."""
  ffi, gl = _load_gl()
  if ffi is None or gl is None:
    return None
  pbo = PboReadback(ffi, gl, data_size)
  return pbo if pbo.ok else None
