"""carrot 전용: Params 변경 감지 게이트.

Params.put()은 임시파일을 파라미터 디렉토리로 rename하며 끝나므로, 디렉토리의
mtime이 바뀌었다는 것은 어떤 파라미터든 값이 바뀌었다는 뜻이다. 이를 이용해
"바뀌었을 때만" 파일 재읽기를 허용한다 — 매 프레임 읽기의 I/O 비용과
고정 TTL의 반영 지연을 동시에 피한다.
"""
import os

from openpilot.common.params import Params

_params_dir: str | None = None


def _dir_mtime() -> int:
  global _params_dir
  if _params_dir is None:
    _params_dir = Params().get_param_path()
  try:
    return os.stat(_params_dir).st_mtime_ns
  except OSError:
    return -1


class ParamsRefreshGate:
  """should_refresh()가 True를 돌려줄 때만 Params를 다시 읽는다.

  비용은 프레임당 stat() 한 번 수준이라 매 프레임 불러도 된다. min_interval은
  다른 데몬이 파라미터를 고빈도로 갱신할 때 재읽기가 프레임마다 일어나는 것을
  막는 하한. 첫 호출은 항상 True(초기 로드).
  """

  def __init__(self, min_interval: float = 0.5):
    self._min_interval = min_interval
    self._mtime = -2
    self._next_check = 0.0

  def should_refresh(self, now: float) -> bool:
    if now < self._next_check:
      return False
    mtime = _dir_mtime()
    if mtime == self._mtime:
      return False
    self._mtime = mtime
    self._next_check = now + self._min_interval
    return True
