import time

from openpilot.common.swaglog import cloudlog


class CarrotPerf:
  """carrot 전용 구간별 렌더 시간 누적기.

  begin() 후 mark(name)를 순차 호출하면 직전 mark 이후 경과 시간이 name에 누적된다.
  frame_done()이 report_frames회 쌓이면 UIPERF2-<tag> 라인을 cloudlog(→rlog logMessage)로 남긴다.
  오버헤드는 mark당 time.monotonic() 1회 수준.
  """

  def __init__(self, tag: str, report_frames: int = 200):
    self._tag = tag
    self._report_frames = report_frames
    self._sums: dict[str, float] = {}
    self._count = 0
    self._t = 0.0

  def begin(self) -> None:
    self._t = time.monotonic()

  def mark(self, name: str) -> None:
    now = time.monotonic()
    self._sums[name] = self._sums.get(name, 0.0) + (now - self._t)
    self._t = now

  def frame_done(self) -> None:
    self._count += 1
    if self._count >= self._report_frames:
      avg = " ".join(f"{k}={v / self._count * 1000:.2f}" for k, v in self._sums.items())
      cloudlog.warning(f"UIPERF2-{self._tag} n={self._count} avg_ms[{avg}]")
      self._sums = {}
      self._count = 0


MODEL_PERF = CarrotPerf("model")
HUD_PERF = CarrotPerf("hud")
