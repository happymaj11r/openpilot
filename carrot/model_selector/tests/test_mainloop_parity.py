"""check_contracts mainloop-parity: mirror engines must keep upstream main()'s call surface."""
from __future__ import annotations

from carrot.model_selector import check_contracts as cc

UPSTREAM = """\
def main():
  sm = SubMaster(['carState', 'livePose'])
  DH = DesireHelper(fingerprint=1, kick_supported=True)
  DH.update(cs, indicator_valid=True, turn_pose=None)
  if DH.manual_turn_indicator_enabled:
    modelv2_send.modelV2.meta.assistantLaneSignalId = DH.assistant_lane.signal_id
"""
STALE_MIRROR = """\
def main():
  sm = SubMaster(['carState'])
  DH = DesireHelper(kick_supported=True)
  DH.update(cs, indicator_valid=True)
"""


def _arrange(tmp_path, monkeypatch, mirror_source, allowed=None):
  (tmp_path / "modeld.py").write_text(UPSTREAM)
  (tmp_path / "genc_modeld.py").write_text(mirror_source)
  monkeypatch.setattr(cc, "MODELD_DIR", tmp_path)
  monkeypatch.setattr(cc, "SELECTOR_DIR", tmp_path)
  monkeypatch.setattr(cc, "MAINLOOP_MIRRORS", {"genc_modeld.py": allowed or {}})


def test_current_tree_has_no_mainloop_drift():
  assert cc.check_mainloop_parity() is None


def test_surface_extraction(tmp_path):
  source = tmp_path / "loop.py"
  source.write_text(UPSTREAM)
  surface = cc._mainloop_surface(source)
  assert surface["SubMaster"] == {"carState", "livePose"}
  assert surface["DesireHelper"] == {"fingerprint", "kick_supported"}
  assert surface["DH.update"] == {"indicator_valid", "turn_pose"}
  assert surface["meta"] == {"assistantLaneSignalId"}
  assert surface["DH.attrs"] == {"update", "manual_turn_indicator_enabled", "assistant_lane"}


def test_identical_mirror_passes(tmp_path, monkeypatch):
  _arrange(tmp_path, monkeypatch, UPSTREAM)
  assert cc.check_mainloop_parity() is None


def test_missing_port_is_reported(tmp_path, monkeypatch):
  _arrange(tmp_path, monkeypatch, STALE_MIRROR)
  message = cc.check_mainloop_parity()
  assert message is not None
  for item in ("livePose", "fingerprint", "turn_pose", "assistantLaneSignalId", "manual_turn_indicator_enabled"):
    assert item in message


def test_allowlisted_gap_passes(tmp_path, monkeypatch):
  _arrange(tmp_path, monkeypatch, STALE_MIRROR, allowed={
    "SubMaster": {"livePose"}, "DesireHelper": {"fingerprint"}, "DH.update": {"turn_pose"},
    "meta": {"assistantLaneSignalId"}, "DH.attrs": {"manual_turn_indicator_enabled", "assistant_lane"}})
  assert cc.check_mainloop_parity() is None


def test_unrecognisable_upstream_loop_fails_closed(tmp_path, monkeypatch):
  _arrange(tmp_path, monkeypatch, STALE_MIRROR)
  (tmp_path / "modeld.py").write_text("def main():\n  pass\n")
  assert "main()" in (cc.check_mainloop_parity() or "")
