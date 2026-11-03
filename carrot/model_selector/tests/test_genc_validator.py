"""Gen C artifact-set validation: model pkl + per-camera warp pkls + metadata json."""
from __future__ import annotations

from pathlib import Path

from carrot.model_selector import validator
from carrot.model_selector.config import GENC_METADATA_NAME, GENC_PKL_NAME, GENC_WARP_PKL_FMT, SUPERCOMBO_PKL_NAME


def _write(base: Path, name: str, content: bytes = b"x") -> None:
  (base / name).write_bytes(content)


def _full_genc_set(base: Path) -> None:
  _write(base, GENC_PKL_NAME)
  _write(base, GENC_METADATA_NAME, b"{}")
  for w, h in validator.GENC_CAMERA_CONFIGS:
    _write(base, GENC_WARP_PKL_FMT.format(w=w, h=h))


def test_full_set_is_valid(tmp_path: Path):
  _full_genc_set(tmp_path)
  assert validator.has_genc(tmp_path)
  assert validator.is_valid_genc_model_dir(tmp_path)
  assert validator.is_valid_model_dir(tmp_path)
  assert validator.describe(tmp_path) == f"{tmp_path}: genc"


def test_missing_warp_or_metadata_is_invalid(tmp_path: Path):
  _full_genc_set(tmp_path)
  (tmp_path / GENC_WARP_PKL_FMT.format(w=1344, h=760)).unlink()
  assert not validator.has_genc(tmp_path)
  _full_genc_set(tmp_path)
  (tmp_path / GENC_METADATA_NAME).unlink()
  assert not validator.has_genc(tmp_path)


def test_empty_artifact_is_invalid(tmp_path: Path):
  _full_genc_set(tmp_path)
  _write(tmp_path, GENC_PKL_NAME, b"")
  assert not validator.has_genc(tmp_path)
  assert not validator.is_valid_model_dir(tmp_path)


def test_genc_wins_over_supercombo_in_describe(tmp_path: Path):
  _full_genc_set(tmp_path)
  _write(tmp_path, SUPERCOMBO_PKL_NAME)
  assert validator.describe(tmp_path) == f"{tmp_path}: genc"


def test_camera_configs_match_installer_devices():
  # installer.CAMERA_CONFIGS derives from these; the Gen C warp set must cover the same cameras.
  from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
  assert validator.GENC_CAMERA_CONFIGS == ((_ar_ox_fisheye.width, _ar_ox_fisheye.height),
                                           (_os_fisheye.width, _os_fisheye.height))


def test_missing_dir(tmp_path: Path):
  missing = tmp_path / "nope"
  assert not validator.is_valid_genc_model_dir(missing)
  assert validator.describe(missing) == f"{missing}: missing"
