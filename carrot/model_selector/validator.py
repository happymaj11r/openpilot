"""Validate an installed custom model directory.

A directory is considered valid when it contains either:
  - new architecture (supercombo/lebowski): a single unified
    driving_tinygrad.pkl (metadata + model JIT + warp JITs bundled inside), or
  - legacy split architecture:
      driving_vision_tinygrad.pkl + driving_vision_metadata.pkl (required)
      driving_on_policy_* or driving_policy_* (either acceptable)
    The off-policy pair is optional and only activates the 3-model architecture.
"""
from __future__ import annotations

from pathlib import Path

from .config import GENC_METADATA_NAME, GENC_PKL_NAME, GENC_WARP_PKL_FMT, SUPERCOMBO_PKL_NAME

# Camera resolutions the Gen C installer compiles warps for (tici/tizi, mici).
GENC_CAMERA_CONFIGS = ((1928, 1208), (1344, 760))


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _pair_exists(base: Path, stem: str) -> bool:
    pkl = base / f"{stem}_tinygrad.pkl"
    meta = base / f"{stem}_metadata.pkl"
    return pkl.exists() and meta.exists() and pkl.stat().st_size > 0 and meta.stat().st_size > 0


def has_vision(base: Path) -> bool:
    return _pair_exists(base, "driving_vision")


def has_on_policy(base: Path) -> bool:
    return _pair_exists(base, "driving_on_policy")


def has_policy(base: Path) -> bool:
    return _pair_exists(base, "driving_policy")


def has_off_policy(base: Path) -> bool:
    return _pair_exists(base, "driving_off_policy")


def has_supercombo(base: Path) -> bool:
    pkl = base / SUPERCOMBO_PKL_NAME
    return pkl.exists() and pkl.stat().st_size > 0


def genc_warp_pkl_names() -> tuple[str, ...]:
    return tuple(GENC_WARP_PKL_FMT.format(w=w, h=h) for w, h in GENC_CAMERA_CONFIGS)


def has_genc(base: Path) -> bool:
    """Gen C artifact set: model JIT pkl + a warp JIT pkl per camera + metadata json."""
    if not _nonempty(base / GENC_PKL_NAME) or not _nonempty(base / GENC_METADATA_NAME):
        return False
    return all(_nonempty(base / name) for name in genc_warp_pkl_names())


def is_valid_legacy_model_dir(base: Path) -> bool:
    if not base.exists() or not base.is_dir():
        return False
    return has_vision(base) and (has_on_policy(base) or has_policy(base))


def is_valid_supercombo_model_dir(base: Path) -> bool:
    if not base.exists() or not base.is_dir():
        return False
    return has_supercombo(base)


def is_valid_genc_model_dir(base: Path) -> bool:
    if not base.exists() or not base.is_dir():
        return False
    return has_genc(base)


def is_valid_model_dir(base: Path) -> bool:
    return is_valid_genc_model_dir(base) or is_valid_supercombo_model_dir(base) or is_valid_legacy_model_dir(base)


def describe(base: Path) -> str:
    """Human-readable status string for logs / API responses."""
    if not base.exists():
        return f"{base}: missing"
    if has_genc(base):
        return f"{base}: genc"
    if has_supercombo(base):
        return f"{base}: supercombo"
    parts = []
    parts.append("vision" if has_vision(base) else "NO_VISION")
    if has_on_policy(base):
        parts.append("on_policy")
    elif has_policy(base):
        parts.append("policy")
    else:
        parts.append("NO_POLICY")
    if has_off_policy(base):
        parts.append("off_policy")
    return f"{base}: " + "+".join(parts)
