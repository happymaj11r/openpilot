"""Tiny hook called once from `system/manager/manager.py::main()` so the
upstream patch stays a single line:

    from openpilot.carrot.model_selector.boot_compile import run as _ms_boot
    _ms_boot()

All heavy lifting lives in `installer`.  This wrapper exists so the patch
site doesn't have to import anything heavy (e.g. tinygrad) unless a pending
model is present or the installed model needs a recompile.
"""
from __future__ import annotations

from openpilot.common.swaglog import cloudlog

from .config import COMPILE_ENV_STAMP_NAME, COMPILE_ENV_TAG, MODELS_DIR, MODELS_TMP_DIR


def run() -> None:
    try:
        if MODELS_TMP_DIR.exists():
            from .installer import compile_pending
            compile_pending()
            return

        # 설치된 커스텀 모델이 구 컴파일 환경(다른 tinygrad/pkl 포맷)에서
        # 빌드된 경우 보존된 onnx 로 부팅 시 자동 재컴파일한다.
        # 스탬프가 현재 태그와 일치하면 heavy import 없이 바로 통과.
        stamp = MODELS_DIR / COMPILE_ENV_STAMP_NAME
        if MODELS_DIR.is_dir() and (not stamp.is_file() or stamp.read_text().strip() != COMPILE_ENV_TAG):
            from .installer import recompile_stale_if_needed
            recompile_stale_if_needed()
    except Exception as e:
        # Never crash manager on install errors — installer already restores
        # the backup and wipes the tmp dir on failure.
        cloudlog.error(f"model_selector boot_compile: {e}")
