"""Paths, filenames, and constants for the Carrot model selector."""
from __future__ import annotations

from pathlib import Path

# Storage locations
MODELS_DIR = Path("/data/models")
MODELS_TMP_DIR = Path("/data/models_tmp")
MODELS_BACKUP_DIR = Path("/data/models_backup")
COMPILE_STATUS_FILE = Path("/data/model_compile_status")

# Default built-in model directory (fallback when no custom model is installed)
OPENPILOT_ROOT = Path("/data/openpilot")
DEFAULT_MODEL_DIR = OPENPILOT_ROOT / "selfdrive" / "modeld" / "models"

# Remote manifest — 이중화:
#  - models_v4.json: 전체 카탈로그 (v4+ 셀렉터용 마스터, supercombo 등 신형 파일명 포함)
#  - models.json:    레거시 파일명 모델만 (v3 이하 구버전용으로 동결)
# 구버전(v3) manifest 파서는 minimum_selector_version 게이트 "이전에" 파일명을
# 검사하고 미지의 파일명이 하나라도 있으면 목록 전체를 실패시키므로, 신형
# 파일명이 든 항목은 models.json 에 절대 실으면 안 된다. openpilot-models
# 저장소의 scripts/update_models.py 가 두 파일을 자동 분리 생성·서명한다.
_MODELS_JSON_BASE = "https://raw.githubusercontent.com/happymaj11r/openpilot-models/main"
MODELS_JSON_URL = f"{_MODELS_JSON_BASE}/models_v4.json"
# models_v4.json fetch 실패 시(파일 부재/일시 오류) 레거시 manifest 로 폴백.
MODELS_JSON_FALLBACK_URL = f"{_MODELS_JSON_BASE}/models.json"
# str.startswith가 튜플을 받으므로 downloader의 검증 로직 변경 없이 동작.
# releases/download 프리픽스는 95MB 초과 파일(GitHub raw 호스팅 불가, 예: Giga의
# driving_vision.onnx 122MB)이 Release 에셋으로 배포되는 경우를 위해 필요.
ALLOWED_URL_PREFIX = (
    "https://raw.githubusercontent.com/happymaj11r/openpilot-models/",
    "https://github.com/happymaj11r/openpilot-models/releases/download/",
)

# Allowed onnx filenames for download (allowlist)
ALLOWED_ONNX_FILES = frozenset({
    "driving_vision.onnx",
    "driving_policy.onnx",
    "driving_on_policy.onnx",
    "driving_off_policy.onnx",
    "driving_supercombo.onnx",
})

# Base names that we compile (.onnx → _tinygrad.pkl + _metadata.pkl)
VISION_BASE = "driving_vision"
ON_POLICY_BASE = "driving_on_policy"
POLICY_BASE = "driving_policy"
OFF_POLICY_BASE = "driving_off_policy"

# New-architecture (lebowski) single-onnx model: compile_modeld.py bundles
# metadata + model JIT + per-resolution warp JITs into one pkl that the
# upstream modeld engine loads directly.
SUPERCOMBO_BASE = "driving_supercombo"
SUPERCOMBO_PKL_NAME = "driving_tinygrad.pkl"

# Env var honored by selfdrive/modeld/helpers.py::modeld_pkl_path() to load
# the unified pkl from a custom directory instead of the built-in models dir.
MODELD_MODELS_DIR_ENV = "MODELD_MODELS_DIR"

# Params keys
PARAM_DRIVING_MODEL_NAME = "DrivingModelName"
PARAM_PENDING_MODEL_NAME = "PendingModelName"

# tinygrad compile flags (must match selfdrive/modeld/SConscript)
TINYGRAD_COMPILE_ENV_QCOM = {
    "DEV": "QCOM",
    "FLOAT16": "1",
    "NOLOCALS": "1",
    "JIT_BATCH_SIZE": "0",
    "IMAGE": "1",
    "OPENPILOT_HACKS": "1",
}
TINYGRAD_COMPILE_ENV_FALLBACK = {
    "DEV": "CPU:LLVM",
    "THREADS": "0",
}

# Model ID validation (matches model_manager.cc isValidModelId())
MODEL_ID_REGEX = r"^[A-Za-z0-9_\-\s]{1,64}$"
