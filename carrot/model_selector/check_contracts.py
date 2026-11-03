#!/usr/bin/env python3
"""모델 셀렉터 ↔ upstream 계약 점검.

upstream 커밋(특히 자동 체리픽)이 셀렉터가 의존하는 계약을 깨뜨렸는지 검사한다.
체리픽/머지 후 실행해서 전부 PASS 인지 확인할 것 — FAIL 은 셀렉터 코드를 함께
적응시켜야 한다는 신호다 (자동 수정 아님).

용법 (어디서든):
    python3 carrot/model_selector/check_contracts.py                    # 계약 점검
    python3 carrot/model_selector/check_contracts.py --sync-baselines  # 미러 리뷰 후 스냅샷 갱신

표준 라이브러리만 사용한다 — openpilot 빌드/무거운 의존성 불필요.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MODELD_DIR = REPO_ROOT / "openpilot" / "selfdrive" / "modeld"
SELECTOR_DIR = REPO_ROOT / "carrot" / "model_selector"
GENC_DIR = SELECTOR_DIR / "genc"
GENC_TINYGRAD_DIR = GENC_DIR / "tinygrad_repo"

# installer.py 가 subprocess 로 실행하는 스크립트들 (저장소 상대 경로 계약)
REQUIRED_SCRIPTS = (
    MODELD_DIR / "get_model_metadata.py",
    MODELD_DIR / "compile_modeld.py",
    REPO_ROOT / "tinygrad_repo" / "examples" / "openpilot" / "compile3.py",
    SELECTOR_DIR / "compile_legacy_warp.py",
    # Gen C: 벤더 tinygrad 로 실행하는 컴파일/프로브 스크립트
    GENC_DIR / "onnx_io.py",
    GENC_TINYGRAD_DIR / "examples" / "openpilot" / "compile_onnx.py",
    GENC_TINYGRAD_DIR / "examples" / "openpilot" / "compile_warp.py",
    GENC_TINYGRAD_DIR / "examples" / "openpilot" / "helpers.py",
    GENC_TINYGRAD_DIR / "PIN",
)

# genc/runtime.py · genc/compile.py 가 벤더 tinygrad 에서 쓰는 API (파일: 정의 문자열)
GENC_VENDOR_API = {
    "tinygrad/engine/realize.py": ("def lower_and_compile", "def compile_linear"),
    "tinygrad/engine/jit.py": ("class CapturedJit",),
    "tinygrad/uop/ops.py": ("def from_buffer",),
    "tinygrad/nn/onnx.py": ("class OnnxPBParser", "def _parse_ValueInfoProto", "def _parse_StringStringEntryProto"),
    "examples/openpilot/helpers.py": ("def load_pickle", "out_of_band"),
    "examples/openpilot/compile_onnx.py": ("--out-of-band", "'input_specs'", "'output_specs'"),
    "examples/openpilot/compile_warp.py": ("--layout", "'yuv420'", "--frames", "'input_specs'"),
}

# installer._compile_supercombo 가 조립하는 compile_modeld CLI 인자 계약
REQUIRED_CLI_FLAGS = ("--model-size", "--camera-resolutions", "--onnx", "--output", "--frame-skip")

# carrot_modeld 가 위치 인자로 호출하는 upstream 함수 시그니처 계약
EXPECTED_SIGNATURES = {
    "fill_model_msg": ["msg", "net_output_data", "action", "publish_state", "vipc_frame_id",
                       "vipc_frame_id_extra", "frame_id", "frame_drop", "timestamp_eof",
                       "model_execution_time", "valid"],
    "fill_pose_msg": ["msg", "net_output_data", "vipc_frame_id", "vipc_dropped_frames",
                      "timestamp_eof", "live_calib_seen"],
    "fill_driving_model_data": ["msg", "modelv2_send"],
}

# installer/_validate 와 carrot_modeld 가 참조하는 상수 계약
REQUIRED_MODEL_CONSTANTS = ("MODEL_RUN_FREQ", "MODEL_CONTEXT_FREQ", "DESIRE_LEN",
                            "TRAFFIC_CONVENTION_LEN", "FEATURE_LEN")

REQUIRED_COMPILE_ENV_PATHS = {
    "tinygrad_repo",
    "carrot/model_selector/genc",
    "carrot/model_selector/config.py",
    "carrot/model_selector/installer.py",
    "carrot/model_selector/compile_legacy_warp.py",
    "carrot/model_selector/carrot_modeld.py",
    "openpilot/selfdrive/modeld/SConscript",
    "openpilot/selfdrive/modeld/compile_modeld.py",
    "openpilot/selfdrive/modeld/compile_dm_warp.py",
    "openpilot/selfdrive/modeld/get_model_metadata.py",
    "openpilot/selfdrive/modeld/helpers.py",
    "openpilot/selfdrive/modeld/constants.py",
    "openpilot/selfdrive/modeld/modeld.py",
    "openpilot/selfdrive/modeld/dmonitoringmodeld.py",
    "openpilot/common/file_chunker.py",
}

# carrot_modeld / carrot_parse_model_outputs 가 "미러링"하는 upstream 파일들.
# fill_model_msg 처럼 임포트하는 파일과 달리, 이 둘의 로직 변경(예: 커밋
# 67b6f17d44 의 has_wide_camera 분기)은 자동 반영되지 않고 사람이 판단해
# 이식해야 한다 — 시그니처가 안 바뀌면 다른 검사로는 잡히지 않는다.
# 마지막 미러 리뷰 시점의 스냅샷(upstream_baseline/)과 비교해 변경을 알린다.
BASELINE_DIR = Path(__file__).resolve().parent / "upstream_baseline"
MIRRORED_UPSTREAM_FILES = {
    MODELD_DIR / "modeld.py": BASELINE_DIR / "modeld.py.baseline",
    MODELD_DIR / "parse_model_outputs.py": BASELINE_DIR / "parse_model_outputs.py.baseline",
}


def _func_params(path: Path, name: str) -> list[str] | None:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    return None


def check_script_paths() -> str | None:
    missing = [str(p.relative_to(REPO_ROOT)) for p in REQUIRED_SCRIPTS if not p.is_file()]
    return f"missing: {missing}" if missing else None


def check_helpers_hook() -> str | None:
    """helpers.modeld_pkl_path() 가 MODELD_MODELS_DIR 오버라이드를 반영하는지."""
    probe = "/tmp/__ms_contract_check__"
    old = os.environ.get("MODELD_MODELS_DIR")
    os.environ["MODELD_MODELS_DIR"] = probe
    try:
        from openpilot.selfdrive.modeld.helpers import modeld_pkl_path
        got = Path(modeld_pkl_path(False))
        big = Path(modeld_pkl_path(True))
        if got.parent != Path(probe) or big.parent != Path(probe):
            return f"MODELD_MODELS_DIR ignored — hook lost (got {got}, big {big})"
        big_ok = big.name.startswith("big_driving_") and big.name.endswith("_tinygrad.pkl")
        if got.name != "driving_tinygrad.pkl" or not big_ok:
            return f"pkl naming changed: {got.name}, {big.name}"
        return None
    finally:
        if old is None:
            os.environ.pop("MODELD_MODELS_DIR", None)
        else:
            os.environ["MODELD_MODELS_DIR"] = old


def check_compile_modeld_cli() -> str | None:
    src = (MODELD_DIR / "compile_modeld.py").read_text()
    missing = [f for f in REQUIRED_CLI_FLAGS if f"'{f}'" not in src and f'"{f}"' not in src]
    return f"argparse flags missing: {missing}" if missing else None


def check_metadata_naming() -> str | None:
    src = (MODELD_DIR / "get_model_metadata.py").read_text()
    if "_metadata.pkl" not in src:
        return "get_model_metadata.py no longer derives {stem}_metadata.pkl output naming"
    return None


def check_pkl_format_pair() -> str | None:
    """installer 가 컴파일한 pkl 을 modeld 가 읽는 직렬화 쌍이 유지되는지."""
    compile_src = (MODELD_DIR / "compile_modeld.py").read_text()
    modeld_src = (MODELD_DIR / "modeld.py").read_text()
    if "dump_oob" not in compile_src:
        return "compile_modeld.py no longer writes with dump_oob"
    # upstream may pass an explicit override first (pkl_path or modeld_pkl_path(...)),
    # so allow anything up to the modeld_pkl_path fallback inside the same call.
    if not re.search(r"load_oob\(open_file_chunked\([^)]*modeld_pkl_path", modeld_src):
        return "modeld.py loader changed (expected load_oob(open_file_chunked(... modeld_pkl_path(...))))"
    return None


def check_tinygrad_pickle_compat() -> str | None:
    """The compiler must use tinygrad's native Buffer pickle implementation."""
    compile_src = (MODELD_DIR / "compile_modeld.py").read_text()
    device_src = (REPO_ROOT / "tinygrad_repo" / "tinygrad" / "device.py").read_text()
    if "_patch_tinygrad_buffer_reduce" in compile_src:
        return "obsolete Buffer pickle monkeypatch restored"
    if "def __reduce_ex__" not in device_src:
        return "vendored tinygrad no longer provides native Buffer pickle support"
    return None


def check_sconscript_flags() -> str | None:
    from carrot.model_selector.config import TINYGRAD_COMPILE_ENV_QCOM
    src = (MODELD_DIR / "SConscript").read_text()
    if "if arch == 'larch64':" not in src or "probe_devices" in src:
        return "TICI backend must be selected from the build architecture"
    m = re.search(r"DEV=\{tg_backend\}\s+([A-Z0-9_= ]+)'", src)
    if not m:
        return "cannot locate QCOM tg_flags line in SConscript"
    sconscript = dict(kv.split("=", 1) for kv in m.group(1).split())
    expected = {k: v for k, v in TINYGRAD_COMPILE_ENV_QCOM.items() if k != "DEV"}
    if sconscript != expected:
        return f"QCOM flags diverged: SConscript={sconscript} config={expected}"
    return None


def check_compile_env_guards() -> str | None:
    """Old tinygrad pickles must be rejected before either model loader runs."""
    from carrot.model_selector.config import _COMPILE_ENV_PATHS

    problems = []
    missing = sorted(REQUIRED_COMPILE_ENV_PATHS - set(_COMPILE_ENV_PATHS))
    if missing:
        problems.append(f"compiler fingerprint missing paths: {missing}")

    runner = (REPO_ROOT / "carrot/model_selector/modeld_runner.py").read_text()
    if "model_compile_env_is_current(CUSTOM_MODELS_DIR)" not in runner:
        problems.append("modeld_runner does not reject stale custom PKLs before load")

    launcher = (REPO_ROOT / "launch_chffrplus.sh").read_text()
    if "from carrot.model_selector.config import compile_env_tag" not in launcher:
        problems.append("built-in model stamp does not use the common compiler fingerprint")
    if "HEAD:openpilot/selfdrive/modeld HEAD:tinygrad_repo" in launcher:
        problems.append("built-in model stamp still hashes the self-changing modeld tree")

    release = (REPO_ROOT / "release/build_carrot.sh").read_text()
    build_at = release.find("prepare_prebuilt_models\n")
    delete_at = release.find('rm -f -- openpilot/selfdrive/modeld/models/*.onnx')
    if build_at < 0 or delete_at < 0 or build_at >= delete_at:
        problems.append("prebuilt model compilation must happen before ONNX removal")
    if 'validate_prebuilt_models "$FINAL_MODEL_TAG"' not in release:
        problems.append("final prebuilt compiler stamp is not verified")

    return "; ".join(problems) if problems else None


def check_fill_model_msg_signatures() -> str | None:
    path = MODELD_DIR / "fill_model_msg.py"
    bad = []
    for name, expected in EXPECTED_SIGNATURES.items():
        actual = _func_params(path, name)
        if actual != expected:
            bad.append(f"{name}: expected {expected}, got {actual}")
    return "; ".join(bad) if bad else None


def check_model_constants() -> str | None:
    src = (MODELD_DIR / "constants.py").read_text()
    missing = [c for c in REQUIRED_MODEL_CONSTANTS if not re.search(rf"^\s*{c}\s*=", src, re.M)]
    return f"ModelConstants missing: {missing}" if missing else None


def check_modeld_mirror() -> str | None:
    stale = []
    for current, baseline in MIRRORED_UPSTREAM_FILES.items():
        if not baseline.is_file():
            stale.append(f"{baseline.name} 없음")
        elif current.read_bytes() != baseline.read_bytes():
            stale.append(current.name)
    if stale:
        return ("마지막 미러 리뷰 이후 변경됨: " + ", ".join(stale)
                + " — upstream_baseline/ 스냅샷과 diff 해서 carrot_modeld.py /"
                  " carrot_parse_model_outputs.py / genc_modeld.py 에 필요한 로직을 이식한 뒤"
                  " `check_contracts.py --sync-baselines` 로 스냅샷을 갱신할 것")
    return None


def check_genc_engine() -> str | None:
    """Gen C 엔진: 벤더 tinygrad 핀·API, 설치/실행 분기 배선, 셀렉터 버전."""
    problems = []
    pin_file = GENC_TINYGRAD_DIR / "PIN"
    pin = pin_file.read_text().split()[0] if pin_file.is_file() else ""
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        problems.append("genc/tinygrad_repo/PIN 이 40자리 커밋 해시로 시작하지 않음")
    for rel, needles in GENC_VENDOR_API.items():
        path = GENC_TINYGRAD_DIR / rel
        if not path.is_file():
            problems.append(f"벤더 tinygrad 파일 없음: {rel}")
            continue
        src = path.read_text()
        missing = [n for n in needles if n not in src]
        if missing:
            problems.append(f"{rel}: {missing} 없음 (runtime/compile 포팅 재검토)")

    # 엔진 두 벌이 한 프로세스에 섞이면 안 된다: genc 런타임은 벤더 활성화를 먼저 하고,
    # upstream modeld(시스템 tinygrad)를 import 하지 않는다.
    runtime = (GENC_DIR / "runtime.py").read_text()
    if "activate_tinygrad()" not in runtime.split("from tinygrad")[0]:
        problems.append("genc/runtime.py 가 tinygrad import 전에 activate_tinygrad() 를 호출하지 않음")
    engine = (SELECTOR_DIR / "genc_modeld.py").read_text()
    if "from .genc.runtime import ModelState" not in engine.split("import openpilot.cereal")[0]:
        problems.append("genc_modeld.py 첫 import 가 genc.runtime(벤더 활성화)이 아님")
    if re.search(r"^\s*from openpilot\.selfdrive\.modeld\.modeld import", engine, re.M):
        problems.append("genc_modeld.py 가 upstream modeld.py(시스템 tinygrad)를 import 함")
    for name in ("get_lat_smooth_seconds_dynamic", "get_action_from_model"):
        if _func_params(MODELD_DIR / "modeld.py", name) != _func_params(SELECTOR_DIR / "genc_modeld.py", name):
            problems.append(f"{name} 시그니처가 upstream modeld.py 와 다름")

    installer = (SELECTOR_DIR / "installer.py").read_text()
    if "_compile_supercombo_by_generation(MODELS_TMP_DIR, env)" not in installer:
        problems.append("installer.compile_pending 이 세대 분기(_compile_supercombo_by_generation)를 거치지 않음")
    runner = (SELECTOR_DIR / "modeld_runner.py").read_text()
    if "is_valid_genc_model_dir" not in runner or '"genc_custom"' not in runner:
        problems.append("modeld_runner 에 genc 엔진 분기 없음")
    validator = (SELECTOR_DIR / "validator.py").read_text()
    if "def has_genc" not in validator or "is_valid_genc_model_dir(base) or" not in validator:
        problems.append("validator.is_valid_model_dir 가 Gen C 산출물 세트를 인식하지 않음")

    keys = (SELECTOR_DIR / "keys.py").read_text()
    m = re.search(r"^MODEL_SELECTOR_VERSION\s*=\s*(\d+)", keys, re.M)
    if not m or int(m.group(1)) < 5:
        problems.append("MODEL_SELECTOR_VERSION < 5 (Gen C 항목은 manifest minimum_selector_version 5 로 게이트)")
    return "; ".join(problems) if problems else None


def sync_baselines() -> None:
    """미러 리뷰(이식) 완료 후 실행 — 현재 upstream 파일을 스냅샷으로 저장."""
    for current, baseline in MIRRORED_UPSTREAM_FILES.items():
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_bytes(current.read_bytes())
        print(f"synced {baseline.relative_to(REPO_ROOT)}")


def check_wiring() -> str | None:
    problems = []
    pc = (REPO_ROOT / "openpilot/system/manager/process_config.py").read_text()
    if "openpilot.carrot.model_selector.modeld_runner" not in pc:
        problems.append("process_config.py: modeld no longer points at modeld_runner")
    mgr = (REPO_ROOT / "openpilot/system/manager/manager.py").read_text()
    if "model_selector.boot_compile" not in mgr:
        problems.append("manager.py: boot_compile hook removed")
    keys = (REPO_ROOT / "openpilot/common/params_keys.h").read_text()
    for k in ("DrivingModelName", "PendingModelName"):
        if k not in keys:
            problems.append(f"params_keys.h: {k} unregistered")
    return "; ".join(problems) if problems else None


CHECKS = (
    ("script-paths", check_script_paths),
    ("helpers-hook", check_helpers_hook),
    ("compile-modeld-cli", check_compile_modeld_cli),
    ("metadata-naming", check_metadata_naming),
    ("pkl-format-pair", check_pkl_format_pair),
    ("tinygrad-pickle-compat", check_tinygrad_pickle_compat),
    ("sconscript-flags", check_sconscript_flags),
    ("compile-env-guards", check_compile_env_guards),
    ("fill-model-msg-signatures", check_fill_model_msg_signatures),
    ("model-constants", check_model_constants),
    ("wiring", check_wiring),
    ("modeld-mirror", check_modeld_mirror),
    ("genc-engine", check_genc_engine),
)


def main() -> int:
    if "--sync-baselines" in sys.argv:
        sync_baselines()
        print("스냅샷 갱신 완료 — 검사를 다시 실행해 PASS 를 확인할 것")
        return 0

    failed = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
        except Exception as e:  # 검사 자체의 오류도 계약 위반으로 취급
            detail = f"check errored: {e!r}"
        if detail is None:
            print(f"PASS {name}")
        else:
            failed += 1
            print(f"FAIL {name} — {detail}")

    from carrot.model_selector.config import compile_env_tag
    print(f"INFO compile_env_tag = {compile_env_tag()}")

    if failed:
        print(f"\n{failed}개 계약 위반 — 셀렉터 코드 적응 필요 (carrot/model_selector/README.md 참고)")
        return 1
    print("\n모든 계약 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
