# Gen C(supercombo state-queue) 모델 지원 구현 계획

> **For Hermes:** subagent-driven-development 로 태스크 단위 실행 가능. 모든 태스크는 worktree
> `/root/worktrees/hermes/default/genc` (branch `genc-support`, base `origin/carrot-ms` dcdc5d88f) 에서 수행.

**Goal:** comma master 최신 세대(Gen C: `new_img` + ONNX 내부 state queue) supercombo 모델을 carrot
model_selector 로 설치·컴파일·실행할 수 있게 한다. 기존 Gen A(Lebowski 계열) 경로와 기본 내장 모델은 그대로 둔다.

**Architecture:** 셀렉터 소유의 새 엔진 `carrot/model_selector/genc/` 를 추가한다. 모델 실행 코어(ModelState)는
comma master `openpilot/selfdrive/modeld/modeld.py`(2026-11-02) 의 Gen C 구현을 이식하고, 프로세스 메인 루프는
carrot-ms `modeld.py` 의 carrot 의미론(DesireHelper 확장, MapInput, 동적 lat smooth, desiredVelocity/vEgoStopping,
RuntimeDiagnostics, CameraGapTrace)을 유지한다. Gen C 는 comma 가 핀한 tinygrad(ec91b74, 2026-11-02)가 필요하므로
`genc/tinygrad_repo/` 에 별도 벤더링하고 **Gen C 엔진 프로세스와 Gen C 컴파일 서브프로세스에서만** sys.path 우선으로
활성화한다(시스템 tinygrad 0.13.0 과 Gen A/DM 경로는 불변). 세대 판정은 ONNX 입력명 집합으로 결정론적으로 한다.

**Tech Stack:** Python 3.12(device) / tinygrad ec91b74 (`examples/openpilot/{compile_onnx,compile_warp,helpers}.py`) /
numpy / openpilot messaging / 기존 selector(installer·validator·modeld_runner·check_contracts).

---

## 세대 정의 (ONNX 입력명)

| Gen | 입력 | 출력 | 엔진 |
|-----|------|------|------|
| A | `img, big_img, desire_pulse, traffic_convention, action_t, features_buffer` | `outputs` | 기존 upstream modeld + compile_modeld.py |
| B | `warped`/`new_img` + `state_img_q, state_big_img_q, state_prev_feat, …` | `outputs, next_state_*` | **미지원** (InstallError) |
| C | `new_img[2,6,128,256], desire[8], traffic_convention[1,2], action_t[1,2], state_img_q[2,5,6,128,256], state_desire_q[100,1,8], state_feat_q[96,1,512]` | `outputs[1,N], next_state_img_q, next_state_desire_q, next_state_feat_q` | **신규 genc 엔진** |

Gen C 런타임 계약(comma modeld.py 기준): `state_pairs = {name: f"next_{name}"}` 로 상태 피드백, `new_img` 는 warp JIT
(`compile_warp.py --layout yuv420 --frames 2`) 출력, 나머지 비이미지 입력은 128B 정렬 packed 버퍼로 한 번에 업로드,
`output_slices` 는 ONNX metadata_props(base64 pickle). `desire` 입력은 `desire_pulse` 의 rising edge 펄스.

## 산출물 파일 세트 (`/data/models`, Gen C)

```
driving_supercombo.onnx                       # 보존 (재컴파일용, 기존 규칙)
driving_genc_tinygrad.pkl                     # compile_onnx.py --out-of-band
driving_genc_warp_1928x1208_tinygrad.pkl      # compile_warp.py (tici/tizi)
driving_genc_warp_1344x760_tinygrad.pkl       # compile_warp.py (mici)
genc_metadata.json                            # {"generation":"C","tinygrad_pin":..., "inputs":{...}, "outputs":{...}, "cameras":[...]}
.compile_env                                  # 기존 스탬프 (경로 목록에 genc 파일 추가 → 1회 재컴파일 유발, 의도된 동작)
```

---

## Task 1: tinygrad 벤더링 (genc 전용)

**Files:**
- Create: `carrot/model_selector/genc/tinygrad_repo/` ← `/root/.hermes/cache/scratch/genc/tinygrad` 의
  `tinygrad/`, `examples/__init__.py`, `examples/openpilot/{__init__.py,helpers.py,compile_onnx.py,compile_warp.py}`, `LICENSE`
- Create: `carrot/model_selector/genc/tinygrad_repo/PIN` = `ec91b74b5b6b0fc924b72d3f27689679ca2438b9 (tinygrad/tinygrad, commaai/openpilot master 2026-11-02)`
- Create: `carrot/model_selector/genc/__init__.py`

```python
"""Gen C (comma master state-queue supercombo) engine. tinygrad is vendored here and
activated per-process so the system tinygrad used by Gen A / DM stays untouched."""
from __future__ import annotations
import os, sys
from pathlib import Path

GENC_DIR = Path(__file__).resolve().parent
TINYGRAD_DIR = GENC_DIR / "tinygrad_repo"
TINYGRAD_PIN = (TINYGRAD_DIR / "PIN").read_text().split()[0] if (TINYGRAD_DIR / "PIN").is_file() else ""

def activate_tinygrad() -> None:
    """Put the vendored tinygrad first on sys.path. Must run before any `import tinygrad`."""
    if "tinygrad" in sys.modules:
        import tinygrad  # noqa
        if Path(tinygrad.__file__).resolve().is_relative_to(TINYGRAD_DIR):
            return
        raise RuntimeError("system tinygrad already imported; activate_tinygrad() must run first")
    sys.path.insert(0, str(TINYGRAD_DIR))

def subprocess_env(base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env["PYTHONPATH"] = str(TINYGRAD_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env
```

**Verify:** `python3 -c "import sys; sys.path.insert(0,'carrot/model_selector/genc/tinygrad_repo'); import tinygrad; print(tinygrad.__file__)"`
→ 벤더 경로. `du -sh carrot/model_selector/genc/tinygrad_repo` ≈ 19MB. Commit: `feat(ms): vendor tinygrad ec91b74 for Gen C engine`.

## Task 2: 세대 판정 모듈 + 테스트 (TDD)

**Files:** Create `carrot/model_selector/genc/generation.py`, Test `carrot/model_selector/tests/test_genc_generation.py`

```python
GEN_A_INPUTS = frozenset({"img", "big_img", "desire_pulse", "traffic_convention", "action_t", "features_buffer"})
GEN_C_INPUTS = frozenset({"new_img", "desire", "traffic_convention", "action_t", "state_img_q", "state_desire_q", "state_feat_q"})
GEN_C_STATE = ("state_img_q", "state_desire_q", "state_feat_q")

def detect_generation(input_names, output_names=()) -> str | None:
    names = frozenset(input_names)
    if names == GEN_A_INPUTS: return "A"
    if names == GEN_C_INPUTS and all(f"next_{s}" in output_names for s in GEN_C_STATE): return "C"
    return None  # Gen B 등 미지원
```
테스트 케이스: Gen A(TSFDO 입력), Gen C(SSHv4 입력/출력), Gen B v1(`warped`…), Gen B v2(`new_img`+`state_big_img_q`), 출력에
`next_state_*` 누락 → None. 실행: `python3 -m unittest carrot.model_selector.tests.test_genc_generation -v`.

## Task 3: config/validator 확장 + 테스트

- `config.py`: `GENC_PKL_NAME`, `GENC_WARP_PKL_FMT`, `GENC_METADATA_NAME`, `_COMPILE_ENV_PATHS` 에
  `carrot/model_selector/genc` 디렉터리(트리 해시 1개로 벤더 tinygrad·엔진·컴파일러 변경 모두 반영) 추가.
- `validator.py`: `has_genc(base)` = pkl + 2개 warp pkl + metadata json 존재·비어있지 않음; `is_valid_genc_model_dir`;
  `is_valid_model_dir` 에 포함; `describe` → `"{base}: genc"`.
- Test `tests/test_genc_validator.py`: 임시 디렉터리로 has_genc 참/거짓, describe 문자열.

## Task 4: Gen C 컴파일 래퍼 + 메타데이터 검증 + 테스트

Create `carrot/model_selector/genc/compile.py`:
- `genc_compile_commands(onnx, out_dir, camera_configs, model_size=(512,256))` → 순수 함수, 명령 리스트 반환(테스트 대상).
  - `[python3, TINYGRAD_DIR/examples/openpilot/compile_onnx.py, onnx, out_dir/GENC_PKL_NAME, --out-of-band, --benchmark-runs, 1]`
  - 카메라별 `[python3, …/compile_warp.py, --frame, f"{w},{h},{stride},{y_h},{uv_h},{size}", --warp-to, 512x256, --layout, yuv420, --frames, 2, --output, out_dir/warp]`
    (stride 등은 `openpilot.system.camerad.cameras.nv12_info.get_nv12_info`)
- `validate_genc_metadata(input_shapes, output_shapes, output_slices)`:
  필수 입력 = GEN_C_INPUTS; `new_img` shape `(2, 6, 128, 256)` → `(w*2,h*2)==MEDMODEL_INPUT_SIZE`; `desire[-1]==DESIRE_LEN`;
  `traffic_convention[-1]==2`; `action_t[-1]==2`; 각 state 에 `next_` 출력 존재 & shape 동일; `output_slices` 에
  `plan, meta, pose, lane_lines, road_edges, lead, desire_state, hidden_state` 포함. 위반 시 `InstallError`.
- `compile_genc(tmp_dir, env, camera_configs)`: 메타데이터는 벤더 tinygrad 서브프로세스(`compile_onnx.onnx_metadata` 재사용한
  작은 스크립트 `genc/onnx_io.py <onnx> --json`)로 뽑아 검증 → 명령 실행(`installer._run` 재사용, env 는 `subprocess_env(env)`)
  → `genc_metadata.json` 기록. 산출물 존재/크기 검사.
- Test `tests/test_genc_compile.py`: 명령 조립(경로·플래그), validate 통과/실패 케이스(SSHv4 실제 shape 상수 사용).

## Task 5: 런타임 코어 `genc/runtime.py` (comma ModelState 이식)

comma `modeld.py` 147-238 행을 이식하되 ① `load_oob(modeld_pkl_path(chestnut), chestnut)` → `load_artifact(models_dir/GENC_PKL_NAME)`
(벤더 `examples.openpilot.helpers.load_pickle(path, out_of_band=True)` 를 `Context(DEV=device)` 안에서; device 는 AGNOS→`QCOM`,
그 외 `CPU:LLVM`/env `DEV`) ② warp pkl 경로 `models_dir / GENC_WARP_PKL_FMT.format(w=cam_w,h=cam_h)` ③ chestnut 분기 제거
④ `vision_input_names = ('img','big_img')` 유지(메인 루프가 bufs/transforms 를 이 키로 넘김) ⑤ `run()` 시그니처는 carrot 메인
루프와 맞춰 `run(bufs, transforms, inputs, prepare_only=False)` (prepare_only 는 무시: Gen C 는 상태가 ONNX 안에 있어 매 프레임 실행).
⑥ `warmup()`, `reset()`(state queue 0 초기화) 제공. 모듈은 tinygrad·numpy·`parse_model_outputs.Parser`·`ModelConstants` 만 import
(messaging 없음 → PC 검증 가능). 파일 첫 줄에서 `activate_tinygrad()` 호출 후 tinygrad import.

## Task 6: PC 검증 하네스 `genc/pc_check.py` (실행 증거)

`python3 carrot/model_selector/genc/pc_check.py --onnx /path/SSHv4/driving_supercombo.onnx --camera 1928x1208 --out /tmp/genc_pc`
→ (1) Task 4 명령으로 CPU 컴파일(`DEV=CPU:LLVM` 또는 `CPU`) (2) `runtime.ModelState(out_dir, 1928, 1208)` 로드 (3) 난수 NV12 프레임
3회 실행 → `plan` shape `(1,33,15)`, 모든 출력 finite, `next_state_*` 가 매 스텝 변함, 실행시간 출력. 실행 환경: CT101
(`/root/op-model-watcher/.venv` python3.11 + numpy, LLVM 필요 여부 확인) 또는 Hermes VM(py3.14 — tinygrad 호환 확인 필요).
**이 단계가 PASS 해야 Task 7 이후 진행.**

## Task 7: 엔진 `carrot/model_selector/genc_modeld.py`

carrot-ms `modeld.py::main()` 을 기반으로: eGPU/Jetlink/signal_shadow/precompiled/egpu_recovery 블록 제거, `ModelState` → `genc.runtime.ModelState(
CUSTOM_MODELS_DIR, cam_w, cam_h)`, `prepare_only` 전달은 유지하되 무시, 나머지(DH, nav_input, boundary_observer, get_lat_smooth_seconds_dynamic,
get_action_from_model(carrot 시그니처), fill_model_msg carrot 버전, diagnostics, camera_gap) 동일. `get_action_from_model`/`get_lat_smooth_seconds_dynamic`
는 `openpilot.selfdrive.modeld.modeld` 에서 import 하지 말 것(그 모듈이 시스템 tinygrad 를 import 함) → 두 함수를 `genc_modeld.py` 에 복사하고
check_contracts 미러 검사 대상에 추가(Task 10). 프로세스 시작 시 `activate_tinygrad()` 가 가장 먼저 실행되도록 `modeld_runner` 에서 import 순서 보장.

## Task 8: installer 통합

`compile_pending()`: supercombo 존재 시 → `genc.onnx_io` 로 입출력명 추출 → `detect_generation` →
`"A"`: 기존 `_compile_supercombo` / `"C"`: `compile_genc` / `None`: `InstallError(f"unsupported supercombo generation (inputs={sorted(names)})")`.
`recompile_stale_if_needed` 는 onnx 보존 규칙 그대로 재사용. 기존 Gen A 검증·경로는 수정하지 않는다.

## Task 9: modeld_runner 분기 + 테스트

`_select_engine`: `has_genc` 를 `has_supercombo` 보다 먼저 검사 → `("genc_custom", "genc_modeld")`. `main()`: `genc_custom` →
`from openpilot.carrot.model_selector import genc_modeld; genc_modeld.main(demo)`. crash-loop breaker 는 커스텀 엔진 공통 적용(기존 코드 그대로).
Test `tests/test_genc_runner_select.py`: validator 함수들을 monkeypatch 해 4-way 선택 검증.

## Task 10: check_contracts / keys / README

- `check_contracts.py`: REQUIRED_SCRIPTS += genc compile 스크립트 3종 + PIN; 새 검사 `genc-tinygrad-pin`(PIN sha 와 `tinygrad_repo/PIN`
  일치, `compile_onnx.py` 에 `--out-of-band`, `helpers.py` 에 `def load_pickle`); REQUIRED_COMPILE_ENV_PATHS += `carrot/model_selector/genc`;
  MIRRORED 에 `genc_modeld.py` 가 복사한 두 함수의 원본(`modeld.py`)은 이미 포함.
- `keys.py`: `MODEL_SELECTOR_VERSION = 5` (manifest `minimum_selector_version 5` = Gen C 항목 게이트).
- README `## 설계 원칙` 에 Gen C 엔진/벤더 tinygrad/세대 표 추가.
- 전체: `python3 carrot/model_selector/check_contracts.py` 모두 PASS, `python3 -m unittest discover -s carrot/model_selector/tests -t . -v` 모두 PASS.

## Task 11: 릴리스 전 manifest/watcher 정리 (별도 저장소, 기기 배포 후)

- openpilot-models: Gen B 항목(`Onnx Model History`, `Onnx Model Historyv2`) → `minimum_selector_version 99`; SSH 4종은 실험 모델이므로 99 유지 권장,
  Gen C 실기 테스트용은 `Onnx Model Historyv3`(=master 스냅샷) msv 5 사용.
- op-model-watcher: `data/selector_version.json` → 5 는 **기기 빌드 배포 후**; 판정 규칙화(세대 감지 1차, AI 2차)는 후속 태스크.

## 구현 상태 (2026-11-03, branch `genc-support`)

| Task | 상태 | 비고 |
|------|------|------|
| 1 tinygrad 벤더링 | 완료 | `genc/tinygrad_repo/` (`tinygrad/` 전체 + `examples/openpilot/{helpers,compile_onnx,compile_warp}.py` + LICENSE + PIN), 19MB. `pyproject.toml` ruff/ty/codespell 제외 추가 |
| 2 세대 판정 | 완료 | `genc/generation.py` + `tests/test_genc_generation.py` |
| 3 config/validator | 완료 | `GENC_*` 상수, `_COMPILE_ENV_PATHS` 에 `carrot/model_selector/genc`, `has_genc`/`is_valid_genc_model_dir`, `GENC_CAMERA_CONFIGS` |
| 4 컴파일 래퍼 | 완료 | `genc/onnx_io.py`(프로브) + `genc/compile.py`(`probe_onnx_io`/`validate_genc_io`/`compile_genc`). 계획 대비 추가: `genc_metadata.json` 에 `compile_dev` 기록 |
| 5 런타임 코어 | 완료 | `genc/runtime.py`. 계획 대비 변경: `prepare_only` 는 무시(매 프레임 full inference, comma master 와 동일), `DEV.value` 를 프로세스 전역으로 고정(아래 교훈) |
| 6 PC 검증 | 완료 | `genc/pc_check.py` — Onnx Model Historyv3(=master 스냅샷) 를 CPU:LLVM 으로 컴파일 15s(모델 11s + warp 2s×2), 결과는 본 문서 하단 "PC 검증 기록" |
| 7 엔진 | 완료 | `genc_modeld.py` (eGPU/Jetlink/shadow 제거, 나머지 carrot 의미론 유지) |
| 8 installer | 완료 | `_compile_supercombo_by_generation` (프로브 1회 → A/C 분기, 그 외 InstallError) + `tests/test_genc_installer_routing.py` |
| 9 runner | 완료 | `genc_custom` 분기 + `tests/test_genc_runner_select.py` |
| 10 contracts/keys/README | 완료 | `genc-engine` 검사 추가, `MODEL_SELECTOR_VERSION=5`, README 갱신. `check_contracts.py` 13/13 PASS, pytest 51 passed |
| 11 manifest/watcher | **미착수** | 기기 테스트 후: Gen B 항목 msv 99, Gen C 테스트 항목 msv 5, watcher `selector_version.json` 5 |
| + 기기 프리플라이트 | 추가 | `genc/device_preflight.py` — 첫 기기 테스트 전에 AGNOS 에서 벤더 tinygrad QCOM/호스트 컴파일러 확인 |

### 구현 중 확인된 사실 / 교훈
- **DEV 는 프로세스 전역으로 고정해야 한다.** tinygrad 는 장치의 renderer(CPU → clang/LLVM)를 *커널 컴파일 시점마다* `DEV` 타깃에서 다시 고른다. `Context(DEV=…)` 로 로드만 감싸면 로드 후 지연 컴파일(state 큐 reset 의 fill 커널 등)이 기본 renderer(clang)로 떨어진다 → `ModelState.__init__` 에서 `DEV.value = compile_dev`.
- **`examples.openpilot.helpers` 는 import 시 `Device.DEFAULT` 를 평가**하고 `_select_device` 가 `os.environ["DEV"]` 를 써 버린다 → DEV 고정 후 `_load` 안에서 지연 import.
- **cwd 가 저장소 루트면 `tinygrad` 심볼릭 링크가 벤더본을 가린다** (`python -c` 의 `sys.path[0]==''`). 런타임은 `activate_tinygrad()` 가 `sys.path[0]` 에 벤더 경로를 넣어 해결하지만, 수동 실험은 저장소 밖 cwd 에서 할 것.
- comma master 는 **AGNOS 20.x**(clang 기본, LLVM 20)이고 carrot-ms 는 **AGNOS 19.8-carrot** 이다. 벤더 tinygrad 의 QCOM 런타임은 명령 제출 루틴을 CPU 장치에서 컴파일(`HCQ_RUNTIME_DEV`, 기본 `CPU:LLVM`; renderer 는 clang→LLVM 순 폴백)하므로 기기에 `clang` 또는 `libLLVM` 이 필요하다 — `device_preflight.py` 1번 항목. kgsl ioctl autogen 은 두 세대가 동일(커널 인터페이스 변화 없음).
- pytest 는 저장소 `addopts` 에 xdist 옵션이 있어 `-o addopts="" --noconftest` 로 실행 (`conftest.py` 가 Cython `params_pyx` 를 요구).

### PC 검증 기록 (Hermes VM, 4 vCPU, py3.12, DEV=CPU:LLVM, libLLVM-18)
모델: `Onnx Model Historyv3/driving_supercombo.onnx` (sha256 65a08adc…, comma master 2026-09-15 스냅샷과 동일 I/O 세대).

```
$ PYTHONPATH=<repo> python3 carrot/model_selector/genc/pc_check.py --onnx .../driving_supercombo.onnx --out <dir> --camera 1928x1208 --runs 4
  onnx_io.py probe            ok (0s)
  compile_onnx.py --out-of-band  ok (11s)  → driving_genc_tinygrad.pkl 61.8MB
  compile_warp.py 1928x1208   ok (2s)   → 573KB ; compile_warp.py 1344x760  ok (2s)
metadata: generation=C compile_dev=CPU:LLVM cameras=[[1928,1208],[1344,760]]
loaded ModelState(1928x1208) on CPU:LLVM (model_device=CPU) in 0.6s ; warmup 1.27s
run 0..3: 265–436 ms, plan(1,33,15), finite, 21 parsed keys
  state_img_q / state_feat_q 매 프레임 변경, state_desire_q 는 desire 펄스(run 1) 이후 변경
PASS   (1344x760 카메라 경로도 PASS)
```
`device_preflight.py` 도 같은 VM 에서 PASS(CPU:LLVM 경로). QCOM 경로는 기기에서 실행해야 한다.

## 위험 / 확인 사항

1. 새 tinygrad QCOM 런타임 vs carrot-ms AGNOS 버전: comma master 와 동일 조합이지만 carrot-ms 의 AGNOS 가 더 오래됐을 수 있음 → 기기 첫 테스트는
   crash-loop breaker(3회 → 격리 → 내장 모델) 보호 하에 진행, 로그 `[MODEL_SELECTOR]`/`modeld` 확인.
2. `.compile_env` 경로 추가로 기존 커스텀 모델 1회 재컴파일(부팅 지연 수 분) — 기존 설계대로.
3. Gen C 모델의 `traffic_units` 등 추가 slice 는 fill_model_msg 가 무시(안전). `plan_stds` 는 parser 가 생성하므로 동적 lat smooth 유지.
4. 디스크: 벤더 tinygrad +19MB(저장소/기기).
