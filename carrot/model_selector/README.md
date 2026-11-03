# Carrot Model Selector

Carrot 파일럿 전용 주행 모델 다운로드/설치/적용 모듈. c3x·c4 하드웨어 공용으로
**carrot web(포트 7000)** 에서 UI를 제공한다.

## 설계 원칙

1. **엔진 분리 + 신구조는 upstream 엔진 재사용**
   - 기본 모델 (`openpilot/selfdrive/modeld/models/`) → **upstream `openpilot/selfdrive/modeld/modeld.py` 가 그대로 담당**
   - 커스텀 **신모델(supercombo, lebowski 구조)** (`/data/models/driving_tinygrad.pkl`)
     → **upstream `modeld.py` 를 그대로 재사용** — `MODELD_MODELS_DIR=/data/models` env 만 설정
     (`openpilot/selfdrive/modeld/helpers.py::modeld_pkl_path()` 가 env 를 읽어 경로 전환, monkey-patch 없음)
   - 커스텀 **구모델(vision+policy 분리)** (`/data/models/`) → **`carrot.model_selector.carrot_modeld` 가 담당** (3-모델 on/off policy 자체 지원)
   - 커스텀 **Gen C 모델(comma master 2026-10+ state-queue supercombo)** (`/data/models/driving_genc_tinygrad.pkl` + warp pkl + `genc_metadata.json`)
     → **`carrot.model_selector.genc_modeld` 가 담당**. comma 가 핀한 신세대 tinygrad 를 `genc/tinygrad_repo/` 에 별도 벤더링하고
     **Gen C 프로세스/컴파일 서브프로세스에서만** `sys.path` 우선 활성화 (시스템 tinygrad 를 쓰는 Gen A 엔진·dmonitoringmodeld 는 불변)
   - `modeld_runner` 가 부팅 시 `/data/models` 내용으로 넷 중 하나를 택일 실행
   - upstream 이 바뀌어도 legacy/Gen C 엔진과 무관하게 진행 가능

   **supercombo 세대 판정은 ONNX 입력명 집합으로 결정론적** (`genc/generation.py`):

   | 세대 | ONNX 입력 | 엔진 |
   |------|-----------|------|
   | A (2026-08, Lebowski/RDF/TSF…) | `img, big_img, desire_pulse, traffic_convention, action_t, features_buffer` | upstream modeld + `compile_modeld.py` |
   | B (2026-09 중간형) | `warped`/`new_img` + `state_big_img_q`, `state_prev_feat`, … | **미지원** — 설치 단계에서 InstallError |
   | C (2026-10+, comma master) | `new_img, desire, traffic_convention, action_t, state_img_q, state_desire_q, state_feat_q` (+ `next_state_*` 출력) | `genc_modeld` + 벤더 tinygrad `compile_onnx.py`/`compile_warp.py` |

2. **upstream 파일 최소 침습** (각 1~3줄)
   - `openpilot/system/manager/process_config.py` — modeld 엔트리를 `openpilot.carrot.model_selector.modeld_runner` 로 교체
   - `openpilot/system/manager/manager.py::main()` 초입 — `boot_compile.run()` 호출
   - `openpilot/selfdrive/modeld/helpers.py::modeld_pkl_path()` — `MODELD_MODELS_DIR` env 오버라이드 (2줄)
   - `openpilot/selfdrive/carrot/server/app.py` — 라우터 등록
   - `openpilot/selfdrive/carrot/web/index.html` — 네비 버튼 + `page_models.js` 로드 (`?v=…` 캐시 버스팅)
   - `openpilot/selfdrive/ui/mici/layouts/home.py` — c4 홈 화면 커밋해시 옆에 `DrivingModelName` 표시
   - `openpilot/common/params_keys.h` — `DrivingModelName`, `PendingModelName` 파람 등록

3. **독립 패키지**
   - 모델 셀렉터 자체 엔진 포함 전부 `carrot/model_selector/` 에 완결
   - `carrot_parse_model_outputs.py` 는 3-모델 분기가 있는 c3-ms 파서 이식본
   - 원본 파일(`openpilot/selfdrive/modeld/*`, `openpilot/common/file_chunker.py` 등)은 읽기/import 전용 (helpers.py env 훅 제외)

4. **파일 세트 호환** (신형 단일 onnx 또는 구형 vision + 정책)
   - **신형(supercombo)**: `driving_supercombo.onnx` 단독 — 존재 시 우선 적용 (Gen A/C 는 설치 시 자동 판별)
   - **구형(legacy)**: `driving_vision` 필수 + (`driving_on_policy` 또는 `driving_policy`) 중 하나 필수
   - `driving_off_policy` 는 선택적, 존재 시 3-모델 아키텍처 활성화 (구형 전용)

## 모듈 구조

| 파일 | 책임 |
|------|------|
| `config.py` | 경로·파일명 상수, 허용 onnx 목록 (supercombo 포함), tinygrad 컴파일 플래그 |
| `keys.py` | Ed25519 공개키 맵 + `MODEL_SELECTOR_VERSION` (v4: supercombo 지원, v5: Gen C 지원) |
| `manifest.py` | `models_v4.json`(전체 카탈로그) fetch → 실패 시 `models.json`(레거시, 동결) 폴백. 캐시버스터 `?t=unixtime` + no-cache 헤더 + canonical JSON Ed25519 검증. 버전 게이트를 파일명 검사보다 먼저 수행하고 해석 불가 항목은 스킵(전방 호환). snake_case/camelCase 필드 둘 다 수용 |
| `downloader.py` | ONNX 스트리밍 다운로드 + SHA256/크기 검증. 공백 포함 파일명 URL percent-encoding. supercombo 단독 또는 legacy 세트 허용 |
| `validator.py` | `/data/models/` 파일 세트 유효성 검증 (`has_genc` / `has_supercombo` / legacy 세트) + `describe()` 라벨 |
| `installer.py` | 신형: ONNX 입력명으로 세대 판별 → Gen A 는 `compile_modeld.py` 통합 pkl, Gen C 는 `genc/compile.py` (벤더 tinygrad). 구형: tinygrad 컴파일 (compile3) + warp pkl. 공통: atomic swap + 실패 시 backup 복원 |
| `carrot_modeld.py` | **자체 modeld 엔진** — 구형 2/3-모델 공용, `/data/models` 전담 |
| `carrot_parse_model_outputs.py` | 자체 parser (c3-ms 이식, `parse_off_policy_outputs` 포함) |
| `genc_modeld.py` | **Gen C modeld 엔진** — carrot modeld.py 의 main 루프(DesireHelper·MapInput·동적 lat smooth·diagnostics)에 `genc.runtime.ModelState` 를 결합. eGPU/Jetlink/shadow 미지원 |
| `genc/` | Gen C 엔진 패키지: `generation.py`(세대 판정), `onnx_io.py`(그래프 I/O 프로브), `compile.py`(compile_onnx+compile_warp 래퍼·검증), `runtime.py`(comma master ModelState 이식), `pc_check.py`(PC 컴파일·추론 스모크), `device_preflight.py`(기기 프리플라이트), `tinygrad_repo/`(벤더 tinygrad, `PIN` 에 커밋) |
| `modeld_runner.py` | 부팅 시 genc(genc_modeld) / supercombo(upstream+env) / legacy(carrot_modeld) / 기본(upstream) 4-way 택일 실행 (dispatcher). `/data/model_selector_status` 에 엔진 정보 기록 |
| `boot_compile.py` | `manager.main` 초입 훅, pending 모델 컴파일 트리거 |
| `jobs.py` | 간단한 in-memory job runner (web 진행률 폴링) |
| `web/routes.py` | aiohttp 라우터 (list / status / install / job / apply / reset) |
| `web/frontend/page_models.html` | carrot web 페이지 템플릿 (Material 3 토큰 기반) |
| `web/frontend/page_models.js` | 프런트엔드 로직 (fragment 자동 주입 + Job 폴링 + 확인창 + 재부팅 카운트다운) |

## Params

| 이름 | 타입 | 의미 |
|------|------|------|
| `DrivingModelName` | STRING | 현재 사용 중인 모델 표시용 (c4 홈 화면 커밋해시 옆에도 표시) |
| `PendingModelName` | STRING | 재부팅 후 컴파일 대기 중인 모델 ID |

## 데이터 흐름

### 설치 (Install)
```
[Web UI: "설치" 클릭]
  └─ appConfirm 다이얼로그 ("모델을 설치합니다. 다운로드 후 자동 재부팅됩니다.")
       └─ POST /api/models/install {id}
            └─ manifest 재검증 → jobs.start("install_model")
                 └─ downloader.download_model(entry, progress_cb)   # 파일별 size+sha256 검증
                      → /data/models_tmp/
                 └─ Params.put(PendingModelName=entry.id)
                 └─ job.finish(ok=True, progress=100)

[클라이언트 폴링] GET /api/models/job?id=<job_id>  (320ms)
  └─ snap.done && status=="done"
       └─ POST /api/models/apply  (서버가 5초 후 sudo reboot 예약)
            └─ rebootCountdown(5)  — 0→100 바, "다운로드 완료 — N초 후 재부팅…"

[재부팅 → manager.main() 초입]
  └─ boot_compile.run() → installer.compile_pending()
       ├─ [신형] driving_supercombo.onnx 존재 시: genc/onnx_io.py 로 입력명 프로브 → 세대 판정
       │    ├─ Gen A → compile_modeld.py --model-size(metadata img 크기) --camera-resolutions 1928x1208 1344x760
       │    │          --frame-skip(MODEL_RUN_FREQ//MODEL_CONTEXT_FREQ)
       │    │          → 통합 driving_tinygrad.pkl (metadata + 모델 JIT + 해상도별 warp JIT)
       │    ├─ Gen C → (벤더 tinygrad, PYTHONPATH 우선) compile_onnx.py --out-of-band → driving_genc_tinygrad.pkl
       │    │          compile_warp.py --layout yuv420 --frames 2 (카메라별) → driving_genc_warp_{w}x{h}_tinygrad.pkl
       │    │          + genc_metadata.json (입출력 shape·output_slices·compile_dev)
       │    └─ 그 외(Gen B 등) → InstallError (백업 복원, pending 해제)
       ├─ [구형] ONNX → tinygrad.pkl + metadata.pkl (QCOM/FLOAT16/NOLOCALS/JIT_BATCH_SIZE=0/IMAGE=1/OPENPILOT_HACKS=1)
       │    └─ selected model metadata의 img 입력 크기로 standalone warp pkl 새로 생성 (빌트인 warp 재사용 안 함)
       ├─ atomic swap: /data/models_tmp → /data/models  (실패 시 backup 복원)
       └─ Params: DrivingModelName 기록, PendingModelName 삭제

[process manager → `modeld` 프로세스 (only_onroad)]
  └─ modeld_runner.main()
       ├─ validator.is_valid_genc_model_dir(/data/models) ?
       │    └─ YES → status 기록(engine=genc_modeld)
       │             genc_modeld.main()          ← Gen C 엔진 (벤더 tinygrad 활성화는 import 시점)
       ├─ validator.is_valid_supercombo_model_dir(/data/models) ?
       │    └─ YES → status 기록(engine=upstream_modeld_custom)
       │             MODELD_MODELS_DIR=/data/models 설정 후 upstream modeld.main()  ← 신형 엔진 재사용
       ├─ validator.is_valid_legacy_model_dir(/data/models) ?
       │    └─ YES → status 기록(engine=carrot_modeld)
       │             carrot_modeld.main()        ← 구형 3-모델 엔진
       └─ 그 외 → status 기록(engine=upstream_modeld)
                  upstream modeld.main()          ← 기본 엔진
```

### 기본 모델 복원 (Reset)
```
[Web UI: "기본 모델 복원" 클릭]
  └─ appConfirm 다이얼로그
       └─ POST /api/models/reset
            ├─ installer.reset_to_default()   # /data/models + /data/models_tmp 삭제, 파람 제거
            └─ 5초 후 sudo reboot 예약
       └─ rebootCountdown(5) — "기본 모델 복원 — N초 후 재부팅…"
```

### 런타임 엔진 확인
```
$ cat /data/model_selector_status
engine=carrot_modeld
pid=53610
started=1776677826
describe=/data/models: vision+on_policy+off_policy
```
`engine` 값: `genc_modeld`(Gen C 커스텀) / `upstream_modeld_custom`(Gen A supercombo 커스텀) / `carrot_modeld`(구형 커스텀) / `upstream_modeld`(기본).
`ps` 는 `setproctitle` 때문에 Python 모듈 경로가 마스킹되므로 이 파일이 결정적 출처.

### Gen C 첫 기기 테스트 절차
1. 프리플라이트 (offroad, 설치 전): `cd /data/openpilot && PYTHONPATH=/data/openpilot python3 carrot/model_selector/genc/device_preflight.py`
   — 호스트 컴파일러(clang/libLLVM), QCOM 장치 오픈, JIT 커널, NPY↔장치 복사가 전부 PASS 여야 한다.
   (comma master 는 AGNOS 20.x 기준, carrot-ms 는 19.8-carrot — 벤더 tinygrad 의 QCOM 런타임이 명령 제출 루틴을 CPU 에서 컴파일하므로 이 확인이 필요)
2. manifest 에서 `minimum_selector_version: 5` 로 게이트된 Gen C 모델(예: comma master 스냅샷) 설치 → 재부팅 → `compile_pending` 로그
   (`model_selector: Gen C supercombo detected … compiling with vendored tinygrad`), `/data/models/genc_metadata.json` 의 `compile_dev=QCOM`.
3. onroad 후 `/data/model_selector_status` 의 `engine=genc_modeld`, `/data/models/.load_attempts` 가 90초 뒤 사라지는지(크래시 루프 차단기 미발동) 확인.
   실패 시 3회 조기 크래시 후 `/data/models_quarantined` 로 격리되고 내장 모델로 자동 폴백된다.
4. PC 에서 같은 파이프라인을 먼저 돌려볼 수 있다: `PYTHONPATH=<repo> python3 carrot/model_selector/genc/pc_check.py --onnx <driving_supercombo.onnx> --out <dir>` (DEV=CPU:LLVM, libLLVM 필요).

## 안전장치

- **manifest 전방 호환 + 이중화** (`manifest.py`)
  — 구버전(v3) 파서는 미지의 파일명이 manifest 항목 하나에라도 있으면 목록 전체가
  죽는 결함이 있어(버전 게이트 이전에 파일명 검사), 원격 저장소는 `models.json`(레거시
  파일명 전용, 동결) + `models_v4.json`(전체 카탈로그)으로 이중화됨. v4 파서는
  버전 게이트를 파싱보다 먼저 수행하고 해석 불가 항목을 스킵하므로 이후 버전에서
  새 파일명이 추가되어도 같은 사고가 재발하지 않음. 신형 파일명이 든 항목을
  `models.json` 에 넣으면 안 됨 (openpilot-models 의 `update_models.py` 가 자동 분리).
- **설치 전 metadata 검증** (`installer._validate_supercombo_metadata`)
  — 필수 입력(img/big_img/desire_pulse/traffic_convention/action_t/features_buffer),
  `hidden_state` output slice, 모델 입력 크기 512x256(MEDMODEL — 런타임 warp 행렬이 이 크기에 고정) 을
  확인하고 미달 시 설치를 중단(백업 복원). 비호환 모델이 설치되어 크래시 루프에 빠지는 것을 사전 차단.
  Gen C 는 `genc/compile.validate_genc_io` 가 같은 역할 (입력명 집합 정확 일치, `new_img` 2×6×128×256,
  `desire`/`traffic_convention`/`action_t` 길이, `state_*_q` ↔ `next_state_*` shape 일치, 필수 output_slices).
- **Gen C 엔진 격리** (`genc/__init__.activate_tinygrad`)
  — 벤더 tinygrad 는 genc_modeld 프로세스와 Gen C 컴파일 서브프로세스(`PYTHONPATH` 선두)에서만 활성.
  manager fork 로 시스템 tinygrad 모듈이 상속돼 있으면 `sys.modules` 에서 제거 후 재해석한다.
  `genc_metadata.json` 의 `compile_dev` 로 런타임 백엔드를 컴파일 시점과 일치시킨다(QCOM / PC CPU:LLVM).
- **크래시 루프 차단기** (`modeld_runner._arm_crash_loop_breaker`)
  — 커스텀 엔진 기동 시 `/data/models/.load_attempts` 카운터 증가, 90초 생존 시 자동 삭제.
  3회 연속 조기 크래시(현재 스탬프지만 손상된 pkl, 출력 slice 불일치 등) 시 `/data/models` 를
  `/data/models_quarantined` 로 격리하고 기본 내장 모델로 자동 폴백. 수동 reset 없이 주행 가능 상태 복구.
- **tinygrad 세대 사전 차단** (`modeld_runner._select_engine`)
  — `/data/models/.compile_env` 가 현재 컴파일러 태그와 다르면 구 pkl 을 역직렬화하지
  않고 즉시 기본 내장 모델을 사용한다. 보존된 onnx 자동 재컴파일이 성공하면 새
  스탬프와 함께 원자적으로 교체되고, 실패하거나 onnx 가 없는 경우에도 구 pkl 로
  인한 재시작 루프 없이 기본 모델로 부팅한다. 기존 커스텀 파일은 다음 버전의 재시도를 위해 보존한다.
- **stale PendingModelName 방지**
  — 설치 시작 시(다운로드 전) 이전 pending 제거(`routes.api_install`),
  부팅 시 tmp 가 불완전하면 pending 도 함께 제거(`installer.compile_pending`).
  실패한 다운로드의 잔여 파일이 예전 모델명으로 컴파일되거나 UI 가 영구 "설치중" 이 되는 문제 방지.
- **PC 컴파일 백엔드 프로브** (`installer._probe_tg_devices`)
  — 비-TICI 모델 설치에서는 tinygrad 디바이스를 프로브해 CUDA > CPU 순으로 선택
  (런타임 tg_input_devices.json 과 백엔드 불일치 방지). SCons 빌드는 아키텍처로 선택하며 실기기는 QCOM 고정.
- **USBGPU 동작** — 커스텀 supercombo 설치 시 커스텀 모델 디렉터리에 `big_driving_<sha16>_tinygrad.pkl` 이 없으므로
  upstream 의 USBGPU 프로브가 자동으로 비활성(작은 모델을 QCOM 에서 실행). 커스텀 big 모델 컴파일은 미지원(의도된 안전 동작).

## 웹 UI 특징

- **Material 3 카드 레이아웃** — `page_models.html` 에 인라인 CSS, carrot web 의 `--md-*` 디자인 토큰 사용
- **고정 높이 + 내부 스크롤** — `body[data-page="models"] { overflow: hidden }` + `.ms-list { flex:1; overflow-y:auto }` 로 리스트만 스크롤
- **정렬** — `added_at` DESC → `name` 자연 정렬 DESC (v11 > v10)
- **확인창** — `window.appConfirm(message, {title, confirmLabel})` (도구 탭 reboot 팝업과 동일)
- **카운트다운 통일** — 다운로드·재부팅 모두 0→100 방향으로 채워짐
- **네비 active 동기화** — `carrot:pagechange` 이벤트 리스너가 `btnModels.active` 토글
- **한글 UI** — "모델 셀렉터", "현재 모델", "설치중", "남은 공간", "사용 가능한 모델", "기본 모델 복원", 탭 라벨 "모델"
- **자동 bootstrap** — `#pageModels` 가 없으면 `/models/page_models.html` 프래그먼트를 `#swipeContainer` 에 주입 (idempotent)
- **PAGE_ELEMENTS 통합** — `showPage("models")` 로 다른 탭과 동일하게 동작, 탭 이탈 시 정상적으로 숨김 처리

## 유지보수 — upstream 자동 체리픽(Hermes) 대응

upstream 커밋은 주로 자동 체리픽으로 들어오므로, 셀렉터의 보호 장치는 사람의
기억(체크리스트, 수동 태그 갱신)에 의존하지 않고 전부 **자가 동작**한다:

- **컴파일 환경 스탬프 자동 태그** (`config.compile_env_tag()`)
  — `tinygrad_repo`, modeld 컴파일·warp·metadata 스크립트, 모델 셀렉터 installer와
  컴파일 플래그의 git 해시에서 태그를 도출. 기본 모델의 `.build_stamp` 도 같은 태그를
  사용한다. 이 경로에 닿는 커밋이 체리픽되면 태그가 저절로
  바뀌어 설치본 스탬프와 불일치 → 부팅 시 보존된 onnx 로 자동 재컴파일.
  git 을 못 쓰는 환경에서만 `_COMPILE_ENV_TAG_FALLBACK` 수동 관리.
- **helpers 훅 자가 점검** (`modeld_runner._upstream_hook_alive`)
  — 커스텀 supercombo 기동 직전에 `modeld_pkl_path()` 가 `MODELD_MODELS_DIR` 를
  실제로 반영하는지 확인. 훅이 스쿼시/체리픽으로 유실되면 (크래시 없이 조용히
  빌트인을 타는 유형이라 격리 폴백으로 못 잡는다) 명시적으로 로그 + 상태 파일
  기록 후 빌트인 폴백.
- **계약 점검 스크립트** (`check_contracts.py`)
  — 체리픽/머지 후 `python3 carrot/model_selector/check_contracts.py` 실행.
  컴파일 스크립트 존재, compile_modeld CLI 인자, 메타데이터 파일명 규약,
  pkl 직렬화 쌍(dump_oob/load_oob), SConscript↔config 플래그, fill_model_msg
  시그니처, ModelConstants, 프로세스/파람 배선, Gen C 엔진(벤더 tinygrad PIN·API,
  설치/실행 분기 배선, 셀렉터 버전 ≥5)을 검사한다. **FAIL 은 셀렉터
  코드를 함께 적응시켜야 한다는 신호다.** 표준 라이브러리만 사용하므로 빌드
  없는 체크아웃에서도 돈다. (Hermes 지시문에 "체리픽 후 이 스크립트 실행 후
  PASS 확인" 한 줄을 넣어둘 것)
- **미러 파일 변경 감지** (`check_contracts.py` 의 `modeld-mirror` 검사)
  — `carrot_modeld.py`/`carrot_parse_model_outputs.py`/`genc_modeld.py` 는 upstream `modeld.py`/
  `parse_model_outputs.py` 를 **미러링**하는 독립 엔진이라, 원본 로직이 바뀌어도
  (예: 커밋 67b6f17d44 의 has_wide_camera 분기) 임포트처럼 자동 반영되지 않는다.
  `upstream_baseline/` 스냅샷과 현재 파일을 비교해 마지막 리뷰 이후의 변경을
  FAIL 로 알린다. 대응: 스냅샷과 diff → 필요한 로직을 미러에 이식(무관하면 이식
  생략) → `python3 carrot/model_selector/check_contracts.py --sync-baselines` 로
  스냅샷 갱신 → 재실행해 PASS 확인.
