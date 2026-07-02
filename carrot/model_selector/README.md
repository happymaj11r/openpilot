# Carrot Model Selector

Carrot 파일럿 전용 주행 모델 다운로드/설치/적용 모듈. c3x·c4 하드웨어 공용으로
**carrot web(포트 7000)** 에서 UI를 제공한다.

## 설계 원칙

1. **엔진 분리 + 신구조는 upstream 엔진 재사용**
   - 기본 모델 (`selfdrive/modeld/models/`) → **upstream `selfdrive/modeld/modeld.py` 가 그대로 담당**
   - 커스텀 **신모델(supercombo, lebowski 구조)** (`/data/models/driving_tinygrad.pkl`)
     → **upstream `modeld.py` 를 그대로 재사용** — `MODELD_MODELS_DIR=/data/models` env 만 설정
     (`selfdrive/modeld/helpers.py::modeld_pkl_path()` 가 env 를 읽어 경로 전환, monkey-patch 없음)
   - 커스텀 **구모델(vision+policy 분리)** (`/data/models/`) → **`carrot.model_selector.carrot_modeld` 가 담당** (3-모델 on/off policy 자체 지원)
   - `modeld_runner` 가 부팅 시 `/data/models` 내용으로 셋 중 하나를 택일 실행
   - upstream 이 바뀌어도 legacy 엔진과 무관하게 진행 가능

2. **upstream 파일 최소 침습** (각 1~3줄)
   - `system/manager/process_config.py` — modeld 엔트리를 `carrot.model_selector.modeld_runner` 로 교체
   - `system/manager/manager.py::main()` 초입 — `boot_compile.run()` 호출
   - `selfdrive/modeld/helpers.py::modeld_pkl_path()` — `MODELD_MODELS_DIR` env 오버라이드 (2줄)
   - `selfdrive/carrot/server/app_factory.py` — 라우터 등록
   - `selfdrive/carrot/web/index.html` — 네비 버튼 + `page_models.js` 로드 (`?v=…` 캐시 버스팅)
   - `selfdrive/ui/mici/layouts/home.py` — c4 홈 화면 커밋해시 옆에 `DrivingModelName` 표시
   - `common/params_keys.h` — `DrivingModelName`, `PendingModelName` 파람 등록

3. **독립 패키지**
   - 모델 셀렉터 자체 엔진 포함 전부 `carrot/model_selector/` 에 완결
   - `carrot_parse_model_outputs.py` 는 3-모델 분기가 있는 c3-ms 파서 이식본
   - 원본 파일(`selfdrive/modeld/*`, `common/file_chunker.py` 등)은 읽기/import 전용 (helpers.py env 훅 제외)

4. **파일 세트 호환** (신형 단일 onnx 또는 구형 vision + 정책)
   - **신형(supercombo)**: `driving_supercombo.onnx` 단독 — 존재 시 우선 적용
   - **구형(legacy)**: `driving_vision` 필수 + (`driving_on_policy` 또는 `driving_policy`) 중 하나 필수
   - `driving_off_policy` 는 선택적, 존재 시 3-모델 아키텍처 활성화 (구형 전용)

## 모듈 구조

| 파일 | 책임 |
|------|------|
| `config.py` | 경로·파일명 상수, 허용 onnx 목록 (supercombo 포함), tinygrad 컴파일 플래그 |
| `keys.py` | Ed25519 공개키 맵 + `MODEL_SELECTOR_VERSION` (v4: supercombo 지원) |
| `manifest.py` | `models.json` fetch (캐시버스터 `?t=unixtime` + no-cache 헤더) + canonical JSON Ed25519 검증. snake_case/camelCase 필드 둘 다 수용 |
| `downloader.py` | ONNX 스트리밍 다운로드 + SHA256/크기 검증. 공백 포함 파일명 URL percent-encoding. supercombo 단독 또는 legacy 세트 허용 |
| `validator.py` | `/data/models/` 파일 세트 유효성 검증 (`has_supercombo` / legacy 세트) + `describe()` 라벨 |
| `installer.py` | 신형: `compile_modeld.py` 로 통합 pkl 생성. 구형: tinygrad 컴파일 (compile3) + warp pkl. 공통: atomic swap + 실패 시 backup 복원 |
| `carrot_modeld.py` | **자체 modeld 엔진** — 구형 2/3-모델 공용, `/data/models` 전담 |
| `carrot_parse_model_outputs.py` | 자체 parser (c3-ms 이식, `parse_off_policy_outputs` 포함) |
| `modeld_runner.py` | 부팅 시 supercombo(upstream+env) / legacy(carrot_modeld) / 기본(upstream) 3-way 택일 실행 (dispatcher). `/data/model_selector_status` 에 엔진 정보 기록 |
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
       ├─ [신형] driving_supercombo.onnx 존재 시:
       │    └─ compile_modeld.py --model-size(metadata img 크기) --camera-resolutions 1928x1208 1344x760
       │       --frame-skip(MODEL_RUN_FREQ//MODEL_CONTEXT_FREQ)
       │       → 통합 driving_tinygrad.pkl (metadata + 모델 JIT + 해상도별 warp JIT)
       ├─ [구형] ONNX → tinygrad.pkl + metadata.pkl (QCOM/FLOAT16/NOLOCALS/JIT_BATCH_SIZE=0/IMAGE=1/OPENPILOT_HACKS=1)
       │    └─ selected model metadata의 img 입력 크기로 standalone warp pkl 새로 생성 (빌트인 warp 재사용 안 함)
       ├─ atomic swap: /data/models_tmp → /data/models  (실패 시 backup 복원)
       └─ Params: DrivingModelName 기록, PendingModelName 삭제

[process manager → `modeld` 프로세스 (only_onroad)]
  └─ modeld_runner.main()
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
`engine` 값: `upstream_modeld_custom`(신형 supercombo 커스텀) / `carrot_modeld`(구형 커스텀) / `upstream_modeld`(기본).
`ps` 는 `setproctitle` 때문에 Python 모듈 경로가 마스킹되므로 이 파일이 결정적 출처.

## 안전장치

- **설치 전 metadata 검증** (`installer._validate_supercombo_metadata`)
  — 필수 입력(img/big_img/desire_pulse/traffic_convention/action_t/features_buffer),
  `hidden_state` output slice, 모델 입력 크기 512x256(MEDMODEL — 런타임 warp 행렬이 이 크기에 고정) 을
  확인하고 미달 시 설치를 중단(백업 복원). 비호환 모델이 설치되어 크래시 루프에 빠지는 것을 사전 차단.
- **크래시 루프 차단기** (`modeld_runner._arm_crash_loop_breaker`)
  — 커스텀 엔진 기동 시 `/data/models/.load_attempts` 카운터 증가, 90초 생존 시 자동 삭제.
  3회 연속 조기 크래시(손상 pkl, 포크 업데이트 후 tinygrad 버전 불일치 등) 시 `/data/models` 를
  `/data/models_quarantined` 로 격리하고 기본 내장 모델로 자동 폴백. 수동 reset 없이 주행 가능 상태 복구.
- **stale PendingModelName 방지**
  — 설치 시작 시(다운로드 전) 이전 pending 제거(`routes.api_install`),
  부팅 시 tmp 가 불완전하면 pending 도 함께 제거(`installer.compile_pending`).
  실패한 다운로드의 잔여 파일이 예전 모델명으로 컴파일되거나 UI 가 영구 "설치중" 이 되는 문제 방지.
- **PC 컴파일 백엔드 프로브** (`installer._probe_tg_devices`)
  — 비-TICI 환경에서는 SConscript 와 동일하게 tinygrad 디바이스를 프로브해 CUDA > CPU 순으로 선택
  (런타임 tg_input_devices.json 과 백엔드 불일치 방지). 실기기는 QCOM 고정.
- **USBGPU 동작** — 커스텀 supercombo 설치 시 `/data/models/big_driving_tinygrad.pkl` 이 없으므로
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
