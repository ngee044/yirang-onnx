# yirang-onnx

모던 C++(C++23)로 작성한 **AI 구동 CLI 엔진 + ONNX 모델 파서**입니다. AI 팀이 전달한
`.onnx` 모델과 텐서(`.pb`)를 **잡 스크립트(`input_project.json`)** 하나로 받아
**ONNX Runtime 기반 추론**을 실행하고(랜덤 입력 생성·반복 벤치마크·출력 통계/JSON 덤프 포함),
부가적으로 모델의 **구조·연산자·텐서·메타데이터**를 추출해 분석/시각화합니다 —
재사용 가능한 라이브러리와 소형 CLI로 제공합니다.

> 포트폴리오 프로젝트. [CppToolkit](https://github.com/ngee044/CppToolkit)(C++23 유틸리티
> 라이브러리)을 git 서브모듈로, Protocol Buffers를 파싱 기반으로 사용합니다.

---

## 목적 / 개요

ONNX 모델은 [Protocol Buffers](https://protobuf.dev)로 직렬화된 단일 `ModelProto`입니다.
yirang-onnx는 공식 `onnx.proto` 스키마를 벤더링하고 `protoc`로 C++ 메시지 클래스를 생성한 뒤,
그 위에 확장하기 쉬운 얇은 파사드(`YirangOnnx::OnnxModel`)를 둡니다. 파사드는 순수 뷰
구조체와 3종 렌더러를 노출하여 다음을 수행합니다.

- **구동(Run)** — `--input-script input_project.json` 하나로 추론 잡을 기술: 모델 경로, 입력(`.pb` 파일 또는 랜덤 생성), 심볼릭 차원 바인딩, 반복/워밍업, 출력 옵션. **엔진 동작의 동적 변수는 대부분 이 스크립트에 담습니다** (스크립트가 CLI 인자보다 우선).
- **추론(Infer)** — ONNX Runtime으로 모델을 실행: 입력 TensorProto(`.pb`) 또는 랜덤 생성 텐서를 넣어 실제 출력 텐서를 얻고, 통계(min/max/mean)·`.pb`/JSON 저장·반복 벤치마크(avg/min/max ms)를 제공합니다.
- **조회(Inspect)** — 모델 메타데이터(IR 버전, producer, opset import, doc string, 커스텀 metadata).
- **분석(Analyze)** — 그래프 입력/출력(dtype + 형상), 노드(op type·I/O·속성 **값**), initializer 텐서(dtype·형상·바이트 크기·inline/external + external_data 메타데이터), 연산자 히스토그램(서브그래프 포함), 파라미터 총수·메모리 풋프린트.
- **시각화(Visualize)** — 계산 그래프를 [Graphviz](https://graphviz.org) DOT(그래프 입출력 노드 포함), 구조화된 JSON 리포트, 또는 사람이 읽는 요약으로 출력.

## 기능

- **잡 스크립트 구동**: `--input-script input_project.json` — model/inspect/inputs/dim_overrides/run/outputs를 한 파일로 기술(스크립트 우선, CLI는 보충). 손상 스크립트는 명시적 오류(exit 2).
- **랜덤 입력 생성**: `inputs` 생략 또는 `random` 스펙으로 그래프 입력 형상·dtype에 맞는 텐서를 자동 생성(FLOAT/DOUBLE/INT32/INT64/BOOL, seed 재현성, 심볼릭 차원은 `dim_overrides`).
- **벤치마크**: `run.warmup`/`run.repeat`로 반복 실행, avg/min/max ms 보고. 엔진은 세션을 1회 로드 후 재사용(`InferenceEngine::load/run` 분리).
- **출력 후처리**: 출력별 통계(min/max/mean), `.pb` 저장 토글, JSON 값 덤프(`dump_json`).
- **세션 튜닝**: 설정 파일에서 ONNX Runtime 세션 옵션(`intra_op_threads`/`inter_op_threads`, `enable_mem_pattern`, `enable_cpu_mem_arena`, `execution_mode`, `graph_optimization`)을 지정. 미지정 시 ORT 기본값을 유지하고, 적용된 값은 추론 시 로그로 보고.
- **입력 검증**: 잡 스크립트는 **엄격 검증**(알 수 없는 키 거부 → exit 2로 오타 차단, `_` 접두어는 주석 예외; `random.data_type`·과대 `shape`·`repeat`/`warmup` 상한을 파스 시점에 거부). 엔진 설정 파일은 **비치명 경고**(타입 불일치·범위 이탈·미지 키는 경고 후 기본값으로 진행 — 어떤 값도 프로세스를 중단시키지 않음).
- 파일 로드(`OnnxModel::load`) 또는 인메모리 버퍼 파싱(`OnnxModel::parse`). ONNX proto2 스키마(v1.17.0) 기반으로 opset에 무관하게 추출.
- 3종 출력 포맷: `summary`(텍스트, 파라미터 총수·총 바이트 포함), `json`(분석 — 노드 속성 값·external_data·총계 포함), `dot`(시각화 — 그래프 입출력 노드 표시). `--weights`로 초기화자(가중치) **실제 값**을 JSON에 포함. 서브그래프(If/Loop)의 노드는 히스토그램·노드 수에 합산.
- **추론 엔진**(`OnnxInference`, 별도 라이브러리·상시 빌드): ONNX Runtime으로 `.onnx` + 입력 텐서 → 출력. 파서와 protobuf-free `Tensor` 계약으로만 접점(느슨한 결합, MSA 지향).
- 파싱/IO 경계에서 오류는 예외를 던지지 않고 반환값으로 표현.
- 의존성이 가벼운 작은 파사드 — 스키마를 건드리지 않고 새 뷰를 추가하도록 설계.

## 아키텍처

```
OnnxCli (yirang-onnx)  ── Configurations + InputProject + InputBuilder + TensorConvert + RunSupport + RunCommand + main
   │ links                  (결과·오류는 Utilities::Logger 콘솔 / 파일로 단일화)
   ├──────────────┐
   ▼              ▼
OnnxParser        OnnxInference (YirangOnnx)
(YirangOnnx)         │ links
   │ links           ▼
   ▼              onnxruntime (Homebrew prebuilt)
onnx_proto (onnx.pb) ──► protobuf::libprotobuf

CppToolkit::Utilities  (서브모듈: File / ArgumentParser / JsonTool / Logger)
   ▲ links: OnnxCli · OnnxParser
```

- **`OnnxParser`** — 코어 라이브러리(네임스페이스 `YirangOnnx`). 파싱·추출·렌더. `onnx_proto`는
  생성된 `onnx.pb.{h,cc}`를 담는 OBJECT 라이브러리입니다.
- **`OnnxInference`** — 별도 추론 라이브러리. ONNX Runtime만 링크하고 protobuf에
  의존하지 않는 순수 엔진(`InferenceEngine`) — 파서와 느슨하게 결합됩니다. **ONNX Runtime
  필요**(`brew install onnxruntime`).
- **`OnnxCli`** — CLI 실행 파일 `yirang-onnx`. `Configurations`는 정적 엔진 설정(로깅·타이틀,
  선택적 `yirang_onnx_configurations.json` + CLI 인자)을, `InputProject`는 잡 스크립트
  (`input_project.json`, **스크립트 우선**)를 담당합니다. `InputBuilder`가 랜덤 입력 텐서를
  생성하고, `TensorConvert`가 TensorProto(`.pb`) ↔ 엔진 텐서 변환을, `RunCommand`가 추론
  실행 흐름(입력 구성 → 엔진 → 벤치마크 → 출력 저장/통계)을 담당합니다.
- **`.CppToolkit`** — 서브모듈. `Utilities` 모듈만 빌드합니다.

## 요구 사항

- C++23 컴파일러, **CMake 3.18+**(CMake Presets 사용 시 3.21+), **Ninja**, **vcpkg**(`~/vcpkg`에 있다고 가정).
- 의존성은 vcpkg(`vcpkg.json`)로 해결: protobuf, boost(json/system/filesystem/asio/dll), lz4, efsw, GoogleTest.
- **ONNX Runtime(추론 엔진에 필요):** 추론 엔진이 상시 빌드되므로 **ONNX Runtime**이 필요합니다 — `brew install onnxruntime`(prebuilt). 다른 위치면 `-DONNXRUNTIME_ROOT=<prefix>`.
- **macOS 참고:** 시스템 Apple clang는 C++23 stdlib(최신 macOS SDK libc++의 `__builtin_clzg`)
  를 컴파일하지 못하므로 **Homebrew LLVM clang**(`brew install llvm`)이 필요합니다. 이를
  적용하는 오버레이 트리플릿 + chainload 툴체인은 `cmake/vcpkg-triplets/`에, 프리셋은
  `CMakePresets.json`(프리셋 `macos-arm64`)에 **커밋**되어 있습니다. `build.sh`는 시스템
  컴파일러를 프로브해 필요 시 이 오버레이로 Homebrew LLVM clang을 사용합니다(vcpkg 포트
  포함).

## 빌드

전제(macOS arm64): **vcpkg**가 `~/vcpkg`에 있고, `brew install llvm onnxruntime`(시스템 Apple clang로는 C++23 stdlib 빌드가 안 되어 Homebrew LLVM clang 사용, ONNX Runtime은 prebuilt 필요).

```bash
git clone https://github.com/ngee044/yirang-onnx.git
cd yirang-onnx
git submodule update --init --recursive   # .CppToolkit 가져오기
brew install llvm onnxruntime             # 전제 (Homebrew LLVM clang + ONNX Runtime)

# 방법 1) 스크립트 (컴파일러 자동 프로브/폴백)
./build.sh
#   FRESH_DEPS=1 ./build.sh               # vcpkg.json 변경 후
#   BUILD_TYPE=Debug ./build.sh

# 방법 2) CMake Presets (VS Code / CLion / 신규 클론에 권장)
cmake --preset macos-arm64                # 커밋된 프리셋: Homebrew clang + vcpkg overlay
cmake --build build -j
```

**VS Code / CLion**: CMake Tools가 커밋된 `CMakePresets.json`을 읽습니다. Configure Preset으로 **`yirang-onnx (macOS arm64 + Homebrew LLVM)`**(`macos-arm64`)를 선택하세요. 자동 감지 kit(→ Apple clang)로는 vcpkg 의존성(boost-context 등)부터 실패합니다. 오버레이 트리플릿·chainload 툴체인은 `cmake/vcpkg-triplets/`에 커밋되어 있어 별도 설정이 필요 없습니다.

산출물:

- `build/out/yirang-onnx` — CLI
- `build/lib/libOnnxParser.a` — 코어 라이브러리
- `build/lib/libOnnxInference.a` — 추론 엔진
- `build/out/onnx_parser_tests` — 테스트

## 빠른 시작 (examples)

`examples/`에 바로 실행 가능한 최소 예제가 커밋되어 있습니다 — 작은 선형 모델
(`models/linear.onnx`, `y = x·W + b`), 고정 입력 텐서(`tensors/input.pb`), 잡 스크립트 4종.
잡 스크립트의 상대 경로가 이 디렉터리 기준이므로 `examples/` 안에서 실행합니다.

```bash
cd examples
BIN=../build/out/yirang-onnx

"$BIN" --model models/linear.onnx        # 분석: 요약
"$BIN" --input-script job_auto.json      # 추론: 전 그래프 입력 자동 랜덤
"$BIN" --input-script job_seed.json      # 추론: seed 고정 (두 번 실행 → 동일 결과)
"$BIN" --input-script job_bench.json     # 추론: 배치 [8,4] + repeat 100/warmup 10 벤치마크
"$BIN" --input-script job_path.json      # 추론: 실제 .pb 입력
```

모델·입력은 `make_linear.py`·`make_input.py`(`pip install onnx numpy`)로 재생성할 수 있습니다.
자세한 설명은 [`examples/README.md`](examples/README.md)를 참고하세요.

## 사용법 (CLI)

### 잡 스크립트 구동 (`--input-script`, 권장)

AI 팀이 전달한 모델·텐서를 구동하는 기본 인터페이스입니다. 동적 변수(모델 경로, 입력,
반복 횟수, 출력 옵션 등)는 전부 `input_project.json`에 담고, CLI의 기본 config
(`yirang_onnx_configurations.json`)는 정적 엔진 설정(로깅·타이틀)만 유지합니다.
같은 값이 스크립트와 CLI 인자에 모두 있으면 **스크립트가 우선**하고, 스크립트에 없는 값만
CLI 인자로 보충됩니다.

```bash
build/out/yirang-onnx --input-script input_project.json
```

```jsonc
// input_project.json — 전체 스키마 (예시: OnnxCli/input_project.sample.json)
{
	"model": "models/denoiser.onnx",                 // 대상 모델 (없으면 --model로 보충)
	"inspect": {                                     // 선택: 분석/시각화 실행
		"format": "summary|json|dot", "out": "report.json", "weights": false
	},
	"inputs": [                                      // 선택: 생략 시 모든 그래프 입력을 랜덤 생성
		"tensors/plain.pb",                          // 축약: 경로 문자열
		{ "name": "audio", "path": "tensors/audio_chunk.pb" },
		{ "name": "mask", "random": { "data_type": "FLOAT", "shape": [1, 257], "seed": 42 } },
		{ "name": "ids" }                            // 이름만: 그래프 입력 형상/dtype으로 랜덤 생성
	],
	"dim_overrides": { "N": 1 },                     // 심볼릭 차원 바인딩 (미지정 심볼릭은 1)
	"run": { "repeat": 10, "warmup": 2 },            // 반복 벤치마크 (avg/min/max ms 보고)
	"outputs": { "dir": "outputs", "save": true, "dump_json": false, "stats": true }
}
```

`inputs`/`run`/`outputs` 중 하나라도 있으면 추론이 실행되고, `inspect`가 있으면 분석이
함께 실행됩니다(둘 다 가능). 스크립트에 `model`만 있으면 요약 분석으로 동작합니다.

### 직접 인자 (개발/디버깅용)

```bash
# 사람이 읽는 요약 (Logger 콘솔 출력)
build/out/yirang-onnx --model model.onnx

# 분석용 JSON
build/out/yirang-onnx --model model.onnx --format json --out model.json

# Graphviz 시각화 (--out 파일은 로그 접두어 없는 원본 — 그대로 dot에 입력 가능)
build/out/yirang-onnx --model model.onnx --format dot --out graph.dot
dot -Tsvg graph.dot -o graph.svg

# 초기화자(가중치) 실제 값을 JSON에 포함
build/out/yirang-onnx --model model.onnx --format json --weights true --out params.json

# 추론 실행: 입력 TensorProto(.pb) → 출력 .pb
build/out/yirang-onnx --model model.onnx --input input.pb --out-dir outputs
#   여러 입력: --input a.pb,b.pb   (각 .pb의 name 이 그래프 입력명과 일치해야 함)
```

인자: `--input-script <input_project.json>`, `--model <path>`,
`--format summary|json|dot`(기본 `summary`), `--out <path>`,
`--weights <true|false>`(JSON에 가중치 값 포함), `--input <a.pb[,b.pb]>`(추론 입력 텐서),
`--out-dir <dir>`(추론 출력 위치), `--title <name>`, `--log_root_path <dir>`,
`--write_console_log <level>`, `--write_file_log <level>`, `--write_interval <ms>`,
`--help`, `--version`. `--input-script` 없이 `--input`이 주어지면 추론 모드로 동작하며,
각 그래프 출력이 `<out-dir>/<name>.pb`로 저장됩니다.

선택적 설정 파일: 바이너리 옆에 `yirang_onnx_configurations.json`을 두면 **정적 엔진
설정**(app_title/log_root_path/write_console/write_file/write_interval)과 **ONNX Runtime
세션 튜닝**(intra/inter_op_threads·enable_mem_pattern·enable_cpu_mem_arena·execution_mode·
graph_optimization)의 기본값을 지정할 수 있습니다. 이 파일 검증은 비치명적이라 잘못된
값은 경고만 남기고 기본값으로 진행합니다(엔진 구동을 막지 않음). 예시는
`OnnxCli/yirang_onnx_configurations.sample.json` 참고.

요약 출력 예시:

```
ONNX model summary
  ir_version    : 9
  producer      : pytorch 2.3.0
  graph         : main_graph
  opset_import  :
      ai.onnx v18
  nodes         : 245
  initializers  : 128
  parameters    : 11689512 (44.59 MiB)
  inputs        : 1
  outputs       : 1
  operators     :
        60  Conv
        58  Relu
        ...
```

## 사용법 (라이브러리)

```cpp
#include "OnnxModel.h"

using namespace YirangOnnx;

auto [model, error] = OnnxModel::load("model.onnx");
if (!model.has_value())
{
    // error.value() 에 실패 사유
    return;
}

const auto meta = model->metadata();
const auto nodes = model->nodes();
for (const auto& [op_type, count] : model->operator_histogram())
{
    // ...
}

const std::string json = model->to_json();
const std::string dot = model->to_dot();
```

CMake 타겟 `OnnxParser`를 링크하면 생성된 proto, protobuf, CppToolkit `Utilities`가
전이적으로 전파됩니다.

## 테스트

```bash
ctest --test-dir build --output-on-failure
```

GoogleTest 스위트(`tests/`, 84개)는 파서(파싱·추출·속성 값·서브그래프·external_data·
파라미터 총계·렌더링·오류 경로), CLI 설정(`Configurations` 기본값/CLI 파싱/엔진 설정 로드/
`--input-script`/`--help`·`--version`/손상 설정 경고 + **비치명 검증**: 타입 불일치·로그 레벨
범위·미지 키 경고·비정수 값 크래시 안전성 + **세션 튜닝** 파싱·검증), 잡 스크립트
(`InputProject` 전체 스키마/축약형/오류 거부 + **엄격 검증**: 미지 키 거부·`_` 주석 허용·
불량 dtype·과대 shape·repeat 상한), 랜덤 입력(`InputBuilder` 형상 해석/시드 재현성/미지원
dtype/과대 shape 거부), **추론 end-to-end**(`InferenceEngine` 세션 재사용·값 정확성·오류
경로·세션 튜닝 적용), 추론 실행 지원(`RunSupport` 입력 해석·출력 후처리·저장 검증),
TensorProto↔Tensor 변환(`TensorConvert` 왕복/미지원 dtype 거부)을 검증합니다.

추가로 **CLI 바이너리 스모크 15종**(`cli_smoke_*`)이 `examples/` 자산으로 실제 실행 파일을
구동해 종료 코드(0/1/2)·출력 패턴·산출물 생성(스크립트 우선순위, CLI 보충, `--input` 직접
추론, 미지 CLI 플래그 거부, 스크립트 상대 경로 해석 포함)을 ctest에서 자동 검증합니다.

## 디렉터리 레이아웃

```
yirang-onnx/
├── OnnxParser/     # 코어 라이브러리 (OnnxModel 파사드, ModelTypes)
├── OnnxInference/  # 추론 엔진 (InferenceEngine, Tensor) — ONNX Runtime
├── OnnxCli/        # CLI (yirang-onnx): OnnxCliCore(Configurations + InputProject + InputBuilder + TensorConvert + RunSupport + RunCommand) + main
├── tests/          # GoogleTest 스위트
├── examples/       # 실행 가능한 예제 (linear.onnx + 잡 스크립트 4종 + 입력 텐서)
├── proto/          # 벤더링된 onnx.proto (ONNX v1.17.0, Apache-2.0)
├── .CppToolkit/    # CppToolkit 서브모듈 (Utilities 모듈 사용)
├── cmake/vcpkg-triplets/  # 커밋된 overlay 트리플릿 + chainload 툴체인 (Homebrew LLVM)
├── build.sh        # vcpkg + Ninja 빌드 (macOS 컴파일러 폴백 포함)
├── vcpkg.json      # 매니페스트 (protobuf, boost, lz4, efsw, gtest)
├── CMakePresets.json  # 커밋 프리셋 (default / macos-arm64) — VS Code / 신규 클론
└── CMakeLists.txt
```

## 문서

- [`examples/README.md`](examples/README.md) — 실행 가능한 예제와 단계별 실습.
- **코드 스타일**: 후행 반환 타입(`auto f(void) -> Ret`), PascalCase 파일/클래스 · snake_case 함수/변수 · 멤버 `_` 접미사, 루트 `.clang-format`(GNU 기반, Tab 폭 4, 170 컬럼)에 위임. 오류는 예외 대신 `std::expected<void, string>` 또는 `tuple<optional<T>, optional<string>>` 반환값으로 표현.
- **CppToolkit 소비 규약**: flat include(경로 접두어 금지), 폴더명 ≠ 링크 타겟명 ≠ 네임스페이스, vcpkg 패키지명 ≠ `find_package` 이름 ≠ 링크 타겟.

## 기술 스택

C++23 · Protocol Buffers · ONNX Runtime · Boost.JSON · CppToolkit · vcpkg · CMake + Ninja · GoogleTest

## 라이선스

MIT — [LICENSE](LICENSE) 참조.
