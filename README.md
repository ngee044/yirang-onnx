# yirang-onnx

모던 C++(C++23)로 작성한 경량·확장형 **ONNX 모델 파서**입니다. `.onnx` 파일을 읽어
모델의 **구조·연산자·텐서·메타데이터**를 추출하여 분석 및 시각화합니다 — 재사용 가능한
라이브러리와 소형 CLI로 제공합니다.

> 포트폴리오 프로젝트. [CppToolkit](https://github.com/ngee044/CppToolkit)(C++23 유틸리티
> 라이브러리)을 git 서브모듈로, Protocol Buffers를 파싱 기반으로 사용합니다.

---

## 목적 / 개요

ONNX 모델은 [Protocol Buffers](https://protobuf.dev)로 직렬화된 단일 `ModelProto`입니다.
yirang-onnx는 공식 `onnx.proto` 스키마를 벤더링하고 `protoc`로 C++ 메시지 클래스를 생성한 뒤,
그 위에 확장하기 쉬운 얇은 파사드(`YirangOnnx::OnnxModel`)를 둡니다. 파사드는 순수 뷰
구조체와 3종 렌더러를 노출하여 다음을 수행합니다.

- **조회(Inspect)** — 모델 메타데이터(IR 버전, producer, opset import, doc string, 커스텀 metadata).
- **분석(Analyze)** — 그래프 입력/출력(dtype + 형상), 노드(op type·I/O·속성), initializer 텐서(dtype·형상·바이트 크기·inline/external), 연산자 히스토그램.
- **시각화(Visualize)** — 계산 그래프를 [Graphviz](https://graphviz.org) DOT, 구조화된 JSON 리포트, 또는 사람이 읽는 요약으로 출력.

## 기능

- 파일 로드(`OnnxModel::load`) 또는 인메모리 버퍼 파싱(`OnnxModel::parse`).
- ONNX proto2 스키마(v1.17.0) 기반으로 opset에 무관하게 추출.
- 3종 출력 포맷: `summary`(텍스트), `json`(분석), `dot`(시각화).
- 파싱/IO 경계에서 오류는 예외를 던지지 않고 반환값으로 표현.
- 의존성이 가벼운 작은 파사드 — 스키마를 건드리지 않고 새 뷰를 추가하도록 설계.

## 아키텍처

```
OnnxCli (yirang-onnx)      Configurations + Logger + main → stdout / --out 파일
      │ links
OnnxParser (YirangOnnx)    OnnxModel 파사드 + ModelTypes 뷰 구조체
      │ links                     │ links
onnx_proto (onnx.pb)         CppToolkit::Utilities   (서브모듈: File / ArgumentParser
proto/onnx.proto 에서 생성                            / JsonTool / Logger)
```

- **`OnnxParser`** — 코어 라이브러리(네임스페이스 `YirangOnnx`). `onnx_proto`는 생성된
  `onnx.pb.{h,cc}`를 담는 OBJECT 라이브러리입니다.
- **`OnnxCli`** — CLI 실행 파일 `yirang-onnx`. `Configurations`가 선택적 JSON 파일과
  CLI 인자를 읽어(CLI 우선) 파서를 구동합니다.
- **`.CppToolkit`** — 서브모듈. `Utilities` 모듈만 빌드합니다.

## 요구 사항

- C++23 컴파일러, **CMake 3.18+**, **Ninja**, **vcpkg**(`~/vcpkg`에 있다고 가정).
- 의존성은 vcpkg(`vcpkg.json`)로 해결: protobuf, boost(json/system/filesystem/asio/dll), lz4, efsw, GoogleTest.
- **macOS 참고:** 시스템 Apple clang가 C++23을 컴파일하지 못하는 경우(최신 macOS SDK
  libc++가 `__builtin_clzg`를 요구), `build.sh`가 프로젝트와 vcpkg 의존성 빌드 모두를
  **Homebrew LLVM clang**(`brew install llvm`)으로 자동 전환합니다. 이를 위한 별도 설정
  파일은 리포에 커밋되지 않습니다(필요 시 `build/` 아래에 자동 생성).

## 빌드

```bash
git clone https://github.com/ngee044/yirang-onnx.git
cd yirang-onnx
git submodule update --init --recursive   # .CppToolkit 가져오기

./build.sh                                # Release 빌드 (vcpkg + Ninja + C++23)
#   FRESH_DEPS=1 ./build.sh               # vcpkg.json 변경 후
#   BUILD_TYPE=Debug ./build.sh
```

산출물:

- `build/out/yirang-onnx` — CLI
- `build/lib/libOnnxParser.a` — 코어 라이브러리
- `build/out/onnx_parser_tests` — 테스트

## 사용법 (CLI)

```bash
# 사람이 읽는 요약 (stdout)
build/out/yirang-onnx --model model.onnx

# 분석용 JSON
build/out/yirang-onnx --model model.onnx --format json

# Graphviz 시각화
build/out/yirang-onnx --model model.onnx --format dot --out graph.dot
build/out/yirang-onnx --model model.onnx --format dot | dot -Tsvg -o graph.svg
```

인자: `--model <path>`(필수), `--format summary|json|dot`(기본 `summary`),
`--out <path>`, `--title <name>`, `--log_root_path <dir>`, `--write_console_log <level>`,
`--write_file_log <level>`, `--write_interval <ms>`.

선택적 설정 파일: 바이너리 옆에 `yirang_onnx_configurations.json`을 두면 기본값을 설정할 수
있습니다(CLI 인자가 우선). 예시는 `OnnxCli/yirang_onnx_configurations.sample.json` 참고.

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

GoogleTest 스위트(`tests/`)는 인메모리로 소형 모델을 만들어 파싱·추출·렌더링과 오류
경로(빈 버퍼, 부재 파일)를 검증합니다.

## 디렉터리 레이아웃

```
yirang-onnx/
├── OnnxParser/     # 코어 라이브러리 (OnnxModel 파사드, ModelTypes)
├── OnnxCli/        # CLI (yirang-onnx): Configurations + main
├── tests/          # GoogleTest 스위트
├── proto/          # 벤더링된 onnx.proto (ONNX v1.17.0, Apache-2.0)
├── .CppToolkit/    # CppToolkit 서브모듈 (Utilities 모듈 사용)
├── docs/           # ISO 문서 (SRS/SAD/STP/RTM) + 코딩·모듈 가이드
├── build.sh        # vcpkg + Ninja 빌드 (macOS 컴파일러 폴백 포함)
├── vcpkg.json      # 매니페스트 (protobuf, boost, lz4, efsw, gtest)
└── CMakeLists.txt
```

## 문서

`docs/`에는 요구사항 추적성을 갖춘 ISO 기반 엔지니어링 문서가 있습니다.

- `SRS.md` (ISO/IEC/IEEE 29148) — 요구사항
- `SAD.md` (ISO/IEC/IEEE 42010) — 아키텍처
- `STP.md` (ISO/IEC/IEEE 29119-3) — 테스트 계획 및 결과
- `RTM.md` — 요구사항 ↔ 설계 ↔ 소스 ↔ 테스트 추적성
- `CODING_CONVENTION.md`, `module_usage_guide.md` — 코드 스타일 및 CppToolkit 사용 규약

## 기술 스택

C++23 · Protocol Buffers · Boost.JSON · CppToolkit · vcpkg · CMake + Ninja · GoogleTest

## 라이선스

MIT — [LICENSE](LICENSE) 참조.
