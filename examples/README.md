# examples — yirang-onnx 실습 예제

`yirang-onnx`를 바로 돌려볼 수 있는 최소 예제 세트입니다. 작은 선형 모델(`Gemm`, `y = x·W + b`)과 잡 스크립트가 들어 있습니다. 개념 설명과 단계별 해설은 `docs/hands_on_guide.md`를 참고하세요.

- 입력 `input`: FLOAT `[N, 4]` (`N`은 심볼릭 배치 차원)
- initializer `W`[4,3], `b`[3] — 모델 내장 가중치(사용자 입력 아님)
- 출력 `output`: FLOAT `[N, 3]`

## 구성

| 파일 | 설명 |
|------|------|
| `make_linear.py` | `models/linear.onnx` 생성 (onnx 파이썬 패키지 필요) |
| `make_input.py` | `tensors/input.pb` 생성 (고정 입력 `[[1,2,3,4]]`) |
| `models/linear.onnx` | 생성된 실습 모델 (파이썬 없이 바로 실습 가능) |
| `tensors/input.pb` | 고정 입력 텐서 |
| `job_auto.json` | 입력 생략 → 전 그래프 입력 자동 랜덤 추론 |
| `job_seed.json` | seed 고정 랜덤 입력 (재현성) |
| `job_bench.json` | inspect + 배치 `[8,4]` + `repeat 100/warmup 10` 벤치마크 |
| `job_path.json` | 실제 `.pb`(`tensors/input.pb`) 입력 추론 |

## 실행

모든 명령은 **이 `examples/` 디렉터리 안에서** 실행합니다(잡 스크립트의 상대 경로가 여기 기준).

```bash
cd examples

# (선택) 모델·입력을 다시 만들려면 — 이미 들어 있으므로 생략 가능
python3 -m pip install onnx numpy
python3 make_linear.py     # models/linear.onnx
python3 make_input.py      # tensors/input.pb

BIN=../build/out/yirang-onnx

# 1) 분석 (inspect)
"$BIN" --model models/linear.onnx                                  # 요약
"$BIN" --model models/linear.onnx --format json --out linear.json  # JSON
"$BIN" --model models/linear.onnx --format json --weights true     # 가중치 값 포함
"$BIN" --model models/linear.onnx --format dot --out linear.dot    # 시각화

# 2) 추론 (infer)
"$BIN" --input-script job_auto.json     # 자동 랜덤
"$BIN" --input-script job_seed.json     # seed 고정 (두 번 실행 → 동일)
"$BIN" --input-script job_bench.json    # 벤치마크
"$BIN" --input-script job_path.json     # 실제 .pb 입력

# 3) CLI 직접 인자로 실제 입력
"$BIN" --model models/linear.onnx --input tensors/input.pb --out-dir outputs
```

추론 산출물은 `outputs/`(잡 스크립트) 또는 `--out-dir` 위치에 `<출력이름>.pb`(+ `dump_json` 시 `.json`)로 저장됩니다. 이 산출물과 inspect 결과 파일은 `.gitignore`로 제외됩니다.

> 전제: 리포 루트에서 `./build.sh`로 `build/out/yirang-onnx`가 빌드되어 있어야 하고, 추론에는 ONNX Runtime(`brew install onnxruntime`)이 필요합니다.
