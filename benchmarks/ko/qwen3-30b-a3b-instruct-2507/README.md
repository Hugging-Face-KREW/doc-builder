# Qwen3-30B-A3B-Instruct-2507 한국어 번역 베이스라인

담당: 좌민주. 이 폴더에는 Qwen 모델의 결과만 저장합니다. 다른 참가자의
모델 폴더와 함께 머지하면 모델 5개 × 원문 5개를 독립적으로 비교할 수 있습니다.

현재 상태: **실측 완료, 파이프라인 최종 판정은 실패**. 5개 문서 중 4개는 기존
구조 검증을 통과했고, `tokenizer_summary.md`는 숫자 범위의 `~` 때문에 거절됐습니다.
전체 실행은 488.62초, API 보고 비용은 $0.00791014입니다.

[결과 보고서](RESULTS.md)에서 5개 결과와 문서별 시간을 확인할 수 있습니다.
`baseline/accepted/ko/`에는 캐시와 일치하는 통과 문서 4개 및 목차가 있고,
`baseline/rejected/ko/tokenizer_summary.md`는 마지막 모델 응답을 조립한 **거절된 진단용
산출물**입니다. 이 파일은 검증 통과본이 아닙니다. 원래 파이프라인은 정책대로
전체 결과의 게시를 중단했고, 측정 후 오프라인 분석만 산출물을 분리 저장했습니다.
번역문, 프롬프트, 용어집, 검증기 및 재시도 로직을 결과에 맞춰 수정하지 않았습니다.

## 고정 조건

- 베이스 브랜치: `translation-korean`, commit `773a807b08ba5a039d6ead6774545e748064f596`
- 실험 브랜치: `codex/qwen3-30b-a3b-instruct-2507`
- OpenRouter 모델: `qwen/qwen3-30b-a3b-instruct-2507` (유료 ID)
- 원문: Transformers commit `469230357aab0f2b303b0d638c1f8d06edb14184`
- 대상: `scripts/translation/ko-poc-pages.txt`의 5개 문서 및 필요한 sidebar
- 언어: `ko`, temperature: `0`, context budget: `16384`, output budget: `4096`
- 용어집, 프롬프트, parser, validation 및 retry: 베이스 브랜치 그대로
- 요청: 동시성 1, cold cache. 기존 어댑터의 순차 호출 동작 유지
- 공급자: OpenRouter 기본 라우팅에 가격 상한만 적용. 실제 공급자는 요청별 기록
- 가격 상한: 입력 $0.15/1M, 출력 $0.60/1M tokens. 실제 가격이 아니라 라우팅 제한
- 이 참가자의 실행 예산: 최대 $1 (팀에서 공유한 $5 전체를 사용하지 않음)

OpenRouter는 API 키가 필요합니다. 무료 모델도 인증은 필요하며, 이 모델의
ID에는 `:free`가 없습니다. 모델이나 공급자가 달라지면 별도 실험으로 취급합니다.
API 모델 ID는 immutable weight revision을 보장하지 않습니다.

## 재현

저장소 루트에서 [기존 설치 안내](../../../docs/openrouter-translation.md)에 따라
`uv venv`, `uv pip install -e .`, `npm ci --prefix kit`를 실행합니다.
`--env-file`을 사용하려면 `uv pip install python-dotenv`도 필요합니다.

```bash
# ../transformers는 아래 SHA로 checkout한 clean repository여야 합니다.
git -C ../transformers checkout --detach 469230357aab0f2b303b0d638c1f8d06edb14184

# 키 없이 수행 가능: 소스 고정, 5개 문서 구조 검사 및 실험 명세 기록
.venv/bin/python scripts/translation/benchmark_qwen3.py \
  --source ../transformers --prepare-only

# .env에는 OPENROUTER_API_KEY만 보관하고 커밋하지 않습니다.
# 이미 환경 변수로 설정한 경우 --env-file을 생략합니다.
.venv/bin/python scripts/translation/benchmark_qwen3.py \
  --source ../transformers --env-file .env

# 성공/구조 검증 실패 실행을 오프라인 재생: 추가 API 호출/과금 없음
.venv/bin/python scripts/translation/replay_qwen3.py --source ../transformers
```

동일 결과 폴더에 측정 요청이 이미 있으면 덮어쓰지 않습니다. 재실험은 새
`--results`와 새 `--output-dir`을 지정해야 하며, 동일 모델의 반복 실험임을 기록합니다.
실패한 실행의 cache는 `translation-output/`에 보존하지만 cold baseline으로 재사용하지 않습니다.
이미 커밋된 이번 결과를 재현할 때는 추론 명령이 아니라 오프라인 재생 명령을
사용합니다. 신규 측정은 예를 들어 `--results benchmarks/ko/qwen3-30b-a3b-instruct-2507/run-02
--output-dir translation-output/qwen3-run-02`처럼 두 경로를 모두 바꿉니다.
신규 실행의 최초 재생에는 `--cache translation-output/qwen3-run-02/.translation-cache.json`도 지정합니다.

## 측정값 해석

- `wall_seconds`: 번역·검증·재시도·파일 저장을 포함한 전체 실행 시간
- `api_roundtrip_seconds`: 각 HTTP 요청의 왕복 시간 합계. 네트워크·대기·생성 포함
- `median_api_roundtrip_seconds`: HTTP 요청 latency 중앙값
- `http_requests`, `http_errors`: HTTP 재시도까지 포함한 요청 수와 오류 수
- `validation_retry_http_requests`: 구조 검증 실패 후 재번역 단계의 HTTP 요청 수
- 토큰 수, 비용, 실제 공급자: API 응답에서 받은 값. 응답에 비용이 없으면 표시
- `per-page-metrics.json`: 문서별 API 왕복 시간, 토큰, 비용 및 구조 보존 결과
- `request-attribution.json`: 각 HTTP 요청이 속하는 문서와 번역 unit

오프라인 재생은 저장된 응답만 사용하며, 모든 요청 본문, 최종 성공/실패 판정,
통과 문서 캐시가 원래 실행과 정확히 같은지 검사한 뒤 문서별 수치를 저장합니다.
측정 로그나 번역을 교정하거나 API 호출을 반복하는 기능이 아닙니다.
거절된 문서는 `rejected/`에만 저장합니다. 이번에 관찰된 `~` 거절 이외의
오류는 진단용 문서 조립도 중단하며, 원래 gate의 성공/실패 판정을 바꾸지 않습니다.

OpenRouter에서 서버의 GPU 연산 시간이나 TTFT를 직접 측정하지 않으므로
API 왕복 시간을 순수 GPU 추론 시간으로 표현하지 않습니다. 서로 다른 공급자와
시간대에서 실행한 속도는 모델 자체의 속도로만 해석할 수 없습니다.
이번 측정은 로컬 CPU 단위 테스트가 병행된 단일 실행이며, 격리된 하드웨어에서
반복 측정한 성능 벤치마크는 아닙니다. 로컬 실행 환경은 Python 3.12.7,
Node.js 22.19.0이며 모델 추론은 OpenRouter 공급자 서버에서 수행했습니다.

`requests.jsonl`에는 실제 prompt와 모델 답변을 보존하고 인증 헤더는 기록하지
않습니다. `run.log`에는 기존 구조 검증의 실패 내역이 들어갑니다. 번역 결과를
사람이나 다른 모델로 교정하지 않아 베이스라인의 결함도 그대로 확인할 수 있습니다.

기존 한국어 문서는 다른 원문 revision을 반영할 수 있으므로 자동 정답으로
간주하지 않습니다. 블라인드 평가는 번역 정확성, 누락, 용어, 자연스러움 등을
동일한 고정 영문 원문과 대조해 사람이 수행합니다. 품질 gate 스킬은 이번에 추가하지 않습니다.

## 프롬프트

`baseline/prompt-examples.json`은 실제 코드가 생성한 프롬프트 예시입니다.
베이스라인에서는 변경하지 않습니다. `pin` 용어는 각 unit에 해당하는 항목만
추가되며, 실패 시 기존 어댑터가 작은 fragment로 나누어 재번역합니다.

이 결과만으로 최적의 프롬프트를 입증할 수는 없습니다. 후속 비교에서는
링크/강조 안의 자연어를 번역하되 tag 구조를 유지한다는 지시를 더 명확히 하는
후보와, 제목은 간결한 명사형으로 쓴다는 후보를 별도의 실험으로 평가할 수 있습니다.
이를 baseline 결과에 섞거나 검증 전 최적 프롬프트로 표시하지 않습니다.
[후속 프롬프트 후보](PROMPT-CANDIDATE.md)는 이번 실패를 토대로 작성했으며 아직 API로 평가하지 않았습니다.
