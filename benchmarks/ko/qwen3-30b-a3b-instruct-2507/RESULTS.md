# Qwen 한국어 번역 베이스라인 결과

실행일: 2026-10-04. 원문 및 코드 revision, 프롬프트, 용어집 해시는
[manifest.json](baseline/manifest.json)에 고정했습니다. 추가적인 품질 gate 스킬,
다른 모델의 교정, 사람의 번역문 수정은 적용하지 않았습니다.

## 요약

- 번역 대상: 지정 문서 5개 + 목차. 번역 대상 unit 326개
- 전체 실행 시간: **488.62초 (8분 8.62초)**
- HTTP 왕복 시간 합: **480.05초**. 요청당 중앙값: **0.80초**
- 요청: **343회**, HTTP 오류: **0회**. 구조 재시도에서 17회 추가 호출
- 입력: **50,113 tokens**, 출력: **16,563 tokens**
- API가 보고한 총 비용: **$0.00791014115**. 비용 누락 응답: 0건
- 최종 gate: **문서 4/5 통과**, 목차 통과. 전체 publish는 **실패/중단**
- 코드 블록 48개: 진단용 거절 문서를 포함해 원문과 바이트 단위 동일
- 블라인드 사람 평가: **미실시**. 구조 통과율은 번역 품질 점수가 아님

숫자의 원본은 [metrics.json](baseline/metrics.json),
문서별 원본은 [per-page-metrics.json](baseline/per-page-metrics.json)입니다.

## 문서별 결과

시간과 비용은 재시도를 포함한 해당 문서의 API 호출 합계입니다. 로컬 parser,
검증, 저장 시간은 문서별로 배분하지 않았으며 전체 wall time에 포함됩니다.

| 문서 / 번역 결과 | API 시간(초) | 호출(재시도 호출) | 비용(USD) | 기존 gate |
| --- | ---: | ---: | ---: | --- |
| [Pipeline tutorial](../../../experiments/qwen-qwen3-30b-a3b-instruct-2507/ko/pipeline_tutorial.md) | 79.66 | 52 (0) | 0.00133367 | 통과 |
| [Tokenizer summary — 거절된 진단용](../../../experiments/qwen-qwen3-30b-a3b-instruct-2507/ko/tokenizer_summary.md) | 65.84 | 46 (4) | 0.00142349 | 실패 |
| [Training](../../../experiments/qwen-qwen3-30b-a3b-instruct-2507/ko/training.md) | 49.15 | 43 (8) | 0.00102074 | 통과 |
| [Selecting a quantization method](../../../experiments/qwen-qwen3-30b-a3b-instruct-2507/ko/quantization/selecting.md) | 104.23 | 113 (3) | 0.00178098 | 통과 |
| [Writing a chat template](../../../experiments/qwen-qwen3-30b-a3b-instruct-2507/ko/chat_templating_writing.md) | 163.64 | 76 (2) | 0.00224195 | 통과 |
| [목차 — 문서 5개에 포함하지 않음](../../../experiments/qwen-qwen3-30b-a3b-instruct-2507/ko/_toctree.yml) | 17.53 | 13 (0) | 0.00010931 | 통과 |

표의 통과 상태는 전체 publish 성공을 의미하지 않습니다. 한 문서라도 실패하면
원래 pipeline은 전체 출력 게시를 중단합니다. 위 4개는 그때 저장된 검증 통과
캐시를 오프라인 재생으로 대조한 결과입니다. 거절 파일은 최종 응답을 단지
검토용으로 조립했으며, `~`를 `-`로 고치는 등의 수정은 하지 않았습니다.

## 관찰한 실패

1차 326개 unit 중 6개가 거절됐습니다. 기존 fragment 재시도 후 4개는 복구되고,
Tokenizer의 2개는 남았습니다. 정확한 원문과 응답은 [run.log](baseline/run.log),
모든 실제 프롬프트와 응답은 [requests.jsonl](baseline/requests.jsonl)에 있습니다.

| 문서 / unit | 1차 실패 | 재시도 후 |
| --- | --- | --- |
| Chat template / 1.25 | `Python` 보호 태그 누락 | 구조 통과, 문장 품질 문제는 남음 |
| Quantization / 1.98 | 여러 링크 태그 누락 | 통과 |
| Tokenizer / 1.21 | `10-20%` → `10~20%`; 새 마크업으로 거절 | 동일 원인으로 실패 |
| Tokenizer / 1.22 | `2-4` → `2~4`; 새 마크업으로 거절 | 동일 원인으로 실패 |
| Training / 1.11 | 강조 태그 누락 | 통과 |
| Training / 1.32 | 링크/보호 태그 누락 및 중복 | 구조 통과, 문장 품질 문제는 남음 |

원인은 기존 `segment.accept_unit`이 보호 토큰 외부의 `~`를 금지한다는 점입니다.
자연스러운 한국어 범위 표기도 이 규칙에 걸릴 수 있으므로, 모델의 의미 정확도와
구조 게이트의 실패를 분리해서 해석해야 합니다. 이번 실험에서는 게이트를 수정하지 않았습니다.

## 사람이 검토할 실제 사례

아래는 형식적 점수나 블라인드 평가가 아니라, 다음 평가에서 확인할 표본입니다.

- Chat template의 재시도 결과 제목은 `비호환성에 대한 호환성Python 진자`입니다.
  보호 토큰은 남았지만 원래 “Compatibility with non-Python Jinja”의 의미와 표현이 깨졌습니다.
- Training의 추가 학습 자료 문장은 `읽기 [하위 클래스화합니다 Trainer 방법들입니다] …`로
  시작합니다. fragment를 다시 이어 붙인 결과가 구조 검사에는 통과해도 부자연스럽습니다.
- 제목에 `파이프라인입니다`, `미세 조정합니다.` 등 문장형 종결이 붙습니다.
  모든 unit에 적용되는 `합니다/입니다` 지시가 제목에도 영향을 주었을 가능성이 있습니다.
- Pipeline의 `accelerate시키고` 같은 표현은 용어집과 문맥 해석도 함께 점검해야 합니다.

따라서 이 모델을 번역 품질이 검증된 최종 후보로 선정할 근거는 아직 없습니다.
모델 간 선택은 같은 원문 revision의 25개 결과를 모아 사람이 블라인드 평가해야 합니다.

## 측정의 한계와 다음 실험

- 공급자는 Alibaba, DekaLLM, Nebius, PrimeIntellect, SiliconFlow, StreamLake가 섞였습니다.
  이는 OpenRouter 경유 시스템의 단일 실행이지 고정 GPU에서 측정한 모델 속도가 아닙니다.
- 로컬 CPU 테스트가 병행됐습니다. 순수 서버 추론 시간/TTFT는 측정하지 않았습니다.
- 기존 한국어 번역은 원문 revision이 달라 자동 정답으로 사용하지 않았습니다.
- 현 프롬프트 예시는 [prompt-examples.json](baseline/prompt-examples.json),
  개선 실험 후보는 [PROMPT-CANDIDATE.md](PROMPT-CANDIDATE.md)에 분리했습니다.
  후보는 미검증이며 이번 baseline에 적용하지 않았습니다.
- 공식 `huggingface/transformers`에는 커밋/PR을 만들지 않습니다.
  이 실험 branch는 `Hugging-Face-KREW/doc-builder`의 `translation-korean`을 기준으로 합니다.
