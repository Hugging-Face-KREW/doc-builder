# NVIDIA Nemotron 3.5 Lightning 한국어 번역 실험

OpenRouter 유료 모델 `nvidia/nemotron-3.5-lightning`로 테스트한 Transformers 문서 5개입니다. **3개는 구조 검증을 통과했고, 2개는 재시도 후에도 실패했습니다.** 실패한 문서도 마지막 모델 응답을 조립한 검토용 초안으로 포함했습니다.

| 문서 | 파이프라인 결과 | 출력 |
| --- | --- | --- |
| Pipeline tutorial | 통과 | [pipeline_tutorial.md](ko/pipeline_tutorial.md) |
| Tokenizer summary | 실패: 1개 단위 미해결 | [tokenizer_summary.md](ko/tokenizer_summary.md) |
| Fine-tuning / Training | 통과 | [training.md](ko/training.md) |
| Selecting a quantization method | 실패: 2개 단위 미해결 | [quantization/selecting.md](ko/quantization/selecting.md) |
| Writing a chat template | 통과 | [chat_templating_writing.md](ko/chat_templating_writing.md) |

[목차](ko/_toctree.yml)도 구조 검증을 통과했습니다. 통과 여부는 번역 품질 승인을 뜻하지 않습니다. 표본에서 영어 잔여, 어색한 조사와 재시도 조각 사이의 문장 단절이 관찰됐습니다. 문서 빌드·렌더링 및 전체 번역 내용 검수는 수행하지 않았습니다. 선택하지 않은 문서로 연결되는 링크의 대상 파일은 포함하지 않습니다.

## 설정과 집계

- 원문: `huggingface/transformers`, 커밋 `469230357aab0f2b303b0d638c1f8d06edb14184`
- 생성 당시 파이프라인: `773a807b08ba5a039d6ead6774545e748064f596`
- 실행: 2026-10-04 15:22:53–15:28:38 KST, 345.2초
- 설정: 컨텍스트 16,384, 출력 한도 4,096, 재시도 출력 한도 8,192, temperature 0, 검증 2회
- 실험 실행기에서 `reasoning.enabled=false`를 요청 본문에 추가했습니다.
- 초기 번역 단위: 본문 313개 + 목차 13개 = 326개
- 실제 API 응답: **427개** = 초기 326개 + 재시도 조각 101개
- 1차 검증 실패: 32개 단위; 재시도 후 미해결: 3개 단위
- 입력 62,459 / 출력 15,800 토큰; API 보고 비용 **$0.006292575**
- HTTP 오류·요청 시간 초과: 0회

무료 모델의 출력은 재사용하지 않았습니다. 현재 공유 브랜치의 최신 파이프라인·용어집으로 재생성한 결과는 아닙니다.

## 실패한 문서 보존

파이프라인은 문서 하나라도 실패하면 전체 문서 저장을 중단합니다. 통과한 3개와 목차는 당시 캐시에서 내용 변경 없이 추출했습니다. 실패한 2개는 원래 파이프라인으로 저장된 427개 응답을 오프라인 재생한 뒤, 각 단위의 마지막 출력과 원문의 보호된 코드·링크·구조를 조립했습니다. 검증을 통과하지 못한 출력도 그대로 남겼으며, 추가 API 호출이나 번역 교정은 하지 않았습니다.

| 실패 위치 | 마지막 출력의 문제 |
| --- | --- |
| 양자화 문서 1.2 | 원문에 없는 `[누락된 내용]`이 추가돼 마크업 검사 실패 |
| 양자화 문서 1.71 | `Activation-aware Weight Quantization`을 원문 그대로 반환 |
| 토크나이저 문서 1.21 | `10~20%`의 `~`가 마크업으로 판정됨; `기본 한글`이라는 의미 추가와 어색한 조각 연결도 관찰됨 |

실패한 파일의 상태는 [results.json](results.json)에 `failed`, 내보내기 방식은 `reconstructed_failed_draft`로 기록했습니다. 문서 5개 모두 원문의 코드 블록을 그대로 유지했는지 확인했으며, 총 48개가 일치합니다.

## 제공자와 기록

OpenRouter 자동 라우팅을 사용했습니다. 제공자는 Darkbloom 152회, DeepInfra 84회, Io Net 73회, Phala 65회, CoreWeave 53회였습니다. [실행 당시 엔드포인트 메타데이터](logs/model-endpoints.json)의 정밀도는 Darkbloom INT4, DeepInfra·CoreWeave BF16, 나머지는 unknown입니다. 따라서 담당 HF NVFP4 체크포인트만의 품질을 분리한 실험으로 해석할 수 없습니다.

- [results.json](results.json): 설정, 문서별 상태·해시, 토큰·비용·시간, 제공자와 내보내기 방식
- [requests.jsonl](logs/requests.jsonl): 427개 요청 본문과 원시 응답; API 키·HTTP 헤더 제외
- [unit-responses.jsonl](logs/unit-responses.jsonl): 조립된 단위 출력 358개, 검증 오류와 원시 응답 ID
- [failures.json](logs/failures.json): 최초 실패와 재시도 후 남은 실패 기록
- [pipeline.log](logs/pipeline.log): 당시 실행 로그; 로컬 저장소 경로는 `<repo>`로 치환
