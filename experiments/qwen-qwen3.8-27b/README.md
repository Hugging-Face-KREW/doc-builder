# qwen/qwen3.8-27b 한국어 번역 실험

OpenRouter의 `qwen/qwen3.8-27b`로 생성한 Transformers 문서 5개의 한국어 번역 초안입니다. 생성된 문서와 사이드바 제목을 수정 없이 보관했습니다.

- 원문: `huggingface/transformers`, 커밋 `469230357aab0f2b303b0d638c1f8d06edb14184`
- 생성 당시 파이프라인: `1167ecd` (`translation-korean-qwen3.8-27b` 브랜치)
- 설정: 컨텍스트 65,536, 기본 출력 한도 16,384, 재시도 출력 한도 32,768, temperature 0, 추론(reasoning) 끔, 동시 요청 8개
- 처리: 본문·목차 326개 초기 번역 단위, 보호 마커 검증 및 실패 단위 재시도, 숫자 범위의 `~`를 `-`로 변환
- API 호출 수·토큰·소요 시간: 어댑터에서 기록하지 않아 확인 불가. 키 전체 사용액 약 $0.42에는 중단한 추론 켬 실행(약 $0.28)과 진단 호출이 포함되어 있어 실제 결과 생성 비용보다 큽니다.

| 원문 문서 | 번역 결과 |
| --- | --- |
| Pipeline tutorial | [pipeline_tutorial.md](ko/pipeline_tutorial.md) |
| Tokenizer summary | [tokenizer_summary.md](ko/tokenizer_summary.md) |
| Fine-tuning / Training | [training.md](ko/training.md) |
| Selecting a quantization method | [quantization/selecting.md](ko/quantization/selecting.md) |
| Writing a chat template | [chat_templating_writing.md](ko/chat_templating_writing.md) |

[results.json](results.json)에 원문 링크, 파일 해시, 생성 설정과 검증 정보를 기록했습니다. 문서 구조와 코드 블록 48개 보존을 검증했습니다. 문서 빌드·렌더링 검증은 수행하지 않았습니다.

이 모델은 추론 모델이라 기본 설정에서는 문장마다 수백~수천 토큰을 추론해 매우 느렸습니다(단위당 평균 약 17초). 그래서 추론을 끄고 생성했으며, 추론을 켠 다른 실험과 조건이 다릅니다. 현재 공유 브랜치의 최신 어댑터(`026ac0a`)로 재생성한 결과는 아닙니다.

알려진 문제(사람 검수 필요):

- `pipeline_tutorial.md` 21번째 줄: 첫 시도에서 마커가 누락되어 조각 단위로 재번역되면서 문장이 어색하게 끊기고 `accelerate`가 영어로 남았습니다.
- `quantization/selecting.md`: calibration을 "교정"과 "보정"으로 혼용했고, 제품명 `compressed-tensors`를 "압축된 텐서"로 번역했으며, "loading specific formats"를 "로드하는 중"으로 옮겼습니다.
- 기존 한국어 문서는 "미세 튜닝"을, 이 결과는 "미세 조정"을 사용합니다.
