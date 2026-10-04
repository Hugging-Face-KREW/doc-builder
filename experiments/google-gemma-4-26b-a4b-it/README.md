# google/gemma-4-26b-a4b-it 한국어 번역 실험

OpenRouter의 유료 모델 `google/gemma-4-26b-a4b-it`로 생성한 Transformers 문서 5개의 한국어 번역 초안입니다. 기존 결과 브랜치의 번역본과 목차를 내용 변경 없이 이 폴더로 옮겼습니다.

- 원문: `huggingface/transformers`, 커밋 `469230357aab0f2b303b0d638c1f8d06edb14184`
- 생성 당시 파이프라인: `773a807` (`translation-korean` 브랜치)
- 설정: 컨텍스트 16,384, 기본 출력 한도 4,096, 재시도 출력 한도 8,192, temperature 0
- 처리: 본문·목차 326개 초기 번역 단위, 보호 마커 검증 및 실패 단위 재시도
- API 호출 수·토큰·비용·소요 시간: 당시 어댑터에서 기록하지 않아 확인 불가

| 원문 문서 | 번역 결과 |
| --- | --- |
| Pipeline tutorial | [pipeline_tutorial.md](ko/pipeline_tutorial.md) |
| Tokenizer summary | [tokenizer_summary.md](ko/tokenizer_summary.md) |
| Fine-tuning / Training | [training.md](ko/training.md) |
| Selecting a quantization method | [quantization/selecting.md](ko/quantization/selecting.md) |
| Writing a chat template | [chat_templating_writing.md](ko/chat_templating_writing.md) |

본문 5개는 모델 출력 그대로입니다. 목차 생성 과정에서는 임시로 `Pipeline API → 파이프라인 API` 용어를 지정했으나, 사용자 요청에 따라 최종 목차 제목을 `Pipeline API`로 복원하고 용어집 추가도 되돌렸습니다. 이 제목은 원문 그대로 두는 명시적 검증 예외입니다.

[results.json](results.json)에 원문 링크, 파일 해시, 생성 설정과 검증 정보를 기록했습니다. 문서 구조와 코드 블록 48개 보존을 검증했습니다. 문서 빌드·렌더링 및 번역 내용 검수는 수행하지 않았습니다. 선택하지 않은 문서로 연결되는 링크의 대상 파일은 포함하지 않습니다.

현재 공유 브랜치의 최신 파이프라인·용어집으로 재생성한 결과는 아닙니다. 무료 모델(`:free`)은 제공자 호출 제한으로 실패했으며 이 폴더는 유료 모델의 결과만 포함합니다.
