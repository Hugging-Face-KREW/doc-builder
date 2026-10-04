# qwen/qwen3-30b-a3b-instruct-2507 한국어 번역 실험

담당: 좌민주. OpenRouter로 생성한 지정 Transformers 문서 5개의 번역 결과입니다.
기존 baseline 산출물을 내용 변경 없이 이 폴더로 옮겼으며, 추가 API 호출은 하지 않았습니다.

**문서 4개는 기존 구조 검증을 통과했고, `tokenizer_summary.md`는 거절된 진단용
산출물입니다.** 파일을 한 폴더에 모았다는 것이 5개 모두 검증을 통과했다는 뜻은 아닙니다.
실패 문서도 모델 비교와 사람 검토를 위해 보존하며, `~`를 고치거나 번역문을 교정하지 않았습니다.

- 모델: `qwen/qwen3-30b-a3b-instruct-2507`
- 원문 revision: `469230357aab0f2b303b0d638c1f8d06edb14184`
- 측정 파이프라인 revision: `773a807b08ba5a039d6ead6774545e748064f596`
- 원본 결과 commit: `714d17b18381250635e8b38f14cb5afb71d9be15` ([PR #1](https://github.com/Hugging-Face-KREW/doc-builder/pull/1))
- 설정: temperature 0, context 16,384, output 4,096 (재시도 8,192), 동시성 1
- 전체 시간: 488.62초. API 왕복 시간 합: 480.05초. API 요청 343회, HTTP 오류 0회
- API 보고 비용: $0.00791014115. 입력 50,113 tokens, 출력 16,563 tokens
- 전체 파이프라인 publish: 실패/중단. 코드 블록 48개는 원문과 바이트 단위 동일

| 원문 문서 | 번역 결과 | 당시 구조 검증 |
| --- | --- | --- |
| Pipeline tutorial | [pipeline_tutorial.md](ko/pipeline_tutorial.md) | 통과 |
| Tokenizer summary | [tokenizer_summary.md](ko/tokenizer_summary.md) | **거절: 진단용** |
| Fine-tuning / Training | [training.md](ko/training.md) | 통과 |
| Selecting a quantization method | [quantization/selecting.md](ko/quantization/selecting.md) | 통과 |
| Writing a chat template | [chat_templating_writing.md](ko/chat_templating_writing.md) | 통과 |

[results.json](results.json)에 파일별 해시, 원문, 당시 판정과 실측값을 기록했습니다.
[상세 보고서](../../benchmarks/ko/qwen3-30b-a3b-instruct-2507/RESULTS.md)와
[원본 로그·실험 명세](../../benchmarks/ko/qwen3-30b-a3b-instruct-2507/baseline/)는 기존 위치에 유지합니다.
현재 공유 브랜치의 변경된 어댑터로 재생성한 결과가 아니며, 사람의 블라인드 평가는 아직 수행하지 않았습니다.

알려진 문제:

- Tokenizer의 `10~20%`, `2~4`를 당시 검증기가 신규 마크업으로 판단해 재시도 후에도 거절했습니다.
- Chat template 제목과 Training의 링크 주변에서 fragment 재시도로 문장이 깨진 사례가 있습니다.
- 구조 검증 통과는 번역 품질 통과가 아닙니다. 모든 파일은 사람 검토가 필요한 기계 번역 결과입니다.
