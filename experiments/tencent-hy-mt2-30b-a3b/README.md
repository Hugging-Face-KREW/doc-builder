# tencent/hy-mt2-30b-a3b 한국어 번역 실험

OpenRouter의 `tencent/hy-mt2-30b-a3b`로 생성한 Transformers 문서 5개의 한국어 번역 초안입니다. 생성된 문서와 사이드바 제목을 수정 없이 보관했습니다.

- 원문: `huggingface/transformers`, 커밋 `469230357aab0f2b303b0d638c1f8d06edb14184`
- 실험 파이프라인 코드: `5964d44` (`translation-korean` 브랜치)
- 설정: 컨텍스트 8,192, 기본 출력 한도 2,048, 재시도 출력 한도 4,096, temperature 0
- 처리: Markdown 번역 단위 분리, 긴 단위 청크 분할, 보호 마커 검증 및 실패 시 재시도
- 총 번역 요청: 899회, 비용 약 $0.021096255 (실패한 번역 시도 포함, 별도 응답 진단 호출 제외)

| 원문 문서 | 번역 결과 |
| --- | --- |
| Pipeline tutorial | [pipeline_tutorial.md](ko/pipeline_tutorial.md) |
| Tokenizer summary | [tokenizer_summary.md](ko/tokenizer_summary.md) |
| Fine-tuning / Training | [training.md](ko/training.md) |
| Selecting a quantization method | [quantization/selecting.md](ko/quantization/selecting.md) |
| Writing a chat template | [chat_templating_writing.md](ko/chat_templating_writing.md) |

[results.json](results.json)에 원문 링크, 파일 해시, 문서별 최종 성공 시도의 요청 수·토큰·비용·시간과 저장된 파이프라인 설정을 기록했습니다. 문서별 비용은 최종 성공 시도 기준이며, 전체 합계는 앞선 실패 시도까지 포함합니다. 어댑터를 개선하며 순차 실행했으므로 문서별 저장 설정에는 차이가 있습니다.

Markdown 구조 검증을 통과한 실험 결과입니다. 제목의 문장형 표현, 제품명 번역, 용어와 문체는 사람이 검토해야 합니다. 문서 빌드와 렌더링 검증은 수행하지 않았습니다.
