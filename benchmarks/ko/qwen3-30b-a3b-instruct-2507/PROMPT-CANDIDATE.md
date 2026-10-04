# 후속 프롬프트 후보 — 미검증, baseline 미적용

이번 baseline만으로 최적 프롬프트를 선정할 수는 없습니다. 다음은 관찰한
범위 기호 실패와 fragment/제목 문제를 줄이기 위한 비교 실험 후보입니다.
같은 원문·모델·용어집으로 별도 실행하고, 구조 결과와 사람 평가를 함께 비교해야 합니다.

## 가장 작은 변경부터 비교

현 system prompt에 아래 한 문장만 추가한 후보 A를 먼저 비교합니다.

```text
For numeric ranges, retain a hyphen as in "10-20%" or "2-4"; never replace it with a tilde (~).
```

이번에 남은 두 구조 실패를 겨냥한 지시입니다. 다른 문장이나 gate를 수정하지
않아 효과를 비교하기 쉽지만, 실제로 실패를 없앤다는 보장은 아직 없습니다.

## 제목/fragment 분리 후보

후보 B는 공통의 `Use natural Korean technical documentation style with polite
declarative endings (합니다/입니다).` 문장을 unit kind에 따라 다음처럼 **대체**합니다.
서로 충돌하는 지시를 단순히 뒤에 덧붙이지 않습니다.

```text
For a heading, use a concise Korean noun phrase without a final period or 합니다/입니다 ending.
For an incomplete fragment, translate only that fragment; preserve its grammatical role and do not turn it into a complete sentence.
For a complete prose sentence, use natural Korean technical-documentation style with 합니다/입니다 endings.
```

fragment retry는 문맥 없이 여러 조각을 번역하기 때문에 프롬프트만으로 문장
재조립 문제를 전부 해결할 수 있다고 단정할 수 없습니다. 재시도 설계 변경은
baseline 비교가 끝난 뒤 별도의 파이프라인 실험으로 다룹니다.

## 링크/강조 지시의 명료화 후보

현재 `Keep the same content inside each link and formatting pair.`는 자연어를
그대로 유지하라는 뜻으로 읽힐 수 있습니다. 후보 C에서는 해당 문장을 다음으로 대체합니다.

```text
Translate natural-language text inside link and formatting pairs, while keeping it inside the same pair. Preserve all tag identities and nesting. Keep the contents of keep tags byte-for-byte unchanged.
```

용어집의 unit별 `pin` 주입과 보호 `keep` 처리, 기존 검증기는 유지합니다.
후보 A/B/C를 한 번에 바꾸면 어떤 변경이 효과를 냈는지 알기 어려우므로,
각각 독립 실험 후 조합 후보를 평가하는 것이 좋습니다. 이 파일의 후보를
“최적 프롬프트”나 “개선이 입증된 프롬프트”로 표시하지 않습니다.
