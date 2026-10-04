<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 토큰화 알고리즘입니다.[[tokenization-algorithms]]

<Youtube id="zHvTiHr506c"/>

Transformers는 바이트 쌍 인코딩(BPE), 유니그램, 그리고 WordPiece의 세 가지 하위어 토큰화 알고리즘을 지원합니다. 이들은 단어와 문자 사이의 단위로 텍스트를 나누어, 어휘를 작게 유지하면서도 의미 있는 조각을 포착합니다. 일반적인 단어는 단일 토큰으로 그대로 유지되며, 드문 또는 알 수 없는 단어는 하위어로 분해됩니다.

예를 들어, `annoyingly`는 어휘에 따라 `["annoying", "ly"]` 또는 `["annoy", "ing", "ly"]`로 분할될 수 있습니다. 하위어 분할은 모델이 알려진 하위어로부터 미지의 단어를 표현할 수 있도록 합니다.

> [!TIP]
> 서브워드 토큰화는 서브워드를 연결하여 긴 복잡한 단어를 만들 수 있는 터키어와 같은 언어에 특히 유용합니다.

## 바이트 쌍 인코딩(BPE)입니다[[byte-pair-encoding-bpe]]

<Youtube id="HEikzVL-lZU"/>

[바이트 쌍 인코딩](https://huggingface.co/papers/1508.07909) (BPE)는 Transformers에서 가장 인기 있는 토큰화 알고리즘으로, [Llama](./model_doc/llama), [Gemma](./model_doc/gemma), [Qwen2](./model_doc/qwen2) 등 여러 모델에서 사용됩니다.

1. 사전 토크나이저는 공백 또는 기타 규칙에 따라 텍스트를 분할하여 고유한 단어와 그 빈도 수를 생성합니다.

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

2. BPE 알고리즘은 모든 문자에서 기본 어휘 `["b", "g", "h", "n", "p", "s", "u"]`를 생성합니다.

```text
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

3. BPE는 개별 문자에서 시작하여 가장 빈번한 인접 쌍을 반복적으로 병합합니다. `"u"`와 `"g"`는 `"hug"`, `"pug"`, `"hugs"`에서 가장 자주 함께 나타나므로 BPE는 이를 `"ug"`로 병합하고 어휘에 추가합니다.

```text
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

4. 다음으로 자주 나타나는 쌍은 `"u"`와 `"n"`이며, 이는 `"pun"`와 `"bun"`에 나타나므로 `"un"`으로 합쳐집니다.

```text
("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
```

5. 어휘는 이제 `["b", "g", "h", "n", "p", "s", "u", "ug", "un"]`입니다. BPE는 목표 어휘 크기에 도달할 때까지 병합 규칙을 학습합니다. 이 목표 어휘 크기는 기본 어휘 크기와 병합 횟수의 합과 같습니다. [GPT](model_doc/openai-gpt)은 어휘 크기가 40,478(기본 토큰 478개 + 병합 40,000회)인 BPE를 사용합니다.

기본 어휘에 포함되지 않은 모든 문자는 `"<unk>"`와 같은 미지 토큰으로 매핑됩니다. 실질적으로 기본 어휘는 학습 중에 나타난 모든 문자를 커버하므로, 미지 토큰은 드물게 발생합니다.

### 바이트 수준 BPE입니다.[[byte-level-bpe]]

모든 유니코드 문자를 포함하면 기본 어휘가 매우 커집니다. 바이트 수준 BPE는 256개의 바이트 값으로 기본 어휘를 구성하여, 모든 단어가 `"<unk>"` 토큰 없이 토큰화될 수 있도록 합니다. [GPT-2](./model_doc/gpt2)는 어휘 크기가 50,257(256개의 바이트 토큰 + 50,000회의 병합 + 특수 종료 텍스트 토큰)인 바이트 수준 BPE를 사용합니다.

## 유니그램입니다.[[unigram]]

<Youtube id="TGZfZVuF9Yc"/>

[유니그램](https://huggingface.co/papers/1804.10959)은 Transformers에서 두 번째로 인기 있는 토큰화 알고리즘으로, [T5](./model_doc/t5), [BigBird](./model_doc/big_bird), [Pegasus](./model_doc/pegasus) 등 여러 모델에서 사용됩니다.

1. 유니그램은 많은 후보 서브워드 집합에서 시작하며, 각 후보는 등장 빈도에 따라 확률 점수를 받습니다.

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
["b", "g", "h", "n", "p", "s", "u", "hu", "ug", "un", "pu", "bu", "gs", "hug", "pug", "pun", "bun", "ugs", "hugs"]
```

2. 유니그램 점수는 각 단계에서 현재 어휘가 학습 데이터를 얼마나 잘 토큰화하는지를 평가합니다.
3. 각 토큰에 대해 Unigram은 해당 토큰을 제거했을 때 전체 손실이 얼마나 증가하는지를 측정합니다. 예를 들어, `"pu"`를 제거해도 손실에 거의 영향을 주지 않으며, 이는 `"pug"`과 `"pun"`가 여전히 `["p", "ug"]`와 `["p", "un"]`로 토큰화될 수 있기 때문입니다.

    그러나 `"ug"`를 제거하면 `"hug"`, `"pug"`, 그리고 `"hugs"` 모두 이 부분에 의존하므로 손실이 크게 증가합니다.

4. 유니그램은 손실 증가량이 가장 낮은 토큰을 제거하며, 일반적으로 하위 10~20%입니다. 기본 문자는 항상 유지되므로 어떤 단어도 토큰화할 수 있습니다. 토큰들처럼 `"bu"`, `"pu"`, `"gs"`, `"pug"`, 그리고 `"bun"` 가장 낮은 전체 가능성 기여도를 보인 항목은 제거됩니다.

```text
["b", "g", "h", "n", "p", "s", "u", "hu", "ug", "un", "hug", "pun", "ugs", "hugs"]
```

5. 단계 2~4는 어휘가 목표 크기에 도달할 때까지 반복합니다.

추론 시, Unigram은 단어를 여러 가지 방식으로 토큰화할 수 있습니다. `"hugs"`는 `["hug", "s"]`, `["h", "ug", "s"]`, 또는 `["h", "u", "g", "s"]`로 변환될 수 있습니다. Unigram은 가장 높은 확률을 가진 토큰화 방식을 선택합니다. BPE와 달리, Unigram은 결정론적이며 병합 규칙에 기반하지만, 확률적이고 학습 중에 다양한 토큰화를 샘플링할 수 있습니다.

## SentencePiece[[sentencepiece]]

[SentencePiece](https://huggingface.co/papers/1808.06226)는 원시 텍스트에 직접 BPE 또는 Unigram을 적용하는 토큰화 라이브러리입니다. 표준 BPE 및 Unigram은 단어를 공백으로 구분한다고 가정하지만, 공백을 사용하지 않는 중국어나 일본어와 같은 언어에는 해당되지 않습니다.

1. SentencePiece는 입력 텍스트를 원시 바이트 또는 문자 스트림으로 간주하며, 어휘에 공백 문자를 `"▁"`로 표현하여 포함합니다.

```text
("▁hug", 10), ("▁pug", 5), ("▁pun", 12), ("▁bun", 4), ("▁hugs", 5)
```

2. SentencePiece는 이후 텍스트에 BPE 또는 Unigram을 적용합니다.

디코딩 시, SentencePiece는 모든 토큰을 연결하고 `"▁"`을 공백으로 대체합니다.

## WordPiece[[wordpiece]]

<Youtube id="qpv6ms_t_1A"/>

[WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)는 [DistilBERT](./model_doc/distilbert) 및 [Electra](./model_doc/electra)와 같은 BERT 계열 모델의 토큰화 알고리즘입니다.

[BPE](#byte-pair-encoding-bpe)와 유사하게 하위에서 상위로 반복적으로 쌍을 병합하지만, 쌍을 선택하는 방식에서 차이가 있습니다.

```text
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

WordPiece는 학습 데이터의 가능성을 최대화하는 쌍을 결합합니다.

```text
score("u", "g") = frequency("ug") / (frequency("u") × frequency("g"))
```

| 페어 | 주파수 | 점수 |
|---|---|---|
| `"u"` + `"g"` | 20 | 20 / (36 × 20) = 0.028 |
| `"u"` + `"n"` | 16 | 16 / (36 × 16) = 0.028 |
| `"h"` + `"u"` | 15 | 15 / (15 × 36) = 0.028 |
| `"g"` + `"s"` | 5 | 5 / (20 × 5) = 0.050 |

점수는 개별 토큰 빈도에서 예상되는 것보다 결합 토큰이 더 자주 나타나는 경우 `"g"`와 `"s"`를 병합하는 것을 선호합니다. BPE는 가장 많이 나타나는 쌍을 단순히 병합합니다. WordPiece는 각 병합이 얼마나 *정보성 있는지* 측정합니다. 우연보다 훨씬 더 자주 함께 나타나는 두 토큰이 먼저 병합됩니다.

## 단어 수준 토큰화입니다.[[word-level-tokenization]]

<Youtube id="nhJxYji1aho"/>

단어 수준 토큰화는 텍스트를 공백, 구두점 또는 언어에 따라 달라지는 규칙에 따라 토큰으로 나눕니다.

```text
["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

어휘 크기가 매우 커집니다. 모든 고유한 단어, 예를 들어 `"love"`, `"loving"`, `"loved"`, `"lovingly"`의 모든 변형에 대해 별도의 토큰이 필요하기 때문입니다. 결과적으로 임베딩 행렬은 매우 커져 메모리와 계산 자원을 증가시킵니다. 어휘에 없는 단어는 `"<unk>"` 토큰으로 매핑되므로 모델은 새로운 단어를 처리할 수 없습니다.

## 문자 수준 토큰화입니다.[[character-level-tokenization]]

문자 수준 토큰화는 텍스트를 개별 문자로 나눕니다.

```text
["D", "o", "n", "'", "t", "y", "o", "u", "l", "o", "v", "e"]
```

어휘가 작고 모든 단어를 표현할 수 있으므로 `"<unk>"` 문제는 발생하지 않습니다. 그러나 시퀀스 길이가 훨씬 길어집니다. `"l"`과 같은 문자는 `"love"`만큼 의미를 담고 있지 못하므로 성능에 악영향을 미칩니다.

## 자원[[resources]]

- LLM 과정의 [제6장](https://huggingface.co/learn/llm-course/chapter6/1)에서는 토크나이저를 처음부터 훈련하는 방법을 가르치며, BPE, Unigram 및 WordPiece 알고리즘 간의 차이점을 설명합니다.
