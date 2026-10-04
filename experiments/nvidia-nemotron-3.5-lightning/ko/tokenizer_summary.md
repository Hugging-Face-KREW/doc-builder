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

# 토큰화 알고리즘[[tokenization-algorithms]]

<Youtube id="zHvTiHr506c"/>

Transformers는 서브워드 토크나이저 세 가지 알고리즘을 지원합니다: 바이트 페어 인코딩(BPE), 유니그램, 그리고 WordPiece. 텍스트를 단어와 문자 사이의 단위로 분할하여 어휘를 콤팩트하게 유지하면서도 의미 있는 조각을 포착합니다. 일반적인 단어는 그대로 단일 토큰으로 유지되며, 드문 단어이거나 알 수 없는 단어는 서브워드로 분해됩니다.

예를 들어, `annoyingly`는 `["annoying", "ly"]` 또는 `["annoy", "ing", "ly"]`로 나뉠 수 있으며, 이는 어휘에 따라 달라집니다. 서브워드 분할은 알 수 없는 단어를 알 수 있는 서브워드로 표현하게 합니다.

> [!TIP]
> 서브워드 토크나이징은 터키어와 같은 언어에서 특히 유용하며, 긴 복잡한 단어를 서브워드를 연결하여 형성할 수 있습니다.

## 바이트 페어 인코딩 (BPE)[[byte-pair-encoding-bpe]]

<Youtube id="HEikzVL-lZU"/>

[Byte pair encoding](https://huggingface.co/papers/1508.07909) (BPE)는 Transformers에서 가장 널리 사용되는 토크나이저 알고리즘으로, [Llama](./model_doc/llama), [Gemma](./model_doc/gemma), [Qwen2](./model_doc/qwen2) 등의 모델에서 사용됩니다.

1. 전처리기는 공백이나 기타 규칙에 따라 텍스트를 분할하여 고유 단어와 그 빈도 집합을 생성합니다.

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

2. BPE 알고리즘은 모든 문자로부터 기본 어휘 `["b", "g", "h", "n", "p", "s", "u"]`을 생성합니다.

```text
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

3. BPE는 개별 문자로 시작하여 가장 빈번한 인접 쌍을 반복적으로 병합합니다. `"u"`와 `"g"`은 `"hug"`, `"pug"`, 그리고 `"hugs"`에서 가장 많이 함께 나타나므로, BPE는 이를 `"ug"`로 병합하고 어휘에 추가합니다.

```text
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

4. 가장 흔한 다음 쌍은 `"u"`와 `"n"`이며, `"pun"`와 `"bun"`에 나타나므로 `"un"`로 병합됩니다.

```text
("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
```

5. 어휘는 이제 `["b", "g", "h", "n", "p", "s", "u", "ug", "un"]`입니다. BPE는 목표 어휘 크기에 도달할 때까지 병합 규칙 학습을 계속하며, 목표 어휘 크기는 기본 어휘 크기에 병합 횟수를 더한 값과 같습니다. [GPT](model_doc/openai-gpt)는 40,478(기본 토큰 478개 + 40,000회의 병합) 크기의 어휘를 사용하여 BPE를 수행합니다.

기반 어휘집에 포함되지 않은 모든 문자는 `"<unk>"`과 같은 알 수 없는 토큰에 매핑됩니다. 실제로는 기본 어휘집이 훈련 중에 본 모든 문자를 커버하므로 알 수 없는 토큰은 드뭅니다.

### 바이트 레벨 BPE[[byte-level-bpe]]

유니코드 전체를 포함하면 기본 어휘가 엄청나게 커집니다. 바이트 레벨 BPE는 256개의 바이트 값을 기본 어휘로 사용하여, 모든 단어가 토크나이징 없이 처리되도록 보장합니다. `"<unk>"` 토큰입니다. [GPT-2](./model_doc/gpt2) 바이트 레벨 BPE를 사용하여 어휘 크기 50,257(256 바이트 토큰 + 50,000 병합 + 특수 종료 토큰)을 사용합니다.

## 유니그램[[unigram]]

<Youtube id="TGZfZVuF9Yc"/>

[언그림](https://huggingface.co/papers/1804.10959)는 Transformers에서 두 번째로 인기 있는 토크나이저 알고리즘으로, [T5](./model_doc/t5), [빅버드](./model_doc/big_bird), [페가수스](./model_doc/pegasus) 등에서 사용됩니다.

1. 유니그램은 큰 후보 서브워드 집합으로 시작하며, 각 후보는 빈도 기반으로 확률 점수를 받습니다.

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
["b", "g", "h", "n", "p", "s", "u", "hu", "ug", "un", "pu", "bu", "gs", "hug", "pug", "pun", "bun", "ugs", "hugs"]
```

2. 현재 어휘가 훈련 데이터의 각 단계에서 얼마나 잘 토크나이징되는지를 측정합니다.
3. 각 토큰에 대해 유니그램은 토큰을 제거했을 때 전체 손실이 얼마나 증가하는지를 측정합니다. 예를 들어, `"pu"`을 제거해도 손실에 거의 영향을 주지 않는데, 이는 `"pug"`과 `"pun"`가 각각 `["p", "ug"]`과 `["p", "un"]`로 토크나이즈될 수 있기 때문입니다.

    하지만 제거하면 `"ug"` 손실이 현저히 증가하게 됩니다. `"hug"`, `"pug"`, 그리고 `"hugs"` 모두 그것에 의존합니다.

4. 유니그램은 손실 증가가 가장 낮은 토큰을 제거하며, 보통 하위 10~20%를 제거합니다. 기본 문자(기본 한글)는 항상 유지되어 어떤 단어든 토큰화할 수 있습니다. '와 같은 토큰은... `"bu"`, `"pu"`, `"gs"`, `"pug"`, 그리고 `"bun"` 가장 전체적인 가능성에 기여도가 낮아 제거되었습니다.

```text
["b", "g", "h", "n", "p", "s", "u", "hu", "ug", "un", "hug", "pun", "ugs", "hugs"]
```

5. 2-4번 단계는 어휘가 목표 크기에 도달할 때까지 반복합니다.

추론 중에는 Unigram이 단어를 여러 가지 방법으로 토큰화할 수 있습니다. `"hugs"`는 `["hug", "s"]`, `["h", "ug", "s"]`, 또는 `["h", "u", "g", "s"]`과 같이 변할 수 있습니다. Unigram은 가장 높은 확률의 토큰화를 선택합니다. BPE가 결정론적이며 병합 규칙에 기반한 것과 달리, Unigram은 확률론적이며 훈련 중에 다양한 토큰화를 샘플링할 수 있습니다.

## SentencePiece[[sentencepiece]]

[SentencePiece](https://huggingface.co/papers/1808.06226)는 BPE 또는 Unigram을 원시 텍스트에 직접 적용하는 토크나이저 라이브러리입니다. 표준 BPE와 Unigram은 공백이 단어를 구분한다는 전제로 작동하지만, 이는 중국어나 일본어와 같은 공백을 사용하지 않는 언어에는 적용되지 않습니다.

1. 문장피스는 입력 텍스트를 원시 바이트 또는 문자 스트림으로 처리하고, 공백 문자를 포함합니다. `"▁"`, 어휘에 있습니다.

```text
("▁hug", 10), ("▁pug", 5), ("▁pun", 12), ("▁bun", 4), ("▁hugs", 5)
```

2. SentencePiece는 텍스트에 BPE 또는 Unigram을 적용합니다.

디코딩 시, SentencePiece는 모든 토큰을 연결하고 10을 대체합니다. `"▁"` 공백이 있습니다.

## WordPiece[[wordpiece]]

<Youtube id="qpv6ms_t_1A"/>

[WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)는 BERT-계열 모델인 [DistilBERT](./model_doc/distilbert)와 [Electra](./model_doc/electra)에 사용되는 토크나이저 알고리즘입니다.

와 비슷합니다. [BPE](#byte-pair-encoding-bpe) 그리고 아래에서 위로 반복적으로 쌍을 병합하지만, 쌍을 선택하는 방식에서 차이가 있습니다.

```text
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

WordPiece는 학습 데이터의 가능성을 최대화하는 쌍을 병합합니다.

```text
score("u", "g") = frequency("ug") / (frequency("u") × frequency("g"))
```

| 짝꿍 | 빈도 | 점수 |
|---|---|---|
| `"u"` + `"g"` | 20 | 20 / (36 × 20) = 0.028 |
| `"u"` + `"n"` | 16 | 16 / (36 × 16) = 0.028 |
| `"h"` + `"u"` | 15 | 15 / (15 × 36) = 0.028 |
| `"g"` + `"s"` | 5 | 5 / (20 × 5) = 0.050 |

점수는 `"g"`와 `"s"`을 병합하는 것을 유리하게 하며, 결합된 토큰이 개별 토큰 빈도에서 예상되는 것보다 더 자주 나타날 때 발생합니다. BPE는 단순히 가장 많이 나타나는 쌍을 병합합니다. WordPiece는 각 병합이 얼마나 *정보가 많은*지를 측정합니다. 우연히 예측되는 것보다 훨씬 더 자주 함께 나타나는 두 토큰이 먼저 병합됩니다.

## 단어 레벨 토큰화[[word-level-tokenization]]

<Youtube id="nhJxYji1aho"/>

공백, 구두점 또는 언어별 규칙에 따라 텍스트를 토큰으로 분할합니다.

```text
["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

어휘 크기가 매우 커지는 이유는 모든 고유한 단어가 자체 토큰을 필요로 하기 때문입니다. `"love"`, `"loving"`, `"loved"`, `"lovingly"`와 같은 모든 변형이 포함됩니다. 결과적으로 임베딩 매트릭스가 엄청나게 커져 메모리와 연산량이 증가합니다. 어휘에 없는 단어는 `"<unk>"` 토큰에 매핑되어, 모델이 새로운 단어를 처리할 수 없게 됩니다.

## 문자 수준 토큰화[[character-level-tokenization]]

문자 단위 토크나이징은 텍스트를 개별 문자로 분할합니다.

```text
["D", "o", "n", "'", "t", "y", "o", "u", "l", "o", "v", "e"]
```

단어 수가 작고 모든 단어가 표현될 수 있어 `"<unk>"` 문제가 없습니다. 하지만 시퀀스가 훨씬 길어집니다. `"l"` 같은 문자는 `"love"`에 비해 의미가 훨씬 적어 성능이 저하됩니다.

## 리소스[[resources]]

- [제6장](https://huggingface.co/learn/llm-course/chapter6/1)의 LLM 강좌는 토크나이저를 처음부터 훈련하는 방법을 가르치고 BPE, Unigram, 그리고 WordPiece 알고리즘의 차이점을 설명합니다.
