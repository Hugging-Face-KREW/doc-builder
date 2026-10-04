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

Transformers는 바이트 페어 인코딩(BPE), 유니그램, 그리고 WordPiece라는 세 가지 서브워드 토큰화 알고리즘을 지원합니다. 이러한 알고리즘들은 텍스트를 단어와 문자 사이의 단위로 분할하여 어휘의 크기를 작게 유지하면서도 의미 있는 조각들을 포착할 수 있게 합니다. 흔히 사용되는 단어들은 하나의 토큰으로 그대로 유지되며, 드물거나 알려지지 않은 단어들은 서브워드로 분해됩니다.

예를 들어, `annoyingly`는 사용되는 어휘에 따라 `["annoying", "ly"]` 또는 `["annoy", "ing", "ly"]`로 분할될 수 있습니다. 서브워드 분할을 통해 모델은 알려진 서브워드들로부터 이전에 본 적 없는 단어를 표현할 수 있습니다.

> [!TIP]
> 서브워드 토큰화는 터키어와 같이 서브워드를 연결하여 길고 복잡한 단어를 만들 수 있는 언어에 특히 유용합니다.

## 바이트 페어 인코딩(BPE)입니다.[[byte-pair-encoding-bpe]]

<Youtube id="HEikzVL-lZU"/>

[바이트 페어 인코딩](https://huggingface.co/papers/1508.07909)(BPE)은 Transformers 모델에서 가장 널리 사용되는 토큰화 알고리즘이며, [Llama](./model_doc/llama), [Gemma](./model_doc/gemma), [Qwen2](./model_doc/qwen2)와 같은 모델들에서 활용됩니다.

1. 사전 토크나이저는 공백이나 기타 규칙에 따라 텍스트를 분할하여 고유한 단어들과 그 출현 빈도를 산출합니다.

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

2. BPE 알고리즘은 모든 문자로부터 기본 어휘인 `["b", "g", "h", "n", "p", "s", "u"]`를 생성합니다.

```text
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

3. BPE는 개별 문자로 시작하여 가장 빈번하게 나타나는 인접한 쌍을 반복적으로 병합합니다. `"u"`와 `"g"`은 `"hug"`, `"pug"`, 그리고 `"hugs"`에서 가장 자주 함께 나타나므로, BPE는 이들을 `"ug"`로 병합한 뒤 어휘에 추가합니다.

```text
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

4. 그 다음으로 흔한 조합은 `"u"`와 `"n"`이며, 이들은 `"pun"`과 `"bun"`에 나타나므로 `"un"`로 합쳐집니다.

```text
("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
```

5. 현재 어휘는 `["b", "g", "h", "n", "p", "s", "u", "ug", "un"]`입니다. BPE는 기본 어휘 크기에 병합된 항목 수를 더한 목표 어휘 크기에 도달할 때까지 병합 규칙을 계속 학습합니다. [GPT](model_doc/openai-gpt)는 어휘 크기가 40,478개인 BPE를 사용하며, 이는 기본 토큰 478개와 병합된 항목 40,000개로 구성됩니다.

기본 어휘에 포함되지 않은 모든 문자는 `"<unk>"`와 같은 알 수 없는 토큰으로 매핑됩니다. 실제로는 학습 과정에서 나타나는 모든 문자가 기본 어휘에 포함되므로 알 수 없는 토큰이 발생하는 경우는 드뭅니다.

### 바이트 수준의 BPE입니다.[[byte-level-bpe]]

모든 유니코드 문자를 포함할 경우 기본 어휘의 크기가 엄청나게 커지게 됩니다. 반면 바이트 단위 BPE는 256개의 바이트 값을 기본 어휘로 사용하여 모든 단어를 `"<unk>"` 토큰 없이도 토큰화할 수 있도록 합니다. [GPT-2](./model_doc/gpt2)는 어휘 크기가 50,257개인 바이트 단위 BPE를 사용하며, 이는 256개의 바이트 토큰과 50,000개의 병합 결과, 그리고 특수한 문장 종료 토큰을 포함합니다.

## 유니그램입니다.[[unigram]]

<Youtube id="TGZfZVuF9Yc"/>

[유니그램](https://huggingface.co/papers/1804.10959)은 Transformers에서 두 번째로 널리 사용되는 토큰화 알고리즘이며, [T5](./model_doc/t5), [빅버드](./model_doc/big_bird), [페가수스](./model_doc/pegasus)와 같은 모델들에서 활용됩니다.

1. 유니그램은 방대한 후보 서브워드 집합으로부터 시작하며, 각 후보는 그 단어가 등장하는 빈도에 따라 확률 점수를 부여받습니다.

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
["b", "g", "h", "n", "p", "s", "u", "hu", "ug", "un", "pu", "bu", "gs", "hug", "pug", "pun", "bun", "ugs", "hugs"]
```

2. 유니그램은 각 단계에서 현재의 어휘가 학습 데이터를 얼마나 잘 토큰화하는지를 평가합니다.
3. 유니그램은 각 토큰을 제거했을 때 전체 손실이 얼마나 증가하는지를 측정합니다. 예를 들어, `"pu"`를 제거하더라도 손실에 미치는 영향이 거의 없는데, 그 이유는 `"pug"`과 `"pun"`가 여전히 `["p", "ug"]` 및 `["p", "un"]`로 토큰화될 수 있기 때문입니다.

    하지만 `"ug"`를 제거하면 손실이 크게 증가하는데, 이는 `"hug"`, `"pug"`, 그리고 `"hugs"`가 모두 이에 의존하기 때문입니다.

4. 유니그램은 손실 증가가 가장 적은 토큰들, 보통 하위 10-20%에 해당하는 토큰들을 제거합니다. 기본 문자들은 항상 유지되므로 어떤 단어든 토큰화할 수 있습니다. `"bu"`, `"pu"`, `"gs"`, `"pug"`, 그리고 `"bun"`와 같은 토큰들은 전체 확률에 가장 적게 기여했기 때문에 제거됩니다.

```text
["b", "g", "h", "n", "p", "s", "u", "hu", "ug", "un", "hug", "pun", "ugs", "hugs"]
```

5. 어휘의 크기가 목표치에 도달할 때까지 2단계부터 4단계까지를 반복합니다.

추론 과정에서 Unigram은 단어를 여러 가지 방식으로 토큰화할 수 있습니다. `"hugs"`는 `["hug", "s"]`, `["h", "ug", "s"]`, 또는 `["h", "u", "g", "s"]`로 변환될 수 있습니다. Unigram은 가장 높은 확률을 가진 토큰화 방식을 선택합니다. 병합 규칙에 따라 결정적인 방식으로 토큰화하는 BPE와 달리, Unigram은 확률적이며 학습 중에 서로 다른 토큰화 결과를 생성할 수 있습니다.

## SentencePiece[[sentencepiece]]

[SentencePiece](https://huggingface.co/papers/1808.06226)는 원시 텍스트에 직접 BPE 또는 Unigram 알고리즘을 적용하는 토크나이즈 라이브러리입니다. 표준적인 BPE와 Unigram 방식은 공백으로 단어를 구분한다고 가정하지만, 공백을 사용하지 않는 중국어나 일본어와 같은 언어에는 적용되지 않습니다.

1. SentencePiece는 입력 텍스트를 원시 바이트 또는 문자 스트림으로 처리하며, 어휘에는 `"▁"`로 표현되는 공백 문자도 포함됩니다.

```text
("▁hug", 10), ("▁pug", 5), ("▁pun", 12), ("▁bun", 4), ("▁hugs", 5)
```

2. 그 다음 SentencePiece는 텍스트에 BPE 또는 유니그램 알고리즘을 적용합니다.

디코딩 시에 SentencePiece는 모든 토큰을 연결하고 `"▁"`를 공백으로 대체합니다.

## WordPiece[[wordpiece]]

<Youtube id="qpv6ms_t_1A"/>

[WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)는 [DistilBERT](./model_doc/distilbert) 및 [Electra](./model_doc/electra)와 같은 BERT 계열 모델에서 사용되는 토큰화 알고리즘입니다.

이는 [BPE](#byte-pair-encoding-bpe)와 유사하며 아래에서 위로 쌍을 반복적으로 병합하지만, 쌍을 선택하는 방식에서는 차이가 있습니다.

```text
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

WordPiece는 학습 데이터의 가능성을 극대화하는 쌍들을 병합합니다.

```text
score("u", "g") = frequency("ug") / (frequency("u") × frequency("g"))
```

| 쌍입니다. | 주파수입니다. | 점수입니다. |
|---|---|---|
| `"u"` + `"g"` | 20 | 20 / (36 × 20) = 0.028 |
| `"u"` + `"n"` | 16 | 16 / (36 × 16) = 0.028 |
| `"h"` + `"u"` | 15 | 15 / (15 × 36) = 0.028 |
| `"g"` + `"s"` | 5 | 5 / (20 × 5) = 0.050 |

점수상으로는 개별 토큰의 출현 빈도로 예측되는 수치보다 결합된 토큰이 더 자주 나타나는 경우에 `"g"`와 `"s"`를 병합하는 것이 유리합니다. BPE는 단순히 가장 자주 등장하는 토큰 쌍을 병합합니다. WordPiece는 각 병합이 얼마나 *정보량이 풍부한지*를 측정합니다. 우연히 발생할 것으로 예상되는 빈도보다 훨씬 자주 함께 나타나는 두 토큰이 먼저 병합됩니다.

## 단어 단위 토큰화입니다.[[word-level-tokenization]]

<Youtube id="nhJxYji1aho"/>

단어 단위 토큰화는 공백, 구두점 또는 언어별 규칙에 따라 텍스트를 토큰으로 분할합니다.

```text
["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

어휘의 크기는 각각의 고유한 단어마다 별도의 토큰이 필요하기 때문에 매우 커지게 됩니다. 여기에는 모든 변형들(`"love"`, `"loving"`, `"loved"`, `"lovingly"`)도 포함됩니다. 그로 인해 생성되는 임베딩 행렬의 크기가 엄청나게 커져서 메모리와 연산 부하가 증가합니다. 어휘에 포함되지 않은 단어들은 `"<unk>"` 토큰으로 매핑되므로 모델이 새로운 단어를 처리할 수 없습니다.

## 문자 단위 토큰화입니다.[[character-level-tokenization]]

문자 단위 토큰화는 텍스트를 개별 문자로 분할합니다.

```text
["D", "o", "n", "'", "t", "y", "o", "u", "l", "o", "v", "e"]
```

어휘의 종류가 적고 모든 단어를 표현할 수 있기 때문에 `"<unk>"`와 같은 문제가 발생하지 않습니다. 하지만 시퀀스의 길이가 훨씬 길어집니다. `"l"`과 같은 문자는 `"love"`에 비해 의미가 현저히 적으므로 성능이 저하됩니다.

## 참고 자료입니다.[[resources]]

- LLM 과정의 [제6장](https://huggingface.co/learn/llm-course/chapter6/1)에서는 토크나이저를 처음부터 훈련하는 방법을 배우고, BPE, 유니그램, 그리고 WordPiece 알고리즘 간의 차이점에 대해 설명합니다.
