<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 파이프라인[[pipeline]]

The [`Pipeline`] 는 다양한 머신러닝 작업에 활용할 수 있는 간단하지만 강력한 추론 API입니다. Hugging Face [Hub](https://hf.co/models).

[`Pipeline`]를 작업에 맞게 조정하려면, 회의록 전사를 위해 자동 음성 인식(ASR) 파이프라인에 타임스탬프를 추가하는 것과 같은 작업별 파라미터를 추가하십시오. [`Pipeline`]은 GPU, Apple Silicon, 그리고 accelerate을 가속화하고 메모리를 절약하기 위해 반정밀 가중치를 지원합니다.

<Youtube id=tiZFewofSLM/>

Transformers는 두 가지 파이프라인 클래스를 가지고 있습니다. 일반적인 [`Pipeline`]과 다양한 개별 태스크 전용 파이프라인, 예를 들어 [`TextGenerationPipeline`]가 그것입니다. 이러한 개별 파이프라인을 로드하려면 [`Pipeline`]의 `task` 파라미터에 태스크 식별자를 설정하면 됩니다. 각 파이프라인에 해당하는 태스크 식별자는 그들의 API 문서에서 확인할 수 있습니다.

각 작업은 기본 사전 학습된 모델 및 전처리기를 사용하도록 구성되어 있지만, `model` 파라미터를 사용하여 다른 모델을 사용하고자 할 경우 이를 재정의할 수 있습니다.

예를 들어, [`TextGenerationPipeline`]를 [Gemma 2](./model_doc/gemma2)와 함께 사용하려면 `task="text-generation"`와 `model="google/gemma-2-2b"`를 설정하십시오.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}]
```

입력이 하나 이상인 경우, 목록으로 전달하십시오.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device)
pipeline(["the secret to baking a really good cake is ", "a baguette is "])
[[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}],
 [{'generated_text': 'a baguette is 100% bread.\n\na baguette is 100%'}]]
```

이 가이드는 [`Pipeline`]을 소개하고, 그 기능을 보여주며, 다양한 파라미터를 구성하는 방법을 보여줍니다.

## 작업[[tasks]]

[`Pipeline`]는 다양한 모달리티에서 많은 머신러닝 작업과 호환됩니다. 적절한 입력을 파이프라인에 전달하면 나머지는 파이프라인이 처리합니다.

[`Pipeline`]를 사용하여 다양한 작업 및 모달리티에 대해 몇 가지 예를 보여드리겠습니다.

<hfoptions id="tasks">
<hfoption id="automatic speech recognition">

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</hfoption>
<hfoption id="image classification">

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/vit-base-patch16-224")
pipeline(images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
[{'label': 'lynx, catamount', 'score': 0.43350091576576233},
 {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
  'score': 0.034796204417943954},
 {'label': 'snow leopard, ounce, Panthera uncia',
  'score': 0.03240183740854263},
 {'label': 'Egyptian cat', 'score': 0.02394474856555462},
 {'label': 'tiger cat', 'score': 0.02288915030658245}]
```

</hfoption>
<hfoption id="visual question answering">

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</hfoption>
</hfoptions>

## 매개변수[[parameters]]

최소한으로, [`Pipeline`]는 작업 식별자, 모델, 그리고 적절한 입력만 있으면 됩니다. 하지만 파이프라인을 구성하기 위해 사용 가능한 많은 파라미터가 있으며, 작업별 파라미터부터 성능 최적화까지 포함합니다.

이 섹션에서는 주요 파라미터 몇 가지를 소개합니다.

### 장치[[device]]

[`Pipeline`] 는 다양한 하드웨어 유형과 호환되며, GPU를 포함합니다. CPUs, Apple Silicon그리고 더 많은 항목. 하드웨어 유형을 구성하십시오. `device` 매개변수입니다. 기본적으로, when `device` 설정되지 않은 경우 [`Pipeline`] 자동으로 첫 번째 사용 가능한 가속기에 모델을 배치합니다.CUDA GPU, Apple Silicon MPS, XPU, ...) 및 해당되지 않는 경우 기본값으로 폴백합니다. CPU 사용 가능한 것이 없을 때. `device="cpu"` 를 강제로 실행하도록 설정합니다. CPU.

<hfoptions id="device">
<hfoption id="GPU">

GPU에서 [`Pipeline`]를 실행하려면 `device`를 해당 CUDA 장치 ID에 설정하십시오. 예를 들어 `device=0`는 첫 번째 GPU에서 실행됩니다.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=0)
pipeline("the secret to baking a really good cake is ")
```

Accelerate 라이브러리를 사용하면 [Accelerate](https://hf.co/docs/accelerate/index)가 분산 학습에 적합한 장치에 모델 가중치를 자동으로 로드하고 저장하도록 할 수 있습니다. 특히 여러 장치를 보유하고 있을 때 유용합니다. Accelerate는 모델 가중치를 가장 빠른 장치에 먼저 로드하고 저장한 뒤, 필요에 따라 CPU나 하드 드라이브와 같은 다른 장치로 이동시킵니다. `device_map="auto"`를 설정하면 Accelerate가 장치를 선택하도록 할 수 있습니다.

> [!TIP]
> [Accelerate](https://hf.co/docs/accelerate/basic_tutorials/install)가 설치되어 있는지 확인하십시오.
>
> ```py
> !pip install -U accelerate
> ```

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device_map="auto")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
<hfoption id="Apple silicon">

[`Pipeline`]을 Apple silicon에서 실행하려면 `device="mps"`를 설정하십시오.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="mps")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
</hfoptions>

### 배치 추론[[batch-inference]]

[`Pipeline`]는 또한 `batch_size` 파라미터를 사용하여 입력 배치도 처리할 수 있습니다. 배치 추론은 GPU에서 특히 속도를 향상시킬 수 있지만, 보장되지는 않습니다. 하드웨어, 데이터, 모델 자체와 같은 기타 변수는 배치 추론이 속도를 향상시키는지에 영향을 미칠 수 있습니다. 이러한 이유로 배치 추론은 기본적으로 비활성화되어 있습니다.

아래 예시에서 입력값이 4개이고 `batch_size`가 2로 설정된 경우, [`Pipeline`]는 모델에 한 번에 2개의 입력값 배치(batch)를 전달합니다.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device, batch_size=2)
pipeline(["the secret to baking a really good cake is", "a baguette is", "paris is the", "hotdogs are"])
[[{'generated_text': 'the secret to baking a really good cake is to use a good cake mix.\n\ni’'}],
 [{'generated_text': 'a baguette is'}],
 [{'generated_text': 'paris is the most beautiful city in the world.\n\ni’ve been to paris 3'}],
 [{'generated_text': 'hotdogs are a staple of the american diet. they are a great source of protein and can'}]]
```

배치 추론의 또 다른 좋은 사용 사례는 [`Pipeline`]에서 스트리밍 데이터에 있습니다.

```py
from transformers import pipeline
from accelerate import Accelerator
from transformers.pipelines.pt_utils import KeyDataset
import datasets

device = Accelerator().device

# KeyDataset is a utility that returns the item in the dict returned by the dataset
dataset = load_dataset("stanfordnlp/imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

배치 추론이 성능을 개선하는 데 도움이 될 수 있는지 판단할 때 다음 일반적인 가이드라인을 염두에 두십시오.

1. 모델, 데이터, 하드웨어에서 성능을 측정하는 것만이 확실하게 알 수 있는 유일한 방법입니다.
2. 지연 시간(실시간 추론 제품의 경우 등)에 제약이 있는 경우 배치 추론을 수행하지 마십시오.
3. CPU를 사용 중이라면 CPU을 하지 마십시오.
4. 배치를 추론하지 마십시오. 데이터의 `sequence_length`를 알 수 없으면요. 성능을 측정하고, `sequence_length`를 반복적으로 늘리며, 메모리 부족(OOM) 체크를 포함하여 실패로부터 복구하십시오.
5. 배치 추론을 수행하려면 `sequence_length`가 규칙적이어야 하며, OOM 오류에 도달할 때까지 계속 밀어붙이십시오. GPU가 클수록 배치 추론이 더 유용합니다.
6. 배치를 수행할 경우 OOM 오류를 처리할 수 있도록 확인해 주세요.

### 작업별 매개변수[[task-specific-parameters]]

[`Pipeline`]는 각 개별 작업 파이프라인에서 지원하는 모든 매개변수를 수락합니다. 사용 가능한 매개변수 유형을 확인하려면 각 개별 작업 파이프라인을 확인해 주시기 바랍니다. 유용한 매개변수를 찾을 수 없는 경우, GitHub [issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml)를 열어 요청해 주시기 바랍니다.!

아래 예제는 사용 가능한 일부 태스크별 파라미터를 보여줍니다.

<hfoptions id="task-specific-parameters">
<hfoption id="automatic speech recognition">

`return_timestamps="word"` 파라미터를 [`Pipeline`]에 전달하여 각 단어가 언제 발화되었는지 반환합니다.

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline(audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac", return_timestamps="word")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.',
 'chunks': [{'text': ' I', 'timestamp': (0.0, 1.1)},
  {'text': ' have', 'timestamp': (1.1, 1.44)},
  {'text': ' a', 'timestamp': (1.44, 1.62)},
  {'text': ' dream', 'timestamp': (1.62, 1.92)},
  {'text': ' that', 'timestamp': (1.92, 3.7)},
  {'text': ' one', 'timestamp': (3.7, 3.88)},
  {'text': ' day', 'timestamp': (3.88, 4.24)},
  {'text': ' this', 'timestamp': (4.24, 5.82)},
  {'text': ' nation', 'timestamp': (5.82, 6.78)},
  {'text': ' will', 'timestamp': (6.78, 7.36)},
  {'text': ' rise', 'timestamp': (7.36, 7.88)},
  {'text': ' up', 'timestamp': (7.88, 8.46)},
  {'text': ' and', 'timestamp': (8.46, 9.2)},
  {'text': ' live', 'timestamp': (9.2, 10.34)},
  {'text': ' out', 'timestamp': (10.34, 10.58)},
  {'text': ' the', 'timestamp': (10.58, 10.8)},
  {'text': ' true', 'timestamp': (10.8, 11.04)},
  {'text': ' meaning', 'timestamp': (11.04, 11.4)},
  {'text': ' of', 'timestamp': (11.4, 11.64)},
  {'text': ' its', 'timestamp': (11.64, 11.8)},
  {'text': ' creed.', 'timestamp': (11.8, 12.3)}]}
```

</hfoption>
<hfoption id="text generation">

`return_full_text=False`를 [`Pipeline`]에 전달하여 프롬프트와 생성된 텍스트 대신 생성된 텍스트만 반환하도록 합니다.

[`~TextGenerationPipeline.__call__`]는 또한 [`~GenerationMixin.generate`] 메서드에서 추가 키워드 인수를 지원합니다. 여러 개의 생성된 시퀀스를 반환하려면 `num_return_sequences`를 1보다 큰 값으로 설정하십시오.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai-community/gpt2")
pipeline("the secret to baking a good cake is", num_return_sequences=4, return_full_text=False)
[{'generated_text': ' how easy it is for me to do it with my hands. You must not go nuts, or the cake is going to fall out.'},
 {'generated_text': ' to prepare the cake before baking. The key is to find the right type of icing to use and that icing makes an amazing frosting cake.\n\nFor a good icing cake, we give you the basics'},
 {'generated_text': " to remember to soak it in enough water and don't worry about it sticking to the wall. In the meantime, you could remove the top of the cake and let it dry out with a paper towel.\n"},
 {'generated_text': ' the best time to turn off the oven and let it stand 30 minutes. After 30 minutes, stir and bake a cake in a pan until fully moist.\n\nRemove the cake from the heat for about 12'}]
```

</hfoption>
</hfoptions>

## 청크 배치[[chunk-batching]]

데이터를 청크로 처리해야 하는 경우가 있습니다.

- 일부 데이터 유형의 경우, 예를 들어 정말 긴 오디오 파일과 같이 단일 입력이 여러 부분으로 분할되어 처리될 수 있습니다.
- 일부 작업, 예를 들어 영 제로-샷 분류나 질문 응답의 경우, 단일 입력에 대해 여러 번의 전파가 필요할 수 있으며, 이는 `batch_size` 파라미터에 문제를 일으킬 수 있습니다.

[ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) 클래스는 이러한 사용 사례를 처리하도록 설계되었습니다. 두 파이프라인 클래스는 동일한 방식으로 사용되지만, [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387)는 자동으로 배치 처리를 처리할 수 있으므로 입력으로 트리거된 포워드 패스의 수에 대해 걱정할 필요가 없습니다. 대신, `batch_size`를 입력과 독립적으로 최적화할 수 있습니다.

아래 예제는 [`Pipeline`]과 어떻게 다른지 보여줍니다.

```py
# ChunkPipeline
all_model_outputs = []
for preprocessed in pipeline.preprocess(inputs):
    model_outputs = pipeline.model_forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs =pipeline.postprocess(all_model_outputs)

# Pipeline
preprocessed = pipeline.preprocess(inputs)
model_outputs = pipeline.forward(preprocessed)
outputs = pipeline.postprocess(model_outputs)
```

## 대규모 데이터세트[[large-datasets]]

대규모 데이터셋에 대한 추론 시, 데이터셋 자체를 직접 반복할 수 있습니다. 이는 전체 데이터셋에 대한 메모리를 즉시 할당하지 않으며, 배치 자체를 생성할 필요도 없습니다. `batch_size` 파라미터와 함께 [배치 추론](#batch-inference)을 시도해 보시면 성능 향상이 도움이 될 수 있습니다.

```py
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from accelerate import Accelerator
from datasets import load_dataset

device = Accelerator().device

dataset = load_dataset("stanfordnlp/imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

[`Pipeline`]를 사용한 대량 데이터셋 추론의 다른 방법으로는 반복자나 제너레이터를 사용하는 것이 있습니다.

```py
def data():
    for i in range(1000):
        yield f"My example {i}"

pipeline = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipeline(data()):
    generated_characters += len(out[0]["generated_text"])
```

## 대형 모델[[large-models]]

[Accelerate](https://hf.co/docs/accelerate/index)는 [`Pipeline`]을 사용하여 대규모 모델을 실행할 때 몇 가지 최적화를 가능하게 합니다. Accelerate를 먼저 설치해 주세요.

```py
!pip install -U accelerate
```

The `device_map="auto"` 설정은 사용 가능한 경우 가장 빠른 장치(GPU)에 먼저 모델을 자동으로 분산시킨 다음 다른 느린 장치에 배포하는 데 유용합니다.CPU, 하드디스크).

[`Pipeline`]는 torch.float16의 반정밀 가중치를 지원하며, 이는 크게 빨라지고 메모리를 절약할 수 있습니다. 대부분의 모델에서 성능 손실은 무시할 수 있으며, 특히 큰 모델의 경우 더욱 그렇습니다. 하드웨어가 지원한다면 torch.bfloat16을 대신 사용하면 더 넓은 범위를 지원할 수 있습니다.

> [!TIP]
> 내부적으로 입력은 torch.float16로 변환되며, PyTorch 백엔드를 사용하는 모델에서만 작동합니다.

마지막으로 [`Pipeline`]도 메모리 사용량을 더욱 줄이기 위해 양자화된 모델을 수락합니다. 먼저 [bitsandbytes](https://hf.co/docs/bitsandbytes/installation) 라이브러리가 설치되어 있는지 확인한 다음, `quantization_config`를 `model_kwargs`에 추가하십시오.

```py
import torch
from transformers import pipeline, BitsAndBytesConfig

pipeline = pipeline(model="google/gemma-7b", dtype=torch.bfloat16, device_map="auto", model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})
pipeline("the secret to baking a good cake is ")
[{'generated_text': 'the secret to baking a good cake is 1. the right ingredients 2. the right'}]
```
