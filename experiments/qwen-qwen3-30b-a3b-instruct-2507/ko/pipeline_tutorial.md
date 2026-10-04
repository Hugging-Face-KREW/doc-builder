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

# 파이프라인입니다[[pipeline]]

[`Pipeline`]는 Hugging Face [Hub](https://hf.co/models)의 어떤 모델을 사용하더라도 다양한 머신러닝 작업에 쉽게 활용할 수 있는 간단하면서도 강력한 추론 API입니다.

작업에 맞게 [`Pipeline`]를 조정하여 회의록을 녹음하는 자동 음성 인식(ASR) 파이프라인에 타임스탬프를 추가하는 등의 작업 전용 매개변수를 설정할 수 있습니다. [`Pipeline`]는 GPU, Apple Silicon, 및 반정밀도 가중치를 지원하여 accelerate시키고 메모리를 절약합니다.

<Youtube id=tiZFewofSLM/>

Transformers에는 일반적인 [`Pipeline`]과 [`TextGenerationPipeline`]와 같은 특정 작업용 파이프라인들이 있습니다. 각각의 파이프라인은 [`Pipeline`]의 `task` 파라미터에 작업 식별자를 설정하여 로드할 수 있습니다. 각 파이프라인의 작업 식별자는 해당 API 문서에서 확인할 수 있습니다.

각 작업은 기본적으로 사전 학습된 모델과 전처리기를 사용하도록 구성되어 있지만, 다른 모델을 사용하고자 할 경우 `model` 매개변수를 통해 이를 재정의할 수 있습니다.

예를 들어, [`TextGenerationPipeline`]를 [Gemma 2](./model_doc/gemma2)와 함께 사용하려면 `task="text-generation"` 및 `model="google/gemma-2-2b"`를 설정합니다.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}]
```

입력이 하나보다 많을 경우, 이를 리스트 형태로 전달합니다.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device)
pipeline(["the secret to baking a really good cake is ", "a baguette is "])
[[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}],
 [{'generated_text': 'a baguette is 100% bread.\n\na baguette is 100%'}]]
```

이 가이드에서는 [`Pipeline`]를 소개하고, 그 기능을 보여주며, 다양한 매개변수를 어떻게 구성할 수 있는지 안내합니다.

## 작업[[tasks]]

[`Pipeline`]는 다양한 모달리티의 많은 머신러닝 작업과 호환됩니다. 적절한 입력을 파이프라인에 전달하면 나머지는 자동으로 처리됩니다.

다양한 작업과 모달리티에 대해 [`Pipeline`]를 사용하는 방법에 대한 몇 가지 예시입니다.

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

## 매개변수입니다[[parameters]]

최소한 [`Pipeline`]는 작업 식별자, 모델 및 적절한 입력만 필요합니다. 그러나 작업에 특화된 매개변수부터 성능 최적화에 이르기까지 구성할 수 있는 다양한 매개변수가 제공됩니다.

이 섹션에서는 더 중요한 파라미터 중 일부를 소개합니다.

### 장치입니다[[device]]

[`Pipeline`]는 GPU, CPUs, Apple Silicon 등 다양한 하드웨어 유형과 호환됩니다. 하드웨어 유형은 `device` 매개변수로 설정할 수 있습니다. 기본적으로 `device`가 설정되지 않은 경우, [`Pipeline`]는 첫 번째 사용 가능한 가속기(CUDA GPU, Apple Silicon MPS, XPU 등)에 모델을 자동으로 배치하며, 가속기가 사용 불가능할 경우에만 CPU로 대체합니다. `device="cpu"`를 전달하여 CPU에서 실행되도록 강제할 수 있습니다.

<hfoptions id="device">
<hfoption id="GPU">

GPU에서 [`Pipeline`]을 실행하려면 `device`를 해당 CUDA 장치 ID로 설정합니다. 예를 들어, `device=0`은 첫 번째 GPU에서 실행합니다.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=0)
pipeline("the secret to baking a really good cake is ")
```

또한 분산 학습을 위한 라이브러리인 [Accelerate](https://hf.co/docs/accelerate/index)를 사용하여 모델 가중치를 적절한 장치에 자동으로 로드하고 저장할 수 있습니다. 이 방법은 여러 장치를 보유하고 있을 경우 특히 유용합니다. Accelerate는 가장 빠른 장치부터 모델 가중치를 로드하고 저장한 후, 필요에 따라 다른 장치(CPU, 하드 드라이브)로 가중치를 이동합니다. `device_map="auto"`를 설정하면 Accelerate가 장치를 자동으로 선택합니다.

> [!TIP]
> [Accelerate](https://hf.co/docs/accelerate/basic_tutorials/install)가 설치되어 있는지 확인합니다.
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

[`Pipeline`]를 Apple silicon에서 실행하려면 `device="mps"`로 설정합니다.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="mps")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
</hfoptions>

### 배치 추론입니다[[batch-inference]]

[`Pipeline`]는 `batch_size` 파라미터를 사용하여 입력의 배치를 처리할 수도 있습니다. 배치 추론은 특히 GPU에서 속도 향상을 가져올 수 있으나, 보장되지는 않습니다. 하드웨어, 데이터 및 모델 자체와 같은 다른 변수들은 배치 추론이 속도 향상에 기여하는지 여부에 영향을 줄 수 있습니다. 이에 따라 배치 추론은 기본적으로 비활성화되어 있습니다.

아래 예제에서 입력이 4개이고 `batch_size`가 2로 설정되어 있을 경우, [`Pipeline`]는 모델에 한 번에 2개의 입력 배치를 전달합니다.

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

배치 추론의 또 다른 좋은 사용 사례는 [`Pipeline`]에서 스트리밍 데이터를 처리하는 것입니다.

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

배치 추론이 성능 향상에 도움이 될 수 있는지 판단하기 위해 다음의 일반적인 원칙을 기억하시기 바랍니다.

1. 확실하게 알 수 있는 유일한 방법은 모델, 데이터 및 하드웨어에서 성능을 측정하는 것입니다.
2. 지연 시간에 제약이 있는 경우(예: 실시간 추론 제품 등) 배치 추론을 하지 마십시오.
3. CPU를 사용하는 경우 배치 추론을 하지 마십시오.
4. 데이터의 `sequence_length`를 모른다면 배치 추론을 하지 마십시오. 성능을 측정하고, 반복적으로 `sequence_length`를 늘리며 메모리 초과(OOM) 확인을 포함하여 실패 시 복구할 수 있도록 하십시오.
5. `sequence_length`가 규칙적인 경우 배치 추론을 수행하고, OOM 오류가 발생할 때까지 계속 진행합니다. GPU가 클수록 배치 추론이 더욱 유용합니다.
6. 배치 추론을 수행하기로 결정한 경우 OOM 오류를 처리할 수 있도록 하십시오.

### 작업 전용 매개변수입니다.[[task-specific-parameters]]

[`Pipeline`]는 각 개별 작업 파이프라인에서 지원하는 모든 매개변수를 수락합니다. 사용 가능한 매개변수 유형을 확인하기 위해 각 개별 작업 파이프라인을 확인해 주시기 바랍니다. 귀하의 사용 사례에 유용한 매개변수가 없을 경우, 자유롭게 GitHub [이슈](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml)를 열어 요청하시기 바랍니다.!

아래의 예시는 사용 가능한 작업 전용 매개변수 중 일부를 보여줍니다.

<hfoptions id="task-specific-parameters">
<hfoption id="automatic speech recognition">

각 단어가 발화된 시점을 반환하려면 `return_timestamps="word"` 매개변수를 [`Pipeline`]에 전달합니다.

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

`return_full_text=False`를 [`Pipeline`]에 전달하여 전체 텍스트(프롬프트와 생성된 텍스트) 대신 생성된 텍스트만 반환합니다.

[`~TextGenerationPipeline.__call__`]는 또한 [`~GenerationMixin.generate`] 메서드에서 제공하는 추가 키워드 인수를 지원합니다. 하나 이상의 생성된 시퀀스를 반환하려면 `num_return_sequences`를 1보다 큰 값으로 설정합니다.

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

## 청크 배치 처리합니다[[chunk-batching]]

데이터를 조각 단위로 처리해야 하는 경우가 있습니다.

- 일부 데이터 유형의 경우, 하나의 입력(예: 매우 긴 오디오 파일)을 처리하기 전에 여러 부분으로 나누어야 할 수 있습니다.
- 일부 작업, 예를 들어 제로샷 분류 또는 질문 응답과 같은 경우, 단일 입력에 대해 여러 번의 전방 전파가 필요할 수 있으며, 이는 `batch_size` 매개변수에 문제를 일으킬 수 있습니다.

[ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) 클래스는 이러한 사용 사례를 처리하도록 설계되었습니다. 두 파이프라인 클래스는 동일한 방식으로 사용되지만, [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387)가 자동으로 배치 처리를 수행하므로 입력값이 유발하는 순전파 횟수에 대해 걱정할 필요가 없습니다. 대신 입력값과 무관하게 `batch_size`를 별도로 최적화할 수 있습니다.

아래 예시는 [`Pipeline`]와 어떻게 다른지 보여줍니다.

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

## 대규모 데이터셋입니다[[large-datasets]]

큰 데이터셋을 사용한 추론의 경우, 데이터셋 자체를 직접 반복할 수 있습니다. 이 방법은 전체 데이터셋에 대한 메모리 할당을 즉시 하지 않으며, 배치를 수동으로 생성할 필요도 없습니다. 성능 향상 여부를 확인하기 위해 [배치 추론](#batch-inference)을 `batch_size` 매개변수와 함께 시도해 보시기 바랍니다.

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

크고 큰 데이터셋에 대해 [`Pipeline`]를 사용하여 추론을 실행하는 다른 방법으로는 반복자나 생성자를 사용하는 것이 있습니다.

```py
def data():
    for i in range(1000):
        yield f"My example {i}"

pipeline = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipeline(data()):
    generated_characters += len(out[0]["generated_text"])
```

## 대규모 모델입니다[[large-models]]

[Accelerate](https://hf.co/docs/accelerate/index)는 [`Pipeline`]를 사용하여 대규모 모델을 실행할 때 몇 가지 최적화 기능을 제공합니다. 먼저 Accelerate가 설치되어 있는지 확인하시기 바랍니다.

```py
!pip install -U accelerate
```

`device_map="auto"` 설정은 모델을 가장 빠른 장치(GPU)에 우선적으로 자동으로 배분한 후, 사용 가능한 경우 더 느린 장치(CPU, 하드 드라이브)로 배분하는 데 유용합니다.

[`Pipeline`]는 반정밀 가중치(torch.float16)를 지원하여 성능이 향상되고 메모리 사용량을 절약할 수 있습니다. 대부분의 모델, 특히 더 큰 모델에서는 성능 저하가 거의 발생하지 않습니다. 하드웨어에서 지원하는 경우 더 넓은 범위를 위해 torch.bfloat16를 사용할 수 있습니다.

> [!TIP]
> 입력은 내부적으로 torch.float16로 변환되며, 이는 오직 PyTorch 백엔드를 사용하는 모델에서만 작동합니다.

마지막으로, [`Pipeline`]는 메모리 사용량을 더욱 줄이기 위해 양자화 모델도 수용합니다. 먼저 [bitsandbytes](https://hf.co/docs/bitsandbytes/installation) 라이브러리가 설치되어 있는지 확인하고, 그 후 파이프라인에 `quantization_config`를 `model_kwargs`에 추가합니다.

```py
import torch
from transformers import pipeline, BitsAndBytesConfig

pipeline = pipeline(model="google/gemma-7b", dtype=torch.bfloat16, device_map="auto", model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})
pipeline("the secret to baking a good cake is ")
[{'generated_text': 'the secret to baking a good cake is 1. the right ingredients 2. the right'}]
```
