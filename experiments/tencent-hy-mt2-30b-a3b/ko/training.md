<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 미세 조정입니다.[[fine-tuning]]

미세 조정은 특정 작업이나 도메인에 맞는 더 작은 데이터셋을 이용하여 사전 학습된 대규모 모델의 학습을 계속하는 과정입니다. 예를 들어, 코딩 예제로 구성된 데이터셋에서 미세 조정을 수행하면 모델이 코딩 능력을 더욱 향상시킬 수 있습니다. 미세 조정은 사전 학습과 동일한 과정이지만, 무작위 가중치에서 시작하지 않는다는 점이 다릅니다. 또한 미세 조정에는 훨씬 적은 연산 자원, 데이터 및 시간이 필요합니다.

아래의 튜토리얼에서는 [`Trainer`]를 사용하여 대규모 언어 모델을 미세 조정하는 방법을 단계별로 설명합니다.

사용자 토큰을 이용하여 Hugging Face 계정에 로그인한 후, 미세 조정된 모델을 Hub에 업로드할 수 있습니다.

```py
from huggingface_hub import login

login()
```

## 토큰화입니다.[[tokenization]]

데이터셋을 로드한 후, 모델이 학습하는 텍스트 열(아래 데이터셋의 `horoscope`)을 [토큰화](./fast_tokenizers)합니다.

<iframe
  src="https://huggingface.co/datasets/karthiksagarn/astro_horoscope/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

토크나이저는 모델 입력값인 `input_ids`와 `attention_mask`를 생성합니다. 모델의 forward 메서드는 오직 `input_ids`와 `attention_mask`만을 받아들이므로, 토큰화 이후에 `horoscope`와 같은 열들을 제거하기 위해 `remove_columns`를 설정해야 합니다.

- 더 긴 시퀀스를 지정된 최대 길이로 잘라내기 위해 `truncation=True`와 `max_length`를 설정하시기 바랍니다.
- 모델을 평가하기 위한 테스트 세트를 생성하려면 [`~datasets.train_test_split`] 메서드를 사용하시기 바랍니다.

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)
```

데이터 콜레이터는 모델이 처리할 수 있도록 데이터셋 샘플을 배치 단위로 구성합니다. [`DataCollatorForLanguageModeling`]는 데이터셋 내의 모든 시퀀스를 동일한 길이로 패딩하는 대신, 각 배치 내에서 가장 긴 시퀀스에 맞게 *동적으로* 패딩을 적용합니다. 이렇게 하면 불필요한 패딩 토큰에 대한 연산을 피함으로써 연산 자원과 메모리를 절약할 수 있습니다.

- 무작위로 토큰을 마스킹하는 것을 방지하려면 `mlm=False`를 설정하시기 바랍니다.

```py
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

## 모델을 로딩합니다.[[loading-a-model]]

사전 학습된 체크포인트를 불러와서 미세 조정을 수행할 수 있습니다(모델 로딩에 관한 보다 자세한 내용은 [모델 로딩](./models) 가이드를 참조하시기 바랍니다).

- `dtype="auto"`를 설정하면 저장된 데이터 타입 그대로 가중치를 로드할 수 있습니다. 이 옵션을 사용하지 않으면 PyTorch는 가중치를 `torch.float32` 형태로 로드하는데, 만약 원래 가중치가 `torch.bfloat16` 형태였다면 메모리 사용량이 두 배로 증가하게 됩니다.

```py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
```

## 학습 설정입니다.[[training-configuration]]

[`TrainingArguments`]는 학습 실행을 맞춤 설정하기 위한 모든 옵션을 제공합니다. 여기서는 가장 일반적으로 사용되는 인수들만 다룹니다. 그 외의 인수들은 적절한 기본값을 가지거나 분산 학습과 같은 특정 시나리오에서만 관련이 있습니다. 인수의 전체 목록은 [`TrainingArguments`] API 문서를 참조하시기 바랍니다.

<hfoptions id="training-args">
<hfoption id="training duration">

- `num_train_epochs`와 `per_device_train_batch_size`는 학습 기간과 배치 크기를 조절합니다. `learning_rate`는 최적화기의 초기 학습률을 설정합니다.

</hfoption>
<hfoption id="training optimizations">

- 하드웨어가 지원하는 경우(Ampere 이상 세대의 GPU)에는 빠른 혼합 정밀도 학습을 위해 `bf16=True`로 설정하시기 바랍니다. 그렇지 않은 경우에는 구형 하드웨어에서 `fp16=True`를 사용하시기 바랍니다.
- `gradient_accumulation_steps`는 가중치를 업데이트하기 전에 여러 번의 순전파 과정에서 기울기를 누적시킴으로써 더 큰 효과적인 배치 크기를 시뮬레이션합니다.
- `gradient_checkpointing`은 중간 활성화 값을 저장하는 대신 역전파 과정에서 이를 재계산함으로써 메모리를 절약하기 위해 연산량을 늘리는 방식을 사용합니다. 메모리와 속도 사이의 균형을 맞추기 위해서는 [부분 체크포인팅](./grad_checkpointing#partial-checkpointing)을 참고하시기 바랍니다.
- `gradient_checkpointing=True`일 경우에는 저장된 활성화 값을 고정된 호스트 메모리에 보관하고 긴 시퀀스 처리 시 GPU 메모리 사용량을 줄이기 위해 `gradient_checkpointing_kwargs={"offload": True}`를 설정하시기 바랍니다. 이로 인한 성능상의 절충점에 대해서는 [저장된 활성화 값의 오프로딩](./grad_checkpointing#offloading-the-saved-activations)을 참조해 주시기 바랍니다.
- `train_sampling_strategy="group_by_length"`를 설정하여 길이가 비슷한 배치 예시들을 그룹화하고 패딩을 줄일 수 있습니다. 프로세서 기반의 멀티모달 데이터셋 및 미리 계산된 길이 정보에 대해서는 [길이별로 샘플을 그룹화하기](./trainer_recipes#group-samples-by-length)를 참조하시기 바랍니다.

</hfoption>
<hfoption id="evaluation and checkpointing">

- `eval_strategy`와 `save_strategy`는 학습 중에 모델을 평가할 시점과 체크포인트를 저장할 시점을 결정합니다.
- `load_best_model_at_end`는 학습이 종료될 때 최적의 체크포인트를 로드합니다. 이를 위해서는 `eval_strategy`가 설정되어 있어야 합니다.

</hfoption>
<hfoption id="logging">

- `logging_steps`는 학습 중에 손실 값을 업데이트하고 반환하는 빈도를 조절합니다.

</hfoption>
</hfoptions>

```py
training_args = TrainingArguments(
    output_dir="qwen3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"every_n_layers": 4},
    bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

## 학습입니다.[[training]]

필요한 모든 구성 요소를 갖춘 [`Trainer`] 인스턴스를 생성한 후, [`~Trainer.train`]을 호출하여 시작하시기 바랍니다.

```py
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.push_to_hub()
```

[`~Trainer.push_to_hub`]는 미세 조정된 가중치, 생성 설정, 토크나이저 및 모델 설정을 Hub에 업로드합니다.

## 다음 단계입니다.[[next-steps]]

- 일반적인 Trainer 기능들, 즉 사용자 정의 손실 함수, 메모리 효율적인 평가, 체크포인팅 등에 대한 최소한의 작동 예제를 확인하려면 [Trainer 기능](./trainer_recipes) 가이드를 읽어보시기 바랍니다.
- 새로운 기능 및 맞춤형 기능을 지원하기 위해 [`Trainer`] 메서드를 서브클래싱하는 방법을 알아보려면 [Subclassing Trainer methods](./trainer_customize) 가이드를 읽어보시기 바랍니다.
- 로깅, 조기 종료 및 기타 맞춤형 동작을 위해 학습 이벤트에 연결하는 방법을 알아보려면 [콜백](./trainer_callbacks) 가이드를 읽어보시기 바랍니다.
- 읽어보세요. [데이터 수집기](./data_collators) 샘플을 배치로 구성하는 방식을 사용자 정의하는 방법을 배우기 위한 안내서입니다.
- 임대한 GPU에서 어떤 예제 스크립트든 실행할 수 있습니다. [Hugging Face 채용 정보](https://huggingface.co/docs/hub/jobs-training) (참조하세요) [스크립트를 실행합니다. Hugging Face 채용 정보](./run_scripts#run-a-script-on-hugging-face-jobs)).
- 다양한 텍스트, 오디오, 비전 및 멀티모달 작업에 대한 추가 학습 예제를 확인하려면 [transformers/examples/pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [노트북](./notebooks) 또는 **리소스 > 태스크 레시피** 섹션을 참조해 주시기 바랍니다.
