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

# 미세 조정합니다.[[fine-tuning]]

미세 조정은 특정 작업이나 도메인에 맞는 작은 데이터셋을 사용하여 대규모 사전 학습된 모델을 추가로 학습하는 과정입니다. 예를 들어, 코드 예제 데이터셋으로 미세 조정하면 모델의 코드 작성 능력이 향상됩니다. 미세 조정은 사전 학습과 동일한 방식이지만, 무작위 가중치에서 시작하지 않습니다. 또한 계산 자원, 데이터, 시간이 훨씬 적게 필요합니다.

아래의 튜토리얼은 [`Trainer`]를 사용하여 대규모 언어 모델의 미세 조정을 안내합니다.

사용자 토큰을 사용하여 Hugging Face 계정에 로그인하여 최적화된 모델을 Hub에 푸시합니다.

```py
from huggingface_hub import login

login()
```

## 토큰화합니다[[tokenization]]

데이터셋을 로드하고 모델이 학습하는 텍스트 열(`horoscope` 아래의 데이터셋에서)을 [토큰화](./fast_tokenizers)합니다.

<iframe
  src="https://huggingface.co/datasets/karthiksagarn/astro_horoscope/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

토크나이저는 모델 입력을 생성하며, `input_ids`와 `attention_mask`를 생성합니다. 모델의 forward 메서드는 오직 `input_ids`와 `attention_mask`만을 수용하므로, 토크나이징 후 `horoscope`와 같은 열을 제거하기 위해 `remove_columns`를 설정합니다.

- `truncation=True`를 설정하고, `max_length`를 지정하여 길이가 긴 시퀀스를 지정된 최대 길이로 자르도록 합니다.
- 모델 평가를 위해 테스트 분할을 생성하려면 [`~datasets.train_test_split`] 메서드를 사용합니다.

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

데이터 콜레이터는 모델이 처리할 수 있도록 데이터셋 샘플을 배치로 조립합니다. [`DataCollatorForLanguageModeling`] *동적으로* 각 배치를 해당 배치 내에서 가장 긴 시퀀스에 맞춰 패딩하며, 데이터셋 내 모든 시퀀스를 동일한 길이로 패딩하는 방식을 사용하지 않습니다. 이는 불필요한 패딩 토큰 계산을 피함으로써 계산 및 메모리 자원을 절약합니다.

- 토큰을 무작위로 마스킹하는 것을 방지하려면 `mlm=False`로 설정합니다.

```py
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

## 모델 로딩합니다[[loading-a-model]]

사전 학습된 체크포인트를 로드하여 미세 조정합니다 (모델 로드에 대한 자세한 내용은 [모델 로드](./models) 가이드를 참조하세요).

- `dtype="auto"`를 설정하면 가중치를 저장된 dtype으로 로드합니다. 이를 설정하지 않으면 PyTorch는 가중치를 `torch.float32`로 로드하게 되며, 가중치가 원래 `torch.bfloat16`였다면 메모리 사용량이 두 배로 증가합니다.

```py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
```

## 학습 구성[[training-configuration]]

[`TrainingArguments`]는 학습 실행을 사용자 정의하는 데 필요한 모든 옵션을 제공합니다. 여기서는 가장 일반적인 인수들만 다룹니다. 나머지 모든 인수는 합리적인 기본값을 가지거나 특정 시나리오(예: 분산 학습)에만 관련이 있습니다. 인수의 완전한 목록은 [`TrainingArguments`] API 문서를 참조하시기 바랍니다.

<hfoptions id="training-args">
<hfoption id="training duration">

- `num_train_epochs`와 `per_device_train_batch_size`는 학습 기간과 배치 크기를 제어합니다. `learning_rate`는 최적화기의 초기 학습률을 설정합니다.

</hfoption>
<hfoption id="training optimizations">

- 하드웨어가 지원하는 경우(Аmpere 이상 GPU), 빠른 혼합 정밀도 학습을 위해 `bf16=True`로 설정합니다. 그렇지 않은 경우, 오래된 하드웨어에서는 `fp16=True`로 대체합니다.
- `gradient_accumulation_steps`는 가중치를 업데이트하기 전에 여러 번의 순전파 동안 그래디언트를 누적함으로써 더 큰 효과적인 배치 크기를 시뮬레이션합니다.
- `gradient_checkpointing`는 중간 활성화 값을 저장하는 대신 역전파 과정에서 다시 계산함으로써 계산량을 늘리고 메모리를 절약합니다. 메모리와 속도 사이의 균형을 원할 경우 [부분 체크포인팅](./grad_checkpointing#partial-checkpointing)을 참고합니다.
- `gradient_checkpointing=True`인 경우, 긴 시퀀스에 대해 GPU 메모리 사용량을 줄이기 위해 저장된 활성화를 핀ned 호스트 메모리에 유지하려면 `gradient_checkpointing_kwargs={"offload": True}`로 설정합니다. 교환 요소에 대해서는 [저장된 활성화를 오프로드하는 방법](./grad_checkpointing#offloading-the-saved-activations)을 참조합니다.
- `train_sampling_strategy="group_by_length"`를 설정하여 유사한 길이를 가진 예제를 배치하고 패딩을 줄입니다. 프로세서 기반의 다중 모달 데이터셋 및 사전 계산된 길이에 대해 [길이별로 샘플 그룹화](./trainer_recipes#group-samples-by-length)를 참조합니다.

</hfoption>
<hfoption id="evaluation and checkpointing">

- `eval_strategy`와 `save_strategy`는 학습 중 모델을 평가할 시점과 체크포인트를 저장할 시점을 결정합니다.
- `load_best_model_at_end`는 학습이 종료될 때 최고의 체크포인트를 로드합니다. 이 기능을 사용하려면 `eval_strategy`를 설정해야 합니다.

</hfoption>
<hfoption id="logging">

- `logging_steps`는 학습 중 손실을 얼마나 자주 업데이트하고 반환할지를 제어합니다.

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

## 학습[[training]]

필요한 모든 구성 요소를 갖춘 [`Trainer`] 인스턴스를 생성한 후, [`~Trainer.train`]을 호출하여 시작합니다.

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

[`~Trainer.push_to_hub`]는 최적화된 가중치, 생성 설정, 토크나이저 및 모델 설정을 Hub에 업로드합니다.

## 다음 단계입니다[[next-steps]]

- 일반적인 Trainer 기능인 사용자 정의 손실 함수, 메모리 효율적인 평가, 체크포인팅 등에 대한 최소한의 작동 예제는 [Trainer 기능 안내서](./trainer_recipes)를 참조하시기 바랍니다.
- 읽기 [하위 클래스화합니다 Trainer 방법들입니다](./trainer_customize) 서브클래싱하는 방법을 배우는 가이드입니다. [`Trainer`] 새로운 기능 및 사용자 정의 기능을 지원하기 위한 방법입니다.
- 로그 기록, 조기 중지 및 기타 사용자 정의 동작을 위해 학습 이벤트에 연결하는 방법을 알아보려면 [콜백](./trainer_callbacks) 가이드를 읽어보십시오.
- 샘플이 배치로 조합되는 방식을 사용자 정의하는 방법을 배우기 위해 [데이터 콜레이터](./data_collators) 가이드를 읽어보십시오.
- 임의의 예제 스크립트를 [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-training)를 통해 임대한 GPU에서 실행합니다 (자세한 내용은 [Hugging Face Jobs에서 스크립트 실행하기](./run_scripts#run-a-script-on-hugging-face-jobs) 참조합니다).
- 다른 텍스트, 오디오, 비전 및 멀티모달 작업에 대한 추가 학습 예제는 [transformers/examples/pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [노트북](./notebooks), 또는 **자원 > 작업 조리법** 섹션을 참조하시기 바랍니다.
