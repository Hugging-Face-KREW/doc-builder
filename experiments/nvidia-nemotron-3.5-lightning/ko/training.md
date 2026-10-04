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

# 미세 조정[[fine-tuning]]

미세 조정은 특정 작업이나 도메인에 특화된 작은 데이터셋으로 대규모 사전 학습 모델의 추가 학습을 진행하는 과정입니다. 예를 들어, 코딩 예시 데이터셋으로 미세 조정하면 모델의 코딩 능력이 향상됩니다. 미세 조정은 사전 학습과 동일하지만, 랜덤 가중치로 시작하지 않는다는 점이 다릅니다. 또한 훨씬 적은 연산, 데이터, 시간을 필요로 합니다.

아래 튜토리얼은 [`Trainer`]를 사용하여 대규모 언어 모델을 미세 조정하는 과정을 안내합니다.

Hugging Face 계정에 사용자 토큰으로 로그인하여 파인튜닝된 모델을 Hub에 푸시하십시오.

```py
from huggingface_hub import login

login()
```

## 토큰화[[tokenization]]

데이터셋을 로드하고 모델이 훈련하는 텍스트 열(`horoscope`)을 [토큰화](./fast_tokenizers)합니다.

<iframe
  src="https://huggingface.co/datasets/karthiksagarn/astro_horoscope/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

토크나이저는 모델 입력인 `input_ids`와 `attention_mask`을 생성합니다. 모델의 forward 메서드는 `input_ids`와 `attention_mask`만 받아들이므로, `remove_columns`를 `horoscope`와 같은 열을 드롭하도록 설정하십시오.

- `truncation=True`와 `max_length`을 설정하여 더 긴 시퀀스를 지정된 최대 길이로 잘라냅니다.
- [`~datasets.train_test_split`] 방법을 사용하여 모델을 평가하기 위한 테스트 분할을 생성하십시오.

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

데이터 콜레이터는 모델 처리를 위해 데이터셋 샘플을 배치로 조립합니다. [`DataCollatorForLanguageModeling`] *동적으로* 각 배치에서 가장 긴 시퀀스까지 패딩을 적용하여, 데이터셋 전체 시퀀스를 동일한 길이로 패딩하는 것보다 효율적으로 처리합니다. 이는 불필요한 패딩 토큰 계산을 피함으로써 연산 및 메모리 사용량을 절약합니다.

- `mlm=False`를 설정하여 토큰을 무작위로 마스킹하지 않도록 합니다.

```py
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

## 모델 로드하기[[loading-a-model]]

사전 학습된 체크포인트를 로드하여 [Loading models](./models) 가이드에서 모델 로드에 대한 자세한 내용을 확인하십시오.

- `dtype="auto"`를 설정하면 저장된 dtype으로 가중치를 로드합니다. `torch.float32`로 로드하면 가중치가 원래 `torch.bfloat16`인 경우 메모리 사용량이 두 배가 됩니다.

```py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
```

## 학습 설정[[training-configuration]]

[`TrainingArguments`]는 훈련 실행의 사용자 지정을 위한 모든 옵션을 제공합니다. 여기서는 가장 일반적인 인자만 다룹니다. 그 외의 인자는 기본값이 합리적이거나 분산 훈련과 같은 특정 시나리오에만 관련이 있습니다. 인자의 전체 목록은 [`TrainingArguments`] API 문서를 참조하십시오.

<hfoptions id="training-args">
<hfoption id="training duration">

- `num_train_epochs`와 `per_device_train_batch_size`는 학습 기간과 배치 크기를 제어합니다. `learning_rate`는 옵티마이저의 초기 학습률을 설정합니다.

</hfoption>
<hfoption id="training optimizations">

- `bf16=True`로 빠른 혼합 정밀도 학습을 설정하려면 하드웨어가 Ampere+ GPU를 지원해야 합니다. 그렇지 않으면 오래된 하드웨어에서는 `fp16=True`로 대체하십시오.
- `gradient_accumulation_steps` 배치 크기를 효과적으로 늘리기 위해 여러 번의 forward pass에서 그래디언트를 누적한 후 가중치를 업데이트합니다.
- `gradient_checkpointing` 연산을 메모리에 저장하는 대신 역전파 시 중간 활성값을 재계산하여 메모리를 절약합니다. 메모리와 속도의 절충안을 원한다면 [partial checkpointing](./grad_checkpointing#partial-checkpointing)을 참조하십시오.
- `gradient_checkpointing=True`인 경우, `gradient_checkpointing_kwargs={"offload": True}`를 설정하여 저장된 활성화를 고정 호스트 메모리에 보관하고 장시간 시퀀스에 대한 GPU 메모리 사용량을 줄이십시오. [offloading the saved activations](./grad_checkpointing#offloading-the-saved-activations)를 참조하여 트레이드오프를 확인하십시오.
- `train_sampling_strategy="group_by_length"`을(를) 설정하여 배치 예시의 길이를 유사하게 그룹화하고 패딩을 줄입니다. [길이별 샘플링](./trainer_recipes#group-samples-by-length)을 참조하세요. 멀티모달 데이터셋 및 사전 계산된 길이에 대한 프로세서 기반 처리에 적용됩니다.

</hfoption>
<hfoption id="evaluation and checkpointing">

- `eval_strategy`와 `save_strategy`는 모델 훈련 중 평가 시점과 체크포인트 저장 시점을 결정합니다.
- `load_best_model_at_end`는 학습이 완료될 때 가장 좋은 체크포인트를 로드합니다. `eval_strategy`가 설정되어 있어야 합니다.

</hfoption>
<hfoption id="logging">

- `logging_steps`는 훈련 중 손실을 업데이트하고 반환하는 빈도를 제어합니다.

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

[`Trainer`] 인스턴스를 생성하여 필요한 모든 컴포넌트를 설정한 뒤, [`~Trainer.train`]을 호출하여 학습을 시작합니다.

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

[`~Trainer.push_to_hub`] 허브에 파인튜닝된 가중치, 생성 설정, 토크나이저, 모델 설정을 업로드합니다.

## 다음 단계[[next-steps]]

- 읽기 [Trainer 기능](./trainer_recipes) 가이드: 최소 실행 예제, 일반적인 ... Trainer 사용자 지정 손실 함수, 메모리 효율적인 평가, 체크포인트 등 다양한 기능을 지원합니다.
- [Subclassing Trainer methods](./trainer_customize) 가이드를 읽고, [`Trainer`] 클래스의 메서드를 서브클래싱하여 새로운 기능과 사용자 지정 기능을 지원하는 방법을 학습하십시오.
- [콜백](./trainer_callbacks) 가이드를 읽고 학습 이벤트에 훅을 걸어 로깅, 조기 종료 및 기타 사용자 정의 동작을 학습하는 방법을 알아보세요.
- [데이터 콜레이터](./data_collators) 가이드를 읽어보아, 샘플이 배치로 어셈블되는 방법을 사용자 정의하는 방법을 학습하십시오.
- 렌트 GPU에서 예제 스크립트를 실행하려면 [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-training)을(를) 참조하세요 ( [Run a script on Hugging Face Jobs](./run_scripts#run-a-script-on-hugging-face-jobs) 참조).
- [transformers/examples/pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [notebooks](./notebooks), 또는 **Resources > Task Recipes** 섹션을 참조하여 다양한 텍스트, 오디오, 비전 및 멀티모달 태스크에 대한 추가 학습 예시를 확인하십시오.
