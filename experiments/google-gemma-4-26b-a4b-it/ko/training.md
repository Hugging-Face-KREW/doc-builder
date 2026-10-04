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

미세 조정은 특정 작업이나 도메인에 특화된 더 작은 데이터셋을 사용하여 대규모 사전 학습된 모델의 학습을 계속하는 것입니다. 예를 들어, 코딩 예시 데이터셋으로 미세 조정을 수행하면 모델이 코딩 능력을 향상시키는 데 도움이 됩니다. 미세 조정은 무작위 가중치에서 시작하지 않는다는 점을 제외하면 사전 학습과 동일합니다. 또한 미세 조정은 훨씬 적은 연산량, 데이터 및 시간이 필요합니다.

아래 튜토리얼은 [`Trainer`]를 사용하여 대규모 언어 모델을 미세 조정하는 과정을 안내합니다.

미세 조정된 모델을 Hub에 푸시하려면 사용자 토큰을 사용하여 Hugging Face 계정에 로그인하십시오.

```py
from huggingface_hub import login

login()
```

## 토큰화[[tokenization]]

데이터셋을 로드하고 모델이 학습하는 텍스트 컬럼(`horoscope` 아래 데이터셋에 포함됨)을 [토큰화](./fast_tokenizers)합니다.

<iframe
  src="https://huggingface.co/datasets/karthiksagarn/astro_horoscope/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

토크나이저는 모델 입력값인 `input_ids`와 `attention_mask`를 생성합니다. 모델의 forward 메서드는 `input_ids`와 `attention_mask`만 허용하므로, 토큰화 후에 `horoscope`와 같은 열을 제거하려면 `remove_columns`를 설정하십시오.

- 긴 시퀀스를 지정된 최대 길이로 자르려면 `truncation=True`와 `max_length`를 설정하십시오.
- 모델을 평가하기 위한 테스트 분할을 생성하려면 [`~datasets.train_test_split`] 메서드를 사용하십시오.

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

데이터 콜레이터는 모델이 처리할 수 있도록 데이터셋 샘플을 배치로 구성합니다. [`DataCollatorForLanguageModeling`]는 데이터셋의 모든 시퀀스를 동일한 길이로 패딩하는 대신, 각 배치를 해당 배치 내의 가장 긴 시퀀스에 맞춰 *동적으로* 패딩합니다. 이는 불필요한 패딩 토큰을 계산하지 않음으로써 연산 및 메모리를 절약합니다.

- 토큰을 무작위로 마스킹하지 않으려면 `mlm=False`로 설정하십시오.

```py
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

## 모델 로드하기[[loading-a-model]]

미세 조정을 위해 사전 학습된 체크포인트를 로드합니다 ([Loading models](./models) 가이드를 참조하여 모델 로드에 대한 자세한 내용을 확인하십시오).

- 가중치를 저장된 dtype으로 로드하려면 `dtype="auto"`를 설정하십시오. 이 설정이 없으면 PyTorch는 가중치를 `torch.float32`로 로드하며, 가중치가 원래 `torch.bfloat16`인 경우 메모리 사용량이 두 배로 증가합니다.

```py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
```

## 학습 설정[[training-configuration]]

[`TrainingArguments`]는 학습 실행을 사용자 정의하기 위한 모든 옵션을 제공합니다. 여기에는 가장 일반적인 인자들만 다룹니다. 그 외의 모든 것들은 합리적인 기본값을 가지고 있거나 분산 학습과 같은 특정 시나리오에만 관련이 있습니다. 전체 인자 목록은 [`TrainingArguments`] API 문서를 참조하십시오.

<hfoptions id="training-args">
<hfoption id="training duration">

- `num_train_epochs`와 `per_device_train_batch_size`는 학습 기간과 배치 크기를 제어합니다. `learning_rate`는 옵티마이저의 초기 학습률을 설정합니다.

</hfoption>
<hfoption id="training optimizations">

- 하드웨어가 이를 지원하는 경우(Ampere+ GPU), 빠른 혼합 정밀도 학습을 위해 `bf16=True`를 설정하십시오. 그렇지 않으면 이전 하드웨어에서는 `fp16=True`로 전환하십시오.
- `gradient_accumulation_steps`는 가중치를 업데이트하기 전에 여러 번의 순전파에 걸쳐 그래디언트를 누적함으로써 더 큰 유효 배치 크기를 시뮬레이션합니다.
- `gradient_checkpointing`은 중간 활성화 함수 값을 저장하는 대신 역전파 과정에서 이를 다시 계산함으로써 메모리 대신 연산량을 사용합니다. 메모리와 속도 사이의 절충안을 확인하려면 [partial checkpointing](./grad_checkpointing#partial-checkpointing)을 참조하십시오.
- `gradient_checkpointing=True`를 설정할 때, 저장된 활성화 함수를 고정된 호스트 메모리에 유지하고 긴 시퀀스에 대한 GPU 메모리 사용량을 줄이려면 `gradient_checkpointing_kwargs={"offload": True}`를 설정하십시오. 트레이드오프에 대해서는 [offloading the saved activations](./grad_checkpointing#offloading-the-saved-activations)를 참조하십시오.
- `train_sampling_strategy="group_by_length"`를 설정하여 길이가 유사한 예시들을 배치로 묶고 패딩을 줄입니다. 프로세서 기반 멀티모달 데이터셋 및 미리 계산된 길이에 대해서는 [group samples by length](./trainer_recipes#group-samples-by-length)를 참조하십시오.

</hfoption>
<hfoption id="evaluation and checkpointing">

- `eval_strategy`와 `save_strategy`는 학습 중에 모델을 언제 평가할지, 그리고 언제 체크포인트를 저장할지를 결정합니다.
- `load_best_model_at_end`는 학습이 완료될 때 가장 좋은 체크포인트를 로드합니다. 이를 위해서는 `eval_strategy`가 설정되어 있어야 합니다.

</hfoption>
<hfoption id="logging">

- `logging_steps`는 학습 중 손실(loss)을 얼마나 자주 업데이트하고 반환할지를 제어합니다.

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

모든 필수 구성 요소를 포함하는 [`Trainer`] 인스턴스를 생성한 다음, [`~Trainer.train`]을 호출하여 시작합니다.

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

## 다음 단계[[next-steps]]

- 커스텀 손실 함수, 메모리 효율적인 평가, 체크포인팅 등 일반적인 Trainer 기능의 최소 작동 예제를 확인하려면 [Trainer 기능](./trainer_recipes) 가이드를 읽어보십시오.
- 새롭고 사용자 정의된 기능을 지원하기 위해 [`Trainer`] 메서드를 서브클래싱하는 방법을 배우려면 [Trainer 메서드 서브클래싱](./trainer_customize) 가이드를 읽어보십시오.
- 로깅, 조기 종료 및 기타 사용자 정의 동작을 위해 학습 이벤트에 연결하는 방법을 배우려면 [Callbacks](./trainer_callbacks) 가이드를 읽어보십시오.
- 샘플이 배치로 조립되는 방식을 사용자 지정하는 방법을 배우려면 [Data collators](./data_collators) 가이드를 읽어보십시오.
- [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-training)를 사용하여 대여한 GPU에서 모든 예제 스크립트를 실행합니다 ( [ Hugging Face Jobs에서 스크립트 실행](./run_scripts#run-a-script-on-hugging-face-jobs)을 참조하십시오).
- 다양한 텍스트, 오디오, 비전 및 멀티모달 작업에 대한 추가 학습 예제는 [transformers/examples/pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [notebooks](./notebooks), 또는 **Resources > Task Recipes** 섹션을 참조하십시오.
