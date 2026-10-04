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

미세 조정은 특정 작업이나 도메인에 특화된 더 작은 데이터셋을 사용하여 대규모 사전 학습된 모델의 학습을 계속하는 과정입니다. 예를 들어, 코딩 예제 데이터셋으로 미세 조정을 수행하면 모델의 코딩 능력이 향상됩니다. 미세 조정은 무작위 가중치로 시작하지 않는다는 점을 제외하면 사전 학습과 동일합니다. 또한, 훨씬 적은 컴퓨트, 데이터, 시간이 필요합니다.

아래 튜토리얼은 [`Trainer`]를 사용하여 대규모 언어 모델을 미세 조정하는 과정을 안내합니다.

사용자 토큰으로 Hugging Face 계정에 로그인하여 Hub에 파인튜닝된 모델을 푸시합니다.

```py
from huggingface_hub import login

login()
```

## 토큰화[[tokenization]]

데이터셋을 로드하고 모델이 학습하는 텍스트 열(`horoscope`은 아래 데이터셋에 해당)을 [토큰화](./fast_tokenizers)합니다.

<iframe
  src="https://huggingface.co/datasets/karthiksagarn/astro_horoscope/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

토크나이저는 모델 입력인 `input_ids`와 `attention_mask`를 생성합니다. 모델의 forward 메서드는 `input_ids`와 `attention_mask`만 허용하므로, 토크나이제이션 후 `horoscope`와 같은 컬럼을 제거하기 위해 `remove_columns`를 설정합니다.

- `truncation=True`를 설정하고 `max_length`를 지정하여 더 긴 시퀀스를 지정된 최대 길이로 잘라냅니다.
- [`~datasets.train_test_split`] 메서드를 사용하여 모델을 평가하기 위한 테스트 분할을 생성합니다.

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

데이터 콜레이터는 모델이 처리할 수 있도록 데이터셋 샘플을 배치로 조립합니다. [`DataCollatorForLanguageModeling`]은 데이터셋의 모든 시퀀스를 동일한 길이로 패딩하는 대신, 각 배치를 해당 배치 내 가장 긴 시퀀스 길이에 맞게 *동적으로* 패딩합니다. 이를 통해 불필요한 패딩 토큰의 계산을 피함으로써 계산 및 메모리를 절약합니다.

- 토큰이 무작위로 마스킹되는 것을 방지하려면 `mlm=False`로 설정합니다.

```py
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

## 모델 로드[[loading-a-model]]

사전 학습된 체크포인트를 로드하여 미세 조정을 수행합니다(모델 로딩에 대한 자세한 내용은 [모델 로드](./models) 가이드를 참조하십시오).

- `dtype="auto"`를 설정하면 가중치를 저장된 dtype로 로드합니다. 이 설정이 없으면 PyTorch는 가중치를 `torch.float32`로 로드하며, 가중치가 원래 `torch.bfloat16`인 경우 메모리 사용량이 두 배로 증가합니다.

```py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
```

## 학습 구성[[training-configuration]]

[`TrainingArguments`]은 학습 실행을 사용자 정의하는 모든 옵션을 제공합니다. 여기서는 가장 일반적인 인자만 다루며, 나머지는 합리적인 기본값을 가지거나 분산 학습과 같은 특정 시나리오에만 관련이 있습니다. 인자의 전체 목록은 [`TrainingArguments`] API 문서를 참조하십시오.

<hfoptions id="training-args">
<hfoption id="training duration">

- `num_train_epochs`와 `per_device_train_batch_size`는 학습 기간과 배치 크기를 제어합니다. `learning_rate`는 옵티마이저의 초기 학습률을 설정합니다.

</hfoption>
<hfoption id="training optimizations">

- 하드웨어가 이를 지원할 경우(Ampere 이상 GPU) 빠른 혼합 정밀도 학습을 위해 `bf16=True`를 설정합니다. 그렇지 않으면 구형 하드웨어에서는 `fp16=True`로 전환합니다.
- `gradient_accumulation_steps`는 가중치를 업데이트하기 전에 여러 번의 전방 전달에 걸쳐 기울기를 누적하여 더 큰 유효 배치 크기를 시뮬레이션합니다.
- `gradient_checkpointing`은 중간 활성화 값을 저장하는 대신 역전파 과정에서 이를 다시 계산함으로써 메모리를 계산으로 교환합니다. 메모리와 속도 사이의 절충점에 대해서는 [부분 체크포인트링](./grad_checkpointing#partial-checkpointing)을 참조하십시오.
- `gradient_checkpointing=True`인 경우, `gradient_checkpointing_kwargs={"offload": True}`를 설정하여 저장된 활성화를 고정된 호스트 메모리에 유지하고 긴 시퀀스에 대한 GPU 메모리 사용량을 줄입니다. 트레이드오프에 대해서는 [저장된 활성화 오프로딩](./grad_checkpointing#offloading-the-saved-activations)을 참조하십시오.
- `train_sampling_strategy="group_by_length"`를 설정하여 유사한 길이의 예시를 배치하고 패딩을 줄입니다. 프로세서 기반 멀티모달 데이터셋 및 사전 계산된 길이에 대해서는 [길이에 따라 샘플 그룹화](./trainer_recipes#group-samples-by-length)를 참조하십시오.

</hfoption>
<hfoption id="evaluation and checkpointing">

- `eval_strategy`와 `save_strategy`는 학습 중에 모델을 평가하는 시점과 체크포인트를 저장하는 시점을 결정합니다.
- `load_best_model_at_end`은 학습이 완료되면 최적의 체크포인트를 로드합니다. `eval_strategy`가 설정되어 있어야 합니다.

</hfoption>
<hfoption id="logging">

- `logging_steps`은 학습 중에 손실을 업데이트하고 반환하는 빈도를 제어합니다.

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

필요한 모든 구성 요소를 갖춘 [`Trainer`] 인스턴스를 생성한 다음, [`~Trainer.train`]을 호출하여 시작합니다.

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

[`~Trainer.push_to_hub`]는 미세 조정된 가중치, 생성 설정, 토크나이저, 모델 설정을 Hub에 업로드합니다.

## 다음 단계[[next-steps]]

- [Trainer 기능](./trainer_recipes) 가이드를 참고하여 커스텀 손실 함수, 메모리 효율적 평가, 체크포인트링 등 일반적인 Trainer 기능의 최소 작동 예제를 확인합니다.
- [Trainer 메서드 서브클래스화](./trainer_customize) 가이드를 읽고 [`Trainer`] 메서드를 서브클래스화하여 새로운 및 사용자 정의된 기능을 지원하는 방법을 학습합니다.
- 학습 이벤트에 대한 로깅, 조기 종료 및 기타 사용자 정의 동작을 연결하는 방법을 배우려면 [콜백](./trainer_callbacks) 가이드를 읽어 보십시오.
- [데이터 콜렉터](./data_collators) 가이드를 읽어 샘플이 배치로 조립되는 방식을 사용자 지정하는 방법을 학습합니다.
- 임대 GPU에서 [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-training)를 사용하여 예제 스크립트를 실행해 보세요([Hugging Face Jobs에서 스크립트 실행](./run_scripts#run-a-script-on-hugging-face-jobs) 참조).
- [transformers/examples/pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [notebooks](./notebooks), 또는 **Resources > Task Recipes** 섹션을 참조하여 다양한 텍스트, 오디오, 비전 및 멀티모달 작업에 대한 추가 학습 예제를 확인합니다.
