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

# 양자화 방법 선택[[selecting-a-quantization-method]]

많은 양자화 방법이 [누락된 내용]에 사용 가능합니다. Transformers 추론 및 미세 조정을 위한 본 가이드는 사용 사례에 따라 가장 일반적인 생산용 양자화 기법을 선택하는 데 도움을 주며, 각 기법의 장단점을 제시합니다.

전체 지원 메서드와 그 기능을 확인하려면 [Overview](./overview)에 있는 표를 참조하십시오.

## 추론[[inference]]

아래에 제시된 양자화 방법을 추론에 적용합니다.

| 양자화 방법 | 사용 사례 |
|---|---|
| bitsandbytes | 사용 편의성 및 QLoRA 미세 조정 on NVIDIA 그리고 인텔 GPU도. |
| 압축 텐서 | FP8 양식 로드 중 |
| GPTQModel 또는 AWQ | 4비트 정확도와 사전 보정 |
| HQQ | 빠른 실시간 양자화, 캘리브레이션 필요 없음 |
| SINQ | 초고속이지만 고품질의 온라인 양자화, 캘리브레이션 불필요 |
| torchao | 유연성과 빠른 추론으로 torch.compile와 함께 |

### 캘리브레이션 불필요 (온더플라이 양자화)[[no-calibration-required-on-the-fly-quantization]]

이 방법들은 일반적으로 별도의 데이터셋이나 단계 없이 사용하기가 더 쉽습니다.

#### bitsandbytes[[bitsandbytes]]

| 장점                                                         | 소속                                                    |
|--------------------------------------------------------------|---------------------------------------------------------|
| 추론에 필요한 캘리브레이션 데이터셋이 매우 간단합니다.  | 주로 최적화된 NVIDIA GPUs (CUDA).             |
| 좋은 커뮤니티 지원 및 널리 사용됨.                   | 추론 속도가 보장되지 않습니다.                     |

[bitsandbytes 문서](./bitsandbytes)를 참조하십시오.

#### HQQ (반-이차 양자화)[[hqq-half-quadratic-quantization]]

| 장점                                                                 | 소비자                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| 빠른 양자화 과정, 캘리브레이션 데이터가 필요하지 않습니다.              | 정확도는 비트 깊이에서 현저히 저하될 수 있습니다. <4-bit.                     |
| 다중 백엔드를 통한 빠른 추론.                                | 추론 속도는 `torch.compile` 또는 백엔드를 사용하지 않는 경우 다른 모델에 비해 따라잡기 어려울 수 있습니다. |
| 호환성 있음 `torch.compile`.                                     |                                                                            |
| 8, 4, 3, 2, 1비트까지 다양한 비트 깊이를 지원합니다.              |                                                                            |

[HQQ 문서](./hqq)에 대해 자세히 알아보십시오.

#### SINQ[[sinq]]

| 장점                                                                 | 소비자 (Consumer)                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| 초고속이며 고품질의 양자화 과정이며, 캘리브레이션 데이터가 필요하지 않습니다.              | 정확도는 비트 폭이 <=2비트일 경우 현저히 저하될 수 있습니다.                     |
| GemLite 백엔드를 통한 빠른 추론.                                  | 3비트 모델의 느린 추론 (gemlite 커널 없음)
| 8비트, 4비트, 3비트, 2비트 폭을 지원합니다.              |                                                                            |

보기를 클릭하세요. [SINQ 문서ation](./sinq) 자세한 내용은 여기를 참조하십시오.

#### torchao[[torchao]]

| 장점                                                                 | 소비자                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| `torch.compile`와 강력한 통합으로 잠재적인 속도 향상.     | 더 새로운 라이브러리이며, 생태계가 여전히 진화하고 있습니다.                             |
| Offers decent CPU 양자화 지원합니다.                              | 성능은 `torch.compile`가 잘 작동하는지에 따라 달라집니다.                 |
| 양자화 스키마의 유연성 (int8, int4, fp8).           | 4비트 양자화 (int4wo)는 GPTQ/AWQ보다 정확도가 낮을 수 있습니다.              |

[torchao 문서](./torchao)를 참조하세요.

### 캘리브레이션 기반 양자화[[calibration-based-quantization]]

이 방법들은 데이터셋을 사용하여 사전 교정 단계를 수행함으로써 정확도를 높일 수 있습니다.

#### GPTQ/GPTQModel[[gptqgptqmodel]]

8B 모델의 캘리브레이션은 ~20분 정도 A100 GPU에서 소요됩니다.

| 장점                                                                 | 소속                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 높은 정확도를 자주 달성합니다.                                        | 캘리브레이션 데이터셋이 필요하며 별도의 캘리브레이션 단계가 필요합니다.      |
| 추론 속도 향상으로 이어질 수 있습니다.                                      | 캘리브레이션 데이터에 과적합될 수 있습니다.                             |
| 많은 [Hugging Face Hub](https://huggingface.co/models?other=gptq)에 있는 사전 양자화된 GPTQ 모델. |                                           |

보기를 참조하십시오. [GPTQ 문서](./gptq) 자세한 내용은 여기를 참조하십시오.

#### AWQ (Activation-aware Weight Quantization)[[awq-activation-aware-weight-quantization]]

8B 모델의 캘리브레이션은 ~10분 정도 A100 GPU에서 소요됩니다.

| 장점                                                                 | 소비자                                                |
|----------------------------------------------------------------------|-----------------------------------------------------|
| 4비트에서 높은 정확도를 자주 달성합니다. (특정 작업에서는 GPTQ를 능가하는 경우도 있습니다.) | 직접 양자화하는 경우 보정이 필요합니다.        |
| 추론 속도 향상으로 이어질 수 있습니다.                                      |                                                     |
| GPTQ보다 짧은 교정 시간입니다.                                  |                                                     |
| 많은 AWQ 모델이 [Hugging Face Hub](https://huggingface.co/models?other=awq)에 있습니다. |                                                     |

[AWQ 문서](./awq)를 참조하십시오.

### 특정 형식 로드[[loading-specific-formats]]

#### 압축된 텐서[[compressed-tensors]]

| 장점                                                         | 소비자                                                        |
|--------------------------------------------------------------|-------------------------------------------------------------|
| FP8와 같은 유연한 형식을 지원합니다.        | 주로 양자화된 모델을 로드하는 데 사용됩니다.                 |
| NVIDIA의 NVIDIA 커널 가속 (SM89+) 및 Intel XPU.    |                                                             |

[압축된 텐서 문서](./compressed_tensors)을(를) 참조하십시오.

## 미세 조정[[fine-tuning]]

아래 양자화 방법을 미세 조정 중에 메모리를 절약하기 위해 사용하십시오.

### bitsandbytes[[training]]

* **설명:** PEFT를 통한 QLoRA 미세 조정의 표준 방법입니다.
* **Pros:** PEFT를 사용한 대형 모델의 소비자 GPU 미세 조정 지원 및 광범위한 문서화.
* **단점:** 주로 위해 NVIDIA GPUs.

다른 방법은 PEFT 호환성을 제공하지만, bitsandbytes가 QLoRA에 가장 정착되어 있고 straightforward한 경로입니다.

[bitsandbytes 문서](./bitsandbytes#qlora)와 [PEFT 설명서](https://huggingface.co/docs/peft/developer_guides/quantization#quantize-a-model)을 참조하세요.

## 연구[[research]]

방법과 같이 [AQLM](./aqlm), [SpQR](./spqr), [VPTQ](./vptq), [HIGGS](./higgs)등, 압축의 경계를 넓히다 (< 2비트) 또는 새로운 기법을 탐색합니다.

* 다음과 같은 경우를 고려하십시오.
  * 극한의 압축(4비트 미만)이 필요합니다.
  * 연구 수행이 필요하거나 해당 논문에서 최신 결과를 요구하는 경우입니다.
  * 컴퓨팅 리소스가 풍부하여 복잡한 양자화 절차를 진행할 수 있습니다. 각 방법의 문서와 관련 논문을 주의 깊게 검토하신 후, 프로덕션용으로 적합한 방법을 선택하시기를 권장합니다.

## 성능 비교[[benchmark-comparison]]

정확도(높을수록 좋음), 토큰/초 단위의 추론 처리량(높을수록 좋음), 피크 VRAM 사용량(낮을수록 좋음), 양자화 시간을 비교하기 위해 Llama 3.1 8B 및 70B 모델에서 여러 인기 기법을 벤치마크했습니다. 아래 표는 각 항목의 결과를 보여줍니다.

성능 지표는 2에서 측정되었습니다. NVIDIA A100 80GB GPU로 Llama 3.1 70B (bfloat16), 1 NVIDIA H100 80GB GPU를 위한 FP8 방법, 그리고 1개 NVIDIA A100 80GB GPU를 사용한 모든 다른 방법들에 대해. 배치 크기가 1이며 64개의 토큰을 생성할 때 측정된 처리량. 64 결과 for `torch.compile` 그리고 Marlin 적용 가능한 경우 및 지원되는 경우 커널이 포함됩니다.

<iframe
  src="https://huggingface.co/datasets/derekl35/quantization-benchmarks/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
  title="benchmarking results dataset"
></iframe>

핵심 요점은 다음과 같습니다.

| 양자화 및 방법                      | bf16 대비 메모리 절약 (Memory Savings vs bf16) | 정확도             | 기타 메모                                                        |
|-------------------------------------------- |------------------------- |--------------------- |------------------------------------------------------------------- |
| **8비트** (bnb-int8, HQQ, Quanto, torchao, fp8) | ~2x             | 기반 bf16 모델과 매우 유사함   |                                                                    |
| **4-bit** (AWQ, GPTQ, HQQ, bnb-nf4, SINQ)    | ~4x                      | 상대적으로 높은 정확도            | AWQ/GPTQ는 정확도가 높지만 보정이 필요합니다. HQQ/bnb-nf4/SINQ 쉽습니다. |
| **4비트 미만** (VPTQ, AQLM, 2비트 GPTQ) | 극심한 (>4x)            | 특히 2비트에서 두드러진 감소 | 양자화 시간은 매우 길 수 있습니다 (AQLM, VPTQ). 성능은 달라집니다. |

> [!TIP]
> 정확도와 속도를 모두 벤치마크하여 양자화된 모델의 성능을 항상 테스트하고, 귀하의 요구 사항을 충족하는지 확인하십시오. 위에 연결된 개별 문서 페이지를 참조하여 자세한 사용 지침을 확인하십시오.
