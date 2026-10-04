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

# 양자화 방법을 선택하기입니다.[[selecting-a-quantization-method]]

Transformers에서는 추론 및 미세 조정을 위해 다양한 양자화 방법들이 제공됩니다. 본 가이드는 사용 사례에 따라 가장 일반적이고 실제 서비스에 적용 가능한 양자화 기법을 선택하는 데 도움을 주며, 각 기법의 장점과 단점을 설명합니다.

지원되는 모든 방법과 그 기능에 대한 종합적인 개요를 확인하려면 [개요](./overview)에 있는 표를 참조해 주시기 바랍니다.

## 추론입니다.[[inference]]

추론을 위해 아래의 양자화 방법들을 고려해 주시기 바랍니다.

| 양자화 방법 | 사용 사례입니다. |
|---|---|
| bitsandbytes | 사용 편의성 및 NVIDIA와 Intel GPU에서의 QLoRA 미세 조정입니다. |
| 압축된 텐서입니다. | 특정 양자화 형식(FP8)을 로딩합니다. |
| GPTQModel 또는 AWQ입니다. | 사전 보정을 통해 우수한 4비트 정확도를 제공합니다. |
| HQQ | 보정 없이 즉시 적용되는 빠른 양자화 기능 |
| SINQ | 보정 없이도 초고속으로 고품질의 즉석 양자화 처리 가능 |
| torchao | torch.compile을 이용한 유연성과 빠른 추론 |

### 보정이 필요하지 않습니다(즉시 양자화 기능).[[no-calibration-required-on-the-fly-quantization]]

이러한 방법들은 별도의 보정 데이터셋이나 단계가 필요하지 않기 때문에 일반적으로 사용하기가 더 쉽습니다.

#### bitsandbytes[[bitsandbytes]]

| 장점                                                         | 단점                                                    |
|--------------------------------------------------------------|---------------------------------------------------------|
| 매우 간단하며, 추론을 위해 별도의 보정 데이터셋이 필요하지 않습니다.  | 주로 NVIDIA GPU(CUDA)에 최적화되어 있습니다.             |
| 훌륭한 커뮤니티 지원이 제공되며 널리 채택되고 있습니다.                   | 추론 속도 향상이 보장되는 것은 아닙니다.                     |

더 자세한 내용은 [bitsandbytes 문서](./bitsandbytes)를 참조하시기 바랍니다.

#### HQQ(하프-이차 양자화)입니다.[[hqq-half-quadratic-quantization]]

| 장점                                                                 | 단점                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| 신속한 양자화 과정이며, 보정 데이터가 필요하지 않습니다.              | 비트 깊이가 낮을 경우 정확도가 현저히 저하될 수 있습니다. <4-bit.                     |
| 신속한 추론을 위한 여러 백엔드를 지원합니다.                                | `torch.compile` 또는 백엔드를 사용하지 않는 경우에는 추론 속도가 다른 경우와 일치하지 않을 수 있습니다. |
| `torch.compile`와 호환됩니다.                                     |                                                                            |
| 8비트, 4비트, 3비트, 2비트, 1비트 등 다양한 비트 깊이를 지원합니다.              |                                                                            |

더 자세한 내용은 [HQQ 문서](./hqq)를 참조하시기 바랍니다.

#### SINQ[[sinq]]

| 장점                                                                 | 단점                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| 매우 빠르면서도 고품질의 양자화 과정을 제공하며, 보정 데이터가 필요하지 않습니다.              | 비트 폭이 특정 수준 이하일 때 정확도가 현저히 저하될 수 있습니다. <=2-bit.                     |
| 더 빠른 추론을 위한 GemLite 백엔드입니다.                                  | 3비트 모델의 경우 추론 속도가 더 느립니다(젬라이트 커널 미적용).
| 8비트, 4비트, 3비트, 2비트 등 다양한 비트 폭을 지원합니다.              |                                                                            |

더 자세한 내용은 [SINQ 문서](./sinq)를 참조하시기 바랍니다.

#### torchao[[torchao]]

| 장점                                                                 | 단점                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 잠재적인 속도 향상을 위해 `torch.compile`와의 강력한 통합이 이루어집니다.     | 더 새로운 라이브러리이며, 생태계는 아직 발전 중입니다.                             |
| 괜찮은 수준의 CPU 양자화 지원을 제공합니다.                              | 성능은 `torch.compile`이 원활하게 작동하는지에 따라 달라집니다.                 |
| 양자화 방식의 유연성(int8, int4, fp8)입니다.           | 4비트 양자화(int4wo)는 정확도 측면에서 GPTQ/AWQ와 동일하지 않을 수 있습니다.              |

더 자세한 내용은 [torchao 문서](./torchao)를 참조하시기 바랍니다.

### 보정 기반 양자화입니다.[[calibration-based-quantization]]

이러한 방법들은 더 높은 정확도를 얻기 위해 데이터셋을 이용한 사전 보정 단계가 필요합니다.

#### GPTQ/GPTQModel[[gptqgptqmodel]]

8B 모델의 보정에는 하나의 A100 GPU를 사용할 경우 ~20분이 소요됩니다.

| 장점                                                                 | 단점                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 종종 높은 정확도를 달성합니다.                                        | 보정용 데이터셋과 별도의 보정 단계가 필요합니다.      |
| 추론 속도를 향상시킬 수 있습니다.                                      | 교정 데이터에 과적합될 가능성이 있습니다.                             |
| 많은 사전 양자화된 GPTQ 모델들이 있습니다. [Hugging Face Hub입니다.](https://huggingface.co/models?other=gptq). |                                           |

더 자세한 내용은 [GPTQ 문서](./gptq)를 참조하시기 바랍니다.

#### AWQ(활성화 인식 가중치 양자화)입니다.[[awq-activation-aware-weight-quantization]]

8B 모델의 보정에는 하나의 A100 GPU에서 ~10분이 소요됩니다.

| 장점                                                                 | 단점                                                |
|----------------------------------------------------------------------|-----------------------------------------------------|
| 종종 4비트에서도 높은 정확도를 달성합니다. (특정 작업에서는 GPTQ보다 더 뛰어난 성능을 보이기도 합니다.) | 자체적으로 양자화하는 경우에는 보정이 필요합니다.        |
| 추론 속도를 향상시킬 수 있습니다.                                      |                                                     |
| GPTQ보다 보정 시간이 더 짧습니다.                                  |                                                     |
| 많은 사전 양자화된 AWQ 모델에 관한 내용입니다. [Hugging Face Hub입니다.](https://huggingface.co/models?other=awq). |                                                     |

더 자세한 내용은 [AWQ 문서](./awq)를 참조하시기 바랍니다.

### 특정 형식 로딩하기[[loading-specific-formats]]

#### 압축된 텐서입니다.[[compressed-tensors]]

| 장점                                                         | 단점                                                        |
|--------------------------------------------------------------|-------------------------------------------------------------|
| FP8과 같은 유연한 형식을 지원합니다.        | 주로 사전에 양자화된 모델을 로드하는 데 사용됩니다.                 |
| NVIDIA (SM89 이상) 및 Intel XPU에서의 FP8 커널 가속 기능입니다.    |                                                             |

더 자세한 내용은 [compressed-tensors 문서](./compressed_tensors)를 참조해 주시기 바랍니다.

## 미세 조정입니다.[[fine-tuning]]

메모리를 절약하기 위해 미세 조정 시 아래의 양자화 방법을 고려하시기 바랍니다.

### bitsandbytes[[training]]

* **설명:** QLoRA 미세 조정을 위한 표준 방법으로 PEFT를 사용합니다.
* **장점:** 일반 소비자용 GPU에서 대규모 모델을 미세 조정할 수 있으며, PEFT에 대해 폭넓은 지원과 문서화가 이루어져 있습니다.
* **단점:** 주로 NVIDIA GPU용으로 설계되었습니다.

다른 방법들도 PEFT와의 호환성을 제공하지만, QLoRA를 위해서는 bitsandbytes가 가장 안정적이고 간편한 경로입니다.

더 자세한 내용은 [bitsandbytes 문서](./bitsandbytes#qlora)와 [PEFT 문서](https://huggingface.co/docs/peft/developer_guides/quantization#quantize-a-model)를 참조하시기 바랍니다.

## 연구입니다.[[research]]

[AQLM](./aqlm), [SpQR](./spqr), [VPTQ](./vptq), [HIGGS](./higgs)와 같은 방법들은 압축의 한계를 넘어서거나(< 2비트), 새로운 기법을 탐구합니다.

* 다음의 경우에 참고하시기 바랍니다:
  * 극도의 압축(4비트 미만)이 필요합니다.
  * 귀하는 연구를 수행 중이거나 해당 논문들에서 최신의 결과를 필요로 합니다.
  * 잠재적으로 복잡한 양자화 절차를 수행하기 위해 충분한 컴퓨팅 리소스를 활용할 수 있습니다. 프로덕션 환경에서 사용할 방법을 선택하기 전에 각 방법에 대한 문서와 관련 논문을 꼼꼼히 확인하시기 바랍니다.

## 벤치마크 비교입니다.[[benchmark-comparison]]

다양한 양자화 방법들을 정량적으로 비교하기 위해, 저희는 Llama 3.1 8B 및 70B 모델에서 여러 인기 있는 기술들을 벤치마킹하였습니다. 다음 표들은 정확도(값이 높을수록 좋음), 초당 토큰 수로 측정된 추론 처리량(값이 높을수록 좋음), GB 단위로 측정된 최대 VRAM 사용량(값이 낮을수록 좋음), 그리고 양자화 시간에 대한 결과를 보여줍니다.

성능 지표는 2에서 측정되었습니다. NVIDIA A100 Llama 3.1 70B를 위한 80GB GPU (bfloat16), 1 NVIDIA H100 FP8 방식을 위한 80GB GPU이며, 1개입니다. NVIDIA A100 다른 모든 방법에는 80GB GPU가 사용되었습니다. 처리량은 배치 크기 1로 설정하고 64개의 토큰을 생성하여 측정되었습니다. 결과는 다음과 같습니다. `torch.compile` 그리고 Marlin 해당되는 경우에 커널이 포함되어 지원됩니다.

<iframe
  src="https://huggingface.co/datasets/derekl35/quantization-benchmarks/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
  title="benchmarking results dataset"
></iframe>

주요 요점은 다음과 같습니다:

| 양자화 및 방법들입니다.                      | 메모리 절약량 (bf16 대비) | 정확도             | 기타 참고 사항입니다.                                                        |
|-------------------------------------------- |------------------------- |--------------------- |------------------------------------------------------------------- |
| **8비트** (bnb-int8, HQQ, Quanto, torchao, fp8)입니다. | ~2x             | 기준선 bf16 모델과 매우 유사합니다.   |                                                                    |
| **4-bit** (AWQ, GPTQ, HQQ, bnb-nf4, SINQ)    | ~4x                      | 비교적 높은 정확도입니다.            | AWQ/GPTQ는 정확도 면에서 우수하지만 보정이 필요합니다. HQQ/bnb-nf4/SINQ는 즉시 사용하기에 용이합니다. |
| **서브-4비트** (VPTQ, AQLM, 2비트 GPTQ)입니다. | 극단적 (>4배)입니다.            | 특히 2비트에서 현저한 성능 저하가 나타납니다. | 양자화 시간은 매우 길 수 있습니다(AQLM, VPTQ). 성능은 경우에 따라 다릅니다. |

> [!TIP]
> 특정 작업 및 하드웨어 환경에서 양자화된 모델의 성능(정확도 및 속도)을 반드시 벤치마킹하여 요구사항을 충족하는지 확인해야 합니다. 자세한 사용 방법은 위에 링크된 각각의 문서 페이지를 참조하시기 바랍니다.
