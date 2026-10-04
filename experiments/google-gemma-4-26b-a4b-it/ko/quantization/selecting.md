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

# 양자화 방법 선택하기[[selecting-a-quantization-method]]

Transformers에는 추론 및 미세 조정을 위한 다양한 양자화 방법이 있습니다. 이 가이드는 사용 사례에 따라 가장 일반적이고 프로덕션에 바로 적용 가능한 양자화 기술을 선택할 수 있도록 도와주며, 각 기술의 장단점을 제시합니다.

지원되는 모든 메서드와 그 기능에 대한 포괄적인 개요를 확인하려면 [Overview](./overview)의 표를 다시 참조하십시오.

## 추론[[inference]]

추론을 위한 아래의 양자화 방법들을 고려하십시오.

| 양자화 방법 | 유스케이스 |
|---|---|
| bitsandbytes | NVIDIA 및 Intel GPU에서의 사용 편의성 및 QLoRA 미세 조정 |
| 압축된 텐서 | 특정 양자화 형식(FP8) 로드하기 |
| GPTQModel 또는 AWQ | 사전 보정을 통한 우수한 4비트 정확도 |
| HQQ | 보정 없는 빠른 온더플라이 양자화 |
| SINQ | 보정(calibration) 없이 실시간으로 수행되는 매우 빠르고 고품질인 양자화 |
| torchao | torch.compile을(를) 통한 유연성 및 빠른 추론 |

### 보정 불필요 (실시간 양자화)[[no-calibration-required-on-the-fly-quantization]]

이러한 방법들은 별도의 보정 데이터셋이나 단계가 필요하지 않기 때문에 일반적으로 사용하기 더 쉽습니다.

#### bitsandbytes[[bitsandbytes]]

| 장점                                                         | 단점                                                    |
|--------------------------------------------------------------|---------------------------------------------------------|
| 매우 간단하며, 추론을 위한 교정 데이터셋이 필요하지 않습니다.  | 주로 NVIDIA GPU(CUDA)에 최적화되어 있습니다.             |
| 커뮤니티 지원이 우수하며 널리 채택되어 있습니다.                   | 추론 속도 향상이 보장되지 않습니다.                     |

더 자세한 내용은 [bitsandbytes documentation](./bitsandbytes)을 참조하십시오.

#### HQQ (Half-Quadratic 양자화)[[hqq-half-quadratic-quantization]]

| 장점                                                                 | 단점                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| 빠른 양자화 프로세스로, 보정 데이터가 필요하지 않습니다.              | 정확도는 비트 깊이가 <4비트 미만일 때 크게 저하될 수 있습니다.                     |
| 빠른 추론을 위한 다중 백엔드.                                | `torch.compile` 또는 백엔드를 사용하지 않으면 추론 속도가 다른 것과 일치하지 않을 수 있습니다. |
| `torch.compile`과 호환됩니다.                                     |                                                                            |
| 다양한 비트 심도(8, 4, 3, 2, 1비트)를 지원합니다.              |                                                                            |

더 자세한 내용은 [HQQ 문서를 참조하십시오.](./hqq)

#### SINQ[[sinq]]

| 장점                                                                 | 단점                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| 매우 빠르면서도 고품질의 양자화 프로세스로, 보정 데이터가 필요하지 않습니다.              | 정확도는 비트 너비가 <=2비트 미만일 때 크게 저하될 수 있습니다.                     |
| 더 빠른 추론을 위한 GemLite 백엔드입니다.                                  | 3비트 모델의 추론 속도 저하 (gemlite 커널 미사용)
| 다양한 비트 폭(8, 4, 3, 2비트)을 지원합니다.              |                                                                            |

자세한 내용은 [SINQ documentation](./sinq)을 참조하십시오.

#### torchao[[torchao]]

| 장점                                                                 | 단점                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 잠재적인 속도 향상을 위해 `torch.compile`과 강력하게 통합됩니다.     | 새로운 라이브러리이며, 생태계가 여전히 발전 중입니다.                             |
| 적절한 CPU 양자화 지원을 제공합니다.                              | 성능은 `torch.compile`이 잘 작동하는지에 달려 있습니다.                 |
| 양자화 방식의 유연성 (int8, int4, fp8).           | 4비트 양자화(int4wo)는 정확도 측면에서 GPTQ/AWQ와 일치하지 않을 수 있습니다.              |

더 자세한 내용은 [torchao 문서를 참조하십시오.](./torchao)

### 보정 기반 양자화[[calibration-based-quantization]]

이러한 방법들은 잠재적으로 더 높은 정확도를 달성하기 위해 데이터셋을 사용하는 사전 보정 단계가 필요합니다.

#### GPTQ/GPTQModel[[gptqgptqmodel]]

8B 모델의 보정은 하나의 A100 GPU에서 ~20분이 소요됩니다.

| 장점                                                                 | 단점                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 종종 높은 정확도를 달성합니다.                                        | 교정 데이터셋과 별도의 교정 단계가 필요합니다.      |
| 추론 속도 향상으로 이어질 수 있습니다.                                      | 교정 데이터에 과적합될 가능성이 있습니다.                             |
| [Hugging Face Hub](https://huggingface.co/models?other=gptq)에서 많은 사전 양자화된 GPTQ 모델을 찾을 수 있습니다. |                                           |

자세한 내용은 [GPTQ documentation](./gptq)을 참조하십시오.

#### AWQ (활성화 인지 가중치 양자화)[[awq-activation-aware-weight-quantization]]

8B 모델의 보정은 하나의 A100 GPU에서 ~10분이 소요됩니다.

| 장점                                                                 | 단점                                                |
|----------------------------------------------------------------------|-----------------------------------------------------|
| 4비트에서 높은 정확도를 자주 달성합니다. (특정 작업에서는 GPTQ를 능가하기도 합니다.) | 직접 양자화를 수행하는 경우 보정이 필요합니다.        |
| 추론 속도 향상으로 이어질 수 있습니다.                                      |                                                     |
| GPTQ보다 짧은 보정 시간입니다.                                  |                                                     |
| [Hugging Face Hub](https://huggingface.co/models?other=awq)에 많은 사전 양자화된 AWQ 모델이 있습니다. |                                                     |

더 자세한 내용은 [AWQ documentation](./awq)을 참조하십시오.

### 특정 형식 로드하기[[loading-specific-formats]]

#### 압축된 텐서[[compressed-tensors]]

| 장점                                                         | 단점                                                        |
|--------------------------------------------------------------|-------------------------------------------------------------|
| FP8과 같은 유연한 형식을 지원합니다.        | 주로 사전 양자화된 모델을 로드하기 위한 용도입니다.                 |
| NVIDIA (SM89 이상) 및 Intel XPU에서의 FP8 커널 가속.    |                                                             |

자세한 내용은 [compressed-tensors documentation](./compressed_tensors)을 참조하십시오.

## 미세 조정[[fine-tuning]]

메모리를 절약하기 위해 미세 조정 과정에서 아래의 양자화 방법을 고려하십시오.

### bitsandbytes[[training]]

* **설명:** PEFT를 통한 QLoRA 미세 조정의 표준 방법입니다.
* **장점:** 소비자용 GPU에서 대규모 모델의 미세 조정을 가능하게 합니다; PEFT에 대해 널리 지원되며 문서화되어 있습니다.
* **단점:** 주로 NVIDIA GPU용입니다.

다른 방법들도 PEFT 호환성을 제공하지만, bitsandbytes가 QLoRA를 위한 가장 확립되고 간단한 경로입니다.

다음의 내용을 확인하십시오. [bitsandbytes 문서화](./bitsandbytes#qlora) 및 [PEFT 문서](https://huggingface.co/docs/peft/developer_guides/quantization#quantize-a-model) 자세한 내용은 다음을 참조하십시오.

## 연구[[research]]

[AQLM](./aqlm), [SpQR](./spqr), [VPTQ](./vptq), [HIGGS](./higgs) 등과 같은 방법들은 압축의 한계(< 2비트)를 넓히거나 새로운 기술을 탐구합니다.

* 다음의 경우를 고려하십시오:
  * 극단적인 압축(4비트 미만)이 필요합니다.
  * 귀하는 연구를 수행 중이거나 각 논문에서 제공하는 최첨단 결과를 필요로 합니다.
  * 잠재적으로 복잡한 양자화 절차를 수행할 수 있는 상당한 컴퓨팅 자원을 사용할 수 있습니다. 운영 환경에서 사용할 방법을 선택하기 전에 각 방법의 문서와 관련 논문을 주의 깊게 검토할 것을 권장합니다.

## 벤치마크 비교[[benchmark-comparison]]

다양한 양자화 방법의 정량적 비교를 제공하기 위해, Llama 3.1 8B 및 70B 모델을 대상으로 몇 가지 대중적인 기술들을 벤치마크했습니다. 다음 표들은 정확도(높을수록 좋음), 초당 토큰 수로 측정된 추론 처리량(높을수록 좋음), GB 단위로 측정된 최대 VRAM 사용량(낮을수록 좋음), 그리고 양자화 시간을 보여줍니다.

성능 지표는 Llama 3.1 70B (bfloat16)를 위해 2개의 NVIDIA A100 80GB GPU, FP8 방식을 위해 1개의 NVIDIA H100 80GB GPU, 그리고 그 외 모든 방식을 위해 1개의 NVIDIA A100 80GB GPU에서 측정되었습니다. 처리량은 배치 크기 1과 64개 토큰 생성 조건으로 측정되었습니다. `torch.compile` 및 Marlin 커널에 대한 결과는 해당되는 경우와 지원되는 경우에 포함되었습니다.

<iframe
  src="https://huggingface.co/datasets/derekl35/quantization-benchmarks/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
  title="benchmarking results dataset"
></iframe>

주요 요점은 다음과 같습니다:

| 양자화 및 방법론                      | 메모리 절감량 (bf16 대비) | 정확도             | 기타 참고 사항                                                        |
|-------------------------------------------- |------------------------- |--------------------- |------------------------------------------------------------------- |
| **8비트** (bnb-int8, HQQ, Quanto, torchao, fp8) | ~2x             | 기본 bf16 모델과 매우 유사합니다.   |                                                                    |
| **4-bit** (AWQ, GPTQ, HQQ, bnb-nf4, SINQ)    | ~4x                      | 상대적으로 높은 정확도            | AWQ/GPTQ는 종종 정확도 면에서 앞서지만 보정 작업이 필요합니다. HQQ/bnb-nf4/SINQ는 즉석에서 사용하기 쉽습니다. |
| **Sub-4-bit** (VPTQ, AQLM, 2비트 GPTQ) | 극심함 (>4배)            | 특히 2비트에서 눈에 띄는 하락이 발생합니다. | 양자화 시간은 매우 길 수 있습니다(AQLM, VPTQ). 성능은 다양합니다. |

> [!TIP]
> 양자화된 모델이 요구 사항을 충족하는지 확인하기 위해, 특정 작업 및 하드웨어에서 모델의 성능(정확도 및 속도)을 항상 벤치마킹하십시오. 자세한 사용 지침은 위에 링크된 개별 문서 페이지를 참조하십시오.
