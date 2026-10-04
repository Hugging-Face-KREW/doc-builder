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

Transformers에는 추론과 미세 조정을 위한 다양한 양자화 방법이 있습니다. 이 가이드는 사용 사례에 따라 가장 일반적이고 프로덕션 환경에서 사용할 수 있는 양자화 기술을 선택하는 데 도움을 주며, 각 기술의 장단점을 제시합니다.

지원되는 모든 방법과 그 기능에 대한 포괄적인 개요는 [개요](./overview)의 표를 참조하십시오.

## 추론[[inference]]

추론을 위해 아래 양자화 방법들을 고려해 보십시오.

| 양자화 방법 | 사용 사례 |
|---|---|
| bitsandbytes | 사용 편의성 및 NVIDIA와 Intel GPU에서의 QLoRA 미세 조정 |
| 압축된 텐서 | 특정 양자화 형식(FP8)을 로드하는 중 |
| GPTQModel 또는 AWQ | 사전 보정을 통한 양호한 4비트 정확도 |
| HQQ | 교정 없이 빠른 온더플라이 양자화 |
| SINQ | 교정 없이 초고속으로 고품질 온더플라이 양자화를 수행합니다 |
| torchao | 유연하고 빠른 추론을 위한 torch.compile |

### 교정 불필요 (온더플라이 양자화)[[no-calibration-required-on-the-fly-quantization]]

이러한 방법들은 별도의 보정 데이터셋이나 단계가 필요하지 않으므로 일반적으로 사용하기가 더 쉽습니다.

#### bitsandbytes[[bitsandbytes]]

| 장점                                                         | 단점                                                    |
|--------------------------------------------------------------|---------------------------------------------------------|
| 매우 간단하며, 추론을 위해 보정 데이터셋이 필요하지 않습니다.  | 주요 최적화 대상은 NVIDIA GPU(CUDA)입니다.             |
| 좋은 커뮤니티 지원과 광범위한 채택                   | 추론 속도 향상이 보장되지 않습니다.                     |

자세한 내용은 [bitsandbytes 문서](./bitsandbytes)를 참조하십시오.

#### HQQ(반이차 양자화)[[hqq-half-quadratic-quantization]]

| 장점                                                                 | 단점                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| 빠른 양자화 프로세스이며, 보정 데이터가 필요하지 않습니다.              | 비트 깊이가 <4비트인 경우 정확도가 크게 저하될 수 있습니다.                     |
| 빠른 추론을 위한 여러 백엔드                                | 추론 속도는 `torch.compile` 또는 백엔드를 사용하지 않는 경우 다른 것과 일치하지 않을 수 있습니다. |
| `torch.compile`와 호환됩니다.                                     |                                                                            |
| 다양한 비트 깊이(8, 4, 3, 2, 1비트)를 지원합니다.              |                                                                            |

자세한 내용은 [HQQ 문서](./hqq)를 참조하십시오.

#### SINQ[[sinq]]

| 장점                                                                 | 단점                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| 매우 빠르면서도 고품질의 양자화 과정이며, 보정 데이터가 필요하지 않습니다.              | 비트 폭에서 정확도가 크게 저하될 수 있습니다 <=2-bit.                     |
| 더 빠른 추론을 위한 GemLite 백엔드.                                  | 3비트 모델의 느린 추론 (gemlite 커널 없음)
| 다양한 비트 폭(8, 4, 3, 2비트)을 지원합니다.              |                                                                            |

자세한 내용은 [SINQ 문서](./sinq)를 참조하십시오.

#### torchao[[torchao]]

| 장점                                                                 | 단점                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 강력한 통합 `torch.compile` 잠재적인 속도 향상을 위해.     | 새로운 라이브러리이며, 생태계는 여전히 발전하고 있습니다.                             |
| 적절한 CPU 양자화 지원을 제공합니다.                              | 성능은 `torch.compile`이 잘 작동하는지에 따라 달라집니다.                 |
| 양자화 방식(int8, int4, fp8)의 유연성           | 4-bit 양자화(int4wo)의 정확도가 GPTQ/AWQ와 일치하지 않을 수 있습니다.              |

자세한 내용은 [torchao 문서](./torchao)를 참조하십시오.

### 교정 기반 양자화[[calibration-based-quantization]]

이러한 방법들은 더 높은 정확도를 달성하기 위해 데이터셋을 사용하여 사전 보정 단계를 요구합니다.

#### GPTQ/GPTQModel[[gptqgptqmodel]]

8B 모델의 보정은 ~1개의 A100 GPU에서 약 20분이 소요됩니다.

| 장점                                                                 | 단점                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 높은 정확도를 자주 달성합니다.                                        | 교정 데이터셋과 별도의 교정 단계가 필요합니다.      |
| 추론 속도 향상에 기여할 수 있습니다.                                      | 교정 데이터에 과적합이 발생할 수 있습니다.                             |
| [Hugging Face Hub](https://huggingface.co/models?other=gptq)에는 사전 양자화된 GPTQ 모델이 많이 있습니다. |                                           |

자세한 내용은 [GPTQ 문서](./gptq)를 참조하십시오.

#### AWQ(활성화 인식 가중치 양자화)[[awq-activation-aware-weight-quantization]]

8B 모델의 보정은 하나의 A100 GPU에서 ~10분이 소요됩니다.

| 장점                                                                 | 단점                                                |
|----------------------------------------------------------------------|-----------------------------------------------------|
| 4비트에서 높은 정확도를 달성하는 경우가 많습니다. (특정 작업에서는 GPTQ를 초과하기도 합니다.) | 자체 양자화를 수행하는 경우 보정이 필요합니다.        |
| 추론 속도 향상에 기여할 수 있습니다.                                      |                                                     |
| GPTQ보다 더 짧은 보정 시간                                  |                                                     |
| [Hugging Face Hub](https://huggingface.co/models?other=awq)에는 사전 양자화된 AWQ 모델이 많이 있습니다. |                                                     |

자세한 내용은 [AWQ 문서](./awq)를 참조하십시오.

### 특정 형식 불러오기[[loading-specific-formats]]

#### 압축된 텐서[[compressed-tensors]]

| 장점                                                         | 단점                                                        |
|--------------------------------------------------------------|-------------------------------------------------------------|
| FP8과 같은 유연한 형식을 지원합니다.        | 주요 용도로는 사전 양자화 모델의 로딩에 사용됩니다.                 |
| NVIDIA (SM89+) 및 Intel XPU에서 FP8 커널 가속.    |                                                             |

자세한 내용은 [압축 텐서 문서](./compressed_tensors)를 참조하십시오.

## 미세 조정[[fine-tuning]]

미세 조정 시 메모리를 절약하기 위해 아래 양자화 방법을 고려하십시오.

### bitsandbytes[[training]]

* **설명:** PEFT를 통한 QLoRA 미세 조정의 표준 방법입니다.
* **장점:** 소비자용 GPU에서 대규모 모델의 미세 조정을 가능하게 하며, PEFT에 대해 광범위하게 지원되고 문서화되어 있습니다.
* **단점:** 주로 NVIDIA GPU에 해당합니다.

다른 방법들은 PEFT 호환성을 제공하지만, bitsandbytes는 QLoRA에 대해 가장 확립되고 직관적인 경로입니다.

자세한 내용은 [bitsandbytes 문서](./bitsandbytes#qlora) 및 [PEFT 문서](https://huggingface.co/docs/peft/developer_guides/quantization#quantize-a-model)를 참조하십시오.

## 연구[[research]]

[AQLM](./aqlm), [SpQR](./spqr), [VPTQ](./vptq), [HIGGS](./higgs) 등의 방법은 압축의 한계(< 2비트)를 확장하거나 새로운 기법을 탐구합니다.

* 다음 if 문들을 고려해 보겠습니다:
  * 극도로 높은 압축률(4비트 미만)이 필요합니다.
  * 연구를 수행 중이거나 해당 논문들에서 최첨단 결과를 필요로 하는 경우입니다.
  * 복잡한 양자화 절차에 사용할 수 있는 상당한 수준의 컴퓨트 리소스가 있습니다. 프로덕션 환경에서 사용할 방법을 선택하기 전에 각 방법의 문서와 관련 논문을 신중하게 검토하는 것을 권장합니다.

## 벤치마크 비교[[benchmark-comparison]]

다양한 양자화 방법의 정량적 비교를 제공하기 위해, Llama 3.1 8B 및 70B 모델에서 여러 인기 있는 기법을 벤치마크했습니다. 다음 표에는 정확도(높을수록 좋음), 토큰/초 단위로 측정된 추론 처리량(높을수록 좋음), GB 단위로 측정된 피크 VRAM 사용량(낮을수록 좋음), 그리고 양자화 시간이 표시됩니다.

성능 지표는 Llama 3.1 70B(bfloat16)의 경우 2개의 NVIDIA A100 80GB GPU, FP8 방식의 경우 1개의 NVIDIA H100 80GB GPU, 그리고 기타 모든 방식의 경우 1개의 NVIDIA A100 80GB GPU에서 측정되었습니다. 처리량은 배치 크기가 1이고 64개의 토큰을 생성하는 조건으로 측정되었습니다. `torch.compile` 및 Marlin 커널에 대한 결과는 해당되고 지원되는 경우에 포함됩니다.

<iframe
  src="https://huggingface.co/datasets/derekl35/quantization-benchmarks/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
  title="benchmarking results dataset"
></iframe>

주요 내용은 다음과 같습니다:

| 양자화 및 방법                      | 메모리 절약 (bf16 대비) | 정확도             | 기타 참고 사항                                                        |
|-------------------------------------------- |------------------------- |--------------------- |------------------------------------------------------------------- |
| **8비트** (bnb-int8, HQQ, Quanto, torchao, fp8) | ~2x             | 베이스라인 bf16 모델과 매우 유사합니다   |                                                                    |
| **4-bit** (AWQ, GPTQ, HQQ, bnb-nf4, SINQ)    | ~4x                      | 상대적으로 높은 정확도            | AWQ/GPTQ는 정확도에서 우위를 점하는 경우가 많지만 보정이 필요합니다. HQQ/bnb-nf4/SINQ는 온더플라이로 쉽게 적용할 수 있습니다. |
| **4비트 미만** (VPTQ, AQLM, 2비트 GPTQ) | 극단적 (>4배)            | 특히 2비트에서 눈에 띄는 감소 | 양자화 시간은 매우 길 수 있습니다(AQLM, VPTQ). 성능은 다양합니다. |

> [!TIP]
> 특정 작업 및 하드웨어에서 양자화 모델의 성능(정확도와 속도)을 항상 벤치마크하여 요구 사항을 충족하는지 확인하십시오. 자세한 사용 방법은 위에서 링크된 개별 문서 페이지를 참조하십시오.
