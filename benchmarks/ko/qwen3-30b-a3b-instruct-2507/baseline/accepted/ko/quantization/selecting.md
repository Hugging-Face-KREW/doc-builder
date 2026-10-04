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

# 양자화 방법 선택하기입니다[[selecting-a-quantization-method]]

Transformers에는 추론 및 미세 조정용으로 다양한 양자화 방법이 제공됩니다. 본 가이드는 사용 사례에 따라 가장 일반적이고 프로덕션 환경에서 사용 가능한 양자화 기법을 선택하는 데 도움을 주며, 각 기법의 장점과 단점을 제시합니다.

모든 지원되는 방법과 그 기능에 대한 포괄적인 개요는 [개요](./overview)에 있는 표를 참조하시기 바랍니다.

## 추론합니다[[inference]]

추론을 위한 아래의 양자화 방법을 고려합니다.

| 양자화 방법입니다 | 사용 사례 |
|---|---|
| bitsandbytes | 사용 편의성 및 QLoRA 미세 조정은 NVIDIA 및 인텔 GPU에서 지원합니다. |
| 압축된 텐서 | 특정 양자화 형식(FP8) 로딩합니다 |
| GPTQModel 또는 AWQ | 사전 캘리브레이션을 통해 우수한 4비트 정확도를 제공합니다. |
| HQQ | 캘리브레이션 없이 빠른 온더플라이 양자화입니다. |
| SINQ | 캘리브레이션 없이 초고속이면서도 고품질의 실시간 양자화를 제공합니다. |
| torchao | 유연성과 빠른 추론을 위한 torch.compile |

### 보정이 필요 없음 (실시간 양자화)합니다.[[no-calibration-required-on-the-fly-quantization]]

이 방법들은 별도의 캘리브레이션 데이터셋이나 단계가 필요하지 않기 때문에 일반적으로 사용하기가 더 쉽습니다.

#### bitsandbytes[[bitsandbytes]]

| 장점                                                         | 단점입니다.                                                    |
|--------------------------------------------------------------|---------------------------------------------------------|
| 매우 간단하며, 추론 시 보정 데이터셋이 필요하지 않습니다.  | 주로 NVIDIA GPU(CUDA)에 최적화되어 있습니다.             |
| 좋은 커뮤니티 지원과 널리 채택되고 있습니다.                   | 추론 속도 향상이 보장되지 않습니다.                     |

자세한 내용은 [bitsandbytes 문서](./bitsandbytes)를 참조하시기 바랍니다.

#### HQQ (반정수 양자화)[[hqq-half-quadratic-quantization]]

| 장점                                                                 | 단점입니다                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| 빠른 양자화 프로세스로 캘리브레이션 데이터가 필요하지 않습니다.              | 비트 깊이가 <4비트 미만일 경우 정확도가 크게 저하됩니다.                     |
| 빠른 추론을 위한 다중 백엔드 지원합니다.                                | 추론 속도는 `torch.compile` 또는 백엔드를 사용하지 않는 한 다른 것들과 일치하지 않을 수 있습니다. |
| `torch.compile`와 호환됩니다.                                     |                                                                            |
| 다양한 비트 깊이(8비트, 4비트, 3비트, 2비트, 1비트)를 지원합니다.              |                                                                            |

자세한 내용은 [HQQ 설명서](./hqq)를 참조하시기 바랍니다.

#### SINQ[[sinq]]

| 장점                                                                 | 단점입니다.                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| 초고속이지만 고품질의 양자화 프로세스로, 보정 데이터가 필요하지 않습니다.              | 비트 폭이 <=2비트보다 작을 경우 정확도가 크게 저하될 수 있습니다.                     |
| GemLite는 더 빠른 추론을 위한 백엔드입니다.                                  | 3비트 모델의 경우 추론 속도가 느립니다 (gemlite 커널 없음)
| 8비트, 4비트, 3비트, 2비트 등 광범위한 비트 폭을 지원합니다.              |                                                                            |

자세한 내용은 [SINQ 문서](./sinq)를 참조하시기 바랍니다.

#### torchao[[torchao]]

| 장점                                                                 | 단점입니다                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 속도 향상을 위한 `torch.compile`와의 강력한 통합이 제공됩니다.     | 최근에 출시된 라이브러리로, 생태계가 여전히 발전 중입니다.                             |
| 합리적인 CPU 양자화 지원을 제공합니다.                              | 성능은 `torch.compile`가 잘 작동하는지에 따라 달라집니다.                 |
| 양자화 방식의 유연성(int8, int4, fp8).           | 4비트 양자화(int4wo)는 정확도 측면에서 GPTQ/AWQ와 일치하지 않을 수 있습니다.              |

자세한 내용은 [torchao 문서](./torchao)를 참조하시기 바랍니다.

### 보정 기반 양자화입니다[[calibration-based-quantization]]

이 방법들은 더 높은 정확도를 달성하기 위해 데이터셋을 사용하여 사전 캘리브레이션 단계를 거쳐야 합니다.

#### GPTQ/GPTQModel[[gptqgptqmodel]]

8B 모델의 캘리브레이션은 하나의 A100 GPU에서 ~20분이 소요됩니다.

| 장점                                                                 | 단점입니다.                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 일반적으로 높은 정확도를 달성합니다.                                        | 교정 데이터셋과 별도의 교정 단계가 필요합니다.      |
| 추론 속도 향상으로 이어질 수 있습니다.                                      | 교정 데이터에 과적합될 수 있습니다.                             |
| 많은 사전 양자화된 GPTQ 모델들이 [Hugging Face Hub](https://huggingface.co/models?other=gptq)에 있습니다. |                                           |

자세한 내용은 [GPTQ 설명서](./gptq)를 참조하시기 바랍니다.

#### AWQ (활성화 인지 가중치 양자화)[[awq-activation-aware-weight-quantization]]

8B 모델의 캘리브레이션은 하나의 A100 GPU에서 ~10분이 소요됩니다.

| 장점                                                                 | 단점입니다                                                |
|----------------------------------------------------------------------|-----------------------------------------------------|
| 4비트에서 높은 정확도를 자주 달성합니다. (때로는 특정 작업에서 GPTQ를 초과하기도 합니다.) | 자체 양자화를 수행하는 경우 보정이 필요합니다.        |
| 추론 속도 향상으로 이어질 수 있습니다.                                      |                                                     |
| GPTQ보다 보정 시간이 짧습니다.                                  |                                                     |
| 많은 사전 양자화된 AWQ 모델이 [Hugging Face Hub](https://huggingface.co/models?other=awq)에 있습니다. |                                                     |

자세한 내용은 [AWQ 문서](./awq)를 참조하시기 바랍니다.

### 특정 형식 로딩합니다[[loading-specific-formats]]

#### 압축된 텐서[[compressed-tensors]]

| 장점                                                         | 단점입니다                                                        |
|--------------------------------------------------------------|-------------------------------------------------------------|
| FP8과 같은 유연한 형식을 지원합니다.        | 주로 사양화된 모델을 로드하는 데 사용합니다.                 |
| NVIDIA (SM89+) 및 인텔 XPU에서 FP8 커널 가속을 지원합니다.    |                                                             |

자세한 내용은 [압축 텐서 문서](./compressed_tensors)를 참조하시기 바랍니다.

## 미세 조정합니다.[[fine-tuning]]

미세 조정 중에 메모리를 절약하기 위해 아래의 양자화 방법을 고려합니다.

### bitsandbytes[[training]]

* **설명:** PEFT를 통한 QLoRA 미세 조정의 표준 방법입니다.
* **장점:** 소비자용 GPU에서 대규모 모델의 미세 조정을 가능하게 합니다; PEFT에 대해 널리 지원되며 문서화되어 있습니다.
* **단점:** 주로 NVIDIA 그래픽 카드용입니다.

다른 방법들도 PEFT 호환성을 제공하지만, bitsandbytes가 QLoRA에 가장 널리 사용되고 직관적인 접근 방식입니다.

자세한 내용은 [bitsandbytes 문서](./bitsandbytes#qlora)와 [PEFT 문서](https://huggingface.co/docs/peft/developer_guides/quantization#quantize-a-model)를 참조하시기 바랍니다.

## 연구합니다[[research]]

메서드와 같은 것들 [AQLM](./aqlm), [SpQR](./spqr), [VPTQ](./vptq), [HIGGS](./higgs)등, 압축의 한계를 넓히는 것< 2비트)이거나 새로운 기법을 탐구합니다.

* 다음 사항들을 고려하십시오:
  * 초기 압축(4비트 미만)이 필요합니다.
  * 당신은 연구를 수행 중이거나 해당 논문들에서 최신 기술 결과를 필요로 하고 있습니다.
  * 복잡한 양자화 절차에 대해 상당한 계산 자원이 사용 가능합니다. 제작 환경에서 사용하기 전에 각 방법의 문서와 관련 논문을 신중히 검토하는 것을 권장합니다.

## 성능 비교 벤치마크[[benchmark-comparison]]

다양한 양자화 방법 간의 정량적 비교를 제공하기 위해, Llama 3.1 8B 및 70B 모델에서 여러 인기 있는 기법을 벤치마킹하였습니다. 다음 표는 정확도(높을수록 좋음), 추론 처리량(토큰/초, 높을수록 좋음), 최대 VRAM 사용량(GB, 낮을수록 좋음), 그리고 양자화 시간을 보여줍니다.

성능 메트릭은 Llama 3.1 70B(bfloat16)에 대해 2개의 NVIDIA A100 80GB GPU, FP8 방법에 대해 1개의 NVIDIA H100 80GB GPU, 기타 모든 방법에 대해 1개의 NVIDIA A100 80GB GPU에서 측정되었습니다. 처리량은 배치 크기 1로 설정하고 64토큰을 생성하여 측정하였습니다. 해당 사항이 적용되고 지원되는 경우 `torch.compile` 및 Marlin 커널의 결과도 포함되어 있습니다.

<iframe
  src="https://huggingface.co/datasets/derekl35/quantization-benchmarks/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
  title="benchmarking results dataset"
></iframe>

핵심 내용은 다음과 같습니다:

| 양자화 및 방법                      | 메모리 절약량 (bf16 대비) | 정확도             | 기타 참고 사항                                                        |
|-------------------------------------------- |------------------------- |--------------------- |------------------------------------------------------------------- |
| **8비트** (bnb-int8, HQQ, Quanto, torchao, fp8) | ~2x             | 베이스라인 bf16 모델에 매우 가까움입니다   |                                                                    |
| **4-bit** (AWQ, GPTQ, HQQ, bnb-nf4, SINQ)    | ~4x                      | 상대적으로 높은 정확도입니다            | AWQ/GPTQ는 정확도에서 뛰어나지만 캘리브레이션을 필요로 합니다. HQQ/bnb-nf4/SINQ는 실시간으로 사용하기에 간편합니다. |
| **4비트 미만** (VPTQ, AQLM, 2비트 GPTQ) | 극한(>4배 이상)입니다.            | 두드러진 하락, 특히 2비트에서 | 양자화 시간이 매우 길 수 있습니다 (AQLM, VPTQ). 성능은 다양합니다. |

> [!TIP]
> 정량화된 모델의 성능(정확도 및 속도)을 특정 작업과 하드웨어에서 항상 벤치마킹하여 요구 사항을 충족하는지 확인합니다. 자세한 사용 방법은 위에서 링크된 개별 문서 페이지를 참조합니다.
