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

# 채팅 템플릿 작성하기[[writing-a-chat-template]]

채팅 템플릿은 토크나이저의 [`~PreTrainedTokenizer.chat_template`] 속성에 저장된 [Jinja](https://jinja.palletsprojects.com/en/stable/templates/) 템플릿입니다. Jinja는 Python-유사한 코드와 구문을 작성할 수 있도록 허용하는 템플릿 언어입니다.

```jinja
{%- for message in messages %}
    {{- '<|' + message['role'] + '|>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

이것을 조금 더 바라보면, 실제로 이 내용이 Python과 매우 유사하다는 것을 알 수 있습니다. 다만 몇 가지 이상한 `{%-` 구문이 있습니다. 템플릿은 메시지 목록을 반복하며, 각 메시지에 대해 메시지의 역할과 내용을 출력한 후 시퀀스 종료 토큰을 추가합니다. `add_generation_prompt=True`인 경우, 대화의 끝에 어시스턴트 메시지의 시작 헤더를 추가합니다.

작성된 템플릿을 문자열로 로드하여 토크나이저의 `chat_template` 속성에 할당합니다. 설정 후에는 언제든지 [`~PreTrainedTokenizerBase.apply_chat_template`]를 호출할 때 해당 템플릿이 사용됩니다. 또한 [`~PreTrainedTokenizer.save_pretrained`] 또는 [`~PreTrainedTokenizer.push_to_hub`]가 호출될 때 토크나이저와 함께 템플릿이 저장됩니다. 템플릿은 토크나이저 디렉터리 내의 `chat_template.jinja` 파일에 저장됩니다. 템플릿을 수정하려면 이 파일을 직접 편집할 수 있으며, 템플릿 문자열을 조작하는 것보다 훨씬 간편합니다. 기타 디스크에 저장되는 형식에 대해서는 아래의 [채팅 템플릿 저장 및 로드](#storing-and-loading-chat-templates)를 참조하시기 바랍니다. Transformers가 지원하는 다른 디스크 기반 형식에 대해서도 확인할 수 있습니다.

## 템플릿 작성 팁입니다.[[template-writing-tips]]

Jinja 템플릿을 작성하기 가장 쉬운 방법은 기존 템플릿을 참조하는 것입니다. 어떤 채팅 모델에서도 `print(tokenizer.chat_template)`를 사용하여 사용 중인 템플릿을 확인할 수 있습니다. 도구를 호출하거나 RAG를 지원하지 않는 간단한 모델부터 시작해 보시기 바랍니다. 도구 사용 모델은 매우 복잡한 템플릿을 가질 수 있으므로 주의가 필요합니다. 더 자세한 포맷팅 및 문법 정보는 [Jinja 문서](https://jinja.palletsprojects.com/en/stable/templates/#synopsis)를 참조하시기 바랍니다.

채팅 템플릿을 작성하는 과정에서 특정한 팁과 함정을 마주할 수 있으며, 본 섹션에서는 이러한 사항들을 더 자세히 다룹니다.

### 다중 모달 채팅 템플릿 작성하기입니다.[[writing-multimodal-chat-templates]]

다중 모달 템플릿의 경우, `chat_template` 속성은 토크나이저가 아니라 **프로세서**에 설정됩니다. 메시지의 `content` 키는 단일 문자열이 아니라 일반적으로 콘텐츠 사전의 리스트입니다. 리스트 내 각 콘텐츠 항목의 타입을 확인하고, 적절히 처리하는 것이 좋습니다.

일반적으로 템플릿은 이미지나 동영상 데이터에 직접 접근해서는 안 됩니다. 이는 보통 템플릿 렌더링이 완료된 후 프로세서에서 처리됩니다. 대신 이미지나 동영상 콘텐츠를 만나면 `<|image|>` 또는 `<|video|>`와 같은 특수 토큰 하나를 출력해야 합니다. 이후 프로세서는 이 특수 토큰을 이미지나 동영상 토큰의 시퀀스로 확장합니다. 출력할 정확한 토큰은 사용 중인 모델에 따라 다릅니다. 기존의 멀티모달 프로세서를 로드하여 데이터 처리 방식을 확인하는 것을 강력히 권장합니다.

아래 예제 템플릿은 이미지와 텍스트의 혼합 콘텐츠를 처리합니다.

```jinja
{%- for message in messages %}
    {%- if loop.index0 == 0 %}
        {{- bos_token }}
    {%- endif %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
    {%- if message['content'] is string %}
        {{- message['content'] }}
    {%- else %}
        {%- for content in message['content'] %}
            {%- if content['type'] == 'image' %}
                {{- '<|image|>' }}
            {%- elif content['type'] == 'text' %}
                {{- content['text'] }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
```

이 다중 모달 템플릿은 위의 더 간단한 템플릿과 매우 유사하지만, `content` 목록을 확인하고 필요에 따라 `<|image|>` 토큰을 렌더링하기 위해 이를 반복 처리합니다. 이를 통해 이미지를 사용자 텍스트의 흐름 속에 삽입할 수 있습니다.

모든 모델이 이 방식으로 작동하는 것은 아닙니다. 예를 들어 일부 모델은 모든 이미지를 사용자 메시지의 끝으로 이동할 수 있습니다. 채팅 템플릿은 항상 모델이 훈련된 형식과 일치해야 합니다.

### 공백 줄이기[[trimming-whitespace]]

Jinja는 텍스트 블록 앞뒤의 공백을 그대로 출력합니다. 이는 채팅 템플릿에서 문제가 될 수 있으며, 모델 학습 시 존재하지 않았던 추가 공백이 성능에 악영향을 미칠 수 있습니다. 공백을 제거하려면 Jinja 라인 구문에 `-`를 추가합니다. 이를 통해 파이썬 스타일의 들여쓰기와 줄바꿈을 사용해 템플릿을 작성할 수 있으며, 렌더링된 출력에서 의도치 않게 들여쓰기가 출력되는 것을 방지할 수 있습니다.

아래 예제 템플릿은 `-`를 사용하지 않아 출력 시 불필요한 공백이 포함됩니다.

```jinja
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

의도한 내용만 출력되도록 하기 위해 `-` 사용을 강력히 권장합니다.

```jinja
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

### 특수 변수 및 호출 가능 객체입니다.[[special-variables-and-callables]]

템플릿에서 유일한 상수는 `messages` 변수와 `add_generation_prompt` 부울 값입니다. 그러나 [`~PreTrainedTokenizerBase.apply_chat_template`] 메서드에 전달되는 **모든 기타 키워드 인수**에 접근할 수 있습니다.

이러한 방식은 사양을 설계할 때 고려하지 못한 사용 사례에 대한 지원을 가능하게 하여 유연성을 제공합니다. 가장 흔한 추가 변수는 `tools`이며, 이는 JSON 스키마 형식으로 구성된 도구 목록을 포함합니다. 변수 이름은 원하는 대로 사용할 수 있지만, 관례를 따르고 이 목적에 `tools`를 사용하는 것을 강력히 권장합니다. 이를 통해 템플릿이 표준 API와 더 잘 호환되게 됩니다.

또한 `tokenizer.special_tokens_map`에 포함된 토큰에 접근할 수 있습니다. 이는 종종 `bos_token` 및 `eos_token`와 같은 특수 토큰을 포함합니다. 이름을 직접 사용하여 접근할 수 있으며, 예를 들어 `{{- bos_token }}`와 같이 사용합니다.

사용할 수 있는 호출 가능한 함수가 두 개 있습니다. 이를 호출하려면 `{{- function_name(argument) }}`를 사용합니다.

- `raise_exception(msg)`는 `TemplateException`를 발생시킵니다. 이는 잘못된 템플릿 사용에 대해 디버깅하거나 사용자에게 경고하는 데 유용합니다.
- `strftime_now(format_str)`는 시스템 메시지에서 자주 요구되는 특정 형식의 현재 날짜와 시간을 가져옵니다. 이는 [datetime.now().strftime(format_str)](https://docs.python.org/3/library/datetime.html#datetime.datetime.now)과 동일하며, Python에서 사용됩니다.

### 비호환성에 대한 호환성Python 진자[[compatibility-with-non-python-jinja]]

Jinja는 여러 언어로 구현되어 있으며 일반적으로 동일한 구문을 사용합니다. Python에서 템플릿을 작성하면 문자열에 [lower](https://docs.python.org/3/library/stdtypes.html#str.lower) 또는 사전에 [items](https://docs.python.org/3/library/stdtypes.html#dict.items)와 같은 Python 메서드를 사용할 수 있습니다. 그러나 템플릿이 Python 이외의 구현에서 사용될 경우, 예를 들어 자바스크립트 또는 루스트로 배포할 때는 이와 같은 기능이 작동하지 않습니다.

모든 Jinja 구현 간 호환성을 보장하기 위해 아래의 변경 사항을 적용합니다.

- Python 메서드를 Jinja 필터로 대체합니다. 예를 들어, `string.lower()`를 `string|lower` 또는 `dict.items()`를 `dict|dictitems`로 대체합니다. 대부분의 변경 사항은 동일한 패턴을 따르지만, `string.strip()`는 `string|trim`으로 대체됩니다. 전체 필터 목록은 [내장 필터](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters)를 참조하시기 바랍니다.
- `True`, `False`, `None` (이들은 Python 전용입니다)를 각각 `true`, `false`, `none`로 변경합니다.
- 사전이나 리스트를 직접 렌더링하면 다른 구현에서 다른 결과를 반환할 수 있습니다. 예를 들어, 문자열 항목이 단일 인용부호에서 이중 인용부호로 변경될 수 있습니다. 이를 방지하기 위해 일관성을 유지하기 위해 [tojson](https://jinja.palletsprojects.com/en/3.1.x/templates/#jinja-filters.tojson) 필터를 추가합니다.

### 큰 템플릿[[big-templates]]

최신 모델이나 [도구 호출](./chat_extras) 및 RAG와 같은 기능을 갖춘 모델은 100줄을 넘는 길이의 템플릿이 필요할 수 있습니다. 더 큰 템플릿은 별도의 파일에 작성하는 것이 더 쉬울 수 있습니다. 별도 파일의 줄 번호는 템플릿 파싱 또는 실행 오류에서 나타나는 줄 번호와 정확히 일치하므로 잠재적인 문제를 보다 쉽게 디버깅할 수 있습니다.

템플릿을 별도의 파일로 작성하고 채팅 템플릿으로 추출합니다.

```py
open("template.jinja", "w").write(tokenizer.chat_template)
```

편집된 템플릿을 다시 토크나이저에 불러올 수도 있습니다.

```py
tokenizer.chat_template = open("template.jinja").read()
```

## 채팅 템플릿 저장 및 로드합니다[[storing-and-loading-chat-templates]]

채팅 템플릿은 디스크에 여러 가지 다른 형식으로 저장됩니다. 최신 체크포인트는 템플릿을 별도의 `.jinja` 파일로 저장하는 반면, 오래된 체크포인트는 토크나이저 또는 프로세서 설정에 템플릿을 내장합니다.

### 저장 형식[[storage-formats]]

템플릿은 다음 중 어느 형식으로든 저장할 수 있습니다.

- `chat_template.jinja` (권장됨). 리포지토리 루트에 위치한 독립적인 Jinja 파일로, 하나의 채팅 템플릿을 포함합니다. 이는 [`~PreTrainedTokenizer.save_pretrained`]가 기본적으로 작성하는 내용입니다. 템플릿을 별도의 파일에 저장하면 검사, 편집, 비교가 용이합니다. 토큰라이저와 프로세서는 `chat_template.jinja`를 동일한 방식으로 로드합니다.
- `additional_chat_templates/<name>.jinja`. 모델이 여러 이름이 지정된 템플릿을 함께 제공할 경우 사용하는 독립형 Jinja 파일 디렉터리입니다 (예: `default` 템플릿과 별도의 `tool_use` 템플릿). `default` 템플릿은 여전히 리포지토리 루트에 있는 `chat_template.jinja`에 위치하지만, 그 외의 모든 이름이 지정된 템플릿은 `additional_chat_templates/<name>.jinja`에 위치하며, 파일 이름의 접두사 부분이 템플릿 이름이 됩니다.

> [!WARNING]
> 아래의 레거시 형식은 후방 호환성 있는 로딩을 위해 유지되고 있습니다. 이 둘 중 어느 것도 채팅 템플릿으로 작성하지 마십시오.

- `chat_template` 필드는 `tokenizer_config.json`에 포함됩니다. 독립적인 `.jinja` 파일이 도입되기 이전에 사용되던 로드 전용 레거시 형식입니다. 템플릿은 `tokenizer_config.json` 내부에 JSON 문자열로 포함됩니다. 모델에 여러 개의 이름이 지정된 템플릿이 있는 경우, 이 필드는 단일 문자열이 아니라 `{"name": ..., "template": ...}` 사전들의 리스트 형태가 됩니다. 기존에 이 형식을 사용하는 저장소는 여전히 로드 가능하지만, [`~PreTrainedTokenizer.save_pretrained`]는 현대적인 `.jinja` 형식을 사용하여 저장합니다.

- `chat_template.json`. 오래된 다중 모달 프로세서 체크포인트에서 사용하는 읽기 전용 레거시 형식입니다. JSON 형식의 파일로, `{"chat_template": "<template string>"}` 형태를 가집니다. 기존에 이 형식을 사용하는 저장소는 계속 로드되지만, [`~ProcessorMixin.save_pretrained`]는 대신 현대적인 `.jinja` 형식을 작성합니다. 레거시 `chat_template.json` 파일과 현대적인 `.jinja` 파일을 혼합한 프로세서 저장소는 로드 시 오류를 발생시킵니다.

### 로딩 우선순위입니다.[[loading-precedence]]

[`~PreTrainedTokenizer.from_pretrained`]를 호출할 때, Transformers는 고정된 우선순위에 따라 저장 형식을 해결합니다. 구성 파일 내에 포함된 템플릿보다 독립적인 `.jinja` 파일이 우선 순위가 높습니다. 로더:

1. `chat_template` 필드가 `tokenizer_config.json`에 포함되어 있는 경우 이를 읽습니다 (프로세서의 경우, 이전 버전의 `chat_template.json`도 포함합니다).
2. 레포지토리 루트에 `chat_template.jinja` 파일이 존재할 경우 이를 읽어 `default` 템플릿으로 사용하며, 단계 1을 무시합니다.
3. `additional_chat_templates/` 폴더 내의 모든 `.jinja` 파일을 파일 이름의 접두사로 키로 하여 읽고, 이를 병합합니다.

결과에 단일한 `default` 템플릿이 있는 경우, [`~PreTrainedTokenizer.chat_template`]은 해당 문자열로 설정됩니다. 여러 개의 이름이 지정된 템플릿이 존재하는 경우, `chat_template`는 `{name: template_string}` 형식의 사전이 됩니다. 이 경우, [`~PreTrainedTokenizer.apply_chat_template`]는 도구가 전달되었을 때 `tool_use` 항목을 선택하고, 그렇지 않을 때는 `default`를 선택합니다.

### 저장하기[[saving]]

[`~PreTrainedTokenizer.save_pretrained`] 및 [`~PreTrainedTokenizer.push_to_hub`]는 기본적으로 `.jinja` 형식을 작성합니다. 단일 문자열 템플릿은 `chat_template.jinja`로 변환됩니다. 이름이 지정된 템플릿의 사전은 `default` 항목을 `chat_template.jinja`에 기록하고, 나머지 항목은 `additional_chat_templates/` 아래에 각각 별도의 파일로 작성합니다. `chat_template` 필드는 중복을 방지하기 위해 `tokenizer_config.json`에서 제거됩니다.

구식 형식 중 하나에 템플릿을 저장하는 지원되는 방법이 없습니다. 이러한 형식은 오래된 저장소를 로드하기 위해만 유지되고 있습니다.

### 오래된 저장소 업데이트하기[[updating-an-older-repository]]

`tokenizer_config.json` 또는 `chat_template.json`에 내장된 템플릿을 다시 로드하고 저장함으로써 권장하는 `.jinja` 형식으로 마이그레이션합니다.

로드 단계는 저장소가 사용하는 임의의 레거시 형식을 `chat_template`에서 정규화하며, [`~PushToHubMixin.push_to_hub`]는 `chat_template.jinja` 파일을 반환합니다.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-org/your-model")
tokenizer.push_to_hub("your-org/your-model")
```

## 도구용 템플릿입니다.[[templates-for-tools]]

도구용 템플릿을 작성하는 데 특정 형식이 있는 것은 아니지만, 표준 API를 따르는 것이 가장 좋습니다. 이를 통해 사용자가 모델과 도구를 사용하기 위해 맞춤형 코드를 작성하지 않아도 되어 템플릿이 다양한 모델에서 널리 접근 가능하게 됩니다.

> [!WARNING]
> 공백 및 특수 토큰과 같은 형식은 모델에 따라 다릅니다. 모델이 훈련된 형식과 정확히 일치하는지 확인합니다.

다음 섹션은 도구용 템플릿을 작성하기 위한 표준 API의 요소를 나열합니다.

### 도구 정의합니다[[tool-definitions]]

[도구](./chat_extras)는 Python 함수 또는 JSON 스키마로 전달됩니다. 함수가 전달될 경우, 자동으로 JSON 스키마가 생성되어 템플릿에 전달됩니다. 템플릿에서 `tools` 변수에 접근할 때는 항상 JSON 스키마의 리스트입니다.

템플릿은 항상 도구를 JSON 스키마 형식으로 받지만, 모델이 훈련된 형식과 일치시키기 위해 렌더링 시 이 형식을 근본적으로 변경해야 할 수 있습니다. 예를 들어, [Command-R](./model_doc/cohere)는 Python 함수 헤더로 정의된 도구를 기반으로 훈련되었습니다. 템플릿은 내부적으로 JSON 스키마 형식을 변환하여 입력 도구를 Python 헤더 형식으로 렌더링합니다.

아래 예제는 도구가 JSON 스키마 형식으로 어떻게 정의되는지를 보여줍니다.

```json
{
  "type": "function",
  "function": {
    "name": "multiply",
    "description": "A function that multiplies two numbers",
    "parameters": {
      "type": "object",
      "properties": {
        "a": {
          "type": "number",
          "description": "The first number to multiply"
        },
        "b": {
          "type": "number",
          "description": "The second number to multiply"
        }
      },
      "required": ["a", "b"]
    }
  }
}
```

채팅 템플릿에서 도구 정의를 처리하는 예시는 아래와 같습니다. 특정 토큰과 레이아웃은 모델이 훈련된 방식에 맞게 변경되어야 합니다.

```jinja
{%- if tools %}
    {%- for tool in tools %}
        {{- '<tool>' + tool['function']['name'] + '\n' }}
        {%- for argument in tool['function']['parameters']['properties'] %}
            {{- argument + ': ' + tool['function']['parameters']['properties'][argument]['description'] + '\n' }}
        {%- endfor %}
        {{- '\n</tool>' }}
    {%- endfor %}
{%- endif %}
```

### 도구 호출합니다[[tool-calls]]

도구 정의를 렌더링하는 것 외에도, 템플릿에서 **도구 호출**과 **도구 응답**도 렌더링해야 합니다.

도구 호출은 일반적으로 `tool_calls` 키에 포함된 `"assistant”` 메시지에 전달됩니다. 대부분의 도구 호출 모델이 단일 도구 호출만 지원하기 때문에, 이는 항상 리스트 형태이지만 보통 리스트에는 단일 요소만 포함되어 있습니다.

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "multiply",
        "arguments": {
          "a": 5,
          "b": 6
        }
      }
    }
  ]
}
```

도구 호출을 처리하는 일반적인 패턴은 아래와 같습니다. 이 예를 시작점으로 사용할 수 있지만, 템플릿이 모델이 훈련된 형식과 실제로 일치하는지 반드시 확인합니다.!

```jinja
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
    {%- endfor %}
{%- endif %}
```

### 도구 응답입니다[[tool-responses]]

도구 응답은 `tool` 역할을 가진 메시지 사전입니다. 도구 호출보다 훨씬 간단하며, 일반적으로 `role`, `name`, `content` 키만 포함합니다.

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```

일부 템플릿은 `name` 키가 필요하지 않을 수도 있으며, 이 경우 템플릿을 단지 `content` 키만 읽도록 작성할 수 있습니다.

```jinja
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

## 기여합니다[[contribute]]

템플릿이 준비되면 토크나이저의 `chat_template` 속성에 설정하고, [`~PreTrainedTokenizerBase.apply_chat_template`]로 테스트합니다. 예상대로 작동한다면, [`~PreTrainedTokenizer.push_to_hub`]를 사용하여 Hub에 업로드합니다.

모델 소유자가 아니더라도 빈 또는 잘못된 채팅 템플릿을 가진 모델에 대한 템플릿을 추가하는 것이 여전히 유용합니다. 모델 리포지토리에서 [풀 리퀘스트](https://hf.co/docs/hub/repositories-pull-requests-discussions)를 열어 템플릿을 추가합니다.!

```py
tokenizer.chat_template = template
tokenizer.push_to_hub("amazing_company/cool_model", commit_message="Add chat template", create_pr=True)
```
