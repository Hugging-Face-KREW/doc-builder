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

# 채팅 템플릿 작성[[writing-a-chat-template]]

채팅 템플릿은 토크나이저의 [`~PreTrainedTokenizer.chat_template`] 속성에 저장된 [Jinja](https://jinja.palletsprojects.com/en/stable/templates/) 템플릿입니다. Jinja는 Python과 유사한 코드와 구문을 작성할 수 있는 템플릿 언어입니다.

```jinja
{%- for message in messages %}
    {{- '<|' + message['role'] + '|>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

이를 잠시 살펴보면, 이는 실제로 Python과 매우 유사하지만, 몇 가지 특이한 `{%-` 구문을 사용한다는 점을 인식하게 됩니다. 템플릿은 메시지 목록을 반복하며, 각 메시지에 대해 메시지의 역할과 내용을 출력한 다음 시퀀스 종료 토큰을 출력합니다. `add_generation_prompt=True`인 경우, 대화의 끝에 어시스턴트 메시지의 시작 헤더를 추가합니다.

작성된 템플릿을 문자열로 로드하여 크나이저의 `chat_template` 속성에 할당합니다. 설정이 완료되면 [`~PreTrainedTokenizerBase.apply_chat_template`]를 호출할 때마다 해당 템플릿이 사용됩니다. 또한 [`~PreTrainedTokenizer.save_pretrained`] 또는 [`~PreTrainedTokenizer.push_to_hub`]가 호출될 때 토크나이저와 함께 저장됩니다. 템플릿은 토크나이저 디렉터리의 `chat_template.jinja` 파일에 저장됩니다. 이 파일을 직접 편집하여 템플릿을 변경할 수 있으며, 이는 템플릿 문자열을 조작하는 것보다 일반적으로 더 쉽습니다. Transformers가 지원하는 다른 디스크 저장 형식에 대해서는 아래 [채팅 템플릿 저장 및 로드](#storing-and-loading-chat-templates)를 참조하십시오.

## 템플릿 작성 팁[[template-writing-tips]]

Jinja 템플릿 작성을 시작하는 가장 쉬운 방법은 기존 템플릿을 참고하는 것입니다. 어떤 채팅 모델이든 `print(tokenizer.chat_template)`을 사용하여 해당 모델이 사용하는 템플릿을 확인할 수 있습니다. 도구 호출을 하지 않거나 RAG를 지원하지 않는 간단한 모델부터 시작해 보십시오. 도구 사용 모델은 매우 복잡한 템플릿을 가질 수 있기 때문입니다. 마지막으로, 포맷팅 및 구문에 대한 자세한 내용은 [Jinja 문서](https://jinja.palletsprojects.com/en/stable/templates/#synopsis)를 참고하십시오.

다만, 채팅 템플릿을 작성할 때 특별히 유의해야 할 팁과 함정이 있으며, 이 섹션에서는 그 중 일부를 보다 자세히 다루겠습니다.

### 멀티모달 채팅 템플릿 작성[[writing-multimodal-chat-templates]]

멀티모달 템플릿의 경우, `chat_template` 속성은 토크나이저가 아닌 **프로세서**에 설정됩니다. 메시지의 `content` 키는 단일 문자열이 아니라 콘텐츠 딕셔너리의 목록인 경우가 많습니다. 목록의 각 콘텐츠 항목의 유형을 확인하고 그에 따라 처리하는 것이 좋습니다.

일반적으로 템플릿은 이미지 또는 비디오 데이터에 직접 접근하지 않아야 합니다. 이는 일반적으로 템플렌트 렌더링이 완료된 후 프로세서에서 처리됩니다. 대신, 템플릿은 이미지 또는 비디오 콘텐츠를 만나면 `<|image|>` 또는 `<|video|>`와 같은 단일 특수 토큰을 출력해야 합니다. 프로세서는 나중에 이 단일 특수 토큰을 이미지 또는 비디오 토큰의 시퀀스로 확장합니다. 출력할 정확한 토큰은 작업 중인 모델에 따라 달라집니다. 기존 멀티모달 프로세서를 로드하여 데이터를 어떻게 처리하는지 확인하는 것을 강력히 권장합니다.

아래 예제 템플릿은 이미지와 텍스트가 혼합된 콘텐츠를 처리합니다.

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

이 멀티모달 템플릿은 위의 보다 단순한 템플릿과 매우 유사하지만, `content` 목록을 확인하고, 필요한 위치에 `<|image|>` 토큰을 렌더링하도록 반복 처리합니다. 이를 통해 이미지를 사용자 텍스트의 "흐름" 안에 삽입할 수 있습니다.

모든 모델이 이러한 방식으로 작동하지는 않습니다. 예를 들어, 일부 모델은 모든 이미지를 사용자 메시지의 끝으로 이동시킬 수 있습니다. 채팅 템플릿은 항상 해당 모델이 학습된 형식과 일치해야 합니다.

### 공백 문자 제거[[trimming-whitespace]]

Jinja는 텍스트 블록 앞뒤의 모든 공백을 출력합니다. 이는 채팅 템플릿에서 문제가 될 수 있으며, 모델 학습 시 존재하지 않았던 추가 공백을 삽입하면 성능에 해를 끼칠 수 있습니다. 공백을 제거하려면 Jinja 줄 구문에서 `-`를 추가합니다. 이를 통해 Python 스타일의 들여쓰기와 줄바꿈으로 템플릿을 작성할 수 있으며, 렌더링된 출력에 들여쓰기가 의도치 않게 출력되는 것을 방지할 수 있습니다.

아래 예제 템플릿은 `-`을 사용하지 않아 출력에 추가적인 공백이 인쇄됩니다.

```jinja
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

의도된 내용만 출력되도록 보장하기 위해 `-` 사용을 강력히 권장합니다.

```jinja
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

### 특별 변수와 콜러블[[special-variables-and-callables]]

템플릿에서 유일한 상수는 `messages` 변수와 `add_generation_prompt` 불리언 값입니다. 그러나 **전달되는 모든 다른 키워드 인자**에 [`~PreTrainedTokenizerBase.apply_chat_template`] 메서드를 통해 접근할 수 있습니다.

이것은 유연성을 제공하며, 사양을 설계할 당시에는 예상하지 못했을 사용 사례에 대한 지원을 가능하게 합니다. 가장 일반적인 추가 변수는 `tools`이며, JSON 스키마 형식의 도구 목록을 포함합니다. 원하는 변수 이름을 사용할 수 있지만, 이 용도로 `tools`를 사용하는 것이 관례를 따르는 것이므로 강력히 권장합니다. 이렇게 하면 템플릿이 표준 API와 더 호환성이 높아집니다.

`tokenizer.special_tokens_map`에 포함된 모든 토큰에도 접근할 수 있으며, 여기에는 `bos_token`과 `eos_token`와 같은 특수 토큰이 자주 포함됩니다. 이러한 토큰은 `{{- bos_token }}`과 같이 이름으로 직접 접근할 수 있습니다.

사용할 수 있는 콜러블 함수가 두 개 있습니다. 호출하려면 `{{- function_name(argument) }}`을(를) 사용합니다.

- `raise_exception(msg)`은 `TemplateException`을 발생시킵니다. 이는 디버깅이나 잘못된 템플릿 사용에 대해 사용자에게 경고하는 데 유용합니다.
- `strftime_now(format_str)`은 특정 형식으로 현재 날짜와 시간을 가져오며, 이는 시스템 메시지에서 자주 필요합니다. 이는 Python의 [datetime.now().strftime(format_str)](https://docs.python.org/3/library/datetime.html#datetime.datetime.now)와 동일합니다.

### Python이 아닌 Jinja와의 호환성[[compatibility-with-non-python-jinja]]

Jinja는 여러 언어로 구현되어 있으며 일반적으로 동일한 구문을 사용합니다. Python에서 템플릿을 작성하면 문자열에 [lower](https://docs.python.org/3/library/stdtypes.html#str.lower)나 딕셔너리에 [items](https://docs.python.org/3/library/stdtypes.html#dict.items)와 같은 Python 메서드를 사용할 수 있습니다. 그러나 템플릿이 Python이 아닌 구현체, 예를 들어 Javascript 또는 Rust로 배포되는 경우에서는 작동하지 않습니다.

모든 Jinja 구현체 간의 호환성을 보장하기 위해 아래 변경 사항을 적용하십시오.

- Python 메서드를 Jinja 필터로 대체합니다. 예를 들어, `string.lower()`은 `string|lower`로, `dict.items()`은 `dict|dictitems`로 대체합니다. 대부분의 변경 사항은 동일한 패턴을 따르며, `string.strip()`만 `string|trim`으로 대체됩니다. 필터의 전체 목록은 [내장 필터](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters) 목록을 참조하십시오.
- `True`, `False`, `None`(이것들은 Python에 특화된 것들)을 각각 `true`, `false`, `none`으로 대체합니다.
- dict 또는 list를 직접 렌더링하면 다른 구현체에서 다른 결과가 반환될 수 있습니다. 예를 들어, 문자열 항목이 작은따옴표에서 큰따옴표로 변경될 수 있습니다. 이를 방지하려면 [tojson](https://jinja.palletsprojects.com/en/3.1.x/templates/#jinja-filters.tojson) 필터를 추가하여 일관성을 유지하십시오.

### 큰 템플릿[[big-templates]]

새로운 모델이나 [tool-calling](./chat_extras) 및 RAG와 같은 기능이 있는 모델은 100줄을 초과할 수 있는 더 큰 템플릿이 필요합니다. 더 큰 템플릿을 별도의 파일에 작성하는 것이 더 쉬울 수 있습니다. 별도 파일의 줄 번호는 템플릿 파싱 또는 실행 오류의 줄 번호와 정확히 일치하므로, 잠재적인 문제를 디버깅하는 것이 더 쉬워집니다.

템플릿을 별도의 파일에 작성하고 채팅 템플릿으로 추출합니다.

```py
open("template.jinja", "w").write(tokenizer.chat_template)
```

편집된 템플릿을 토크나이저에 다시 로드할 수도 있습니다.

```py
tokenizer.chat_template = open("template.jinja").read()
```

## 채팅 템플릿 저장 및 로드[[storing-and-loading-chat-templates]]

채팅 템플릿은 디스크에 여러 가지 다른 형식으로 저장됩니다. 최신 체크포인트는 템플릿을 독립적인 `.jinja` 파일로 저장하는 반면, 이전 체크포인트에서는 토크나이저 또는 프로세서 구성에 내장되어 있습니다.

### 저장 형식[[storage-formats]]

템플릿은 다음 형식 중 하나로 저장될 수 있습니다.

- `chat_template.jinja` (권장). 저장소 루트에 위치한 독립적인 Jinja 파일로, 단일 채팅 템플릿을 포함합니다. 이는 [`~PreTrainedTokenizer.save_pretrained`]가 기본적으로 작성하는 형식입니다. 템플릿을 별도의 파일로 저장하면 검사, 편집, 차이를 확인하기가 용이합니다. 토크나이저와 프로세서 모두 `chat_template.jinja`를 동일한 방식으로 로드합니다.
- `additional_chat_templates/<name>.jinja`. 모델이 여러 개의 이름이 지정된 템플릿을 함께 제공하는 경우(예: `default` 템플릿과 별도의 `tool_use` 템플릿)에 사용되는 독립형 Jinja 파일의 디렉터리입니다. `default` 템플릿은 여전히 저장소 루트의 `chat_template.jinja`에 위치하지만, 그 외의 모든 이름이 지정된 템플릿은 `additional_chat_templates/<name>.jinja`에 위치하며, 여기서 파일 이름의 줄기(stem)가 템플릿 이름이 됩니다.

> [!WARNING]
> 아래의 레거시 형식은 하위 호환성 로딩을 위해만 유지됩니다. 이 두 형식에 채팅 템플릿을 작성하지 마십시오.

- `tokenizer_config.json`의 `chat_template` 필드입니다. 독립적인 `.jinja` 파일이 사용되기 이전의 읽기 전용 레거시 형식입니다. 템플릿은 `tokenizer_config.json` 내에 JSON 문자열로 내장됩니다. 모델에 여러 개의 이름이 지정된 템플릿이 있는 경우, 해당 필드는 단일 문자열 대신 `{"name": ..., "template": ...}` 딕셔너리의 목록입니다. 이 형식을 사용하는 기존 저장소는 계속 로드되지만, [`~PreTrainedTokenizer.save_pretrained`]는 대신 현대적인 `.jinja` 형식을 작성합니다.

- `chat_template.json`. 구형 멀티모달 프로세서 체크포인트에서 사용되던 읽기 전용 레거시 형식입니다. `{"chat_template": "<template string>"}` 형태의 JSON 파일입니다. 이 형식을 사용하는 기존 저장소는 계속 로드될 수 있지만, [`~ProcessorMixin.save_pretrained`]는 대신 현대적인 `.jinja` 형식으로 저장합니다. 레거시 `chat_template.json` 파일과 현대적인 `.jinja` 파일이 혼합된 프로세서 저장소는 로드 시 오류가 발생합니다.

### 로딩 우선순위[[loading-precedence]]

[`~PreTrainedTokenizer.from_pretrained`]를 호출할 때, Transformers는 고정된 우선순위에 따라 저장 형식을 해석합니다. 독립적인 `.jinja` 파일은 설정에 내장된 템플릿보다 우선합니다. 로더는 다음과 같습니다:

1. `tokenizer_config.json`(또는 프로세서의 경우, 레거시 `chat_template.json`)에 존재하는 `chat_template` 필드를 읽습니다.
2. 리포지토리 루트에 `chat_template.jinja` 파일이 존재하는 경우 이를 읽고 `default` 템플릿으로 사용하여 1단계의 설정을 덮어씁니다.
3. `.jinja` 파일의 모든 파일을 `additional_chat_templates/`에서 파일명 줄기 기준으로 읽고 병합합니다.

결과에 `default` 템플릿이 하나만 있는 경우, [`~PreTrainedTokenizer.chat_template`]은 해당 문자열로 설정됩니다. 여러 개의 이름이 지정된 템플릿이 존재하는 경우, `chat_template`은 `{name: template_string}`의 딕셔너리가 됩니다. 이 경우, [`~PreTrainedTokenizer.apply_chat_template`]은 도구가 전달되면 `tool_use` 항목을 선택하고, 그렇지 않으면 `default`를 선택합니다.

### 저장[[saving]]

[`~PreTrainedTokenizer.save_pretrained`] 및 [`~PreTrainedTokenizer.push_to_hub`]은 기본적으로 `.jinja` 형식을 작성합니다. 단일 문자열 템플릿은 `chat_template.jinja`로 저장됩니다. 이름이 지정된 템플릿의 딕셔너리인 경우, `default` 항목은 `chat_template.jinja`에, 나머지 각 항목은 `additional_chat_templates/` 아래에 각각 하나의 파일로 저장됩니다. 중복을 방지하기 위해 `chat_template` 필드는 `tokenizer_config.json`에서 제거됩니다.

레거시 형식으로 템플릿을 저장할 수 있는 지원되는 방법이 없습니다. 이러한 형식은 이전 저장소를 로드하기 위해만 유지됩니다.

### 기존 저장소 업데이트[[updating-an-older-repository]]

`tokenizer_config.json` 또는 `chat_template.json`에 내장된 템플릿을 권장되는 `.jinja` 형식으로 마이그레이션하려면, 해당 템플릿을 다시 로드하고 저장하면 됩니다.

로드 단계는 저장소가 사용하는 레거시 형식이 `chat_template`에서 어떤 것이든 정규화하며, [`~PushToHubMixin.push_to_hub`]는 `chat_template.jinja` 파일을 반환합니다.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-org/your-model")
tokenizer.push_to_hub("your-org/your-model")
```

## 도구 템플릿[[templates-for-tools]]

도구용 템플릿 작성에 대한 특정 형식은 없지만, 표준 API를 따르는 것이 가장 좋습니다. 이를 통해 사용자가 모델과 함께 도구를 사용할 때 사용자 정의 코드를 작성할 필요가 없이, 다양한 모델에서 템플릿을 널리 활용할 수 있도록 보장합니다.

> [!WARNING]
> 공백과 특수 토큰과 같은 포맷은 모델별로 다릅니다. 모델이 학습된 포맷과 모든 것이 정확히 일치하는지 확인하십시오.

다음 섹션에서는 도구의 템플릿을 작성하기 위한 표준 API의 요소를 나열합니다.

### 도구 정의[[tool-definitions]]

[도구](./chat_extras)는 Python 함수 또는 JSON 스키마로 전달됩니다. 함수가 전달되면 JSON 스키마가 자동으로 생성되어 템플릿에 전달됩니다. 템플릿이 `tools` 변수에 액세스할 때, 해당 변수는 항상 JSON 스키마의 목록입니다.

템플릿은 항상 도구를 JSON 스키마로 수신하지만, 모델이 학습된 형식에 맞게 렌더링할 때 이 형식을 근본적으로 변경해야 할 수 있습니다. 예를 들어, [Command-R](./model_doc/cohere)는 Python 함수 헤더로 정의된 도구로 학습되었습니다. 템플릿은 내부적으로 JSON 스키마 유형을 변환하고, 입력 도구를 Python 헤더로 렌더링합니다.

아래 예시는 도구가 JSON 스키마 형식으로 정의되는 방법을 보여 줍니다.

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

채팅 템플릿에서 도구 정의를 처리하는 예시는 아래에 나와 있습니다. 구체적인 토큰과 레이아웃은 모델이 학습된 것과 일치하도록 변경해야 합니다.

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

### 도구 호출[[tool-calls]]

도구 정의 렌더링 외에도 템플릿에서 **도구 호출**과 **도구 응답**도 렌더링해야 합니다.

도구 호출은 일반적으로 `"assistant”` 메시지의 `tool_calls` 키를 통해 전달됩니다. 대부분의 도구 호출 모델은 단일 도구 호출만 지원하지만, 이 값은 항상 목록입니다. 따라서 이 목록에는 일반적으로 단일 요소만 포함됩니다.

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

도구 호출을 처리하는 일반적인 패턴은 아래에 나와 있습니다. 이를 시작점으로 사용할 수 있지만, 템플릿이 모델이 학습된 형식과 실제로 일치하는지 확인하십시오.!

```jinja
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
    {%- endfor %}
{%- endif %}
```

### 도구 응답[[tool-responses]]

도구 응답은 `tool` 역할을 가진 메시지 딕셔너리입니다. 도구 호출보다 훨씬 단순하며, 일반적으로 `role`, `name`, `content` 키만 포함합니다.

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```

일부 템플릿은 `name` 키가 전혀 필요하지 않을 수 있으며, 이 경우 템플릿을 `content` 키만 읽도록 작성할 수 있습니다.

```jinja
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

## 기여[[contribute]]

템플릿이 준비되면 토크나이저의 `chat_template` 속성에 설정하고 [`~PreTrainedTokenizerBase.apply_chat_template`]으로 테스트합니다. 예상대로 작동하면 [`~PreTrainedTokenizer.push_to_hub`]를 사용하여 Hub에 업로드합니다.

모델 소유자가 아니더라도, 빈 채팅 템플릿이나 잘못된 채팅 템플릿을 가진 모델에 템플릿을 추가하는 것은 여전히 유용합니다. 모델 저장소에서 템플릿을 추가하기 위해 [풀 리퀘스트](https://hf.co/docs/hub/repositories-pull-requests-discussions)를 열어 주세요.!

```py
tokenizer.chat_template = template
tokenizer.push_to_hub("amazing_company/cool_model", commit_message="Add chat template", create_pr=True)
```
