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

채팅 템플릿은 토크나이저의 [`~PreTrainedTokenizer.chat_template`] 속성에 저장된 [Jinja](https://jinja.palletsprojects.com/en/stable/templates/) 템플릿입니다. Jinja는 Python과 유사한 코드와 구문을 작성할 수 있게 해주는 템플릿 언어입니다.

```jinja
{%- for message in messages %}
    {{- '<|' + message['role'] + '|>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

이것을 잠시 동안 응시하다 보면, 비록 약간의 이상한 `{%-` 구문이 있기는 하지만, 이것이 실제로는 Python과 매우 유사하다는 것을 깨닫게 될 것입니다. 이 템플릿은 메시지 리스트를 반복하며, 각 메시지에 대해 메시지의 역할과 내용을 출력한 뒤 시퀀스 종료 토큰을 출력합니다. 만약 `add_generation_prompt=True`라면, 대화 끝에 어시스턴트 메시지를 위한 시작 헤더를 추가합니다.

작성된 템플릿을 문자열로 로드하여 토크나이저의 `chat_template` 속성에 할당합니다. 설정이 완료되면, [`~PreTrainedTokenizerBase.apply_chat_template`]을 호출할 때마다 해당 템플릿이 사용됩니다. 또한 [`~PreTrainedTokenizer.save_pretrained`] 또는 [`~PreTrainedTokenizer.push_to_hub`]가 호출될 때마다 토크나이저와 함께 저장됩니다. 템플릿은 토크나이저 디렉터리의 `chat_template.jinja` 파일에 저장됩니다. 템플릿 문자열을 조작하는 것보다 이 파일을 직접 편집하여 템플릿을 변경하는 것이 더 쉬운 경우가 많습니다. Transformers가 지원하는 다른 디스크 저장 형태에 대해서는 아래의 [Storing and loading chat templates](#storing-and-loading-chat-templates)를 참조하십시오.

## 템플릿 작성 팁[[template-writing-tips]]

Jinja 템플릿 작성을 시작하는 가장 쉬운 방법은 기존 템플릿을 참조하는 것입니다. 어떤 채팅 모델이든 `print(tokenizer.chat_template)`를 사용하여 해당 모델이 사용 중인 템플릿을 확인하십시오. 도구 사용(tool-use) 모델은 템플릿이 매우 복잡할 수 있으므로, 도구를 호출하거나 RAG를 지원하지 않는 간단한 모델부터 시작해 보시기 바랍니다. 마지막으로, 서식 및 구문에 대한 자세한 내용은 [Jinja documentation](https://jinja.palletsprojects.com/en/stable/templates/#synopsis)을 참조하십시오.

다만 채팅 템플릿을 작성할 때 특별히 마주할 수 있는 몇 가지 구체적인 팁과 주의 사항이 있으며, 이 섹션에서는 그중 일부를 더 자세히 다룹니다.

### 멀티모달 채팅 템플릿 작성하기[[writing-multimodal-chat-templates]]

멀티모달 템플릿의 경우, `chat_template` 속성은 토크나이저가 아닌 **processor**에 설정됩니다. 메시지의 `content` 키는 단순히 단일 문자열이 아니라 콘텐츠 딕셔너리들의 리스트인 경우가 많습니다. 리스트 내 각 콘텐츠 항목의 타입을 확인하고 그에 따라 적절히 처리하는 것이 좋습니다.

일반적으로 템플릿은 이미지 또는 비디오 데이터에 직접 접근해서는 안 됩니다. 이는 보통 템플릿 렌더링이 완료된 후 프로세서에 의해 처리됩니다. 대신, 템플릿은 이미지 또는 비디오 콘텐츠를 만났을 때 `<|image|>` 또는 `<|video|>`와 같은 단일 특수 토큰을 방출해야 합니다. 프로세서는 나중에 이 단일 특수 토큰을 이미지 또는 비디오 토큰의 시퀀스로 확장합니다. 방출할 정확한 토큰은 작업 중인 모델에 따라 달라집니다. 데이터 처리 방식을 확인하기 위해 기존의 멀티모달 프로세서를 로드하는 것을 강력히 권장합니다.

아래의 예시 템플릿은 이미지와 텍스트가 혼합된 콘텐츠를 처리합니다.

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

이 멀티모달 템플릿은 위의 더 단순한 템플릿과 매우 유사하지만, `content` 리스트를 확인하고 이를 반복하여 필요한 위치에 `<|image|>` 토큰을 렌더링합니다. 이를 통해 사용자 텍스트의 "흐름 속으로" 이미지를 삽입할 수 있습니다.

모든 모델이 이와 같은 방식으로 작동하는 것은 아닙니다. 예를 들어, 어떤 모델은 모든 이미지를 사용자 메시지의 끝으로 이동할 수도 있습니다. 채팅 템플릿은 항상 모델이 학습된 형식과 일치해야 합니다.

### 공백 제거[[trimming-whitespace]]

Jinja는 텍스트 블록의 앞이나 뒤에 있는 모든 공백을 출력합니다. 이는 모델 학습 중에 존재하지 않았던 추가 공백이 성능을 저하시킬 수 있기 때문에 채팅 템플릿에서 문제가 될 수 있습니다. 공백을 제거하려면 Jinja 라인 구문에 `-`를 추가하십시오. 이를 통해 렌더링된 출력물에 실수로 들여쓰기가 출력되지 않도록 하면서, 파이썬 스타일의 들여쓰기와 줄바꿈을 사용하여 템플릿을 작성할 수 있습니다.

아래의 예시 템플릿은 `-`를 사용하지 않아, 출력 시 추가적인 공백이 출력됩니다.

```jinja
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

의도한 콘텐츠만 출력되도록 `-`를 사용할 것을 강력히 권장합니다.

```jinja
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

### 특수 변수 및 호출 가능한 객체[[special-variables-and-callables]]

템플릿에서 유일하게 변하지 않는 것은 `messages` 변수와 `add_generation_prompt` 불리언 값입니다. 하지만 [`~PreTrainedTokenizerBase.apply_chat_template`] 메서드에 **전달되는 다른 모든 키워드 인자**를 사용할 수 있습니다.

이는 유연성을 제공하며 사양을 설계할 때 미처 생각하지 못했을 수도 있는 사용 사례에 대한 지원을 가능하게 합니다. 가장 일반적인 추가 변수는 `tools`이며, 여기에는 JSON 스키마 형식의 도구 목록이 포함됩니다. 원하는 어떤 변수 이름을 사용할 수도 있지만, 관례를 따라 이 용도로 `tools`를 사용하는 것을 강력히 권장합니다. 이렇게 하면 템플릿이 표준 API와 더 높은 호환성을 갖게 됩니다.

또한 `tokenizer.special_tokens_map`에 포함된 모든 토큰에 접근할 수 있으며, 여기에는 종종 `bos_token` 및 `eos_token`와 같은 특수 토큰이 포함됩니다. `{{- bos_token }}`와 같이 이름을 통해 이들에 직접 접근하십시오.

사용할 수 있는 두 개의 호출 가능한 함수가 있습니다. 이 함수들을 호출하려면 `{{- function_name(argument) }}`를 사용하십시오.

- `raise_exception(msg)`는 `TemplateException`을(를) 발생시킵니다. 이는 디버깅을 하거나 사용자에게 잘못된 템플릿 사용에 대해 경고를 주는 데 유용합니다.
- `strftime_now(format_str)`는 시스템 메시지에서 자주 요구되는 특정 형식의 현재 날짜와 시간을 가져옵니다. 이는 Python의 [datetime.now().strftime(format_str)](https://docs.python.org/3/library/datetime.html#datetime.datetime.now)과 동일합니다.

### 비-Python Jinja와의 호환성[[compatibility-with-non-python-jinja]]

Jinja는 여러 언어로 구현되어 있으며 일반적으로 동일한 구문을 가집니다. Python에서 템플릿을 작성하면 문자열에 대한 [lower](https://docs.python.org/3/library/stdtypes.html#str.lower) 또는 딕셔너리에 대한 [items](https://docs.python.org/3/library/stdtypes.html#dict.items)와 같은 Python 메서드를 사용할 수 있습니다. 하지만 템플릿이 비-Python 구현체에서 사용되는 경우, 예를 들어 Javascript나 Rust로 배포하는 경우에는 이것이 작동하지 않습니다.

모든 Jinja 구현체 간의 호환성을 보장하기 위해 아래의 변경 사항을 적용하십시오.

- Python 메서드를 Jinja 필터로 교체하십시오. 예를 들어, `string.lower()`를 `string|lower`로, 또는 `dict.items()`를 `dict|dictitems`로 교체합니다. 대부분의 변경 사항은 동일한 패턴을 따르지만, `string.strip()`는 `string|trim`로 교체된다는 점에 유의하십시오. 필터의 전체 목록은 [내장 필터](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters) 목록을 참조하십시오.
- `True`, `False`, 그리고 `None`(이것들은 Python 전용입니다)를 각각 `true`, `false`, 그리고 `none`으로 교체합니다.
- dict 또는 list를 직접 렌더링하면 다른 구현체에서 다른 결과가 반환될 수 있습니다. 예를 들어, 문자열 항목이 작은따옴표에서 큰따옴표로 변경될 수 있습니다. 이를 방지하려면 [tojson](https://jinja.palletsprojects.com/en/3.1.x/templates/#jinja-filters.tojson) 필터를 추가하여 일관성을 유지하십시오.

### 대형 템플릿[[big-templates]]

[tool-calling](./chat_extras) 및 RAG와 같은 기능을 갖춘 최신 모델은 100줄보다 길어질 수 있는 더 큰 템플릿을 필요로 합니다. 더 큰 템플릿은 별도의 파일에 작성하는 것이 더 쉬울 수 있습니다. 별도 파일의 줄 번호는 템플릿 파싱 또는 실행 오류의 줄 번호와 정확히 일치하므로 잠재적인 문제를 디버깅하기가 더 쉽습니다.

템플릿을 별도의 파일에 작성하고 이를 채팅 템플릿으로 추출하십시오.

```py
open("template.jinja", "w").write(tokenizer.chat_template)
```

수정된 템플릿을 토크나이저에 다시 로드할 수도 있습니다.

```py
tokenizer.chat_template = open("template.jinja").read()
```

## 채팅 템플릿 저장 및 로드[[storing-and-loading-chat-templates]]

채팅 템플릿은 디스크에 여러 가지 다른 형식으로 저장됩니다. 최신 체크포인트는 템플릿을 독립적인 `.jinja` 파일로 저장하는 반면, 이전 체크포인트는 이를 토크나이저 또는 프로세서 설정에 포함합니다.

### 저장 형식[[storage-formats]]

템플릿은 다음 중 어느 형식으로도 저장할 수 있습니다.

- `chat_template.jinja`(을/를) 권장합니다. 저장소의 루트에 위치하며 단일 채팅 템플릿을 포함하는 독립적인 Jinja 파일입니다. 이것은 [`~PreTrainedTokenizer.save_pretrained`]가 기본적으로 작성하는 파일입니다. 템플릿을 별도의 파일로 저장하면 검사, 편집 및 차이점 확인(diff)이 용이합니다. 토크나이저와 프로세서 모두 `chat_template.jinja`를 동일한 방식으로 로드합니다.
- `additional_chat_templates/<name>.jinja`. 모델이 여러 개의 명명된 템플릿(예를 들어, `default` 템플릿과 별도의 `tool_use` 템플릿)을 함께 제공하는 경우에 사용되는 독립적인 Jinja 파일들의 디렉토리입니다. `default` 템플릿은 여전히 저장소 루트의 `chat_template.jinja`에 위치하지만, 그 외의 모든 명명된 템플릿은 파일 이름의 본체가 템플릿 이름이 되는 `additional_chat_templates/<name>.jinja`에 위치합니다.

> [!WARNING]
> 아래의 레거시 형식은 하위 호환성을 위한 로딩 목적으로만 유지됩니다. 두 형식 중 어느 곳에도 채팅 템플릿을 작성하지 마십시오.

- `tokenizer_config.json`의 `chat_template` 필드입니다. 독립적인 `.jinja` 파일이 사용되기 전에 사용되었던 로드 전용 레거시 형식입니다. 템플릿은 `tokenizer_config.json` 내에 JSON 문자열로 임베드됩니다. 모델이 여러 개의 이름이 지정된 템플릿을 가진 경우, 이 필드는 단일 문자열 대신 `{"name": ..., "template": ...}` 딕셔너리들의 리스트가 됩니다. 이 형식을 사용하는 기존 저장소는 계속해서 로드되지만, [`~PreTrainedTokenizer.save_pretrained`]는 대신 현대적인 `.jinja` 형식으로 저장합니다.

- `chat_template.json`. 이전의 멀티모달 프로세서 체크포인트에서 사용되는 로드 전용 레거시 형식입니다. `{"chat_template": "<template string>"}` 형태의 JSON 파일입니다. 이 형식을 사용하는 기존 저장소는 계속해서 로드되지만, [`~ProcessorMixin.save_pretrained`]는 대신 현대적인 `.jinja` 형식을 작성합니다. 레거시 `chat_template.json`와 현대적인 `.jinja` 파일을 혼용하는 프로세서 저장소는 로드 시 오류가 발생합니다.

### 로드 우선순위[[loading-precedence]]

[`~PreTrainedTokenizer.from_pretrained`]를 호출할 때, Transformers는 고정된 우선순위에 따라 저장 형식을 결정합니다. 독립된 `.jinja` 파일이 설정에 포함된 템플릿보다 우선순위를 갖습니다. 로더는 다음과 같이 동작합니다:

1. `tokenizer_config.json`(또는 프로세서의 경우 레거시 `chat_template.json`)에 존재하는 모든 `chat_template` 필드를 읽습니다.
2. 저장소 루트에 `chat_template.jinja` 파일이 존재하면 이를 읽어 들여 1단계를 무시하고 `default` 템플릿으로 사용합니다.
3. `additional_chat_templates/`에 있는 모든 `.jinja` 파일을 파일 이름의 stem을 키로 하여 읽어 들인 후, 이를 병합합니다.

결과에 단일 `default` 템플릿이 있는 경우, [`~PreTrainedTokenizer.chat_template`]은 해당 문자열로 설정됩니다. 여러 개의 이름이 지정된 템플릿이 존재하는 경우, `chat_template`는 `{name: template_string}` 형태의 딕셔너리가 됩니다. 이 경우, [`~PreTrainedTokenizer.apply_chat_template`]는 도구가 전달되면 `tool_use` 항목을 선택하고, 그렇지 않으면 `default`를 선택합니다.

### 저장하기[[saving]]

[`~PreTrainedTokenizer.save_pretrained`] 및 [`~PreTrainedTokenizer.push_to_hub`]는 기본적으로 `.jinja` 형식으로 저장합니다. 단일 문자열 템플릿은 `chat_template.jinja`가 됩니다. 이름이 지정된 템플릿들의 딕셔너리는 `default` 항목을 `chat_template.jinja`에 기록하고, 나머지 각 항목은 `additional_chat_templates/` 아래에 개별 파일로 기록합니다. 중복을 방지하기 위해 `tokenizer_config.json`에서 `chat_template` 필드는 제거됩니다.

레거시 형식 중 하나로 템플릿을 저장하는 지원되는 방법은 없습니다. 해당 형식들은 이전 저장소를 불러오기 위한 용도로만 유지됩니다.

### 이전 저장소 업데이트하기[[updating-an-older-repository]]

`tokenizer_config.json` 또는 `chat_template.json`에 포함된 임베디드 템플릿을 다시 로드하고 저장함으로써 권장되는 `.jinja` 형식으로 마이그레이션합니다.

로드 단계는 저장소가 `chat_template`에서 사용하는 기존 형식이 무엇이든 이를 정규화하며, [`~PushToHubMixin.push_to_hub`]은 `chat_template.jinja` 파일을 반환합니다.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-org/your-model")
tokenizer.push_to_hub("your-org/your-model")
```

## 도구용 템플릿[[templates-for-tools]]

도구용 템플릿을 작성하는 데 있어 특정된 형식은 없으나, 표준 API를 따르는 것이 가장 좋습니다. 이를 통해 사용자가 귀하의 모델과 도구를 사용하기 위해 별도의 커스텀 코드를 작성할 필요 없이, 다양한 모델에서 템플릿을 폭넓게 사용할 수 있도록 보장할 수 있습니다.

> [!WARNING]
> 공백 및 특수 토큰과 같은 서식은 모델마다 다릅니다. 모든 내용이 모델이 학습된 서식과 정확히 일치하는지 확인하십시오.

다음 섹션은 도구용 템플릿 작성을 위한 표준 API의 요소들을 나열합니다.

### 도구 정의[[tool-definitions]]

[Tools](./chat_extras)는 Python 함수 또는 JSON 스키마로 전달됩니다. 함수가 전달되면 JSON 스키마가 자동으로 생성되어 템플릿에 전달됩니다. 템플릿이 `tools` 변수에 접근할 때, 이는 항상 JSON 스키마의 리스트입니다.

템플릿은 항상 JSON 스키마로 도구를 받지만, 모델이 학습된 형식에 맞추기 위해 렌더링할 때 이 형식을 근본적으로 변경해야 할 수도 있습니다. 예를 들어, [Command-R](./model_doc/cohere)은 Python 함수 헤더로 정의된 도구로 학습되었습니다. 템플릿은 내부적으로 JSON 스키마 유형을 변환하여 입력된 도구를 Python 헤더로 렌더링합니다.

아래 예시는 도구가 JSON 스키마 형식으로 어떻게 정의되는지 보여줍니다.

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

채팅 템플릿에서 도구 정의를 처리하는 예시가 아래에 나와 있습니다. 특정 토큰과 레이아웃은 모델이 학습할 때 사용한 것과 일치하도록 변경해야 합니다.

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

도구 정의를 렌더링하는 것 외에도, 템플릿에 **tool calls** 및 **tool responses**를 렌더링해야 합니다.

도구 호출은 일반적으로 `"assistant”` 메시지의 `tool_calls` 키를 통해 전달됩니다. 대부분의 도구 호출 모델은 단일 도구 호출만 지원하므로 이 리스트에는 보통 하나의 요소만 포함되지만, 이 값은 항상 리스트 형태입니다.

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

도구 호출을 처리하는 일반적인 패턴이 아래에 표시되어 있습니다. 이를 시작점으로 사용할 수 있지만, 사용 중인 템플릿이 모델이 학습된 형식과 실제로 일치하는지 확인해야 합니다.!

```jinja
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
    {%- endfor %}
{%- endif %}
```

### 도구 응답[[tool-responses]]

도구 응답은 `tool` 역할을 가진 메시지 딕셔너리입니다. 이는 도구 호출보다 훨씬 단순하며, 대개 `role`, `name`, `content` 키만을 포함합니다.

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```

일부 템플릿은 `name` 키가 필요하지 않을 수도 있으며, 이 경우 `content` 키만 읽도록 템플릿을 작성할 수 있습니다.

```jinja
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

## 기여하기[[contribute]]

템플릿이 준비되면 토크나이저의 `chat_template` 속성에 설정하고 [`~PreTrainedTokenizerBase.apply_chat_template`]를 사용하여 테스트합니다. 예상대로 작동한다면 [`~PreTrainedTokenizer.push_to_hub`]를 사용하여 Hub에 업로드합니다.

모델 소유자가 아니더라도, 채팅 템플릿이 비어 있거나 잘못된 모델에 템플릿을 추가하는 것은 여전히 도움이 됩니다. 템플릿을 추가하려면 모델 저장소에 [pull request](https://hf.co/docs/hub/repositories-pull-requests-discussions)를 생성하십시오.!

```py
tokenizer.chat_template = template
tokenizer.push_to_hub("amazing_company/cool_model", commit_message="Add chat template", create_pr=True)
```
