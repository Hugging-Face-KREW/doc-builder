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

챗 템플릿은 토크나이저의 [`~PreTrainedTokenizer.chat_template`] 속성에 저장된 [Jinja](https://jinja.palletsprojects.com/en/stable/templates/) 템플릿입니다. Jinja는 Python-와 유사한 코드와 문법을 사용하여 작성할 수 있는 템플릿 언어입니다.

```jinja
{%- for message in messages %}
    {{- '<|' + message['role'] + '|>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

이 코드를 잠시 살펴보면, 이것이 사실상 Python와 매우 비슷하다는 것을 깨닫게 될 것입니다. 다만 약간 이상한 `{%-` 구문만 다를 뿐입니다. 템플릿은 메시지 목록을 반복하며, 각 메시지에 대해 역할(role)과 내용(content)을 출력한 뒤 종료 토큰(end-of-sequence token)을 추가합니다. `add_generation_prompt=True`인 경우, 대화의 끝에 어시스턴트 메시지의 시작 헤더를 추가합니다.

작성된 템플릿을 문자열로 로드하여 토크나이저의 `chat_template` 속성에 할당합니다. 설정한 후에는 [`~PreTrainedTokenizerBase.apply_chat_template`]을 호출할 때마다 템플릿이 사용됩니다. 또한 템플릿은 토크나이저와 함께 [`~PreTrainedTokenizer.save_pretrained`] 또는 [`~PreTrainedTokenizer.push_to_hub`]이 호출될 때마다 저장됩니다. 템플릿은 `chat_template.jinja` 파일에 토크나이저 디렉터리에 저장됩니다. 템플릿을 직접 이 파일을 편집하여 변경할 수 있는데, 이는 종종 템플릿 문자열을 조작하는 것보다 더 쉽습니다. [Storing and loading chat templates](#storing-and-loading-chat-templates) 아래를 참조하십시오. Transformers가 지원하는 다른 온디스크 형태가 있습니다.

## 템플릿 작성 팁[[template-writing-tips]]

jinja 템플릿을 작성하는 가장 쉬운 방법은 기존 템플릿을 참고하는 것입니다. `print(tokenizer.chat_template)`를 사용하여 어떤 채팅 모델이 사용하고 있는 템플릿을 확인해 보세요. 도구 사용 모델은 매우 복잡한 템플릿을 가질 수 있으므로, 도구를 호출하지 않거나 RAG를 지원하지 않는 간단한 모델부터 시작해 보세요. 마지막으로 [Jinja 문서](https://jinja.palletsprojects.com/en/stable/templates/#synopsis)를 살펴보아 서식과 문법에 대한 자세한 내용을 확인해 보세요.

챗 템플릿을 작성하는 동안 특정 팁과 함정을 마주할 수 있으며, 이 섹션에서는 그중 일부를 더 자세하게 다룹니다.

### 멀티모달 채팅 템플릿 작성[[writing-multimodal-chat-templates]]

멀티모달 템플릿의 경우, `chat_template` 속성이 설정되었습니다. **프로세서**, 토크나이저가 아닙니다. The `content` 메시지의 키는 종종 단일 문자열이 아니라 콘텐츠 딕셔너리 목록인 경우가 많습니다. 각 콘텐츠 항목의 타입을 확인하고 그에 따라 처리하는 것이 좋습니다.

일반적으로 템플릿은 이미지나 비디오 데이터에 직접 접근해서는 안 됩니다. 이는 템플릿 렌더링이 완료된 후 프로세서에 의해 정상적으로 처리됩니다. 대신 템플릿은 단일 특수 토큰을 방출해야 합니다. `<|image|>` 또는 `<|video|>` 이미지 또는 비디오 콘텐츠를 만나면 프로세서는 단일 특수 토큰을 이미지 또는 비디오 토큰 시퀀스로 확장합니다. 정확히 어떤 토큰을 방출할지는 사용 중인 모델에 따라 달라집니다. 데이터가 어떻게 처리되는지 확인하려면 기존 멀티모달 프로세서를 로드하는 것을 강력히 권장합니다.

아래 예제 템플릿은 혼합된 이미지 및 텍스트 콘텐츠를 처리합니다.

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

이 멀티모달 템플릿은 위에 있는 더 단순한 템플릿과 매우 유사하지만, `content` 목록을 확인하고 이를 반복하여 `<|image|>` 토큰을 필요한 위치에 렌더링합니다. 이를 통해 이미지를 사용자 텍스트 흐름 안으로 삽입할 수 있습니다.

모든 모델이 이 방식으로 작동하는 것은 아닙니다. 예를 들어, 일부 모델은 모든 이미지를 사용자 메시지의 끝에 이동시킬 수 있습니다. 챗 템플릿은 항상 모델이 훈련된 형식과 일치해야 합니다.

### 공백 제거[[trimming-whitespace]]

진자(Jinja)는 텍스트 블록 앞이나 뒤에 있는 공백을 모두 출력합니다. 이는 채팅 템플릿에서 모델 학습 시 포함되지 않았던 여분의 공백이 추가되어 성능에 악영향을 줄 수 있기 때문에 문제가 될 수 있습니다. 공백을 제거하려면, `-` 자바스크립트 템플릿을 Jinja 라인 문법으로 변환하면 Pythonic한 들여쓰기와 줄바꿈을 사용하여 템플릿을 작성할 수 있으며, 렌더링된 출력에 의도하지 않은 들여쓰기가 인쇄되지 않도록 할 수 있습니다.

아래 예제 템플릿은 `-`을 사용하지 않아 출력에 여백이 추가로 인쇄됩니다.

```jinja
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

`-`를 사용하여 인쇄하려는 콘텐츠가 정확히 표시되도록 권장합니다.

```jinja
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

### 특수 변수 및 콜러블s[[special-variables-and-callables]]

템플릿의 유일한 상수는 `messages` 변수와 `add_generation_prompt` 불리언입니다. 그러나 [`~PreTrainedTokenizerBase.apply_chat_template`] 메서드에 전달된 **any other keyword arguments**에 액세스할 수 있습니다.

이 기능은 유연성을 제공하며, 스펙을 설계할 때 고려하지 못했던 사용 사례를 지원합니다. 가장 일반적인 추가 변수는 `tools`이며, 이는 JSON 스키마 형식의 도구 목록을 포함합니다. 변수 이름은 원하는 대로 사용할 수 있지만, 관례에 따라 `tools`를 이 목적에 사용하는 것이 좋습니다. 이는 템플릿을 표준 API와 더 호환되게 만듭니다.

또한 `tokenizer.special_tokens_map`에 포함된 모든 토큰에 액세스할 수 있습니다. 여기에는 `bos_token`과 `eos_token`와 같은 특수 토큰이 자주 포함됩니다. `{{- bos_token }}`과 같이 직접 이름으로 액세스할 수 있습니다.

사용 가능한 호출 가능한 함수가 두 개 있습니다. 이를 호출하려면 `{{- function_name(argument) }}`.

- `raise_exception(msg)`는 `TemplateException`을 발생시킵니다. 이는 디버깅이나 템플릿 사용법에 대한 경고에 유용합니다.
- `strftime_now(format_str)`는 시스템 메시지에서 자주 필요한 현재 날짜와 시간을 특정 형식으로 반환합니다. Python에서 [datetime.now().strftime(format_str)](https://docs.python.org/3/library/datetime.html#datetime.datetime.now)와 동등합니다.

### 비Python Jinja와의 호환성[[compatibility-with-non-python-jinja]]

Jinja는 여러 언어에 구현되어 있으며, 대체로 동일한 문법을 사용합니다. Python으로 템플릿을 작성하면 Python의 [lower](https://docs.python.org/3/library/stdtypes.html#str.lower) 메서드를 문자열에, [items](https://docs.python.org/3/library/stdtypes.html#dict.items)를 딕셔너리에 적용할 수 있습니다. 하지만 템플릿이 비-Python 구현에서 사용되는 경우 작동하지 않습니다. 예를 들어, Javascript나 Rust로 배포할 때와 같습니다.

모든 Jinja 구현에서 호환성을 보장하기 위해 아래와 같은 변경 사항을 적용합니다.

- Python 메서드를 Jinja 필터로 교체합니다. 예를 들어, `string.lower()`을 `string|lower` 또는 `dict.items()`을 `dict|dictitems`로 교체합니다. 대부분의 변경 사항은 동일한 패턴을 따르지만, `string.strip()`는 `string|trim`로 교체되는 예외입니다. 사용 가능한 필터의 전체 목록은 [built-in filters](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters)을 참조하십시오.
- `True`, `False`, 그리고 `None` (이것은 Python 전용입니다)는 각각 `true`, `false`, 그리고 `none`로 대체합니다.
- 직접 dict나 list를 렌더링하면 다른 구현체에서 결과가 다를 수 있습니다. 예를 들어, 문자열 항목이 홑따옴표에서 쌍따옴표로 바뀔 수 있습니다. 이를 방지하려면 [tojson](https://jinja.palletsprojects.com/en/3.1.x/templates/#jinja-filters.tojson) 필터를 추가하여 일관성을 유지하도록 합니다.

### 큰 템플릿[[big-templates]]

새로운 모델이나 [tool-calling](./chat_extras)와 같은 기능을 사용하는 모델은 100줄을 초과할 수 있는 더 큰 템플릿이 필요합니다. 더 큰 템플릿을 작성하는 것은 별도의 파일에 작성하는 것이 더 쉬울 수 있습니다. 별도 파일의 줄 번호는 템플릿 파싱 또는 실행 오류의 줄 번호와 정확히 일치하므로, 잠재적인 문제를 디버깅하는 데 더 편리합니다.

별도 파일에 템플릿을 작성하고 채팅 템플릿으로 추출합니다.

```py
open("template.jinja", "w").write(tokenizer.chat_template)
```

편집된 템플릿을 토크나이저에 다시 로드할 수도 있습니다.

```py
tokenizer.chat_template = open("template.jinja").read()
```

## 챗 템플릿 저장 및 로드[[storing-and-loading-chat-templates]]

챗 템플릿은 디스크에 여러 가지 형식으로 저장됩니다. 최신 체크포인트는 `.jinja` 파일을 별도로 저장하며, 구형 체크포인트는 토크나이저나 프로세서 설정 안에 포함합니다.

### 저장 형식[[storage-formats]]

템플릿은 다음 형식 중 하나에 저장될 수 있습니다.

- `chat_template.jinja` (권장). 리포지토리 루트에 있는 단일 Jinja 파일에 챗 템플릿이 포함된 경우. [`~PreTrainedTokenizer.save_pretrained`] 기본적으로 작성합니다. 템플릿을 별도의 파일에 저장하면 검사, 편집, diff가 용이합니다. 토크나이저와 프로세서 모두 로드합니다. `chat_template.jinja` 같은 방식으로.
- `additional_chat_templates/<name>.jinja`. 모델이 여러 개의 명명된 템플릿을 배포할 때 사용되는 독립형 Jinja 파일 디렉터리입니다(예: `default` 템플릿과 별도의 `tool_use` 템플릿). `default` 템플릿은 여전히 리포트 루트의 `chat_template.jinja`에 위치하지만, 그 외의 다른 명명된 템플릿은 `additional_chat_templates/<name>.jinja`에 위치하며, 파일 이름은 템플릿 이름이 됩니다.

> [!WARNING]
> 아래 레거시 형식은 호환성 로드만 위해 유지됩니다. 이 두 형식에 대해 챗 템플릿을 작성하지 마십시오.

- `chat_template` 필드 in `tokenizer_config.json`로드 전용 레거시 형식은 스탠드얼론 이전의... `.jinja` 파일입니다. 템플릿은 임베디드된 상태로 JSON 문자열 in `tokenizer_config.json`모델이 여러 개의 네이밍된 템플릿을 가질 때, 필드는 템플릿의 목록입니다. `{"name": ..., "template": ...}` 딕셔너리 대신 단일 문자열을 사용합니다. 기존 저장소는 이 형식을 계속 로드하지만, [`~PreTrainedTokenizer.save_pretrained`] 현대적인 `.jinja` 형식으로 변경하십시오.

- `chat_template.json`더 이상 사용되지 않는 멀티모달 프로세서 체크포인트에서 사용되는 로드 전용 레거시 형식입니다. A JSON 형식의 파일을 선택하십시오. `{"chat_template": "<template string>"}`기존 저장소는 이 형식을 계속 로드하지만, [`~ProcessorMixin.save_pretrained`] 현대적인 것을 씁니다. `.jinja` 형식 대신. 유산 프로세서 저장소가 혼합된 `chat_template.json` 현대적인 `.jinja` 파일을 로드하면 오류가 발생합니다.

### 로딩 우선순위[[loading-precedence]]

[`~PreTrainedTokenizer.from_pretrained`]를 호출하면, Transformers는 고정된 우선순위로 저장 형식을 해결합니다. `.jinja` 파일이 단독으로 존재할 경우, 설정 내에 내장된 템플릿보다 우선합니다. 로더는:

1. `chat_template` 필드가 `tokenizer_config.json`에 포함된 경우 이를 읽으며, 프로세서의 경우 레거시 `chat_template.json`를 읽습니다.
2. 리포지토리 루트에 `chat_template.jinja`가 존재하면 읽고, `default` 템플릿으로 사용해 1번 단계를 대체합니다.
3. 읽습니다. `.jinja` 파일 in `additional_chat_templates/`파일명 스템에 의해 키가 지정되고, 병합됩니다.

결과가 단일 `default` 템플릿을 포함하면, [`~PreTrainedTokenizer.chat_template`]은 해당 문자열로 설정됩니다. 여러 개의 이름이 지정된 템플릿이 존재하면, `chat_template`는 `{name: template_string}` 형태의 딕셔너리가 됩니다. 이 경우, [`~PreTrainedTokenizer.apply_chat_template`]는 도구가 전달되면 `tool_use` 항목을, 그렇지 않으면 `default`을 선택합니다.

### 저장합니다[[saving]]

[`~PreTrainedTokenizer.save_pretrained`]와 [`~PreTrainedTokenizer.push_to_hub`]는 기본적으로 `.jinja` 형식을 씁니다. 단일 문자열 템플릿은 `chat_template.jinja`가 됩니다. 명명된 템플릿의 사전은 `default` 항목을 `chat_template.jinja`에 쓰고, 나머지 항목은 `additional_chat_templates/` 아래에 개별 파일로 씁니다. `chat_template` 필드는 `tokenizer_config.json`에서 중복을 피하기 위해 제거됩니다.

지원되는 방식으로 레거시 형식의 템플릿을 저장할 수 없습니다. 이들은 오래된 리포지토리를 로드하기 위해 보존된 것입니다.

### 기존 저장소 업데이트[[updating-an-older-repository]]

내장 템플릿을 `tokenizer_config.json` 또는 `chat_template.json`에서 `.jinja` 형식으로 마이그레이션하려면 로드하고 다시 저장하십시오.

로드 단계는 저장소가 사용하는 어떤 레거시 형식도 `chat_template`에서 정규화합니다. [`~PushToHubMixin.push_to_hub`]는 `chat_template.jinja` 파일을 반환합니다.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-org/your-model")
tokenizer.push_to_hub("your-org/your-model")
```

## 도구 템플릿[[templates-for-tools]]

도구에 대한 템플릿을 작성하는 특정 형식은 없지만, 표준 API를 따르는 것이 가장 좋습니다. 이는 사용자가 모델과 도구를 사용하기 위해 사용자 지정 코드를 작성할 필요 없이 템플릿이 다양한 모델에서 널리 활용될 수 있도록 보장합니다.

> [!WARNING]
> 공백 및 특수 토큰은 모델에 따라 다릅니다. 모델이 훈련된 형식과 정확히 일치하도록 확인하십시오.

다음 섹션은 도구용 템플릿을 작성하는 표준 API의 요소를 나열합니다.

### 도구 정의[[tool-definitions]]

[도구](./chat_extras) 인수는 다음과 같이 전달됩니다. Python 함수 또는 a JSON 스키마입니다. 함수가 전달되면, a JSON 스키마는 자동으로 생성되어 템플릿에 전달됩니다. 템플릿이 ... `tools` 변수는 항상 리스트입니다. JSON 스키마.

템플릿은 항상 JSON 스키마로 도구를 수신하지만, 모델에 맞게 렌더링할 때 이 형식을 근본적으로 변경해야 할 수 있습니다. 예를 들어 [Command-R](./model_doc/cohere)은 Python 함수 헤더로 정의된 도구로 훈련되었습니다. 템플릿은 내부적으로 JSON 스키마 유형을 변환하고 Python 헤더로 입력을 렌더링합니다.

아래 예제는 JSON 스키마 형식에서 도구가 정의된 방법을 보여줍니다.

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

도구 정의(Tool definitions)를 처리하는 예시가 아래에 표시됩니다. 모델이 훈련된 것과 일치하도록 특정 토큰과 레이아웃을 변경해야 합니다.

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

도구 정의를 렌더링하는 것 외에도 템플릿에서 **도구 호출** 및 **도구 응답**도 렌더링해야 합니다.

도구 호출은 `tool_calls` 메시지의 `"assistant”` 키에 일반적으로 전달됩니다. 이는 대부분의 도구 호출 모델이 단일 도구 호출만 지원하지만, 목록 형태이므로 목록은 보통 단일 요소를 포함한다는 의미입니다.

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

도구 호출을 처리하는 일반적인 패턴이 아래에 표시되어 있습니다. 이 내용을 시작점으로 사용해도 되지만, 모델이 훈련된 실제 형식과 템플릿이 정확히 일치하는지 확인하시기 바랍니다.!

```jinja
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
    {%- endfor %}
{%- endif %}
```

### 도구 응답[[tool-responses]]

도구 응답은 `tool` 역할을 가진 메시지 딕셔너리입니다. 이들은 tool call보다 훨씬 간단하며, 보통 `role`, `name` 및 `content` 키만 포함합니다.

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```

일부 템플릿은 `name` 키가 필요하지 않을 수도 있습니다. 이 경우, `content` 키만 읽도록 템플릿을 작성할 수 있습니다.

```jinja
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

## 기여하기[[contribute]]

템플릿이 준비되면, 토크나이저의 `chat_template` 속성에 설정하고 [`~PreTrainedTokenizerBase.apply_chat_template`]로 테스트합니다. 예상대로 작동하면 [`~PreTrainedTokenizer.push_to_hub`]를 사용하여 Hub에 업로드합니다.

모델 소유자가 아니더라도, 템플릿이 비어 있거나 잘못된 모델에 템플릿을 추가하는 것은 여전히 유용합니다. 모델 저장소에 [풀 리퀘스트](https://hf.co/docs/hub/repositories-pull-requests-discussions)를 열어 템플릿을 추가해 주세요.!

```py
tokenizer.chat_template = template
tokenizer.push_to_hub("amazing_company/cool_model", commit_message="Add chat template", create_pr=True)
```
