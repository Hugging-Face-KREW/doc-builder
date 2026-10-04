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

채팅 템플릿은 토크나이저의 [`~PreTrainedTokenizer.chat_template`] 속성에 저장된 [Jinja](https://jinja.palletsprojects.com/en/stable/templates/) 템플릿입니다. Jinja는 Python과 유사한 코드 및 구문을 작성할 수 있게 해주는 템플릿 언어입니다.

```jinja
{%- for message in messages %}
    {{- '<|' + message['role'] + '|>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

잠시 이 내용을 살펴보면, 이것이 사실상 Python과 매우 유사하다는 것을 알 수 있습니다. 다만 약간 특이한 `{%-` 문법이 사용되었습니다. 이 템플릿은 메시지 목록을 순회하며, 각 메시지에 대해 해당 메시지의 역할과 내용을 출력한 뒤 시퀀스 종료 토큰을 추가합니다. 만약 `add_generation_prompt=True`로 설정되어 있다면, 대화의 끝에 어시스턴트 메시지의 시작 헤더가 추가됩니다.

작성된 템플릿을 문자열로 로드하여 토크나이저의 `chat_template` 속성에 할당합니다. 이렇게 설정되면 [`~PreTrainedTokenizerBase.apply_chat_template`]을 호출할 때마다 해당 템플릿이 사용됩니다. 또한 [`~PreTrainedTokenizer.save_pretrained`] 또는 [`~PreTrainedTokenizer.push_to_hub`]가 호출될 때마다 토크나이저와 함께 해당 템플릿이 저장됩니다. 템플릿은 토크나이저 디렉터리 내의 `chat_template.jinja` 파일에 저장됩니다. 템플릿을 변경하려면 이 파일을 직접 편집하는 것이 템플릿 문자열을 조작하는 것보다 더 쉬운 경우가 많습니다. [채팅 템플릿 저장 및 로드하기](#storing-and-loading-chat-templates)를 참조하여 Transformers에서 지원하는 다른 디스크 저장 형태에 대해 알아볼 수 있습니다.

## 템플릿 작성 팁입니다.[[template-writing-tips]]

Jinja 템플릿 작성을 시작하는 가장 쉬운 방법은 기존 템플릿을 참고하는 것입니다. 어떤 채팅 모델에서든 `print(tokenizer.chat_template)`를 사용하면 해당 모델이 사용하는 템플릿을 확인할 수 있습니다. 도구를 호출하거나 RAG를 지원하지 않는 간단한 모델부터 시작해 보시기 바랍니다. 왜냐하면 도구 사용 모델의 경우 매우 복잡한 템플릿을 가질 수 있기 때문입니다. 마지막으로, 서식 및 구문에 대한 보다 자세한 내용은 [Jinja 문서](https://jinja.palletsprojects.com/en/stable/templates/#synopsis)를 참조하시기 바랍니다.

특히 채팅 템플릿을 작성할 때는 마주칠 수 있는 몇 가지 특정한 팁과 주의사항이 있으며, 이 섹션에서는 그중 일부를 보다 상세히 다룰 것입니다.

### 다중 모달 채팅 템플릿 작성하기[[writing-multimodal-chat-templates]]

멀티모달 템플릿의 경우, `chat_template` 속성은 토크나이저가 아닌 **프로세서**에 설정됩니다. 메시지의 `content` 키는 단순한 문자열이 아니라 보통 콘텐츠 딕셔너리들의 목록으로 구성됩니다. 목록 내 각 콘텐츠 항목의 타입을 확인하여 그에 맞게 처리하는 것이 좋습니다.

일반적으로 템플릿은 이미지나 비디오 데이터에 직접 접근해서는 안 됩니다. 이러한 처리는 보통 템플릿 렌더링이 완료된 후에 프로세서가 담당합니다. 대신 템플릿에서는 이미지나 비디오 콘텐츠를 만날 때 `<|image|>` 또는 `<|video|>`와 같은 특수 토큰 하나만을 출력해야 합니다. 이후 프로세서는 이 특수 토큰을 이미지나 비디오 토큰들의 시퀀스로 확장합니다. 출력할 정확한 토큰은 사용 중인 모델에 따라 달라집니다. 데이터 처리 방식을 확인하기 위해 기존의 멀티모달 프로세서를 활용하는 것을 적극 권장합니다.

아래의 예제 템플릿은 이미지와 텍스트가 혼합된 콘텐츠를 처리합니다.

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

이 멀티모달 템플릿은 위에 설명된 보다 간단한 템플릿과 매우 유사하지만, `content` 목록을 확인하고 필요한 곳에 `<|image|>` 토큰을 렌더링하기 위해 이를 반복 처리합니다. 이를 통해 이미지를 사용자 텍스트의 흐름 속에 삽입할 수 있습니다.

모든 모델이 이렇게 작동하는 것은 아닙니다. 예를 들어, 일부 모델은 모든 이미지를 사용자 메시지의 끝으로 옮길 수도 있습니다. 채팅 템플릿은 항상 모델이 학습된 형식과 일치해야 합니다.

### 공백 제거하기입니다.[[trimming-whitespace]]

Jinja는 텍스트 블록의 앞뒤에 있는 모든 공백을 출력합니다. 이는 모델 학습 시에는 존재하지 않았던 추가적인 공백이 성능 저하를 초래할 수 있기 때문에 채팅 템플릿에서 문제가 될 수 있습니다. 이러한 공백을 제거하려면 Jinja 문법에 `-`를 추가하면 됩니다. 이렇게 하면 파이썬식 들여쓰기와 줄바꿈을 사용하여 템플릿을 작성할 수 있으며, 렌더링된 출력 결과에 의도치 않은 들여쓰기가 포함되는 것을 방지할 수 있습니다.

아래의 예제 템플릿은 `-`를 사용하지 않으므로 출력 결과에 불필요한 공백이 포함됩니다.

```jinja
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

의도한 내용만 출력되도록 하려면 `-`를 사용하시기를 적극 권장합니다.

```jinja
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

### 특수 변수 및 호출 가능한 객체입니다.[[special-variables-and-callables]]

템플릿 내에서 유일한 상수는 `messages` 변수와 `add_generation_prompt` 불리언 값입니다. 하지만 [`~PreTrainedTokenizerBase.apply_chat_template`] 메서드에 전달되는 **다른 모든 키워드 인수들에도 접근할 수 있습니다.**

이를 통해 유연성을 확보할 수 있으며, 사양을 설계할 때 미처 생각하지 못했던 사용 사례도 지원할 수 있습니다. 가장 일반적으로 사용되는 추가 변수는 `tools`이며, 이는 JSON 스키마 형식으로 된 도구 목록을 포함합니다. 원하는 변수명을 사용할 수 있지만, 관례를 따르고 이 용도로는 `tools`를 사용할 것을 적극 권장합니다. 이렇게 하면 템플릿이 표준 API와 더 잘 호환됩니다.

또한 `tokenizer.special_tokens_map`에 포함된 모든 토큰에도 접근할 수 있으며, 여기에는 보통 `bos_token` 및 `eos_token`와 같은 특수 토큰들이 포함됩니다. 이러한 토큰들은 이름을 통해 직접 접근할 수 있으며, 예로는 `{{- bos_token }}`와 같이 사용됩니다.

사용하실 수 있는 호출 가능한 함수는 두 가지가 있습니다. 이 함수들을 호출하려면 `{{- function_name(argument) }}`를 사용하시면 됩니다.

- `raise_exception(msg)`는 `TemplateException`을 발생시킵니다. 이는 디버깅이나 잘못된 템플릿 사용에 대해 사용자에게 경고하는 데 유용합니다.
- `strftime_now(format_str)`는 시스템 메시지에서 자주 필요로 하는 특정 형식의 현재 날짜와 시간을 반환합니다. 이는 Python에서 [datetime.now().strftime(format_str)](https://docs.python.org/3/library/datetime.html#datetime.datetime.now)과 동일한 기능을 합니다.

### 비Python 기반 Jinja와의 호환성입니다.[[compatibility-with-non-python-jinja]]

Jinja는 여러 언어로 구현되어 있으며, 일반적으로 동일한 문법을 가집니다. Python으로 템플릿을 작성하면 문자열에 대해 [lower](https://docs.python.org/3/library/stdtypes.html#str.lower)와 같은 Python 메서드나 딕셔너리에 대해 [items](https://docs.python.org/3/library/stdtypes.html#dict.items)를 사용할 수 있습니다. 하지만 이 템플릿이 비Python 환경에서 사용될 경우, 예를 들어 JavaScript나 Rust로 배포할 때에는 이러한 기능이 작동하지 않습니다.

모든 Jinja 구현체와의 호환성을 보장하기 위해 아래의 변경 사항을 적용합니다.

- Python 메서드를 Jinja 필터로 대체해야 합니다. 예를 들어, `string.lower()`는 `string|lower`로, `dict.items()`는 `dict|dictitems`로 대체할 수 있습니다. `string.strip()`를 제외한 대부분의 변경 사항은 동일한 패턴을 따르며, 이 경우에는 `string|trim`으로 대체됩니다. 사용 가능한 필터의 전체 목록은 [내장 필터](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters) 목록을 참조하시기 바랍니다.
- `True`, `False`, 그리고 `None`(이들은 Python에 특화된 값입니다)를 각각 `true`, `false`, `none`으로 교체해 주시기 바랍니다.
- 딕셔너리나 리스트를 직접 렌더링하면 다른 구현체에서는 서로 다른 결과가 나타날 수 있습니다. 예를 들어, 문자열 항목의 따옴표가 단일 따옴표에서 이중 따옴표로 변경될 수 있습니다. 이러한 문제를 방지하려면 일관성을 유지하기 위해 [tojson](https://jinja.palletsprojects.com/en/3.1.x/templates/#jinja-filters.tojson) 필터를 추가해야 합니다.

### 대형 템플릿입니다.[[big-templates]]

최신 모델이나 [tool-calling](./chat_extras) 및 RAG와 같은 기능을 갖춘 모델의 경우 100줄 이상에 달하는 더 큰 템플릿이 필요합니다. 이러한 큰 템플릿은 별도의 파일에 작성하는 것이 더 용이할 수 있습니다. 별도의 파일에 있는 줄 번호는 템플릿 파싱 또는 실행 오류에서 나타나는 줄 번호와 정확히 일치하므로 잠재적인 문제를 디버깅하는 데 도움이 됩니다.

템플릿을 별도의 파일에 작성한 후 채팅 템플릿으로 추출합니다.

```py
open("template.jinja", "w").write(tokenizer.chat_template)
```

편집된 템플릿을 토크나이저에 다시 로드할 수도 있습니다.

```py
tokenizer.chat_template = open("template.jinja").read()
```

## 채팅 템플릿을 저장하고 불러오기합니다.[[storing-and-loading-chat-templates]]

채팅 템플릿은 디스크에 여러 가지 형식으로 저장됩니다. 최신 체크포인트에서는 템플릿을 독립적인 `.jinja` 파일로 저장하는 반면, 이전 체크포인트에서는 토크나이저 또는 프로세서 설정에 템플릿을 포함시킵니다.

### 저장 형식입니다.[[storage-formats]]

템플릿은 다음 형식 중 하나로 저장될 수 있습니다.

- `chat_template.jinja` 파일을 사용하는 것이 권장됩니다. 이 파일은 저장소의 루트에 위치한 독립적인 Jinja 템플릿 파일로서 단일 채팅 템플릿을 포함하고 있습니다. 이는 [`~PreTrainedTokenizer.save_pretrained`] 함수가 기본적으로 작성하는 형태입니다. 템플릿을 별도의 파일에 저장하면 검토, 편집 및 비교가 용이해집니다. 토크나이저와 프로세서 모두 `chat_template.jinja` 파일을 동일한 방식으로 로드합니다.
- `additional_chat_templates/<name>.jinja`는 모델이 여러 개의 명명된 템플릿을 제공할 때 사용되는 독립적인 Jinja 파일들이 저장되는 디렉터리입니다(예를 들어, `default` 템플릿과 별도의 `tool_use` 템플릿). `default` 템플릿은 여전히 저장소 루트에 있는 `chat_template.jinja`에 위치하지만, 그 외의 모든 명명된 템플릿들은 `additional_chat_templates/<name>.jinja`에 저장되며, 여기서 파일명의 핵심 부분이 바로 템플릿 이름이 됩니다.

> [!WARNING]
> 아래의 레거시 형식들은 하위 호환성을 위한 로딩 목적으로만 유지됩니다. 이들 중 어느 쪽에도 채팅 템플릿을 작성하지 마십시오.

- `chat_template` 필드는 `tokenizer_config.json`에 포함되어 있습니다. 이는 독립적인 `.jinja` 파일이 등장하기 전에 사용되던 로드 전용의 레거시 형식입니다. 해당 템플릿은 `tokenizer_config.json` 내에 JSON 문자열로 내장됩니다. 모델에 여러 개의 명명된 템플릿이 있는 경우, 이 필드는 단일 문자열이 아닌 `{"name": ..., "template": ...}` 형태의 딕셔너리 목록으로 표현됩니다. 이 형식을 사용하는 기존 저장소들은 계속해서 정상적으로 로드될 수 있지만, [`~PreTrainedTokenizer.save_pretrained`] 메서드는 대신 최신 형식인 `.jinja` 형식을 작성합니다.

- `chat_template.json`은 이전의 멀티모달 프로세서 체크포인트에서 사용되던 로드 전용 레거시 형식입니다. 이는 `{"chat_template": "<template string>"}` 형태의 JSON 파일입니다. 이 형식을 사용하는 기존 저장소들은 계속해서 로드될 수 있지만, [`~ProcessorMixin.save_pretrained`] 메서드는 대신 현대적인 `.jinja` 형식을 작성합니다. 레거시 `chat_template.json` 파일과 현대적인 `.jinja` 파일이 혼재된 프로세서 저장소에서는 로드 시 오류가 발생합니다.

### 로딩 우선순위입니다.[[loading-precedence]]

[`~PreTrainedTokenizer.from_pretrained`]를 호출할 때, Transformers는 고정된 우선순위에 따라 저장 형식을 결정합니다. 독립적인 `.jinja` 파일이 설정에 내장된 템플릿보다 우선적으로 적용됩니다. 로더는 다음과 같습니다:

1. `tokenizer_config.json`에 포함된 모든 `chat_template` 필드를 읽어들입니다(프로세서의 경우에는 기존의 `chat_template.json`도 해당됩니다).
2. 읽기 `chat_template.jinja` 저장소 루트에 해당 파일이 존재할 경우 사용하며, 그것을…로 활용합니다. `default` 템플릿, 오버라이딩 단계 1입니다.
3. `additional_chat_templates/` 디렉터리에 있는 모든 `.jinja` 파일을 읽어서 파일명의 기본 부분을 키로 사용하여 이를 병합합니다.

만약 결과에 단 하나의 `default` 템플릿이 존재한다면, [`~PreTrainedTokenizer.chat_template`]은 해당 문자열로 설정됩니다. 만약 여러 개의 명명된 템플릿이 존재한다면, `chat_template`는 `{name: template_string}` 형태의 딕셔너리가 됩니다. 이 경우, [`~PreTrainedTokenizer.apply_chat_template`]는 도구가 전달될 때는 `tool_use` 항목을 선택하고, 그렇지 않으면 `default`를 사용합니다.

### 저장하기[[saving]]

[`~PreTrainedTokenizer.save_pretrained`]와 [`~PreTrainedTokenizer.push_to_hub`]는 기본적으로 `.jinja` 형식을 사용하여 파일을 저장합니다. 단일 문자열 템플릿의 경우 `chat_template.jinja`가 생성됩니다. 명명된 템플릿이 포함된 딕셔너리의 경우 `default` 항목은 `chat_template.jinja`에 저장되고, 나머지 항목들은 각각 `additional_chat_templates/` 디렉터리 내에 별도의 파일로 저장됩니다. 중복을 방지하기 위해 `tokenizer_config.json`에서 `chat_template` 필드는 제거됩니다.

레거시 형식 중 하나로 템플릿을 저장하는 데에는 지원되는 방법이 없습니다. 이러한 형식들은 오래된 저장소를 로드하기 위해서만 유지됩니다.

### 이전 버전의 저장소를 업데이트하기[[updating-an-older-repository]]

`tokenizer_config.json` 또는 `chat_template.json`에 포함된 임베디드 템플릿을 다시 로드하여 저장함으로써 권장되는 `.jinja` 형식으로 마이그레이션할 수 있습니다.

로드 단계에서는 저장소가 사용하는 레거시 형식을 `chat_template` 내에서 정규화하며, [`~PushToHubMixin.push_to_hub`]은 `chat_template.jinja` 파일을 반환합니다.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-org/your-model")
tokenizer.push_to_hub("your-org/your-model")
```

## 도구용 템플릿입니다.[[templates-for-tools]]

도구용 템플릿을 작성하기 위한 특정한 형식은 존재하지 않지만, 표준 API를 따르는 것이 가장 좋습니다. 이렇게 하면 사용자가 귀사의 모델에서 도구를 사용하기 위해 별도의 코드를 작성할 필요 없이도 해당 템플릿을 다양한 모델에서 쉽게 활용할 수 있습니다.

> [!WARNING]
> 공백 및 특수 토큰과 같은 서식은 모델별로 상이합니다. 모델이 학습할 때 사용된 형식과 완전히 일치하는지 반드시 확인해야 합니다.

다음 섹션에서는 도구용 템플릿을 작성하기 위한 표준 API의 요소들을 나열합니다.

### 도구 정의입니다.[[tool-definitions]]

[도구](./chat_extras)는 Python 함수 또는 JSON 스키마 형태로 전달됩니다. 함수가 전달될 경우, JSON 스키마가 자동으로 생성되어 템플릿에 전달됩니다. 템플릿에서 `tools` 변수에 접근할 때에는 항상 JSON 스키마의 목록 형태가 됩니다.

템플릿은 항상 도구들을 JSON 스키마 형태로 받지만, 모델이 학습된 형식에 맞추기 위해 렌더링 시에는 이 형식을 크게 변경해야 할 수도 있습니다. 예를 들어, [Command-R](./model_doc/cohere)은 Python 함수 헤더로 정의된 도구들을 사용하여 학습되었습니다. 템플릿은 내부적으로 JSON 스키마 타입을 변환하여 입력된 도구들을 Python 헤더 형태로 렌더링합니다.

아래 예시는 JSON 스키마 형식으로 도구가 어떻게 정의되는지를 보여줍니다.

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

채팅 템플릿에서 도구 정의를 처리하는 방법의 예시가 아래에 나와 있습니다. 모델이 학습할 때 사용된 토큰과 레이아웃에 맞게 구체적인 내용을 변경해야 합니다.

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

도구 정의를 렌더링하는 것 외에도, 템플릿 내에서 **도구 호출**과 **도구 응답**도 함께 렌더링해야 합니다.

도구 호출은 일반적으로 `"assistant”` 메시지의 `tool_calls` 키에 포함됩니다. 대부분의 도구 호출 모델이 단일 도구 호출만을 지원하더라도 이 값은 항상 리스트 형태이며, 따라서 보통 리스트에는 단 하나의 요소만 포함됩니다.

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

도구 호출을 처리하는 일반적인 패턴이 아래에 제시되어 있습니다. 이를 시작점으로 활용할 수 있지만, 템플릿이 모델이 학습된 형식과 실제로 일치하는지 반드시 확인해야 합니다.!

```jinja
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
    {%- endfor %}
{%- endif %}
```

### 도구 응답입니다.[[tool-responses]]

도구의 응답은 `tool` 역할을 가진 메시지 딕셔너리입니다. 이는 도구 호출보다 훨씬 간단하며, 보통 `role`, `name` 및 `content` 키만을 포함합니다.

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```

일부 템플릿의 경우 `name` 키가 전혀 필요하지 않을 수도 있으며, 이럴 때는 템플릿이 `content` 키만을 읽도록 작성하시면 됩니다.

```jinja
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

## 기여하기[[contribute]]

템플릿이 준비되면 토크나이저의 `chat_template` 속성에 설정하고 [`~PreTrainedTokenizerBase.apply_chat_template`]을 사용하여 테스트해 보세요. 예상대로 작동한다면 [`~PreTrainedTokenizer.push_to_hub`]를 이용하여 Hub에 업로드하면 됩니다.

비록 모델 소유자가 아니더라도, 빈 채팅 템플릿이나 잘못된 채팅 템플릿을 가진 모델에 대해 템플릿을 추가하는 것은 여전히 유용합니다. 해당 템플릿을 추가하기 위해 모델 저장소에서 [풀 리퀘스트](https://hf.co/docs/hub/repositories-pull-requests-discussions)를 열어 주시기 바랍니다.!

```py
tokenizer.chat_template = template
tokenizer.push_to_hub("amazing_company/cool_model", commit_message="Add chat template", create_pr=True)
```
