"""Exercise HTTP responses, syntax preservation and resumable local publication."""

import json
from types import SimpleNamespace

import httpx
import pytest

from doc_builder.translate import local, pipeline
from doc_builder.translate.openrouter import OpenRouterGenerator
from doc_builder.translate.segment import accept_unit, extract_pages, render_page, validate_pages
from tests.translate_harness import files


def options(tmp_path):
    return SimpleNamespace(
        model="test/model",
        output_dir=tmp_path / "ko",
        source=None,
        lang="ko",
        source_revision="a" * 40,
        context_window=16384,
        max_tokens=4096,
    )


def korean(units, config, retry=False):
    import re

    return [
        re.sub(
            r"¤\d+¤|[^¤]+", lambda m: m[0] if m[0].startswith("¤") or not m[0].strip() else " 번역합니다. ", u["text"]
        )
        for u in units
    ]


def response(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def test_http_adapter_preserves_nested_links_code_and_names():
    plan = extract_pages(["Use [the **guide**](https://example.com) with `code` and CUDA."], keep=["CUDA"])[0]
    unit = plan["units"][0]
    cfg = pipeline.configuration("ko", pipeline.MODEL_REVISION)
    cfg["model"] = "test/model"

    def handle(request):
        payload = json.loads(request.content)
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert payload["model"] == "test/model"
        assert "Korean" in payload["messages"][0]["content"]
        assert "¤" not in payload["messages"][1]["content"]
        tags = pipeline.xml_tags(unit)
        return httpx.Response(
            200,
            json=response(
                "사용합니다. "
                + tags[0]
                + "이 "
                + tags[1]
                + "가이드"
                + tags[2]
                + tags[3]
                + " 와 "
                + tags[4]
                + " 및 "
                + tags[5]
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        generator = OpenRouterGenerator("test-key", client=client)
        translated = generator([unit], cfg)
    accept_unit(unit, translated[0])
    text = render_page(plan, translated)
    validate_pages([plan], [text])
    assert "https://example.com" in text and "`code`" in text and "CUDA" in text


@pytest.mark.parametrize(
    "bad", [response("partial", "length"), response("", "stop"), {"error": {"message": "failed"}}]
)
def test_incomplete_or_error_response_is_not_accepted(bad):
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=bad))) as client:
        generator = OpenRouterGenerator("secret", client=client)
        with pytest.raises(RuntimeError, match="invalid, empty, or truncated"):
            generator.complete([], {"model": "test/model"}, 50)


def test_rate_limit_retries_but_auth_error_does_not_leak_credentials():
    statuses = iter([429, 503, 200])
    sleeps = []
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(next(statuses), json=response("번역")))
    ) as client:
        generator = OpenRouterGenerator("secret", client=client, sleep=sleeps.append)
        assert generator.complete([], {"model": "test/model"}, 50) == "번역"
    assert sleeps == [1, 2]
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="secret"))) as client:
        generator = OpenRouterGenerator("secret", client=client)
        with pytest.raises(OSError, match="HTTP 401") as error:
            generator.complete([], {"model": "test/model"}, 50)
        assert "secret" not in str(error.value)


def test_retry_keeps_markers_outside_model():
    plan = extract_pages(["Read [the guide](url) with `code`."])[0]

    def handle(request):
        source = json.loads(request.content)["messages"][1]["content"]
        assert not any(marker in source for marker in ("¤", "<", "`"))
        return httpx.Response(200, json=response("번역합니다."))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = OpenRouterGenerator("key", client=client)(
            plan["units"], pipeline.configuration("ko", pipeline.MODEL_REVISION), retry=True
        )
    text = render_page(plan, result)
    validate_pages([plan], [text])
    assert "(url)" in text and "`code`" in text


def test_local_publish_and_warm_cache_need_no_hf_or_api_key(tmp_path, monkeypatch):
    args = options(tmp_path)
    local.run(files(), args, korean)
    output = (args.output_dir / "guide.mdx").read_text()
    assert "번역" in output and "`code`" in output
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    local.run(files(), args)
    assert (args.output_dir / "guide.mdx").read_text() == output
    args.model = "test/other"
    calls = []

    def generate(units, cfg, retry=False):
        calls.append(units)
        return korean(units, cfg, retry)

    local.run(files(), args, generate)
    assert calls


def test_failed_run_preserves_previous_output_and_caches_complete_pages(tmp_path):
    args = options(tmp_path)
    local.run(files(), args, korean)
    previous = (args.output_dir / "index.md").read_bytes()
    source = {**files(), "index.md": b"# Updated\n\nBroken page.\n", "guide.mdx": b"# Updated guide\n\nValid body.\n"}

    def generate(units, cfg, retry=False):
        return [None if "Broken" in u["text"] else korean([u], cfg)[0] for u in units]

    with pytest.raises(ValueError, match="output was not published"):
        local.run(source, args, generate)
    assert (args.output_dir / "index.md").read_bytes() == previous
    seen = []

    def resume(units, cfg, retry=False):
        seen.extend(u["text"] for u in units)
        return korean(units, cfg, retry)

    local.run(source, args, resume)
    assert not any("Valid body" in text for text in seen)


def test_output_protects_unowned_files_and_symlinks(tmp_path):
    args = options(tmp_path)
    args.output_dir.mkdir()
    (args.output_dir / "index.md").write_text("user document")
    with pytest.raises(ValueError, match="not owned"):
        local.run(files(), args, korean)
    assert (args.output_dir / "index.md").read_text() == "user document"
    (args.output_dir / "index.md").unlink()
    external = tmp_path / "external.json"
    external.write_text("{}")
    (args.output_dir / ".translation-cache.json").symlink_to(external)
    with pytest.raises(ValueError, match="Unsafe"):
        local.run(files(), args, korean)
    assert external.read_text() == "{}"


def test_korean_glossary_and_cache_are_separate_from_japanese():
    ko, ja = (pipeline.configuration(lang, pipeline.MODEL_REVISION) for lang in ("ko", "ja"))
    assert ko["glossary"]["pin"]["inference"] == "추론"
    assert pipeline.cache_key("index.md", b"inference", ko) != pipeline.cache_key("index.md", b"inference", ja)
    assert "Korean" in pipeline.prompt({"text": "inference"}, ko)


def test_cli_openrouter_path_and_dry_run_do_not_use_hf(tmp_path, monkeypatch):
    from doc_builder.commands import translate as command

    monkeypatch.setattr(pipeline, "inventory", lambda *a: (files(), None))
    monkeypatch.setattr(command, "HfApi", lambda: pytest.fail("OpenRouter must not construct HfApi"))
    parser = command.translate_command_parser()
    args = parser.parse_args(
        [
            "transformers",
            "--source",
            str(tmp_path / "source"),
            "--source-revision",
            "a" * 40,
            "--lang",
            "ko",
            "--provider",
            "openrouter",
            "--model",
            "test/model",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    report = command.run(args, generate_fn=korean)
    assert report["language"] == "ko"
    args.dry_run = True
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    report = command.run(args)
    assert not report["errors"]
