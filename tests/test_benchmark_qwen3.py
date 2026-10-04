"""Measurement must preserve completions, protect credentials, and bound spending."""

import json
import runpy
from pathlib import Path

import httpx
import pytest

benchmark = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/translation/benchmark_qwen3.py"))


def test_record_preserves_completion_without_credentials(tmp_path):
    payload = {
        "model": benchmark["MODEL"],
        "messages": [{"role": "user", "content": "Translate this text."}],
        "max_tokens": 100,
        "temperature": 0,
    }

    def handle(request):
        body = json.loads(request.content)
        assert body["messages"] == payload["messages"]
        assert body["temperature"] == 0
        assert body["provider"]["max_price"] == benchmark["MAX_PRICE"]
        return httpx.Response(
            200,
            json={
                "provider": "test-provider",
                "model": payload["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": "번역합니다."}}],
                "usage": {"cost": 0.0001, "prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        measured = benchmark["MeasuredClient"](client, tmp_path / "requests.jsonl", 1)
        response = measured.post(benchmark["ENDPOINT"], headers={"Authorization": "Bearer private-key"}, json=payload)
    assert response.json()["choices"][0]["message"]["content"] == "번역합니다."
    saved = (tmp_path / "requests.jsonl").read_text()
    assert "private-key" not in saved and "Authorization" not in saved
    assert json.loads(saved)["accounted_cost_usd"] == 0.0001
    assert measured.rows[0]["elapsed_seconds"] >= 0


def test_budget_stops_before_http_and_transport_errors_are_accounted(tmp_path):
    def fail(request):
        raise httpx.ConnectError("offline")

    payload = {"model": benchmark["MODEL"], "messages": [{"content": "Hello"}], "max_tokens": 4096}
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        measured = benchmark["MeasuredClient"](client, tmp_path / "requests.jsonl", 0.003)
        with pytest.raises(httpx.ConnectError):
            measured.post(benchmark["ENDPOINT"], headers={}, json=payload)
        assert measured.reserved_usd > 0
        with pytest.raises(OSError, match="cost ceiling"):
            measured.post(benchmark["ENDPOINT"], headers={}, json=payload)
    assert len(measured.rows) == 1


def test_offline_replay_rejects_changed_prompts(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts/translation"
    monkeypatch.syspath_prepend(str(scripts))
    replay = runpy.run_path(str(scripts / "replay_qwen3.py"))
    payload = {"model": benchmark["MODEL"], "messages": [{"content": "Translate."}]}
    client = replay["ReplayClient"](
        [
            {
                "request": 1,
                "payload": {**payload, "provider": {"max_price": benchmark["MAX_PRICE"]}},
                "validation_retry": False,
                "http_status": 200,
                "choices": [{"finish_reason": "stop", "content": "번역합니다."}],
            }
        ]
    )
    client.unit = {"page": "example.md", "part": 1, "unit": 2}
    with pytest.raises(AssertionError, match="changed during replay"):
        client.post(benchmark["ENDPOINT"], headers={}, json={**payload, "messages": []})
    assert client.offset == 0
    response = client.post(benchmark["ENDPOINT"], headers={}, json=payload)
    assert response.json()["choices"][0]["message"]["content"] == "번역합니다."
    assert client.attribution == [{"request": 1, **client.unit}]


def test_diagnostic_export_keeps_rejected_tildes_without_changing_gate(monkeypatch):
    from doc_builder.translate import segment

    scripts = Path(__file__).resolve().parents[1] / "scripts/translation"
    monkeypatch.syspath_prepend(str(scripts))
    replay = runpy.run_path(str(scripts / "replay_qwen3.py"))
    unit = {"text": "Use the bottom 10-20%.", "tokens": []}
    plan = {"pieces": ["", "\n"], "units": [unit]}
    response = "하위 10~20%를 사용합니다."
    with pytest.raises(ValueError, match="introduced markup"):
        segment.render_page(plan, [response])
    assert replay["diagnostic_page"](plan, [response]) == response + "\n"
    with pytest.raises(ValueError, match="introduced markup"):
        segment.render_page(plan, [response])
    with pytest.raises(ValueError, match="introduced markup"):
        replay["diagnostic_page"](plan, [response + " **설명**"])
