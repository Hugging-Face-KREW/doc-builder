"""Replay recorded responses offline to attribute latency and tokens to each document."""

import argparse
import contextlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import httpx
from benchmark_qwen3 import MAX_PRICE, MODEL, ROOT, SOURCE_REVISION, save_json, sha

from doc_builder.translate import local, pipeline, preflight, segment
from doc_builder.translate.openrouter import ENDPOINT, OpenRouterGenerator


class ReplayClient:
    """No network client: refuse any prompt that differs from the measured run."""

    def __init__(self, rows):
        self.rows, self.offset, self.attribution = rows, 0, []
        self.unit = None
        self.retry = False

    def post(self, url, *, headers, json):
        assert url == ENDPOINT
        row = self.rows[self.offset]
        assert row["payload"] == {**json, "provider": {"max_price": MAX_PRICE}}, (
            f"Request {self.offset + 1} changed during replay"
        )
        assert row["validation_retry"] == self.retry
        self.offset += 1
        self.attribution.append({"request": row["request"], **self.unit})
        if row.get("transport_error"):
            raise httpx.ReadError("Recorded transport error")
        return httpx.Response(
            row["http_status"],
            json={
                "model": row.get("returned_model"),
                "provider": row.get("provider"),
                "choices": [
                    {"finish_reason": c["finish_reason"], "message": {"content": c["content"]}}
                    for c in row.get("choices", [])
                ],
            },
        )


def diagnostic_page(plan, answers):
    """Assemble a rejected artifact for inspection, never a publishable translation.

    Only the observed range-tilde rejection is renderable here. Other validation
    failures still abort this diagnostic export. The actual tildes are retained.
    """
    pieces = []
    for prefix, unit, answer in zip(plan["pieces"][:-1], plan["units"], answers, strict=True):
        if not segment.required(unit):
            text = unit["text"]
        else:
            try:
                text = segment.accept_unit(unit, answer)
            except ValueError as exc:
                if str(exc) != "Translation introduced markup" or "~" not in answer:
                    raise
                # Probe the cause only; this substituted string is NEVER exported.
                segment.accept_unit(unit, answer.replace("~", "-"))
                text = re.match(r"\s*", unit["text"])[0] + " ".join(answer.split())
                text += re.search(r"\s*$", unit["text"])[0]
        pieces.append(prefix + segment.PLACEHOLDER_RE.sub(lambda m, u=unit: u["tokens"][int(m[1])]["raw"], text))
    return "".join(pieces) + plan["pieces"][-1]


def replay(source, results, cache_path):
    rows = [json.loads(line) for line in (results / "requests.jsonl").read_text().splitlines()]
    metrics = json.loads((results / "metrics.json").read_text())
    manifest = json.loads((results / "manifest.json").read_text())
    for path, expected in manifest["implementation_sha256"].items():
        assert sha((ROOT / path).read_bytes()) == expected, f"Measured implementation changed: {path}"
    config = pipeline.configuration("ko", pipeline.MODEL_REVISION)
    config.update(
        provider="openrouter",
        model=MODEL,
        model_revision="openrouter",
        tokenizer_revision="none",
        attention="none",
        transformers="none",
        context=manifest["context_window"],
        output=manifest["max_tokens"],
    )
    if metrics["status"] == "completed":
        assert config == json.loads((results / "translation-state.json").read_text())["config"]
    assert config["model"] == MODEL
    pages = (ROOT / "scripts/translation/ko-poc-pages.txt").read_text().splitlines()
    files, _ = pipeline.inventory(source, SOURCE_REVISION, pages)
    prepare = pipeline.prepare_documents
    documents = {}

    def annotated_documents(files, config):
        documents.update(prepare(files, config))
        for name, plans in documents.items():
            for pi, plan in enumerate(plans):
                for ui, unit in enumerate(plan["units"]):
                    # Metadata is never included in a model prompt or structural fingerprint.
                    unit["benchmark_location"] = {"page": name, "part": pi + 1, "unit": ui + 1}
        return documents

    client = ReplayClient(rows)
    generator = OpenRouterGenerator(api_key="offline-replay-no-network", client=client, sleep=lambda _: None)
    latest = {}

    def generate(units, config, retry=False):
        answers = []
        for unit in units:
            client.unit, client.retry = unit["benchmark_location"], retry
            response = generator([unit], config, retry)[0]
            latest[tuple(client.unit.values())] = response
            answers.append(response)
        return answers

    with patch.object(pipeline, "prepare_documents", annotated_documents), contextlib.redirect_stdout(io.StringIO()):
        translated, candidate, failures = pipeline.translate(files, config, {}, generate)
    assert bool(failures) == (metrics["status"] == "failed"), "Replayed gate outcome changed"
    if failures:
        assert metrics["error"].endswith("\n".join(failures)), "Replayed failures changed"
    assert client.offset == len(rows), "Unconsumed responses"
    saved_cache = results / "complete-page-cache.json"
    original_cache = json.loads((saved_cache if saved_cache.exists() else cache_path).read_text())
    assert candidate == original_cache, "Replayed accepted-page cache changed"
    save_json(saved_cache, original_cache)
    report = []
    for name in [*pages, "_toctree.yml"]:
        accepted = pipeline.cache_key(name, files[name], config) in candidate
        if accepted:
            data = translated[name]
            artifact = Path("ko" if not failures else "accepted/ko") / name
            if not failures:
                assert data == (results / artifact).read_bytes(), f"Replayed output changed: {name}"
        else:
            assert name != "_toctree.yml" and len(documents[name]) == 1
            plan = documents[name][0]
            data = diagnostic_page(
                plan,
                [latest.get((name, 1, ui + 1), u["text"]) for ui, u in enumerate(plan["units"])],
            ).encode()
            artifact = Path("rejected/ko") / name
        local.atomic_write(results / artifact, data)
        selected = [r for r, a in zip(rows, client.attribution, strict=True) if a["page"] == name]
        before = Counter(preflight.fences(files[name].decode()))
        after = Counter(preflight.fences(data.decode()))
        report.append(
            {
                "page": name,
                "status": "accepted_by_existing_pipeline" if accepted else "rejected_diagnostic_only",
                "artifact": str(artifact),
                "http_requests": len(selected),
                "http_errors": sum(r.get("http_status", 599) >= 400 for r in selected),
                "validation_retry_http_requests": sum(r["validation_retry"] for r in selected),
                "api_roundtrip_seconds": sum(r["elapsed_seconds"] for r in selected),
                "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in selected),
                "completion_tokens": sum(r.get("completion_tokens") or 0 for r in selected),
                "reported_cost_usd": sum(r.get("reported_cost_usd") or 0 for r in selected),
                "providers": sorted({r["provider"] for r in selected if r.get("provider")}),
                "fenced_blocks": sum(before.values()),
                "fenced_blocks_byte_identical": before == after,
                "replayed_accepted_cache_byte_identical": True if accepted else None,
                "replayed_gate_outcome_identical": True,
                "human_quality_review": "pending",
            }
        )
    save_json(results / "request-attribution.json", client.attribution)
    save_json(results / "per-page-metrics.json", report)
    save_json(results / "replay-config.json", config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/ko/qwen3-30b-a3b-instruct-2507/baseline")
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "translation-output/qwen3-baseline/.translation-cache.json"
    )
    args = parser.parse_args()
    replay(args.source, args.results, args.cache)
