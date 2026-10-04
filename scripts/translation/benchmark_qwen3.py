"""Measure the team's unchanged Korean OpenRouter baseline; never publish upstream."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from json import dumps
from pathlib import Path
from types import SimpleNamespace

import httpx

from doc_builder.translate import local, pipeline, preflight, segment
from doc_builder.translate.openrouter import ENDPOINT, OpenRouterGenerator

ROOT = Path(__file__).resolve().parents[2]
MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
BASE_REVISION = "773a807b08ba5a039d6ead6774545e748064f596"
SOURCE_REVISION = "469230357aab0f2b303b0d638c1f8d06edb14184"
# USD per million tokens. These are routing ceilings, not claimed actual prices.
MAX_PRICE = {"prompt": 0.15, "completion": 0.60}


def save_json(path, value):
    local.atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def sha(data):
    return hashlib.sha256(data).hexdigest()


class MeasuredClient:
    """Wrap HTTP only: keep the original prompt, splitting and validation intact."""

    def __init__(self, client, path, max_usd):
        self.client, self.path, self.max_usd = client, path, max_usd
        self.rows = []
        self.reserved_usd = 0.0
        self.validation_retry = False

    def post(self, url, *, headers, json):
        if url != ENDPOINT:
            raise ValueError("Unexpected inference endpoint")
        body = {**json, "provider": {"max_price": MAX_PRICE}}
        # UTF-8 bytes conservatively bound content tokens; reserve framing overhead too.
        input_bound = sum(len(m["content"].encode()) + 128 for m in body["messages"]) + 256
        ceiling = (input_bound * MAX_PRICE["prompt"] + body["max_tokens"] * MAX_PRICE["completion"]) / 1e6
        if self.reserved_usd + ceiling > self.max_usd:
            raise OSError("Experiment cost ceiling reached; no further API request sent")
        started = time.perf_counter()
        row = {"request": len(self.rows) + 1, "validation_retry": self.validation_retry, "payload": body}
        charged = ceiling
        try:
            response = self.client.post(url, headers=headers, json=body)
            row["http_status"] = response.status_code
            try:
                data = response.json()
                if not isinstance(data, dict):
                    data = {}
            except ValueError:
                data = {}
            usage = data.get("usage") or {}
            row.update(
                provider=data.get("provider"),
                returned_model=data.get("model"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                reported_cost_usd=usage.get("cost"),
                choices=[
                    {"finish_reason": c.get("finish_reason"), "content": (c.get("message") or {}).get("content")}
                    for c in data.get("choices", [])
                ],
            )
            cost = usage.get("cost")
            if response.is_success and isinstance(cost, (float, int)) and cost >= 0:
                charged = cost
            return response
        except httpx.TransportError:
            row["transport_error"] = True
            raise
        finally:
            row["elapsed_seconds"] = time.perf_counter() - started
            row["accounted_cost_usd"] = charged
            self.reserved_usd += charged
            self.rows.append(row)
            with self.path.open("a", encoding="utf-8") as stream:
                # Explicit allowlist above: never serialize request headers or credentials.
                stream.write(dumps(row, ensure_ascii=False) + "\n")
            print(
                f"API {row['request']}: HTTP {row.get('http_status', 'transport error')}, "
                f"{row['elapsed_seconds']:.2f}s, accounted ${self.reserved_usd:.4f}",
                flush=True,
            )


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(args):
    result_dir = args.results.resolve()
    if (result_dir / "requests.jsonl").exists() or (result_dir / "metrics.json").exists():
        raise ValueError("Use a fresh results directory; do not overwrite an earlier measured run")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory for a cold-cache baseline")
    if args.env_file:
        from dotenv import dotenv_values

        key = dotenv_values(args.env_file).get("OPENROUTER_API_KEY")
        if key:
            os.environ["OPENROUTER_API_KEY"] = key
    pages = (ROOT / "scripts/translation/ko-poc-pages.txt").read_text().splitlines()
    files, _ = pipeline.inventory(args.source, SOURCE_REVISION, pages)
    config = pipeline.configuration("ko", pipeline.MODEL_REVISION)
    documents = pipeline.prepare_documents(files, config)
    report = preflight.check(files, config)
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir / "preflight.json", report)
    metadata = {
        "status": "prepared_not_run",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "base_revision": BASE_REVISION,
        "source_repository": "https://github.com/huggingface/transformers",
        "source_revision": SOURCE_REVISION,
        "language": "ko",
        "temperature": 0,
        "context_window": 16384,
        "max_tokens": 4096,
        "concurrency": 1,
        "cold_cache": True,
        "provider_routing": {"max_price": MAX_PRICE},
        "experiment_cost_ceiling_usd": args.max_usd,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("httpx", "pyyaml", "huggingface_hub")},
        "prompt_sha256": sha(inspect.getsource(pipeline.prompt).encode()),
        "glossary_sha256": sha((ROOT / "src/doc_builder/glossaries/ko.yml").read_bytes()),
        "implementation_sha256": {
            str(p.relative_to(ROOT)): sha(p.read_bytes())
            for p in [
                ROOT / "scripts/translation/benchmark_qwen3.py",
                ROOT / "src/doc_builder/translate/pipeline.py",
                ROOT / "src/doc_builder/translate/openrouter.py",
                ROOT / "src/doc_builder/translate/segment.py",
                ROOT / "src/doc_builder/translate/local.py",
                ROOT / "kit/preprocessors/translate.cjs",
                ROOT / "kit/package-lock.json",
            ]
        },
        "documents": [
            {
                "path": name,
                "source_sha256": sha(files[name]),
                "units": sum(len(p["units"]) for p in documents[name]),
                "required_units": sum(segment.required(u) for p in documents[name] for u in p["units"]),
                "existing_korean_reference": name != "_toctree.yml"
                and (args.source / "docs/source/ko" / name).exists(),
            }
            for name in [*pages, "_toctree.yml"]
        ],
    }
    save_json(result_dir / "manifest.json", metadata)
    save_json(
        result_dir / "prompt-examples.json",
        {
            "plain": pipeline.prompt({"text": "The tokenizer handles padding.", "kind": "paragraph"}, config),
            "protected": pipeline.prompt({"text": "Use ¤0¤ for inference.", "kind": "paragraph"}, config),
            "retry_fragment": pipeline.prompt({"text": "for inference", "kind": "fragment"}, config, retry=True),
            "note": "Exact first-pass and retry messages are recorded per API call in requests.jsonl.",
        },
    )
    print(
        f"Prepared {len(pages)} pages; {report['fenced_blocks']} fenced blocks; {len(report['errors'])} preflight errors"
    )
    if report["errors"]:
        raise ValueError("Preflight failed")
    if args.prepare_only:
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("Preparation complete. Set OPENROUTER_API_KEY or supply --env-file before inference")
    options = SimpleNamespace(
        source=args.source,
        source_revision=SOURCE_REVISION,
        model=MODEL,
        lang="ko",
        output_dir=args.output_dir,
        context_window=16384,
        max_tokens=4096,
    )
    started = time.perf_counter()
    status, error = "completed", None
    with httpx.Client(timeout=120) as client, (result_dir / "run.log").open("w", encoding="utf-8") as log:
        measured = MeasuredClient(client, result_dir / "requests.jsonl", args.max_usd)
        generator = OpenRouterGenerator(client=measured)

        def generate(units, cfg, retry=False):
            measured.validation_retry = retry
            return generator(units, cfg, retry=retry)

        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            try:
                local.run(files, options, generate)
            except (ValueError, OSError, RuntimeError) as exc:
                status, error = "failed", str(exc)
                print(error)
    rows = measured.rows
    durations = [r["elapsed_seconds"] for r in rows]
    metrics = {
        "status": status,
        "error": error,
        "wall_seconds": time.perf_counter() - started,
        "api_roundtrip_seconds": sum(durations),
        "median_api_roundtrip_seconds": statistics.median(durations) if durations else None,
        "http_requests": len(rows),
        "http_errors": sum(r.get("http_status", 599) >= 400 for r in rows),
        "validation_retry_http_requests": sum(r["validation_retry"] for r in rows),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rows),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rows),
        "reported_cost_usd": sum(r.get("reported_cost_usd") or 0 for r in rows),
        "requests_missing_cost": sum(r.get("reported_cost_usd") is None for r in rows),
        "accounted_cost_usd": measured.reserved_usd,
        "providers": sorted({r["provider"] for r in rows if r.get("provider")}),
        "timing_definition": "Client wall clock and full HTTP round-trip latency; not GPU-only inference or TTFT.",
        "human_quality_review": "pending",
    }
    if status == "completed":
        for name in [*pages, "_toctree.yml"]:
            local.atomic_write(result_dir / "ko" / name, (args.output_dir / name).read_bytes())
        save_json(
            result_dir / "translation-state.json", local.read_json(args.output_dir / ".translation-state.json", {})
        )
    save_json(result_dir / "metrics.json", metrics)
    metadata["status"] = status
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_json(result_dir / "manifest.json", metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-usd", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "translation-output/qwen3-baseline")
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/ko/qwen3-30b-a3b-instruct-2507/baseline")
    args = parser.parse_args()
    if not 0 < args.max_usd <= 1:
        parser.error("This participant's experiment budget must be greater than $0 and at most $1")
    run(args)
