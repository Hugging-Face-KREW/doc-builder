# Copyright 2026 The HuggingFace Team. Licensed under the Apache License, Version 2.0.
"""Local PoC output and complete-page cache, independent of Hub storage."""

import json
import os
import tempfile
from pathlib import Path

from . import pipeline
from .openrouter import OpenRouterGenerator


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(data)
        temporary = Path(temp.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot read local translation metadata: {path}") from exc


def destination(root, name):
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"Unsafe translation path: {name}")
    return path


def run(files, args, generate_fn=None):
    if not args.model:
        raise ValueError("OpenRouter requires --model provider/model (or OPENROUTER_MODEL)")
    if not args.output_dir:
        raise ValueError("OpenRouter requires --output-dir")
    root = args.output_dir.resolve()
    if args.source and (root.is_relative_to(args.source.resolve()) or args.source.resolve().is_relative_to(root)):
        raise ValueError("Keep --output-dir separate from the source checkout")
    config = pipeline.configuration(args.lang, pipeline.MODEL_REVISION)
    config.update(
        provider="openrouter",
        model=args.model,
        model_revision="openrouter",
        tokenizer_revision="none",
        attention="none",
        transformers="none",
        context=args.context_window,
        output=args.max_tokens,
    )
    if config["output"] <= 0 or config["context"] <= config["output"] * 2 + 256:
        raise ValueError("Context window must exceed twice --max-tokens plus 256")
    state_path = destination(root, ".translation-state.json")
    cache_path = destination(root, ".translation-cache.json")
    state = read_json(state_path, {})
    if not isinstance(state, dict):
        raise ValueError("Local translation state must be a JSON object")
    if state and (state.get("provider") != "openrouter" or state.get("language") != args.lang):
        raise ValueError("Use a separate output directory for each language and provider")
    previous = state.get("files", [])
    for name in set(files) | set(previous):
        path = destination(root, name)
        if path.exists() and name not in previous:
            raise ValueError(f"Refusing to overwrite a file not owned by this translation run: {path}")
    cache = read_json(cache_path, {})
    if not isinstance(cache, dict):
        raise ValueError("Local translation cache must be a JSON object")
    generator = None
    try:
        if generate_fn is None:
            # Fully cached runs need neither credentials nor a network connection.
            def generate_fn(units, cfg, retry=False):
                nonlocal generator
                if generator is None:
                    generator = OpenRouterGenerator()
                return generator(units, cfg, retry=retry)

        translated, candidate, failures = pipeline.translate(files, config, cache, generate_fn)
    finally:
        if generator is not None:
            generator.close()
    atomic_write(cache_path, json.dumps({**cache, **candidate}, ensure_ascii=False).encode())
    if failures:
        raise ValueError(
            "Translation failed; completed pages were cached, output was not published:\n" + "\n".join(failures)
        )
    for name, data in translated.items():
        atomic_write(destination(root, name), data)
    for name in set(previous) - set(translated):
        destination(root, name).unlink(missing_ok=True)
    state = {
        "provider": "openrouter",
        "language": args.lang,
        "model": args.model,
        "source_revision": args.source_revision,
        "config": config,
        "files": sorted(translated),
    }
    atomic_write(state_path, json.dumps(state, ensure_ascii=False, indent=2).encode())
    result = {
        "output_dir": str(root),
        "language": args.lang,
        "pages": len(candidate),
        "source_revision": args.source_revision,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
