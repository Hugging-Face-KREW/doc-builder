# Copyright 2026 The HuggingFace Team. Licensed under the Apache License, Version 2.0.
"""OpenRouter inference adapter; no model downloads or Hugging Face calls."""

import os
import re
import time

import httpx

from . import pipeline
from .segment import PLACEHOLDER_RE, required

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterGenerator:
    def __init__(self, api_key=None, client=None, timeout=120, retries=2, sleep=time.sleep):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY before generating translations")
        self.client = client or httpx.Client(timeout=timeout)
        self.owns_client = client is None
        self.retries, self.sleep = retries, sleep

    def close(self):
        if self.owns_client:
            self.client.close()

    def complete(self, messages, config, budget):
        for attempt in range(self.retries + 1):
            try:
                response = self.client.post(
                    ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "X-OpenRouter-Title": "doc-builder translation",
                    },
                    json={
                        "model": config["model"],
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": budget,
                        "stream": False,
                    },
                )
            except httpx.TransportError:
                if attempt == self.retries:
                    raise RuntimeError("OpenRouter connection failed after retries") from None
                self.sleep(2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self.retries:
                    self.sleep(2**attempt)
                    continue
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise OSError(
                    f"OpenRouter request rejected (HTTP {response.status_code}); check credentials and model ID"
                )
            if response.is_error:
                # Do not expose API keys, source text, or provider payloads in exceptions.
                raise RuntimeError(f"OpenRouter request failed (HTTP {response.status_code})")
            try:
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
                    raise ValueError("Incomplete completion")
            except (KeyError, IndexError, TypeError, ValueError):
                raise RuntimeError("OpenRouter returned an invalid, empty, or truncated completion") from None
            return content
        raise RuntimeError("OpenRouter request failed")

    def __call__(self, units, config, retry=False):
        budget = config["output"] * (2 if retry else 1)

        def estimate(unit):
            # Conservative UTF-8 byte upper bound; no HF tokenizer download is needed.
            tags = pipeline.xml_tags(unit)
            source = PLACEHOLDER_RE.sub(lambda m, tags=tags: tags[int(m[1])], unit["text"])
            return source.encode() + pipeline.prompt(unit, config, retry).encode()

        results = []
        for unit in units:
            chunks = pipeline.split_unit(unit, estimate, config["context"] - budget - 256, retry=retry)
            parts = []
            for chunk in chunks:
                if not required(chunk):
                    parts.append(chunk["text"])
                    continue
                tags = pipeline.xml_tags(chunk)
                source = PLACEHOLDER_RE.sub(lambda m, tags=tags: tags[int(m[1])], chunk["text"])
                try:
                    text = self.complete(
                        [
                            {"role": "system", "content": pipeline.prompt(chunk, config, retry)},
                            {"role": "user", "content": source},
                        ],
                        config,
                        budget,
                    )
                except RuntimeError as exc:
                    print(str(exc), flush=True)
                    # Other units in this group remain reusable; the pipeline retries this one.
                    parts.append(None)
                    continue
                text = re.sub(
                    r"<\s*(/?)\s*((?:link|image|em|strong|del|span|keep|ph)\d+)\s*(/?)\s*>", r"<\1\2\3>", text
                )
                for i, tag in enumerate(tags):
                    text = (
                        re.sub(rf"<keep{i}>(?:(?!</?keep\d+\b)[\s\S])*?</keep{i}>", f"¤{i}¤", text)
                        if tag.startswith("<keep")
                        else text.replace(tag, f"¤{i}¤")
                    )
                parts.append(re.match(r"\s*", chunk["text"])[0] + text.strip() + re.search(r"\s*$", chunk["text"])[0])
            results.append(None if None in parts else "".join(parts))
        return results
