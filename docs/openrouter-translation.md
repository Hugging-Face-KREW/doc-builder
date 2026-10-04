# Korean translation PoC with OpenRouter

This implementation adapts the translation modules and parser from upstream
[doc-builder PR #813](https://github.com/huggingface/doc-builder/pull/813), pinned at
`f58f569b8afa1dd1019df49b7ae491675e03f805`. The upstream PR is not merged.
The existing build and deployment workflows are unchanged. This PoC runs locally:
OpenRouter handles inference, and the filesystem holds translations and cache.
No Hugging Face token, GPU, Jobs access, or Bucket is required for this path.

## Setup

Use Python 3.10+ and a supported Node version (22.14+ or 24+).
Run from this repository, installed in editable mode so its translation parser is available:

```bash
uv venv
uv pip install -e .
npm ci --prefix kit
source .venv/bin/activate
```

Set `OPENROUTER_API_KEY` in your shell. The command reads it from the environment;
it does not load `.env` automatically. Never put credentials in CLI arguments or commits.
Specify an OpenRouter model ID with `--model provider/model`, or set `OPENROUTER_MODEL`.
The adapter uses OpenRouter's
[chat completions API](https://openrouter.ai/docs/api_reference/overview).
Each prose unit is one API request, so start with a small page selection.

## Select and check source pages

Use a clean local checkout of `huggingface/transformers` with an exact commit SHA.
Page paths are relative to `docs/source/en`. The full SHA must match the checkout's HEAD.
Only the listed pages and their pruned sidebar are translated when `--pages-file` is supplied.
Omit it to translate all tracked English pages, including ones absent from the sidebar.

```bash
git -C ../transformers rev-parse HEAD
printf '%s\n' index.md > /tmp/translation-pages.txt

doc-builder translate transformers \
  --provider openrouter --lang ko \
  --source ../transformers --source-revision <FULL_COMMIT_SHA> \
  --pages-file /tmp/translation-pages.txt --dry-run
```

The dry run audits syntax preservation using synthetic translations. It does not
call OpenRouter, load a model, or write translation output. It does not evaluate
Korean translation quality.

## Generate and preview

```bash
doc-builder translate transformers \
  --provider openrouter --lang ko --model <PROVIDER/MODEL> \
  --source ../transformers --source-revision <FULL_COMMIT_SHA> \
  --pages-file /tmp/translation-pages.txt \
  --output-dir ./translation-output/ko-preview

doc-builder preview transformers ./translation-output/ko-preview \
  --language ko --version main
```

Previewing Transformers docs containing `[[autodoc]]` requires an installed
Transformers package and its doc-build dependencies. Translation itself does not.
For Japanese use `--lang ja` and a separate output directory.
If `--source` is omitted, the command clones the pinned Transformers commit from GitHub.

## Five-page Korean evaluation set

The agreed test selection is checked in at `scripts/translation/ko-poc-pages.txt`:

| English source path | Evaluation purpose |
| --- | --- |
| `pipeline_tutorial.md` | Compare with the existing Korean translation |
| `tokenizer_summary.md` | Compare with the existing Korean translation |
| `training.md` | Compare with the existing Korean translation |
| `quantization/selecting.md` | Review a newly generated Korean translation |
| `chat_templating_writing.md` | Review a newly generated Korean translation |

These reference-availability labels reflect the user's evaluation plan. Existing
Korean translations may cover an older English revision; compare their content
before treating them as ground truth. The generator always reads English sources,
not the existing Korean documents.

Use the selection in both the dry run and generation commands above:

```bash
--pages-file scripts/translation/ko-poc-pages.txt
```

Use `--output-dir ./translation-output/ko-five-pages` for this set. It also generates
an appropriately pruned Korean sidebar. Review terminology, omissions, code and
link preservation, and Markdown rendering on all five pages. Record the exact source
SHA and OpenRouter model so results can be compared across runs.

## Output, caching, and validation

The output directory contains translated `.md`/`.mdx` pages, `_toctree.yml`,
unchanged source assets, `.translation-cache.json`, and `.translation-state.json`.
Keep it separate from the source checkout and use separate directories for preview
and full-document runs: successful runs remove previously generated files no longer
in the selected inventory. Unowned destination files are never overwritten.

Cache keys cover source content, language, model, parser, prompt settings and relevant
glossary terms. Completed pages are revalidated before reuse. A fully cached run does
not need an API key. OpenRouter model IDs are not immutable model revisions, so
choose a new output directory when you need to force fresh translations after a
provider/model update.

Code, URLs, API references, inline formatting, anchors and document structure are
protected by the upstream parser and validators. Invalid units get one validation
retry using smaller fragments. HTTP 429 and 5xx responses and connection failures
get bounded retries. Empty, malformed and truncated completions are rejected.
If any required page fails, the command exits unsuccessfully, saves completed pages
in the cache for resumption, and leaves the previously published documents alone.
Filesystem publication is per-file atomic; an OS interruption during publication
may leave a mix of old and new files, so rerun before previewing interrupted output.

Korean terminology is in `src/doc_builder/glossaries/ko.yml`. It is an initial PoC
vocabulary and should be reviewed by a Korean technical editor before publication.
The prompt uses Korean technical documentation style with 합니다/입니다 endings.
Structural validation does not guarantee translation accuracy or Korean-language
coverage; review the generated text. `--context-window` (default 16384) and
`--max-tokens` (default 4096, doubled on validation retry) must fit the selected model.
The adapter conservatively estimates prompt tokens using UTF-8 byte length and splits
long units instead of downloading an HF tokenizer.

The original HF generation, Bucket publication and Job coordinator are also retained.
For GPU execution, install `uv pip install -e '.[translate]'` and run
`python -m doc_builder.translate.job --help`. This path still requires HF credentials,
Jobs and Bucket access. Automatic HF workflows from the draft PR were not imported.
