#!/usr/bin/env bash
# Translate the five-page Korean evaluation set with one OpenRouter model and
# record the result under translation-results/<model-slug>/ on the current branch.
#
# Usage:
#   scripts/translation/run-model.sh [--dry-run] [--push]
#
# Environment (all optional; .env in the repo root is loaded automatically):
#   OPENROUTER_API_KEY   prompted for (and saved to .env) when missing
#   OPENROUTER_MODEL     default: qwen/qwen3.8-27b
#   TRANSFORMERS_SOURCE  default: ../transformers (clean checkout)
#   CONTEXT_WINDOW       default: 65536
#   MAX_TOKENS           default: 16384 (reasoning tokens count toward this)
#   REASONING            default | off (default: off; thinking is slow and costly for translation)
#   CONCURRENCY          parallel requests (default: 8)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DRY_RUN=0
PUSH=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --push) PUSH=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL="${OPENROUTER_MODEL:-qwen/qwen3.8-27b}"
SOURCE="${TRANSFORMERS_SOURCE:-$ROOT/../transformers}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-65536}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
REASONING="${REASONING:-off}"
CONCURRENCY="${CONCURRENCY:-8}"
PAGES="scripts/translation/ko-poc-pages.txt"
SLUG="$(echo "$MODEL" | tr '/:' '__')"
OUTPUT="translation-results/$SLUG"
[[ $REASONING == off ]] && OUTPUT="$OUTPUT-no-reasoning"

SOURCE="$(cd "$SOURCE" && pwd)"
SHA="$(git -C "$SOURCE" rev-parse HEAD)"

# shellcheck disable=SC1091
source .venv/bin/activate
if ! command -v node >/dev/null && [[ -x "$HOME/.local/node/bin/node" ]]; then
  export PATH="$HOME/.local/node/bin:$PATH"
fi

if [[ $DRY_RUN == 1 ]]; then
  exec doc-builder translate transformers --provider openrouter --lang ko \
    --source "$SOURCE" --source-revision "$SHA" --pages-file "$PAGES" --dry-run
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    echo "OPENROUTER_API_KEY is not set. Add it to $ROOT/.env or run this script in a terminal." >&2
    exit 1
  fi
  read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
  echo
  umask 077
  grep -v '^OPENROUTER_API_KEY=' .env 2>/dev/null > .env.tmp || true
  echo "OPENROUTER_API_KEY=$OPENROUTER_API_KEY" >> .env.tmp
  mv .env.tmp .env
  echo "Saved to .env (git-ignored)."
  export OPENROUTER_API_KEY
fi

echo "Model: $MODEL | reasoning: $REASONING | concurrency: $CONCURRENCY | source: $SHA | output: $OUTPUT"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
doc-builder translate transformers --provider openrouter --lang ko --model "$MODEL" \
  --source "$SOURCE" --source-revision "$SHA" --pages-file "$PAGES" \
  --context-window "$CONTEXT_WINDOW" --max-tokens "$MAX_TOKENS" \
  --reasoning "$REASONING" --concurrency "$CONCURRENCY" \
  --output-dir "$OUTPUT"

printf '%s\t%s\tmodel=%s\tsource=%s\tcontext=%s\tmax_tokens=%s\treasoning=%s\n' \
  "$STARTED" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODEL" "$SHA" "$CONTEXT_WINDOW" "$MAX_TOKENS" "$REASONING" \
  >> "$OUTPUT/runs.tsv"

git add "$OUTPUT"
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "Add Korean translation results for $MODEL (transformers ${SHA:0:12})"
fi
if [[ $PUSH == 1 ]]; then
  git push -u origin HEAD
fi
