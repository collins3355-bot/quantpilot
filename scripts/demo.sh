#!/usr/bin/env bash
# End-to-end demo: download a small model + eval corpus, then run the autotuner.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf"
MODEL="models/qwen2.5-0.5b-instruct-fp16.gguf"
CORPUS_URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
CORPUS="data/wikitext-2-raw/wiki.test.raw"

mkdir -p models data

if [ ! -f "$MODEL" ]; then
  echo "downloading demo model (~1 GB, Qwen2.5-0.5B-Instruct F16)..."
  curl -L --progress-bar -o "$MODEL" "$MODEL_URL"
fi

if [ ! -f "$CORPUS" ]; then
  echo "downloading wikitext-2 eval corpus (~5 MB)..."
  curl -L --progress-bar -o data/wikitext-2-raw-v1.zip "$CORPUS_URL"
  unzip -oq data/wikitext-2-raw-v1.zip -d data
fi

PYTHONPATH=src python3 -m quantpilot bench --source "$MODEL" --corpus "$CORPUS" "$@"
