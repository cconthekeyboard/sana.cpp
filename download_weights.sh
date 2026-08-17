#!/usr/bin/env bash
# Downloads the pre-converted .gguf weights (denoiser, VAE decoder, caption
# projection, and Gemma-2 text encoder) from Hugging Face over plain HTTP --
# no Python, no huggingface_hub, just curl -- laid out exactly as sana_infer
# and the C++ benchmarks expect:
#
#   <outdir>/
#     denoiser_weights.gguf
#     model_weights.gguf
#     caption_weights.gguf
#     vae_decoder_weights.gguf
#     gemma2_text_encoder/gemma-2-2b-it-f32.gguf
#
# Usage: ./download_weights.sh [outdir]
#   outdir defaults to weights (matches sana_infer's --weights-dir default
#   of ../weights).
#
# Set SANA_WEIGHTS_REPO to point at a different Hugging Face repo id.

set -euo pipefail

REPO_ID="${SANA_WEIGHTS_REPO:-doobluhc/sana-cpp-weights}"
OUTDIR="${1:-weights}"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"

FILES=(
    "denoiser_weights.gguf"
    "model_weights.gguf"
    "caption_weights.gguf"
    "vae_decoder_weights.gguf"
    "gemma2_text_encoder/gemma-2-2b-it-f32.gguf"
)

for f in "${FILES[@]}"; do
    dest="${OUTDIR}/${f}"
    mkdir -p "$(dirname "$dest")"
    echo "downloading ${f}"
    curl -L --fail --continue-at - -o "$dest" "${BASE_URL}/${f}"
done

echo "done: weights are in ${OUTDIR}"
