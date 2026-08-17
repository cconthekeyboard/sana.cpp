#pragma once

#include "tensor.h"

#include <memory>
#include <string>

// Native C++ replacement for tools/prepare_conditioning.py's Gemma-2 text
// encoding step, backed by llama.cpp. Reproduces diffusers' SanaPipeline
// encode_prompt() contract exactly (CHI instruction prefix, padding, the
// select_index slice, and CFG batch ordering) -- see AGENTS.md/the approved
// plan for the derivation of each step.

struct GemmaEncoderConfig {
    std::string gguf_path;
    size_t max_sequence_length = 300;  // matches SanaPipeline's default
    int32_t n_threads = 4;
};

// Owns the loaded llama.cpp model/context so a process can encode many
// prompts (e.g. across denoising runs) without reloading the GGUF each time.
struct GemmaEncoder {
    explicit GemmaEncoder(const GemmaEncoderConfig & config);
    ~GemmaEncoder();
    GemmaEncoder(const GemmaEncoder &) = delete;
    GemmaEncoder & operator=(const GemmaEncoder &) = delete;

    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Batch order matches diffusers: negative prompt first, then positive.
struct GemmaPromptPair {
    Tensor embeddings;      // (2, max_sequence_length, hidden_size)
    Tensor attention_bias;  // (2, 1, max_sequence_length), (1 - mask) * -10000.0
};

GemmaPromptPair encode_prompt_pair(
    GemmaEncoder & encoder,
    const std::string & prompt,
    const std::string & negative_prompt,
    size_t max_sequence_length = 300
);

// Exposed for unit testing: select_index = [0] + [-(keep-1), ..., -1], i.e.
// keep row 0 and the last (keep-1) rows, dropping everything in between.
// Mirrors diffusers' `select_index = [0] + list(range(-max_length + 1, 0))`.
Tensor select_rows_first_and_last(const Tensor & input, size_t keep);
