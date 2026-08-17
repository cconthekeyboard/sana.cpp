#include "gemma_encoder.h"

#include "llama.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

const std::string kComplexHumanInstruction =
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\n"
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n"
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n"
    "Here are examples of how to transform or refine prompts:\n"
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n"
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\n"
    "User Prompt: ";

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int32_t n_needed = llama_tokenize(
        vocab, text.c_str(), static_cast<int32_t>(text.size()),
        nullptr, 0, /*add_special=*/true, /*parse_special=*/true
    );
    if (n_needed >= 0) {
        throw std::runtime_error("expected llama_tokenize size probe to return a negative count");
    }
    std::vector<llama_token> tokens(-n_needed);
    int32_t n_tokens = llama_tokenize(
        vocab, text.c_str(), static_cast<int32_t>(text.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        /*add_special=*/true, /*parse_special=*/true
    );
    if (n_tokens < 0) {
        throw std::runtime_error("llama_tokenize failed on second pass");
    }
    tokens.resize(n_tokens);
    return tokens;
}

}  // namespace

Tensor select_rows_first_and_last(const Tensor & input, size_t keep) {
    if (input.rank() != 2) {
        throw std::invalid_argument("select_rows_first_and_last expects a rank-2 tensor");
    }
    const size_t rows = input.dim_size(0);
    const size_t cols = input.dim_size(1);
    if (keep == 0 || keep > rows) {
        throw std::invalid_argument("select_rows_first_and_last: keep must be in (0, rows]");
    }

    // select_index = [0] + [-(keep-1), ..., -1]  ->  row 0, then the last (keep-1) rows.
    Tensor out({keep, cols});
    const float * src = input.data().data();
    float * dst = out.data().data();

    std::copy(src, src + cols, dst);  // row 0

    const size_t tail_count = keep - 1;
    const size_t tail_start = rows - tail_count;
    std::copy(
        src + tail_start * cols,
        src + rows * cols,
        dst + cols
    );

    return out;
}

struct GemmaEncoder::Impl {
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    int32_t n_embd = 0;
    int32_t n_threads = 4;

    explicit Impl(const GemmaEncoderConfig & config) {
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model = llama_model_load_from_file(config.gguf_path.c_str(), model_params);
        if (!model) {
            throw std::runtime_error("failed to load Gemma-2 GGUF: " + config.gguf_path);
        }
        vocab = llama_model_get_vocab(model);
        n_embd = llama_model_n_embd(model);
        n_threads = config.n_threads;
    }

    ~Impl() {
        if (model) {
            llama_model_free(model);
        }
        llama_backend_free();
    }

    // Runs Gemma-2 in embeddings mode on already-tokenized, already-padded
    // input. Returns a (n_tokens, n_embd) Tensor of per-token hidden states
    // (the graph's final output, post output_norm -- matches HF's
    // last_hidden_state semantics).
    Tensor decode_embeddings(const std::vector<llama_token> & tokens) {
        const int32_t n_tokens = static_cast<int32_t>(tokens.size());

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_tokens;
        ctx_params.n_batch = n_tokens;
        ctx_params.n_ubatch = n_tokens;
        ctx_params.n_threads = n_threads;
        ctx_params.n_threads_batch = n_threads;
        ctx_params.embeddings = true;
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            throw std::runtime_error("failed to create llama_context for Gemma-2 encoding");
        }

        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens.data()), n_tokens
        );
        const int32_t rc = llama_decode(ctx, batch);
        if (rc != 0) {
            llama_free(ctx);
            throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
        }

        Tensor out({static_cast<size_t>(n_tokens), static_cast<size_t>(n_embd)});
        for (int32_t i = 0; i < n_tokens; ++i) {
            const float * emb = llama_get_embeddings_ith(ctx, i);
            if (!emb) {
                llama_free(ctx);
                throw std::runtime_error("missing embeddings for token " + std::to_string(i));
            }
            std::copy(emb, emb + n_embd, out.data().data() + static_cast<size_t>(i) * n_embd);
        }

        llama_free(ctx);
        return out;
    }
};

GemmaEncoder::GemmaEncoder(const GemmaEncoderConfig & config)
    : impl(std::make_unique<Impl>(config)) {}

GemmaEncoder::~GemmaEncoder() = default;

namespace {

struct PaddedTokens {
    std::vector<llama_token> tokens;  // size == padded_length
    size_t num_real_tokens = 0;
};

PaddedTokens tokenize_and_pad(const llama_vocab * vocab, const std::string & text, size_t padded_length) {
    std::vector<llama_token> real_tokens = tokenize(vocab, text);
    if (real_tokens.size() > padded_length) {
        throw std::invalid_argument("tokenized text exceeds padded_length; truncation not implemented");
    }

    PaddedTokens result;
    result.num_real_tokens = real_tokens.size();
    result.tokens = std::move(real_tokens);
    result.tokens.resize(padded_length, /*pad_token_id=*/0);
    return result;
}

void write_attention_bias(Tensor & bias, size_t batch_index, size_t num_real_tokens, size_t max_sequence_length) {
    const size_t base = batch_index * max_sequence_length;
    float * data = bias.data().data();
    for (size_t t = 0; t < max_sequence_length; ++t) {
        const float mask = (t < num_real_tokens) ? 1.0f : 0.0f;
        data[base + t] = (1.0f - mask) * -10000.0f;
    }
}

}  // namespace

GemmaPromptPair encode_prompt_pair(
    GemmaEncoder & encoder,
    const std::string & prompt,
    const std::string & negative_prompt,
    size_t max_sequence_length
) {
    GemmaEncoder::Impl & impl = *encoder.impl;
    const size_t hidden_size = static_cast<size_t>(impl.n_embd);
    const std::vector<llama_token> chi_tokens = tokenize(impl.vocab, kComplexHumanInstruction);
    const size_t max_length_all = chi_tokens.size() + max_sequence_length - 2;

    PaddedTokens positive_padded = tokenize_and_pad(
        impl.vocab, kComplexHumanInstruction + prompt, max_length_all
    );
    Tensor positive_full = impl.decode_embeddings(positive_padded.tokens);
    Tensor positive_embeddings = select_rows_first_and_last(positive_full, max_sequence_length);

    size_t positive_real_after_slice = 0;
    {
        const size_t tail_start = max_length_all - (max_sequence_length - 1);
        if (0 < positive_padded.num_real_tokens) {
            positive_real_after_slice += 1;  // row 0 (BOS) is always real
        }
        for (size_t pos = tail_start; pos < max_length_all; ++pos) {
            if (pos < positive_padded.num_real_tokens) {
                positive_real_after_slice += 1;
            }
        }
    }

    // Negative branch: no CHI prefix, padded straight to max_sequence_length,
    // no slicing needed.
    PaddedTokens negative_padded = tokenize_and_pad(impl.vocab, negative_prompt, max_sequence_length);
    Tensor negative_embeddings = impl.decode_embeddings(negative_padded.tokens);

    GemmaPromptPair result{
        Tensor({2, max_sequence_length, hidden_size}),
        Tensor({2, 1, max_sequence_length}),
    };

    const size_t branch_stride = max_sequence_length * hidden_size;
    std::copy(
        negative_embeddings.data().data(),
        negative_embeddings.data().data() + branch_stride,
        result.embeddings.data().data()
    );
    std::copy(
        positive_embeddings.data().data(),
        positive_embeddings.data().data() + branch_stride,
        result.embeddings.data().data() + branch_stride
    );

    write_attention_bias(result.attention_bias, 0, negative_padded.num_real_tokens, max_sequence_length);
    write_attention_bias(result.attention_bias, 1, positive_real_after_slice, max_sequence_length);

    return result;
}
