#include "transformer_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common_ops.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace {

float gelu_tanh(float x) {
    const float kAlpha = std::sqrt(2.0f / static_cast<float>(M_PI));
    return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + 0.044715f * x * x * x)));
}

// Vectorized GELU (tanh approximation), applied to the whole buffer in place:
//   gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
void gelu_tanh_inplace(Tensor & tensor) {
#if defined(__APPLE__)
    const size_t count = tensor.numel();
    static thread_local std::vector<float> x3;
    static thread_local std::vector<float> inner;
    x3.resize(count);
    inner.resize(count);
    float * x = tensor.data().data();

    const float coeff = 0.044715f;
    const float alpha = std::sqrt(2.0f / static_cast<float>(M_PI));
    const float one = 1.0f;
    const float half = 0.5f;

    vDSP_vsq(x, 1, x3.data(), 1, count);                             // x3 = x*x
    vDSP_vmul(x3.data(), 1, x, 1, x3.data(), 1, count);               // x3 = x3*x = x^3
    vDSP_vsma(x3.data(), 1, &coeff, x, 1, inner.data(), 1, count);    // inner = x^3*0.044715 + x
    vDSP_vsmul(inner.data(), 1, &alpha, inner.data(), 1, count);      // inner *= alpha
    const int vv_count = static_cast<int>(count);
    vvtanhf(inner.data(), inner.data(), &vv_count);                   // inner = tanh(inner)
    vDSP_vsadd(inner.data(), 1, &one, inner.data(), 1, count);        // inner += 1
    vDSP_vmul(inner.data(), 1, x, 1, x, 1, count);                    // x = inner * x
    vDSP_vsmul(x, 1, &half, x, 1, count);                             // x *= 0.5
#else
    for (float & value : tensor.data()) {
        value = gelu_tanh(value);
    }
#endif
}

}  // namespace

//   hidden     = gelu_tanh(encoder_hidden_states @ linear_1_weight + linear_1_bias)
//   projected  = hidden @ linear_2_weight + linear_2_bias
//   out        = rms_norm(projected, norm_weight)
//   encoder_hidden_states: [batch, tokens, in_features]
//   linear_1_weight:       [in_features, hidden_features]
//   linear_1_bias:         [hidden_features] or nullptr
//   linear_2_weight:       [hidden_features, out_features]
//   linear_2_bias:         [out_features] or nullptr
//   norm_weight:           [out_features]
//   out:                   [batch, tokens, out_features]
Tensor project_caption_context(
    const Tensor & encoder_hidden_states,
    const Tensor & linear_1_weight,
    const Tensor * linear_1_bias,
    const Tensor & linear_2_weight,
    const Tensor * linear_2_bias,
    const Tensor & norm_weight
) {
    Tensor hidden = linear_3d_lastdim(encoder_hidden_states, linear_1_weight, linear_1_bias);
    gelu_tanh_inplace(hidden);
    Tensor projected = linear_3d_lastdim(hidden, linear_2_weight, linear_2_bias);
    return projected.rms_norm_3d_lastdim(1e-5f, &norm_weight);
}

// Splits the hidden dim into per-head slices for multi-head attention: reshape
// [hidden] into [num_heads, head_dim], then move num_heads in front of tokens.
//   out[b, head, t, d] = input[b, t, head*head_dim + d]
//   input:  [batch, tokens, hidden]  (hidden = num_heads * head_dim)
//   out:    [batch, num_heads, tokens, head_dim]
Tensor split_heads_3d(const Tensor & input, size_t num_heads) {
    if (input.rank() != 3) {
        throw std::invalid_argument("input must be rank 3");
    }
    if (num_heads == 0) {
        throw std::invalid_argument("num_heads must be greater than 0");
    }

    const size_t batch_size = input.dim_size(0);
    const size_t tokens = input.dim_size(1);
    const size_t hidden = input.dim_size(2);
    if (hidden % num_heads != 0) {
        throw std::invalid_argument("hidden size must be divisible by num_heads");
    }

    const size_t head_dim = hidden / num_heads;
    Tensor out({batch_size, num_heads, tokens, head_dim});
    const float * input_data = input.data().data();
    float * out_data = out.data().data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t head = 0; head < num_heads; ++head) {
#if defined(__APPLE__)
            vDSP_mmov(
                input_data + b * tokens * hidden + head * head_dim,
                out_data + (b * num_heads + head) * tokens * head_dim,
                head_dim,
                tokens,
                hidden,
                head_dim
            );
#else
            for (size_t t = 0; t < tokens; ++t) {
                std::copy(
                    input_data + (b * tokens + t) * hidden + head * head_dim,
                    input_data + (b * tokens + t) * hidden + head * head_dim + head_dim,
                    out_data + ((b * num_heads + head) * tokens + t) * head_dim
                );
            }
#endif
        }
    }
    return out;
}

// Inverse of split_heads_3d: move num_heads back after tokens, then merge
// [num_heads, head_dim] back into one [hidden] dim.
//   out[b, t, head*head_dim + d] = input[b, head, t, d]
//   input: [batch, num_heads, tokens, head_dim]
//   out:   [batch, tokens, hidden]  (hidden = num_heads * head_dim)
Tensor merge_heads_4d(const Tensor & input) {
    if (input.rank() != 4) {
        throw std::invalid_argument("input must be rank 4");
    }

    const size_t batch_size = input.dim_size(0);
    const size_t num_heads = input.dim_size(1);
    const size_t tokens = input.dim_size(2);
    const size_t head_dim = input.dim_size(3);
    const size_t hidden = num_heads * head_dim;

    Tensor out({batch_size, tokens, hidden});
    const float * input_data = input.data().data();
    float * out_data = out.data().data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t head = 0; head < num_heads; ++head) {
#if defined(__APPLE__)
            vDSP_mmov(
                input_data + (b * num_heads + head) * tokens * head_dim,
                out_data + b * tokens * hidden + head * head_dim,
                head_dim,
                tokens,
                head_dim,
                hidden
            );
#else
            for (size_t t = 0; t < tokens; ++t) {
                std::copy(
                    input_data + ((b * num_heads + head) * tokens + t) * head_dim,
                    input_data + ((b * num_heads + head) * tokens + t) * head_dim + head_dim,
                    out_data + (b * tokens + t) * hidden + head * head_dim
                );
            }
#endif
        }
    }
    return out;
}

// Raw (unscaled) attention scores. Per (batch, head): scores = q @ k^T.
//   scores[b, h, i, j] = sum_d q[b, h, i, d] * k[b, h, j, d]
//   q:      [batch, heads, q_tokens, head_dim]
//   k:      [batch, heads, k_tokens, head_dim]
//   out:    [batch, heads, q_tokens, k_tokens]
Tensor attention_scores_4d(const Tensor & q, const Tensor & k) {
    if (q.rank() != 4) {
        throw std::invalid_argument("q must be rank 4");
    }
    if (k.rank() != 4) {
        throw std::invalid_argument("k must be rank 4");
    }
    if (q.dim_size(0) != k.dim_size(0)) {
        throw std::invalid_argument("q and k batch size must match");
    }
    if (q.dim_size(1) != k.dim_size(1)) {
        throw std::invalid_argument("q and k head count must match");
    }
    if (q.dim_size(3) != k.dim_size(3)) {
        throw std::invalid_argument("q and k head_dim must match");
    }

    const size_t batch_size = q.dim_size(0);
    const size_t heads = q.dim_size(1);
    const size_t q_tokens = q.dim_size(2);
    const size_t k_tokens = k.dim_size(2);
    const size_t head_dim = q.dim_size(3);

    Tensor out({batch_size, heads, q_tokens, k_tokens});
    const std::vector<float> & q_data = q.data();
    const std::vector<float> & k_data = k.data();
    std::vector<float> & out_data = out.data();
    const size_t q_head_stride = q_tokens * head_dim;
    const size_t k_head_stride = k_tokens * head_dim;
    const size_t out_head_stride = q_tokens * k_tokens;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < heads;  ++h) {
            const size_t q_head_base = (b * heads + h) * q_head_stride;
            const size_t k_head_base = (b * heads + h) * k_head_stride;
            const size_t out_head_base = (b * heads + h) * out_head_stride;
            // O = Q * K^T
            gemm_row_major(
                q_data.data() + q_head_base,
                k_data.data() + k_head_base,
                nullptr,
                out_data.data() + out_head_base,
                q_tokens, head_dim, k_tokens,
                true
            );
        }
    }
    return out;
}

// Weighted sum of values by attention weights. Per (batch, head): out = weights @ v.
//   out[b, h, i, d] = sum_j weights[b, h, i, j] * v[b, h, j, d]
//   weights: [batch, heads, q_tokens, k_tokens]
//   v:       [batch, heads, k_tokens, head_dim]
//   out:     [batch, heads, q_tokens, head_dim]
Tensor attention_values_4d(const Tensor & weights, const Tensor & v) {
    if (weights.rank() != 4) {
        throw std::invalid_argument("weights must be rank 4");
    }
    if (v.rank() != 4) {
        throw std::invalid_argument("v must be rank 4");
    }
    if (weights.dim_size(0) != v.dim_size(0)) {
        throw std::invalid_argument("weights and v batch size must match");
    }
    if (weights.dim_size(1) != v.dim_size(1)) {
        throw std::invalid_argument("weights and v head count must match");
    }
    if (weights.dim_size(3) != v.dim_size(2)) {
        throw std::invalid_argument("weights k_tokens must match v token count");
    }

    const size_t batch_size = weights.dim_size(0);
    const size_t heads = weights.dim_size(1);
    const size_t q_tokens = weights.dim_size(2);
    const size_t k_tokens = weights.dim_size(3);
    const size_t head_dim = v.dim_size(3);
    Tensor out({batch_size, heads, q_tokens, head_dim});
    const std::vector<float> & weights_data = weights.data();
    const std::vector<float> & v_data = v.data();
    std::vector<float> & out_data = out.data();
    const size_t weights_head_stride = q_tokens * k_tokens;
    const size_t v_head_stride = k_tokens * head_dim;
    const size_t out_head_stride = q_tokens * head_dim;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < heads;  ++h) {
            const size_t weights_head_base = (b * heads + h) * weights_head_stride;
            const size_t v_head_base = (b * heads + h) * v_head_stride;
            const size_t out_head_base = (b * heads + h) * out_head_stride;
            // O = W * V
            gemm_row_major(weights_data.data() + weights_head_base, v_data.data() + v_head_base, nullptr, out_data.data() + out_head_base, q_tokens, k_tokens, head_dim);
        }
    }
    return out;

}

// Softmax over the last dimension of a rank-4 tensor (e.g. attention scores -> probabilities over k_tokens)
//   out[b, h, t, d] = exp(input[b,h,t,d] - max_d) / sum_d' exp(input[b,h,t,d'] - max_d)
//   input, out: [batch, heads, tokens, last_dim]  (softmax applied along last_dim)
Tensor softmax_lastdim_4d(const Tensor & input) {
    if (input.rank() != 4) {
        throw std::invalid_argument("input must be rank 4");
    }

    const size_t batch_size = input.dim_size(0);
    const size_t heads = input.dim_size(1);
    const size_t tokens = input.dim_size(2);
    const size_t head_dim = input.dim_size(3);
    Tensor out({batch_size, heads, tokens, head_dim});
    const std::vector<float> & input_data = input.data();
    std::vector<float> & out_data = out.data();
    const size_t head_stride = tokens * head_dim;

#if defined(__APPLE__)
    const bool use_accelerate_softmax = head_dim >= 4;
    std::vector<float> tmp(head_dim);
    const int count = static_cast<int>(head_dim);
#endif

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < heads;  ++h) {
            const size_t head_base = (b * heads + h) * head_stride;
            for (size_t t = 0; t < tokens; ++t) {
                const size_t row_base = head_base + t * head_dim;
#if defined(__APPLE__)
                if (use_accelerate_softmax) {
                    float max_value = 0.0f;
                    vDSP_maxv(input_data.data() + row_base, 1, &max_value, head_dim);
                    for (size_t d = 0; d < head_dim; ++d) {
                        tmp[d] = input_data[row_base + d] - max_value;
                    }
                    vvexpf(tmp.data(), tmp.data(), &count);
                    float denom = 0.0f;
                    vDSP_sve(tmp.data(), 1, &denom, head_dim);
                    const float recip = 1.0f / denom;
                    vDSP_vsmul(tmp.data(), 1, &recip, out_data.data() + row_base, 1, head_dim);
                    continue;
                }
#endif
                float max = input_data[row_base];
                for (size_t d = 0; d < head_dim; ++d) {
                    const float tmp_value = input_data[row_base + d];
                    if (tmp_value > max) {
                        max = tmp_value;
                    }
                }
                float denom = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    const float e = std::exp(input_data[row_base + d] - max);
                    out_data[row_base + d] = e;
                    denom += e;
                }
                const float recip = 1.0f / denom;
                for (size_t d = 0; d < head_dim; ++d) {
                    out_data[row_base + d] *= recip;
                }
            }
        }
    }
    return out;
}

// Standard scaled dot-product attention.
//   scores = (q @ k^T) / sqrt(head_dim) + bias
//   probs  = softmax_lastdim(scores)
//   out    = probs @ v
//   q, k, v:        [batch, heads, tokens, head_dim]
//   attention_bias: [batch, 1, key_tokens] or nullptr (e.g. caption padding mask)
//   out:            [batch, heads, q_tokens, head_dim]
Tensor attention_4d(
    const Tensor & q,
    const Tensor & k,
    const Tensor & v,
    const Tensor * attention_bias
) {
    Tensor scores = attention_scores_4d(q, k);
    scores.mul_inplace_scalar(1.0f/std::sqrt(q.dim_size(3)));

    if (attention_bias) {
        if (attention_bias->rank() != 3 ||
            attention_bias->dim_size(0) != scores.dim_size(0) ||
            attention_bias->dim_size(1) != 1 ||
            attention_bias->dim_size(2) != scores.dim_size(3)) {
            throw std::invalid_argument(
                "attention_bias shape must be {batch, 1, key_tokens}"
            );
        }

        std::vector<float> & score_data = scores.data();
        const std::vector<float> & bias_data = attention_bias->data();
        const size_t batch_size = scores.dim_size(0);
        const size_t heads = scores.dim_size(1);
        const size_t q_tokens = scores.dim_size(2);
        const size_t k_tokens = scores.dim_size(3);
        for (size_t b = 0; b < batch_size; ++b) {
            const float * bias_row = bias_data.data() + b * k_tokens;
            for (size_t h = 0; h < heads; ++h) {
                for (size_t q_tok = 0; q_tok < q_tokens; ++q_tok) {
                    float * score_row = score_data.data() + ((b * heads + h) * q_tokens + q_tok) * k_tokens;
#if defined(__APPLE__)
                    vDSP_vadd(score_row, 1, bias_row, 1, score_row, 1, k_tokens);
#else
                    for (size_t k_tok = 0; k_tok < k_tokens; ++k_tok) {
                        score_row[k_tok] += bias_row[k_tok];
                    }
#endif
                }
            }
        }
    }

    Tensor probs = softmax_lastdim_4d(scores);
    return attention_values_4d(probs, v);
}

// Linear attention (ReLU-kernel approximation)
//   phi(q), phi(k) = relu(q), relu(k)                     elementwise
//   S     = phi(k)^T @ v                                   [head_dim, head_dim]
//   ksum  = sum_t phi(k)[t, :]                             [head_dim]
//   out[t, :] = (phi(q)[t, :] @ S) / (phi(q)[t, :] . ksum + eps)
//   q:      [batch, heads, q_tokens, head_dim]
//   k, v:   [batch, heads, kv_tokens, head_dim]
//   out:    [batch, heads, q_tokens, head_dim]
Tensor linear_attention_4d(const Tensor & q, const Tensor & k, const Tensor & v) {
    if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4) {
        throw std::invalid_argument("q, k, and v must be rank 4");
    }
    if (q.dim_size(0) != k.dim_size(0) || q.dim_size(0) != v.dim_size(0)) {
        throw std::invalid_argument("q, k, and v batch size must match");
    }
    if (q.dim_size(1) != k.dim_size(1) || q.dim_size(1) != v.dim_size(1)) {
        throw std::invalid_argument("q, k, and v head count must match");
    }
    if (q.dim_size(3) != k.dim_size(3) || q.dim_size(3) != v.dim_size(3)) {
        throw std::invalid_argument("q, k, and v head_dim must match");
    }
    if (k.dim_size(2) != v.dim_size(2)) {
        throw std::invalid_argument("k and v token count must match");
    }

    const size_t batch_size = q.dim_size(0);
    const size_t heads = q.dim_size(1);
    const size_t q_tokens = q.dim_size(2);
    const size_t kv_tokens = k.dim_size(2);
    const size_t head_dim = q.dim_size(3);

    Tensor out({batch_size, heads, q_tokens, head_dim});
    const std::vector<float> & q_data = q.data();
    const std::vector<float> & k_data = k.data();
    const std::vector<float> & v_data = v.data();
    std::vector<float> & out_data = out.data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < heads; ++h) {
#if defined(__APPLE__)
            std::vector<float> k_relu(kv_tokens * head_dim);
            std::vector<float> q_relu(q_tokens * head_dim);
            std::vector<float> kv_summary(head_dim * head_dim);
            std::vector<float> k_sum(head_dim, 0.0f);

            const size_t k_base = (b * heads + h) * kv_tokens * head_dim;
            const size_t q_base = (b * heads + h) * q_tokens * head_dim;
            const size_t out_base = (b * heads + h) * q_tokens * head_dim;

            for (size_t i = 0; i < kv_tokens * head_dim; ++i) {
                k_relu[i] = std::max(0.0f, k_data[k_base + i]);
            }
            for (size_t i = 0; i < q_tokens * head_dim; ++i) {
                q_relu[i] = std::max(0.0f, q_data[q_base + i]);
            }

            cblas_sgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                static_cast<int>(head_dim),
                static_cast<int>(head_dim),
                static_cast<int>(kv_tokens),
                1.0f,
                k_relu.data(),
                static_cast<int>(head_dim),
                v_data.data() + k_base,
                static_cast<int>(head_dim),
                0.0f,
                kv_summary.data(),
                static_cast<int>(head_dim)
            );

            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(q_tokens),
                static_cast<int>(head_dim),
                static_cast<int>(head_dim),
                1.0f,
                q_relu.data(),
                static_cast<int>(head_dim),
                kv_summary.data(),
                static_cast<int>(head_dim),
                0.0f,
                out_data.data() + out_base,
                static_cast<int>(head_dim)
            );

            for (size_t t = 0; t < kv_tokens; ++t) {
                vDSP_vadd(k_sum.data(), 1, k_relu.data() + t * head_dim, 1, k_sum.data(), 1, head_dim);
            }

            for (size_t t = 0; t < q_tokens; ++t) {
                const size_t q_row_base = t * head_dim;
                const size_t out_row_base = out_base + q_row_base;
                float denom = 0.0f;
                vDSP_dotpr(q_relu.data() + q_row_base, 1, k_sum.data(), 1, &denom, head_dim);
                const float recip = 1.0f / (denom + 1e-15f);
                vDSP_vsmul(out_data.data() + out_row_base, 1, &recip, out_data.data() + out_row_base, 1, head_dim);
            }
#else
            Tensor scores({head_dim + 1, head_dim});
            scores.fill(0.0f);
            std::vector<float> & scores_data = scores.data();
            const size_t k_base = b * heads + h;
            const size_t v_base = b * heads + h;

            for (size_t t = 0; t < kv_tokens; ++t) {
                for (size_t d = 0; d < head_dim; ++d) {
                    const float key_value = std::max(0.0f, k_data[(k_base * kv_tokens + t) * head_dim + d]);

                    for (size_t row = 0; row < head_dim; ++row) {
                        const float value_component = v_data[(v_base * kv_tokens + t) * head_dim + row];
                        scores_data[row * head_dim + d] += value_component * key_value;
                    }

                    scores_data[head_dim * head_dim + d] += key_value;
                }
            }
            const size_t q_base = b * heads + h;
            const size_t out_base = b * heads + h;
            for (size_t t = 0; t < q_tokens; ++t) {
                for (size_t row = 0; row < head_dim; ++row) {
                    float numer = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        const float query_value = std::max(0.0f, q_data[(q_base * q_tokens + t) * head_dim + d]);
                        numer += scores_data[row * head_dim + d] * query_value;
                    }
                    out_data[(out_base * q_tokens + t) * head_dim + row] = numer;
                }

                float denom = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    const float query_value = std::max(0.0f, q_data[(q_base * q_tokens + t) * head_dim + d]);
                    denom += scores_data[head_dim * head_dim + d] * query_value;
                }
                denom += 1e-15f;

                for (size_t row = 0; row < head_dim; ++row) {
                    out_data[(out_base * q_tokens + t) * head_dim + row] /= denom;
                }
            }
#endif
        }
    }

    return out;
}

Tensor attention_block_3d(
    const Tensor & input,
    const Tensor & q_weight,
    const Tensor * q_bias,
    const Tensor & k_weight,
    const Tensor * k_bias,
    const Tensor & v_weight,
    const Tensor * v_bias,
    const Tensor & out_weight,
    const Tensor * out_bias,
    size_t num_heads
) {
    Tensor q_proj = linear_3d_lastdim(input, q_weight, q_bias);
    Tensor k_proj = linear_3d_lastdim(input, k_weight, k_bias);
    Tensor v_proj = linear_3d_lastdim(input, v_weight, v_bias);

    Tensor q_heads = split_heads_3d(q_proj, num_heads);
    Tensor k_heads = split_heads_3d(k_proj, num_heads);
    Tensor v_heads = split_heads_3d(v_proj, num_heads);

    Tensor attn_heads = linear_attention_4d(q_heads, k_heads, v_heads);
    Tensor merged = merge_heads_4d(attn_heads);

    return linear_3d_lastdim(merged, out_weight, out_bias);

}

Tensor cross_attention_block_3d(
    const Tensor & input,
    const Tensor & context,
    const Tensor * attention_bias,
    const Tensor & q_weight,
    const Tensor * q_bias,
    const Tensor & k_weight,
    const Tensor * k_bias,
    const Tensor & v_weight,
    const Tensor * v_bias,
    const Tensor & out_weight,
    const Tensor * out_bias,
    size_t num_heads
) {
    if (input.rank() != 3) {
        throw std::invalid_argument("input must be rank 3");
    }
    if (context.rank() != 3) {
        throw std::invalid_argument("context must be rank 3");
    }
    if (input.dim_size(0) != context.dim_size(0)) {
        throw std::invalid_argument("input and context batch size must match");
    }

    Tensor q_proj = linear_3d_lastdim(input, q_weight, q_bias);
    Tensor k_proj = linear_3d_lastdim(context, k_weight, k_bias);
    Tensor v_proj = linear_3d_lastdim(context, v_weight, v_bias);

    Tensor q_heads = split_heads_3d(q_proj, num_heads);
    Tensor k_heads = split_heads_3d(k_proj, num_heads);
    Tensor v_heads = split_heads_3d(v_proj, num_heads);

    Tensor attn_heads = attention_4d(q_heads, k_heads, v_heads, attention_bias);
    Tensor merged = merge_heads_4d(attn_heads);

    return linear_3d_lastdim(merged, out_weight, out_bias);
}
