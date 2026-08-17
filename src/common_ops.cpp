#include "common_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#endif

// C = A @ B + bias, or C = A @ B^T + bias if b_transposed
void gemm_row_major(
    const float* A,     // [M, K]
    const float* B,     // [K, N], or [N, K] if b_transposed
    const float* bias,  // [N] or nullptr
    float* C,           // [M, N]
    size_t M,
    size_t K,
    size_t N,
    bool b_transposed
) {
#if defined(__APPLE__)
    for (size_t m = 0; m < M; ++m) {
        float * c_row = C + m * N;
        if (bias) {
            cblas_scopy(static_cast<int>(N), bias, 1, c_row, 1);
        } else {
            std::fill_n(c_row, N, 0.0f);
        }
    }

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        b_transposed ? CblasTrans : CblasNoTrans,
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        1.0f,
        A,
        static_cast<int>(K),
        B,
        static_cast<int>(b_transposed ? K : N),
        1.0f,
        C,
        static_cast<int>(N)
    );
#else
    for (size_t m = 0; m < M; ++m) {
        const size_t C_base = m * N;
        const size_t A_base = m * K;
        for (size_t n = 0; n < N; ++n) {
            C[C_base + n] = bias ? bias[n] : 0.0f;
        }
        for (size_t k = 0; k < K; ++k) {
            const size_t B_base = k * N;
            const float a = A[A_base + k];
            for (size_t n = 0; n < N; ++n) {
                C[C_base + n] += a * (b_transposed ? B[n * K + k] : B[B_base + n]);
            }
        }
    }
#endif
}

// C += A @ B
void gemm_accumulate_row_major(
    const float* A,     // [M, K]
    const float* B,     // [K, N]
    float* C,           // [M, N], accumulated into
    size_t M,
    size_t K,
    size_t N
) {
#if defined(__APPLE__)
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        1.0f,
        A,
        static_cast<int>(K),
        B,
        static_cast<int>(N),
        1.0f,
        C,
        static_cast<int>(N)
    );
#else
    for (size_t m = 0; m < M; ++m) {
        const size_t C_base = m * N;
        const size_t A_base = m * K;
        for (size_t k = 0; k < K; ++k) {
            const size_t B_base = k * N;
            const float a = A[A_base + k];
            for (size_t n = 0; n < N; ++n) {
                C[C_base + n] += a * B[B_base + n];
            }
        }
    }
#endif
}

// out = input @ weight + bias
// out[row, out_feature] = bias[out_feature] + sum_in_feature input[row, in_feature] * weight[in_feature, out_feature]
//   input:  [rows, in_features]
//   weight: [in_features, out_features] -- note PyTorch's native nn.Linear.weight layout (out_features, in_features)
//   bias:   [out_features] or nullptr, broadcast-added to every row.
//   out:    [rows, out_features]
Tensor linear_2d(const Tensor & input, const Tensor & weight, const Tensor * bias) {
    if (input.rank() != 2) {
        throw std::invalid_argument("input must be rank 2");
    }
    if (weight.rank() != 2) {
        throw std::invalid_argument("weight must be rank 2");
    }
    if (bias && bias->rank() != 1) {
        throw std::invalid_argument("bias must be rank 1");
    }
    if (input.dim_size(1) != weight.dim_size(0)) {
        throw std::invalid_argument("inner dim must match for multiplication");
    }
    if (bias && bias->dim_size(0) != weight.dim_size(1)) {
        throw std::invalid_argument("dim for bias mismatch");
    }

    const size_t rows = input.dim_size(0);
    const size_t in_features = weight.dim_size(0);
    const size_t out_features = weight.dim_size(1);
    Tensor out({rows, out_features});

    const float * a = input.data().data();
    const float * b = weight.data().data();
    const float * bias_data = bias ? bias->data().data() : nullptr;
    float * c = out.data().data();

    gemm_row_major(a, b, bias_data, c, rows, in_features, out_features);
    return out;
}

// Two-layer MLP: up-projection,SiLU nonlinearity, down-projection.
//   hidden = silu(input @ up_weight + up_bias)
//   out    = hidden @ down_weight + down_bias
//   silu(x) = x * sigmoid(x) = x / (1 + exp(-x)), applied elementwise to `hidden`.
//   input:       [rows, in_features]
//   up_weight:   [in_features, hidden_features]
//   up_bias:     [hidden_features] or nullptr
//   down_weight: [hidden_features, out_features]
//   down_bias:   [out_features] or nullptr
//   out:         [rows, out_features]
Tensor mlp_2d(
    const Tensor & input,
    const Tensor & up_weight,
    const Tensor * up_bias,
    const Tensor & down_weight,
    const Tensor * down_bias
) {
    Tensor hidden = linear_2d(input, up_weight, up_bias);
    hidden.silu_inplace();
    return linear_2d(hidden, down_weight, down_bias);
}

// Same linear layer as linear_2d, applied independently to every (batch, token)
//   out[b, t, out_feature] = bias[out_feature]
//                            + sum_in_feature input[b, t, in_feature] * weight[in_feature, out_feature]
//   input:  [batch, tokens, in_features]
//   weight: [in_features, out_features]
//   bias:   [out_features] or nullptr, broadcast-added to every (batch, token) row.
//   out:    [batch, tokens, out_features]
// batch and tokens are flattened into one gemm_row_major "rows" dimension
// (m = batch * tokens) since the weight matrix is identical for every token
// therefore one GEMM call instead of `tokens` separate 2D ones.
Tensor linear_3d_lastdim(const Tensor & input, const Tensor & weight, const Tensor * bias) {
    if (input.rank() != 3) {
        throw std::invalid_argument("input must be rank 3");
    }

    if (weight.rank() != 2) {
        throw std::invalid_argument("weight must be rank 2");
    }

    if (bias && bias->rank() != 1) {
        throw std::invalid_argument("bias must be rank 1");
    }
    if (input.dim_size(2) != weight.dim_size(0)) {
        throw std::invalid_argument("inner dim must match for multiplication");
    }
    if (bias && bias->dim_size(0) != weight.dim_size(1)) {
        throw std::invalid_argument("dim for bias mismatch");
    }
    const size_t batch_size = input.dim_size(0);
    const size_t tokens = input.dim_size(1);
    const size_t out_features = weight.dim_size(1);
    const size_t in_features = weight.dim_size(0);
    Tensor out({batch_size, tokens, out_features});

    const size_t m = batch_size * tokens;
    const float * a = input.data().data();
    const float * b = weight.data().data();
    const float * bias_data = bias ? bias->data().data() : nullptr;
    float * c = out.data().data();

    gemm_row_major(a, b, bias_data, c, m, in_features, out_features);
    return out;
}

//   hidden = silu(input @ up_weight + up_bias)
//   out    = hidden @ down_weight + down_bias
//   input:       [batch, tokens, in_features]
//   up_weight:   [in_features, hidden_features]
//   up_bias:     [hidden_features] or nullptr
//   down_weight: [hidden_features, out_features]
//   down_bias:   [out_features] or nullptr
//   out:         [batch, tokens, out_features]
Tensor mlp_3d_lastdim(
    const Tensor & input,
    const Tensor & up_weight,
    const Tensor * up_bias,
    const Tensor & down_weight,
    const Tensor * down_bias
) {
    Tensor hidden = linear_3d_lastdim(input, up_weight, up_bias);
    hidden.silu_inplace();
    return linear_3d_lastdim(hidden, down_weight, down_bias);
}

namespace {

std::vector<float> transpose_conv1x1_weight(
    const std::vector<float> & weight_data,
    size_t out_channel,
    size_t in_channel
) {
    std::vector<float> transposed_weight(in_channel * out_channel);
    for (size_t oc = 0; oc < out_channel; ++oc) {
        for (size_t ic = 0; ic < in_channel; ++ic) {
            transposed_weight[ic * out_channel + oc] = weight_data[oc * in_channel + ic];
        }
    }
    return transposed_weight;
}

template <size_t OC_BLOCK, size_t SP_BLOCK>
// 1x1 conv (a per-pixel matmul across channels). Per batch:
//   out = transposed_weight^T @ input + bias  ([out_channel,in_channel] @ [in_channel,spatial])
// non-Apple fallback does this elementwise:
//   out[b, oc, s] = bias[oc] + sum_ic transposed_weight[ic, oc] * input[b, ic, s]
// OC_BLOCK x SP_BLOCK is the output tile size: small enough that its accumulator
// (c_tile) stays resident in registers/L1 across the entire reduction over
// in_channel, so the output buffer is written once per tile instead of being
// read-modify-written once per input channel.
//   input_data:        [batch, in_channel, spatial]
//   transposed_weight: [in_channel, out_channel]
//   bias_ptr:          [out_channel] or nullptr
//   out_data:          [batch, out_channel, spatial]
void conv2d_1x1_blocked(
    const float * input_data,
    const float * transposed_weight,
    const float * bias_ptr,
    float * out_data,
    size_t batch_size,
    size_t out_channel,
    size_t in_channel,
    size_t spatial
) {
    for (size_t b = 0; b < batch_size; ++b) {
        const float * input_batch = input_data + b * in_channel * spatial;
        float * output_batch = out_data + b * out_channel * spatial;

        for (size_t oc0 = 0; oc0 < out_channel; oc0 += OC_BLOCK) {
            const size_t oc_end = std::min(oc0 + OC_BLOCK, out_channel);
            const size_t oc_len = oc_end - oc0;

            for (size_t sp0 = 0; sp0 < spatial; sp0 += SP_BLOCK) {
                const size_t sp_end = std::min(sp0 + SP_BLOCK, spatial);
                const size_t sp_len = sp_end - sp0;

                float c_tile[OC_BLOCK][SP_BLOCK] = {};
                for (size_t oc = 0; oc < oc_len; ++oc) {
                    for (size_t s = 0; s < sp_len; ++s) {
                        c_tile[oc][s] = bias_ptr ? bias_ptr[oc0 + oc] : 0.0f;
                    }
                }

                for (size_t ic = 0; ic < in_channel; ++ic) {
                    const float * in_strip = input_batch + ic * spatial + sp0;
                    const float * w_row = transposed_weight + ic * out_channel + oc0;
                    for (size_t oc = 0; oc < oc_len; ++oc) {
                        const float w = w_row[oc];
                        for (size_t s = 0; s < sp_len; ++s) {
                            c_tile[oc][s] += w * in_strip[s];
                        }
                    }
                }

                for (size_t oc = 0; oc < oc_len; ++oc) {
                    float * out_row = output_batch + (oc0 + oc) * spatial + sp0;
                    for (size_t s = 0; s < sp_len; ++s) {
                        out_row[s] = c_tile[oc][s];
                    }
                }
            }
        }
    }
}

}  // namespace

// A 1x1 conv applied independently to every pixel. Per batch this is a single GEMM:
//   out = weight @ input + bias  ([out_channel,in_channel] @ [in_channel,spatial])
// elementwise:
//   out[b, oc, h, w] = bias[oc] + sum_ic weight[oc, ic] * input[b, ic, h, w]
//   
//   input:  [batch, in_channels, height, width]
//   weight: [out_channels, in_channels, 1, 1]
//   bias:   [out_channels] or nullptr, broadcast-added to every pixel.
//   out:    [batch, out_channels, height, width]
Tensor conv2d_1x1(const Tensor & input, const Tensor & weight, const Tensor * bias) {
    if (input.rank() != 4) {
        throw std::invalid_argument("input must be rank 4");
    }
    if (weight.rank() != 4) {
        throw std::invalid_argument("weight must be rank 4");
    }
    if (weight.dim_size(2) != 1 || weight.dim_size(3) != 1) {
        throw std::invalid_argument("weight kernel must be 1x1");
    }
    if (input.dim_size(1) != weight.dim_size(1)) {
        throw std::invalid_argument("input channels must match weight in_channels");
    }
    if (bias) {
        if (bias->rank() != 1) {
            throw std::invalid_argument("bias must be rank 1");
        }
        if (bias->dim_size(0) != weight.dim_size(0)) {
            throw std::invalid_argument("bias size must match out_channels");
        }
    }

    const size_t out_channel = weight.dim_size(0);
    const size_t in_channel = weight.dim_size(1);
    const size_t batch_size = input.dim_size(0);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t spatial = height * width;
    Tensor out({batch_size, out_channel, height, width});
    const std::vector<float> & weight_data = weight.data();
    const std::vector<float> & input_data = input.data();
    std::vector<float> & out_data = out.data();
    const float* bias_ptr = bias ? bias->data().data() : nullptr;

#if defined(__APPLE__)
    for (size_t b = 0; b < batch_size; ++b) {
        const float * input_batch = input_data.data() + b * in_channel * spatial;
        float * output_batch = out_data.data() + b * out_channel * spatial;

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            static_cast<int>(out_channel),
            static_cast<int>(spatial),
            static_cast<int>(in_channel),
            1.0f,
            weight_data.data(),
            static_cast<int>(in_channel),
            input_batch,
            static_cast<int>(spatial),
            0.0f,
            output_batch,
            static_cast<int>(spatial)
        );

        if (bias_ptr) {
            for (size_t oc = 0; oc < out_channel; ++oc) {
                float * out_row = output_batch + oc * spatial;
                const float bias_value = bias_ptr[oc];
                for (size_t s = 0; s < spatial; ++s) {
                    out_row[s] += bias_value;
                }
            }
        }
    }
    return out;
#else
    std::vector<float> transposed_weight =
        transpose_conv1x1_weight(weight_data, out_channel, in_channel);

    conv2d_1x1_blocked<8, 64>(
        input_data.data(),
        transposed_weight.data(),
        bias_ptr,
        out_data.data(),
        batch_size,
        out_channel,
        in_channel,
        spatial
    );
    return out;
#endif
}
// Depthwise 3x3 conv, same (zero) padding. High level: for every channel c,
//   out[c] = conv2d(input[c], weight[c]) + bias[c]
// each channel convolved independently with its own single 3x3 filter.
//   out[b, c, y, x] = bias[c]
//                    + sum_ky sum_kx weight[c, 0, ky, kx] * input[b, c, y+ky-1, x+kx-1]
//   (input indices outside [0,height) x [0,width) contribute 0 -- zero padding)
//   input:  [batch, channels, height, width]
//   weight: [channels, 1, 3, 3] -- one 3x3 filter per channel, no in_channel dim
//   bias:   [channels] or nullptr
//   out:    [batch, channels, height, width]
namespace {

struct DepthwiseConvPlaneContext {
    const float * input_data;
    const float * weight_data;
    float * out_data;
    const float * bias_data;  // or nullptr
    size_t channel;
    size_t height;
    size_t width;
    size_t filter_size;
};

// Computes one (b,c) output plane in full
void depthwise_conv3x3_process_plane(void * raw_context, size_t idx) {
    const auto * ctx = static_cast<const DepthwiseConvPlaneContext *>(raw_context);
    const size_t channel = ctx->channel;
    const size_t height = ctx->height;
    const size_t width = ctx->width;
    const size_t filter_size = ctx->filter_size;
    const size_t b = idx / channel;
    const size_t c = idx % channel;

#if defined(__APPLE__)
    const bool use_accelerate_depthwise = height >= 3 && width >= 3;
#else
    const bool use_accelerate_depthwise = false;
#endif
    if (use_accelerate_depthwise) {
#if defined(__APPLE__)
        const float * input_plane = ctx->input_data + (b * channel + c) * height * width;
        float * output_plane = ctx->out_data + (b * channel + c) * height * width;
        const float * filter = ctx->weight_data + c * filter_size * filter_size;
        vDSP_f3x3(input_plane, height, width, filter, output_plane);

        if (ctx->bias_data) {
            const float bias_value = ctx->bias_data[c];
            for (size_t i = 0; i < height * width; ++i) {
                output_plane[i] += bias_value;
            }
        }

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                if (y > 0 && y + 1 < height && x > 0 && x + 1 < width) {
                    continue;
                }
                float sum = 0.0f;
                for (size_t ky = 0; ky < filter_size; ++ky) {
                    for (size_t kx = 0; kx < filter_size; ++kx) {
                        const int in_y = static_cast<int>(y) + static_cast<int>(ky) - 1;
                        const int in_x = static_cast<int>(x) + static_cast<int>(kx) - 1;
                        if (in_y >= 0 && in_y < static_cast<int>(height) &&
                            in_x >= 0 && in_x < static_cast<int>(width)) {
                            sum += input_plane[static_cast<size_t>(in_y) * width +
                                               static_cast<size_t>(in_x)] *
                                filter[ky * filter_size + kx];
                        }
                    }
                }
                if (ctx->bias_data) {
                    sum += ctx->bias_data[c];
                }
                output_plane[y * width + x] = sum;
            }
        }
#endif
    } else {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (size_t ky = 0; ky < filter_size; ++ky) {
                    for (size_t kx = 0; kx < filter_size; ++kx) {
                        const int in_y = static_cast<int>(y) + static_cast<int>(ky) - 1;
                        const int in_x = static_cast<int>(x) + static_cast<int>(kx) - 1;
                        if (in_y >= 0 && in_y < static_cast<int>(height) &&
                            in_x >= 0 && in_x < static_cast<int>(width)) {
                            sum += ctx->input_data[((b * channel + c) * height +
                                static_cast<size_t>(in_y)) * width + static_cast<size_t>(in_x)] *
                                ctx->weight_data[(c * filter_size + ky) * filter_size + kx];
                        }
                    }
                }
                if (ctx->bias_data) {
                    sum += ctx->bias_data[c];
                }
                ctx->out_data[((b * channel + c) * height + y) * width + x] = sum;
            }
        }
    }
}

}  // namespace

Tensor depthwise_conv2d_3x3(const Tensor & input, const Tensor & weight, const Tensor * bias) {
    if (input.rank() != 4) {
        throw std::invalid_argument("input must be rank 4");
    }
    if (weight.rank() != 4) {
        throw std::invalid_argument("weight must be rank 4");
    }
    if (weight.dim_size(1) != 1 || weight.dim_size(2) != 3 || weight.dim_size(3) != 3) {
        throw std::invalid_argument("weight shape must be {channels, 1, 3, 3}");
    }

    const size_t channel = input.dim_size(1);
    if (weight.dim_size(0) != channel) {
        throw std::invalid_argument("weight channels must match input channels");
    }
    if (bias) {
        if (bias->rank() != 1) {
            throw std::invalid_argument("bias must be rank 1");
        }
        if (bias->dim_size(0) != channel) {
            throw std::invalid_argument("bias size must match channels");
        }
    }

    const size_t batch_size = input.dim_size(0);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t filter_size = weight.dim_size(2);
    Tensor out({batch_size, channel, height, width});
    const std::vector<float> & input_data = input.data();
    const std::vector<float> & weight_data = weight.data();
    std::vector<float> & out_data = out.data();
    const std::vector<float> * bias_data = bias ? &bias->data() : nullptr;

    DepthwiseConvPlaneContext ctx{
        input_data.data(),
        weight_data.data(),
        out_data.data(),
        bias_data ? bias_data->data() : nullptr,
        channel,
        height,
        width,
        filter_size,
    };
    const size_t total_planes = batch_size * channel;

#if defined(__APPLE__)
    constexpr size_t kParallelDepthwiseThreshold = 8;
    if (total_planes >= kParallelDepthwiseThreshold) {
        dispatch_apply_f(total_planes, DISPATCH_APPLY_AUTO, &ctx, depthwise_conv3x3_process_plane);
    } else {
        for (size_t idx = 0; idx < total_planes; ++idx) {
            depthwise_conv3x3_process_plane(&ctx, idx);
        }
    }
#else
    for (size_t idx = 0; idx < total_planes; ++idx) {
        depthwise_conv3x3_process_plane(&ctx, idx);
    }
#endif

    return out;
}

namespace {

void depthwise_conv5x5_process_plane(void * raw_context, size_t idx) {
    const auto * ctx = static_cast<const DepthwiseConvPlaneContext *>(raw_context);
    const size_t channel = ctx->channel;
    const size_t height = ctx->height;
    const size_t width = ctx->width;
    const size_t filter_size = ctx->filter_size;
    const int pad = static_cast<int>(filter_size / 2);
    const size_t b = idx / channel;
    const size_t c = idx % channel;

#if defined(__APPLE__)
    const bool use_accelerate_depthwise = height >= 5 && width >= 5;
#else
    const bool use_accelerate_depthwise = false;
#endif
    if (use_accelerate_depthwise) {
#if defined(__APPLE__)
        const float * input_plane = ctx->input_data + (b * channel + c) * height * width;
        float * output_plane = ctx->out_data + (b * channel + c) * height * width;
        const float * filter = ctx->weight_data + c * filter_size * filter_size;
        vDSP_f5x5(input_plane, height, width, filter, output_plane);

        if (ctx->bias_data) {
            const float bias_value = ctx->bias_data[c];
            for (size_t i = 0; i < height * width; ++i) {
                output_plane[i] += bias_value;
            }
        }

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                if (y >= 2 && y + 2 < height && x >= 2 && x + 2 < width) {
                    continue;
                }
                float sum = 0.0f;
                for (size_t ky = 0; ky < filter_size; ++ky) {
                    for (size_t kx = 0; kx < filter_size; ++kx) {
                        const int in_y = static_cast<int>(y) + static_cast<int>(ky) - pad;
                        const int in_x = static_cast<int>(x) + static_cast<int>(kx) - pad;
                        if (in_y >= 0 && in_y < static_cast<int>(height) &&
                            in_x >= 0 && in_x < static_cast<int>(width)) {
                            sum += input_plane[static_cast<size_t>(in_y) * width +
                                               static_cast<size_t>(in_x)] *
                                filter[ky * filter_size + kx];
                        }
                    }
                }
                if (ctx->bias_data) {
                    sum += ctx->bias_data[c];
                }
                output_plane[y * width + x] = sum;
            }
        }
#endif
    } else {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (size_t ky = 0; ky < filter_size; ++ky) {
                    for (size_t kx = 0; kx < filter_size; ++kx) {
                        const int in_y = static_cast<int>(y) + static_cast<int>(ky) - pad;
                        const int in_x = static_cast<int>(x) + static_cast<int>(kx) - pad;
                        if (in_y >= 0 && in_y < static_cast<int>(height) &&
                            in_x >= 0 && in_x < static_cast<int>(width)) {
                            sum += ctx->input_data[((b * channel + c) * height +
                                static_cast<size_t>(in_y)) * width + static_cast<size_t>(in_x)] *
                                ctx->weight_data[(c * filter_size + ky) * filter_size + kx];
                        }
                    }
                }
                if (ctx->bias_data) {
                    sum += ctx->bias_data[c];
                }
                ctx->out_data[((b * channel + c) * height + y) * width + x] = sum;
            }
        }
    }
}

}  // namespace

// same operations as depthwise_conv2d_3x3 but with 5x5 filters
Tensor depthwise_conv2d_5x5(const Tensor & input, const Tensor & weight, const Tensor * bias) {
    if (input.rank() != 4) {
        throw std::invalid_argument("input must be rank 4");
    }
    if (weight.rank() != 4) {
        throw std::invalid_argument("weight must be rank 4");
    }
    if (weight.dim_size(1) != 1 || weight.dim_size(2) != 5 || weight.dim_size(3) != 5) {
        throw std::invalid_argument("weight shape must be {channels, 1, 5, 5}");
    }

    const size_t channel = input.dim_size(1);
    if (weight.dim_size(0) != channel) {
        throw std::invalid_argument("weight channels must match input channels");
    }
    if (bias) {
        if (bias->rank() != 1) {
            throw std::invalid_argument("bias must be rank 1");
        }
        if (bias->dim_size(0) != channel) {
            throw std::invalid_argument("bias size must match channels");
        }
    }

    const size_t batch_size = input.dim_size(0);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t filter_size = weight.dim_size(2);
    Tensor out({batch_size, channel, height, width});
    const std::vector<float> & input_data = input.data();
    const std::vector<float> & weight_data = weight.data();
    std::vector<float> & out_data = out.data();
    const std::vector<float> * bias_data = bias ? &bias->data() : nullptr;

    DepthwiseConvPlaneContext ctx{
        input_data.data(),
        weight_data.data(),
        out_data.data(),
        bias_data ? bias_data->data() : nullptr,
        channel,
        height,
        width,
        filter_size,
    };
    const size_t total_planes = batch_size * channel;

#if defined(__APPLE__)
    constexpr size_t kParallelDepthwiseThreshold = 8;
    if (total_planes >= kParallelDepthwiseThreshold) {
        dispatch_apply_f(total_planes, DISPATCH_APPLY_AUTO, &ctx, depthwise_conv5x5_process_plane);
    } else {
        for (size_t idx = 0; idx < total_planes; ++idx) {
            depthwise_conv5x5_process_plane(&ctx, idx);
        }
    }
#else
    for (size_t idx = 0; idx < total_planes; ++idx) {
        depthwise_conv5x5_process_plane(&ctx, idx);
    }
#endif

    return out;
}

// GLUMBConv: a Gated Linear Unit wrapped around an inverted-bottleneck MBConv.
//   hidden      = silu(conv1x1_inverted(input))       -- expand channels
//   hidden      = depthwise_conv3x3(hidden)            -- per-channel spatial mixing
//   data, gate  = split_channels(hidden)                -- split expanded channels in half
//   out         = conv1x1_point(data * silu(gate))     -- GLU gate, then project down
//   input:               [batch, in_channels, H, W]
//   conv_inverted_weight: [expanded_channels, in_channels, 1, 1] -- expanded_channels must
//                          be even (split in half below)
//   conv_depth_weight:    [expanded_channels, 1, 3, 3] -- depthwise, channel count unchanged
//   conv_point_weight:    [out_channels, expanded_channels/2, 1, 1] -- no bias
//   out:                 [batch, out_channels, H, W]
Tensor glumb_conv(
    const Tensor & input,
    const Tensor & conv_inverted_weight,
    const Tensor * conv_inverted_bias,
    const Tensor & conv_depth_weight,
    const Tensor * conv_depth_bias,
    const Tensor & conv_point_weight
) {
    Tensor hidden = conv2d_1x1(input, conv_inverted_weight, conv_inverted_bias);
    hidden.silu_inplace();
    hidden = depthwise_conv2d_3x3(hidden, conv_depth_weight, conv_depth_bias);

    auto split = hidden.split_channels_4d();
    Tensor data = split.first;
    Tensor gate = split.second;
    gate.silu_inplace();

    if (!data.same_shape(gate)) {
        throw std::invalid_argument("data and gate shapes must match");
    }

    std::vector<float> & data_values = data.data();
    const std::vector<float> & gate_values = gate.data();
    for (size_t i = 0; i < data.numel(); ++i) {
        data_values[i] *= gate_values[i];
    }

    return conv2d_1x1(data, conv_point_weight, nullptr);
}
