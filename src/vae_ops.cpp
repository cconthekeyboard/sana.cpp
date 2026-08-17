#include "vae_ops.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "common_ops.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

//   im2col = flatten each 3x3 receptive field into a column      [in_channels*9, spatial]
//   out    = weight @ im2col + bias                              [out_channels, spatial], per batch
//   out[b,oc,y,x] = bias[oc] + sum_{ic,ky,kx} weight[oc,ic,ky,kx] * input[b,ic,y+ky-1,x+kx-1]
//   input:   [batch, in_channels, height, width]
//   weight:  [out_channels, in_channels, 3, 3]
//   bias:    [out_channels] or nullptr
//   out:     [batch, out_channels, height, width]
Tensor conv2d_3x3_im2col_gemm(
    const Tensor & input,
    const Tensor & weight,
    const Tensor * bias
) {
    const size_t batch = input.dim_size(0);
    const size_t in_channels = input.dim_size(1);
    const size_t out_channels = weight.dim_size(0);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t spatial = height * width;
    const size_t k_dim = in_channels * 9;

    const float * weight_data = weight.data().data();
    const float * input_data = input.data().data();
    const float * bias_ptr = bias ? bias->data().data() : nullptr;

    Tensor out({batch, out_channels, height, width});
    float * out_data = out.data().data();
    std::vector<float> im2col_t(k_dim * spatial);

    for (size_t b = 0; b < batch; ++b) {
        const float * input_batch = input_data + b * in_channels * spatial;

        for (size_t ic = 0; ic < in_channels; ++ic) {
            const float * input_channel = input_batch + ic * spatial;
            for (size_t ky = 0; ky < 3; ++ky) {
                const int dy = static_cast<int>(ky) - 1;
                for (size_t kx = 0; kx < 3; ++kx) {
                    const int dx = static_cast<int>(kx) - 1;
                    const size_t k_index = (ic * 3 + ky) * 3 + kx;
                    float * row_ptr = im2col_t.data() + k_index * spatial;

                    for (size_t y = 0; y < height; ++y) {
                        const int in_y = static_cast<int>(y) + dy;
                        float * dst_row = row_ptr + y * width;
                        if (in_y < 0 || in_y >= static_cast<int>(height)) {
                            std::fill_n(dst_row, width, 0.0f);
                            continue;
                        }
                        const float * src_row = input_channel + static_cast<size_t>(in_y) * width;
                        if (dx == 0) {
                            std::copy(src_row, src_row + width, dst_row);
                        } else if (dx < 0) {
                            dst_row[0] = 0.0f;
                            std::copy(src_row, src_row + (width - 1), dst_row + 1);
                        } else {
                            std::copy(src_row + 1, src_row + width, dst_row);
                            dst_row[width - 1] = 0.0f;
                        }
                    }
                }
            }
        }

        float * out_batch = out_data + b * out_channels * spatial;
        gemm_row_major(weight_data, im2col_t.data(), nullptr, out_batch, out_channels, k_dim, spatial);

        if (bias_ptr) {
            for (size_t oc = 0; oc < out_channels; ++oc) {
                float * out_row = out_batch + oc * spatial;
                const float bias_value = bias_ptr[oc];
#if defined(__APPLE__)
                vDSP_vsadd(out_row, 1, &bias_value, out_row, 1, spatial);
#else
                for (size_t s = 0; s < spatial; ++s) {
                    out_row[s] += bias_value;
                }
#endif
            }
        }
    }

    return out;
}

//   for each of the 9 taps (ky,kx): shifted = input shifted by (ky-1,kx-1), zero-padded  [in_channels, spatial]
//   out += weight_tap @ shifted, accumulated over all 9 taps                              [out_channels, spatial], per batch
//   out[b,oc,y,x] = bias[oc] + sum_{ic,ky,kx} weight[oc,ic,ky,kx] * input[b,ic,y+ky-1,x+kx-1]
//   input:   [batch, in_channels, height, width]
//   weight:  [out_channels, in_channels, 3, 3]
//   bias:    [out_channels] or nullptr
//   out:     [batch, out_channels, height, width]
Tensor conv2d_3x3_tap_accumulate(
    const Tensor & input,
    const Tensor & weight,
    const Tensor * bias
) {
    const size_t batch = input.dim_size(0);
    const size_t in_channels = input.dim_size(1);
    const size_t out_channels = weight.dim_size(0);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t spatial = height * width;

    const float * weight_data = weight.data().data();
    const float * input_data = input.data().data();
    const float * bias_ptr = bias ? bias->data().data() : nullptr;

    Tensor out({batch, out_channels, height, width});
    float * out_data = out.data().data();

    std::vector<float> shifted(in_channels * spatial);
    std::vector<float> weight_tap(out_channels * in_channels);

    for (size_t b = 0; b < batch; ++b) {
        const float * input_batch = input_data + b * in_channels * spatial;
        float * out_batch = out_data + b * out_channels * spatial;
        for (size_t oc = 0; oc < out_channels; ++oc) {
            const float bias_value = bias_ptr ? bias_ptr[oc] : 0.0f;
#if defined(__APPLE__)
            vDSP_vfill(&bias_value, out_batch + oc * spatial, 1, spatial);
#else
            std::fill_n(out_batch + oc * spatial, spatial, bias_value);
#endif
        }

        for (size_t ky = 0; ky < 3; ++ky) {
            const int dy = static_cast<int>(ky) - 1;
            for (size_t kx = 0; kx < 3; ++kx) {
                const int dx = static_cast<int>(kx) - 1;

                for (size_t ic = 0; ic < in_channels; ++ic) {
                    const float * input_channel = input_batch + ic * spatial;
                    float * dst_channel = shifted.data() + ic * spatial;
                    for (size_t y = 0; y < height; ++y) {
                        const int in_y = static_cast<int>(y) + dy;
                        float * dst_row = dst_channel + y * width;
                        if (in_y < 0 || in_y >= static_cast<int>(height)) {
                            std::fill_n(dst_row, width, 0.0f);
                            continue;
                        }
                        const float * src_row = input_channel + static_cast<size_t>(in_y) * width;
                        if (dx == 0) {
                            std::copy(src_row, src_row + width, dst_row);
                        } else if (dx < 0) {
                            dst_row[0] = 0.0f;
                            std::copy(src_row, src_row + (width - 1), dst_row + 1);
                        } else {
                            std::copy(src_row + 1, src_row + width, dst_row);
                            dst_row[width - 1] = 0.0f;
                        }
                    }
                }

                for (size_t oc = 0; oc < out_channels; ++oc) {
                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        weight_tap[oc * in_channels + ic] =
                            weight_data[(oc * in_channels + ic) * 9 + ky * 3 + kx];
                    }
                }

                gemm_accumulate_row_major(
                    weight_tap.data(),
                    shifted.data(),
                    out_batch,
                    out_channels,
                    in_channels,
                    spatial
                );
            }
        }
    }

    return out;
}

Tensor conv2d_3x3_same_dispatch(
    const Tensor & input,
    const Tensor & weight,
    const Tensor * bias
) {
    if (input.rank() != 4 || weight.rank() != 4) {
        throw std::invalid_argument("conv2d_3x3_same_dispatch expects rank-4 input and weight");
    }
    if (weight.dim_size(2) != 3 || weight.dim_size(3) != 3) {
        throw std::invalid_argument("conv2d_3x3_same_dispatch requires 3x3 kernels");
    }
    if (weight.dim_size(1) != input.dim_size(1)) {
        throw std::invalid_argument("conv2d_3x3_same_dispatch weight in_channels must match input channels");
    }
    if (bias && (bias->rank() != 1 || bias->dim_size(0) != weight.dim_size(0))) {
        throw std::invalid_argument("conv2d_3x3_same_dispatch bias must match output channels");
    }
    constexpr size_t kTapAccumulateThresholdFloats = 900'000'000;
    const size_t im2col_floats =
        input.dim_size(1) * 9 * input.dim_size(2) * input.dim_size(3);

    if (im2col_floats > kTapAccumulateThresholdFloats) {
        return conv2d_3x3_tap_accumulate(input, weight, bias);
    }
    return conv2d_3x3_im2col_gemm(input, weight, bias);
}


Tensor conv2d_same(const Tensor & input, const Tensor & weight, const Tensor * bias) {
    if (input.rank() != 4 || weight.rank() != 4) {
        throw std::invalid_argument("conv2d_same expects rank-4 input and weight");
    }
    if (weight.dim_size(1) != input.dim_size(1)) {
        throw std::invalid_argument("conv2d_same weight in_channels must match input channels");
    }
    const size_t kernel_h = weight.dim_size(2);
    const size_t kernel_w = weight.dim_size(3);
    if (kernel_h == 0 || kernel_w == 0 || kernel_h % 2 == 0 || kernel_w % 2 == 0) {
        throw std::invalid_argument("conv2d_same requires odd non-zero kernels");
    }
    if (bias && (bias->rank() != 1 || bias->dim_size(0) != weight.dim_size(0))) {
        throw std::invalid_argument("conv2d_same bias must match output channels");
    }

    if (kernel_h == 1 && kernel_w == 1) {
        return conv2d_1x1(input, weight, bias);
    }
    if (kernel_h == 3 && kernel_w == 3) {
        return conv2d_3x3_same_dispatch(input, weight, bias);
    }

    throw std::invalid_argument("conv2d_same only supports 1x1 or 3x3 kernels");
}

Tensor depthwise_conv2d_same(const Tensor & input, const Tensor & weight, const Tensor * bias) {
    if (input.rank() != 4 || weight.rank() != 4) {
        throw std::invalid_argument("depthwise_conv2d_same expects rank-4 input and weight");
    }
    if (weight.dim_size(1) != 1 || weight.dim_size(0) != input.dim_size(1)) {
        throw std::invalid_argument("depthwise_conv2d_same weight shape must be {channels,1,kh,kw}");
    }
    const size_t kernel_h = weight.dim_size(2);
    const size_t kernel_w = weight.dim_size(3);
    if (kernel_h == 0 || kernel_w == 0 || kernel_h % 2 == 0 || kernel_w % 2 == 0) {
        throw std::invalid_argument("depthwise_conv2d_same requires odd non-zero kernels");
    }
    if (bias && (bias->rank() != 1 || bias->dim_size(0) != input.dim_size(1))) {
        throw std::invalid_argument("depthwise_conv2d_same bias must match channels");
    }

    if (kernel_h == 3 && kernel_w == 3) {
        return depthwise_conv2d_3x3(input, weight, bias);
    }
    if (kernel_h == 5 && kernel_w == 5) {
        return depthwise_conv2d_5x5(input, weight, bias);
    }

    throw std::invalid_argument("depthwise_conv2d_same only supports 3x3 or 5x5 kernels");
}

//   groups partition channels: group g maps input channels [g*in_per_group, (g+1)*in_per_group)
//   to output channels [g*out_per_group, (g+1)*out_per_group) via its own 1x1 conv, no cross-group mixing
//   out[b,oc,y,x] = sum_{ic in group(oc)} weight[oc,ic-in_base,0,0] * input[b,ic,y,x]
//   input:   [batch, in_channels, height, width]
//   weight:  [out_channels, in_channels/groups, 1, 1]
//   groups:  in_channels and out_channels must each be divisible by groups
//   out:     [batch, out_channels, height, width]
Tensor grouped_conv2d_1x1(
    const Tensor & input,
    const Tensor & weight,
    size_t groups
) {
    if (input.rank() != 4 || weight.rank() != 4) {
        throw std::invalid_argument("grouped_conv2d_1x1 expects rank-4 input and weight");
    }
    if (groups == 0) {
        throw std::invalid_argument("groups must be positive");
    }
    if (weight.dim_size(2) != 1 || weight.dim_size(3) != 1) {
        throw std::invalid_argument("grouped_conv2d_1x1 requires 1x1 kernels");
    }
    const size_t in_channels = input.dim_size(1);
    const size_t out_channels = weight.dim_size(0);
    const size_t in_per_group = weight.dim_size(1);
    if (in_channels % groups != 0 || out_channels % groups != 0) {
        throw std::invalid_argument("input/output channels must be divisible by groups");
    }
    if (in_channels / groups != in_per_group) {
        throw std::invalid_argument("grouped_conv2d_1x1 weight in_channels do not match grouped input");
    }

    const size_t out_per_group = out_channels / groups;
    const size_t batch = input.dim_size(0);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t spatial = height * width;
    Tensor out({batch, out_channels, height, width});

    const float * weight_data = weight.data().data();
    const float * input_data = input.data().data();
    float * out_data = out.data().data();

    for (size_t g = 0; g < groups; ++g) {
        const size_t out_base = g * out_per_group;
        const size_t in_base = g * in_per_group;
        const float * weight_group = weight_data + out_base * in_per_group;

        for (size_t b = 0; b < batch; ++b) {
            const float * input_group = input_data + (b * in_channels + in_base) * spatial;
            float * out_group = out_data + (b * out_channels + out_base) * spatial;

            gemm_row_major(
                weight_group,
                input_group,
                nullptr,
                out_group,
                out_per_group,
                in_per_group,
                spatial
            );
        }
    }

    return out;
}

//  each input pixel is replicated into the 2x2 output block it maps to (nearest-neighbor,
//  no interpolation): out[b,c,2y,2x] = out[b,c,2y,2x+1] = out[b,c,2y+1,2x] = out[b,c,2y+1,2x+1] = input[b,c,y,x]
//  input:  [batch, channels, height, width]
//  out:    [batch, channels, height*2, width*2]
Tensor nearest_upsample_2x(const Tensor & input) {
    if (input.rank() != 4) {
        throw std::invalid_argument("nearest_upsample_2x expects rank-4 input");
    }
    const size_t batch = input.dim_size(0);
    const size_t channels = input.dim_size(1);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t out_width = width * 2;
    const size_t in_spatial = height * width;
    const size_t out_spatial = (height * 2) * out_width;
    Tensor out({batch, channels, height * 2, width * 2});
    const float * input_data = input.data().data();
    float * out_data = out.data().data();
    for (size_t b = 0; b < batch; ++b) {
        const float * input_batch = input_data + b * channels * in_spatial;
        float * out_batch = out_data + b * channels * out_spatial;
        for (size_t c = 0; c < channels; ++c) {
            const float * input_channel = input_batch + c * in_spatial;
            float * out_channel = out_batch + c * out_spatial;
            for (size_t y = 0; y < height; ++y) {
                const float * input_row = input_channel + y * width;
                float * out_row0 = out_channel + (y * 2) * out_width;
                float * out_row1 = out_row0 + out_width;
#if defined(__APPLE__)
                cblas_scopy(static_cast<int>(width), input_row, 1, out_row0, 2);
                cblas_scopy(static_cast<int>(width), input_row, 1, out_row0 + 1, 2);
                std::copy(out_row0, out_row0 + out_width, out_row1);
#else
                for (size_t x = 0; x < width; ++x) {
                    const float value = input_row[x];
                    const size_t out_x = x * 2;
                    out_row0[out_x] = value;
                    out_row0[out_x + 1] = value;
                    out_row1[out_x] = value;
                    out_row1[out_x + 1] = value;
                }
#endif
            }
        }
    }
    return out;
}

//   adds a residual shortcut into an already 2x-upsampled tensor: each output channel's
//   2x2 block accumulates from up to 4 input channels, repeated across output channels
//   when out_channels*4 > in_channels (repeats = out_channels*4 / in_channels)
//   upsampled[b,oc,2y,2x]     += input[b, (oc*4+0)/repeats, y, x]
//   upsampled[b,oc,2y,2x+1]   += input[b, (oc*4+1)/repeats, y, x]
//   upsampled[b,oc,2y+1,2x]   += input[b, (oc*4+2)/repeats, y, x]
//   upsampled[b,oc,2y+1,2x+1] += input[b, (oc*4+3)/repeats, y, x]
//   input:      [batch, in_channels, height, width]
//   upsampled:  [batch, out_channels, height*2, width*2], modified in place
void add_upsample_shortcut_2x_inplace(
    Tensor & upsampled,
    const Tensor & input
) {
    if (upsampled.rank() != 4 || input.rank() != 4) {
        throw std::invalid_argument("add_upsample_shortcut_2x_inplace expects rank-4 tensors");
    }

    const size_t batch = input.dim_size(0);
    const size_t in_channels = input.dim_size(1);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t out_channels = upsampled.dim_size(1);

    if (upsampled.dim_size(0) != batch ||
        upsampled.dim_size(2) != height * 2 ||
        upsampled.dim_size(3) != width * 2) {
        throw std::invalid_argument("upsampled tensor shape does not match 2x shortcut shape");
    }
    if ((out_channels * 4) % in_channels != 0) {
        throw std::invalid_argument("VAE upsample shortcut channel ratio is invalid");
    }

    const size_t repeats = (out_channels * 4) / in_channels;
    const size_t in_spatial = height * width;
    const size_t out_width = width * 2;
    const size_t out_spatial = (height * 2) * out_width;
    const float * input_data = input.data().data();
    float * upsampled_data = upsampled.data().data();
    for (size_t b = 0; b < batch; ++b) {
        const float * input_batch = input_data + b * in_channels * in_spatial;
        float * upsampled_batch = upsampled_data + b * out_channels * out_spatial;
        for (size_t oc = 0; oc < out_channels; ++oc) {
            const size_t ic0 = (oc * 4 + 0) / repeats;
            const size_t ic1 = (oc * 4 + 1) / repeats;
            const size_t ic2 = (oc * 4 + 2) / repeats;
            const size_t ic3 = (oc * 4 + 3) / repeats;
            const float * input_channel0 = input_batch + ic0 * in_spatial;
            const float * input_channel1 = input_batch + ic1 * in_spatial;
            const float * input_channel2 = input_batch + ic2 * in_spatial;
            const float * input_channel3 = input_batch + ic3 * in_spatial;
            float * upsampled_channel = upsampled_batch + oc * out_spatial;
            for (size_t y = 0; y < height; ++y) {
                const float * input_row0 = input_channel0 + y * width;
                const float * input_row1 = input_channel1 + y * width;
                const float * input_row2 = input_channel2 + y * width;
                const float * input_row3 = input_channel3 + y * width;
                float * out_row0 = upsampled_channel + (y * 2) * out_width;
                float * out_row1 = out_row0 + out_width;
#if defined(__APPLE__)
                vDSP_vadd(input_row0, 1, out_row0, 2, out_row0, 2, width);
                vDSP_vadd(input_row1, 1, out_row0 + 1, 2, out_row0 + 1, 2, width);
                vDSP_vadd(input_row2, 1, out_row1, 2, out_row1, 2, width);
                vDSP_vadd(input_row3, 1, out_row1 + 1, 2, out_row1 + 1, 2, width);
#else
                for (size_t x = 0; x < width; ++x) {
                    const size_t out_x = x * 2;
                    out_row0[out_x] += input_row0[x];
                    out_row0[out_x + 1] += input_row1[x];
                    out_row1[out_x] += input_row2[x];
                    out_row1[out_x + 1] += input_row3[x];
                }
#endif
            }
        }
    }
}

//   each input channel is added into `repeats` consecutive output channels (no spatial
//   resize, unlike add_upsample_shortcut_2x_inplace): a plain channel-repeat residual
//   output[b, c*repeats+r, y, x] += input[b, c, y, x], for each r in [0, repeats)
//   input:   [batch, channels, height, width]
//   output:  [batch, channels*repeats, height, width], modified in place
void add_repeated_channels_inplace(
    Tensor & output,
    const Tensor & input,
    size_t repeats
) {
    if (output.rank() != 4 || input.rank() != 4) {
        throw std::invalid_argument("add_repeated_channels_inplace expects rank-4 tensors");
    }
    if (repeats == 0) {
        throw std::invalid_argument("channel repeat count must be greater than 0");
    }
    if (output.dim_size(0) != input.dim_size(0) ||
        output.dim_size(2) != input.dim_size(2) ||
        output.dim_size(3) != input.dim_size(3) ||
        output.dim_size(1) != input.dim_size(1) * repeats) {
        throw std::invalid_argument("output shape does not match repeated input shape");
    }

    const size_t batch = input.dim_size(0);
    const size_t channels = input.dim_size(1);
    const size_t spatial = input.dim_size(2) * input.dim_size(3);
    const float * input_data = input.data().data();
    float * output_data = output.data().data();

    for (size_t b = 0; b < batch; ++b) {
        const float * input_batch = input_data + b * channels * spatial;
        float * output_batch = output_data + b * channels * repeats * spatial;
        for (size_t c = 0; c < channels; ++c) {
            const float * input_channel = input_batch + c * spatial;
            for (size_t r = 0; r < repeats; ++r) {
                float * output_channel = output_batch + (c * repeats + r) * spatial;
#if defined(__APPLE__)
                vDSP_vadd(input_channel, 1, output_channel, 1, output_channel, 1, spatial);
#else
                for (size_t i = 0; i < spatial; ++i) {
                    output_channel[i] += input_channel[i];
                }
#endif
            }
        }
    }
}

// input/output: {batch, channels, height, width}; weight: {channels}; bias (optional): {channels}.
// out[b,c,y,x] = x[b,c,y,x] / sqrt(mean_c(x[b,c,y,x]^2) + eps) * weight[c] + bias[c],
// mean reduced over channels (per pixel), not over height/width.
Tensor rms_norm_4d_channelwise(
    const Tensor & input,
    const Tensor & weight,
    const Tensor * bias,
    float eps
) {
    if (input.rank() != 4) {
        throw std::invalid_argument("rms_norm_4d_channelwise expects rank-4 input");
    }
    const size_t channels = input.dim_size(1);
    if (weight.rank() != 1 || weight.dim_size(0) != channels) {
        throw std::invalid_argument("rms_norm_4d_channelwise weight must match channels");
    }
    if (bias && (bias->rank() != 1 || bias->dim_size(0) != channels)) {
        throw std::invalid_argument("rms_norm_4d_channelwise bias must match channels");
    }

    const size_t batch = input.dim_size(0);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t spatial = height * width;
    Tensor out({batch, channels, height, width});
    const float * input_data = input.data().data();
    const float * weight_data = weight.data().data();
    const float * bias_data = bias ? bias->data().data() : nullptr;
    float * out_data = out.data().data();

    std::vector<float> sum_sq(spatial);
    std::vector<float> inv_rms(spatial);
    for (size_t b = 0; b < batch; ++b) {
        const float * input_batch = input_data + b * channels * spatial;
        float * out_batch = out_data + b * channels * spatial;

        std::fill(sum_sq.begin(), sum_sq.end(), 0.0f);
        for (size_t c = 0; c < channels; ++c) {
            const float * channel = input_batch + c * spatial;
#if defined(__APPLE__)
            vDSP_vma(channel, 1, channel, 1, sum_sq.data(), 1, sum_sq.data(), 1, spatial);
#else
            for (size_t i = 0; i < spatial; ++i) {
                sum_sq[i] = std::fma(channel[i], channel[i], sum_sq[i]);
            }
#endif
        }
        const float inv_channels = 1.0f / static_cast<float>(channels);
#if defined(__APPLE__)
        vDSP_vsmsa(sum_sq.data(), 1, &inv_channels, &eps, inv_rms.data(), 1, spatial);
        int spatial_count = static_cast<int>(spatial);
        vvrsqrtf(inv_rms.data(), inv_rms.data(), &spatial_count);
#else
        for (size_t i = 0; i < spatial; ++i) {
            inv_rms[i] = 1.0f / std::sqrt(sum_sq[i] * inv_channels + eps);
        }
#endif

        for (size_t c = 0; c < channels; ++c) {
            const float * channel = input_batch + c * spatial;
            float * out_channel = out_batch + c * spatial;
            const float w = weight_data[c];
#if defined(__APPLE__)
            vDSP_vmul(channel, 1, inv_rms.data(), 1, out_channel, 1, spatial);
            vDSP_vsmul(out_channel, 1, &w, out_channel, 1, spatial);
            if (bias_data) {
                const float bv = bias_data[c];
                vDSP_vsadd(out_channel, 1, &bv, out_channel, 1, spatial);
            }
#else
            if (bias_data) {
                const float bv = bias_data[c];
                for (size_t i = 0; i < spatial; ++i) {
                    float v = channel[i] * inv_rms[i] * w;
                    v += bv;
                    out_channel[i] = v;
                }
            } else {
                for (size_t i = 0; i < spatial; ++i) {
                    out_channel[i] = channel[i] * inv_rms[i] * w;
                }
            }
#endif
        }
    }
    return out;
}

Tensor linear_weight_to_conv1x1(const Tensor & weight) {
    if (weight.rank() != 2) {
        throw std::invalid_argument("linear_weight_to_conv1x1 expects rank-2 weight");
    }
    Tensor out({weight.dim_size(0), weight.dim_size(1), 1, 1});
    std::copy(weight.data().begin(), weight.data().end(), out.data().begin());
    return out;
}

Tensor run_vae_res_block(
    const Tensor & input,
    const Tensor & conv1_weight,
    const Tensor * conv1_bias,
    const Tensor & conv2_weight,
    const Tensor & norm_weight,
    const Tensor & norm_bias
) {
    Tensor hidden = conv2d_same(input, conv1_weight, conv1_bias);
    hidden.silu_inplace();
    hidden = conv2d_same(hidden, conv2_weight, nullptr);
    hidden = rms_norm_4d_channelwise(hidden, norm_weight, &norm_bias, 1e-5f);
    hidden.add_inplace(input);
    return hidden;
}

Tensor run_vae_glumb_conv(
    const Tensor & input,
    const Tensor & conv_inverted_weight,
    const Tensor * conv_inverted_bias,
    const Tensor & conv_depth_weight,
    const Tensor * conv_depth_bias,
    const Tensor & conv_point_weight,
    const Tensor & norm_weight,
    const Tensor & norm_bias
) {
    Tensor hidden = glumb_conv(
        input,
        conv_inverted_weight,
        conv_inverted_bias,
        conv_depth_weight,
        conv_depth_bias,
        conv_point_weight
    );
    hidden = rms_norm_4d_channelwise(hidden, norm_weight, &norm_bias, 1e-5f);
    hidden.add_inplace(input);
    return hidden;
}

Tensor run_vae_multiscale_attention(
    const Tensor & input,
    const Tensor & to_q_weight,
    const Tensor & to_k_weight,
    const Tensor & to_v_weight,
    const Tensor & proj_in_weight,
    const Tensor & proj_out_weight,
    const Tensor & to_out_weight,
    const Tensor & norm_out_weight,
    const Tensor & norm_out_bias,
    size_t attention_head_dim
) {
    const size_t inner_dim = to_q_weight.dim_size(0);
    const size_t num_heads = inner_dim / attention_head_dim;
    if (inner_dim % attention_head_dim != 0) {
        throw std::invalid_argument("VAE attention inner_dim must be divisible by attention_head_dim");
    }

    Tensor query = conv2d_1x1(input, to_q_weight, nullptr);
    Tensor key = conv2d_1x1(input, to_k_weight, nullptr);
    Tensor value = conv2d_1x1(input, to_v_weight, nullptr);

    const size_t batch = input.dim_size(0);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t spatial = height * width;
    Tensor qkv({batch, inner_dim * 3, height, width});
    {
        const size_t plane = inner_dim * spatial;
        const float * query_data = query.data().data();
        const float * key_data = key.data().data();
        const float * value_data = value.data().data();
        float * qkv_data = qkv.data().data();
        for (size_t b = 0; b < batch; ++b) {
            float * qkv_batch = qkv_data + b * 3 * plane;
            std::copy(query_data + b * plane, query_data + (b + 1) * plane, qkv_batch);
            std::copy(key_data + b * plane, key_data + (b + 1) * plane, qkv_batch + plane);
            std::copy(value_data + b * plane, value_data + (b + 1) * plane, qkv_batch + 2 * plane);
        }
    }

    Tensor multiscale = depthwise_conv2d_same(qkv, proj_in_weight, nullptr);
    multiscale = grouped_conv2d_1x1(multiscale, proj_out_weight, 3 * num_heads);

    const size_t tokens = height * width;
    const size_t qkv_heads = qkv.dim_size(1) / (3 * attention_head_dim);
    const size_t multiscale_heads = multiscale.dim_size(1) / (3 * attention_head_dim);
    const size_t merged_heads = qkv_heads + multiscale_heads;
    Tensor merged({batch, merged_heads * attention_head_dim, height, width});
    std::vector<float> query_matrix_buf(attention_head_dim * tokens);
    std::vector<float> key_matrix_buf(attention_head_dim * tokens);
    std::vector<float> value_matrix_buf((attention_head_dim + 1) * tokens);
    std::vector<float> key_transposed_buf(tokens * attention_head_dim);
    std::vector<float> scores_buf((attention_head_dim + 1) * attention_head_dim);
    std::vector<float> hidden_matrix_buf((attention_head_dim + 1) * tokens);
    std::vector<float> denom_buf(tokens);
    float * query_matrix = query_matrix_buf.data();
    float * key_matrix = key_matrix_buf.data();
    float * value_matrix = value_matrix_buf.data();
    float * key_transposed = key_transposed_buf.data();
    float * scores = scores_buf.data();
    float * hidden_matrix = hidden_matrix_buf.data();
    float * denom = denom_buf.data();
    const float * qkv_data = qkv.data().data();
    const float * multiscale_data = multiscale.data().data();
    float * merged_data = merged.data().data();
    const size_t qkv_batch_stride = qkv.dim_size(1) * spatial;
    const size_t multiscale_batch_stride = multiscale.dim_size(1) * spatial;
    const size_t merged_batch_stride = merged_heads * attention_head_dim * spatial;

    for (size_t b = 0; b < batch; ++b) {
        const float * qkv_batch = qkv_data + b * qkv_batch_stride;
        const float * multiscale_batch = multiscale_data + b * multiscale_batch_stride;
        float * merged_batch = merged_data + b * merged_batch_stride;
        for (size_t g = 0; g < merged_heads; ++g) {
            const bool use_multiscale = g >= qkv_heads;
            const size_t local_group = use_multiscale ? (g - qkv_heads) : g;
            const size_t group_channel_base = local_group * 3 * attention_head_dim;
            const float * source_batch = use_multiscale ? multiscale_batch : qkv_batch;
            const float * q_group = source_batch + group_channel_base * spatial;
            const float * k_group = source_batch + (group_channel_base + attention_head_dim) * spatial;
            const float * v_group = source_batch + (group_channel_base + 2 * attention_head_dim) * spatial;
            for (size_t d = 0; d < attention_head_dim; ++d) {
                const float * q_plane = q_group + d * spatial;
                const float * k_plane = k_group + d * spatial;
                const float * v_plane = v_group + d * spatial;
                float * query_row = query_matrix + d * tokens;
                float * key_row = key_matrix + d * tokens;
                float * value_row = value_matrix + d * tokens;
#if defined(__APPLE__)
                constexpr float zero = 0.0f;
                vDSP_vthr(q_plane, 1, &zero, query_row, 1, tokens);
                vDSP_vthr(k_plane, 1, &zero, key_row, 1, tokens);
                std::copy(v_plane, v_plane + tokens, value_row);
#else
                for (size_t t = 0; t < tokens; ++t) {
                    query_row[t] = std::max(0.0f, q_plane[t]);
                    key_row[t] = std::max(0.0f, k_plane[t]);
                    value_row[t] = v_plane[t];
                }
#endif
            }
#if defined(__APPLE__)
            {
                constexpr float one = 1.0f;
                vDSP_vfill(&one, value_matrix + attention_head_dim * tokens, 1, tokens);
            }
            // key_matrix is [attention_head_dim][tokens]; transpose to [tokens][attention_head_dim].
            vDSP_mtrans(key_matrix, 1, key_transposed, 1, tokens, attention_head_dim);
#else
            for (size_t t = 0; t < tokens; ++t) {
                value_matrix[attention_head_dim * tokens + t] = 1.0f;
            }
            for (size_t t = 0; t < tokens; ++t) {
                for (size_t d = 0; d < attention_head_dim; ++d) {
                    key_transposed[t * attention_head_dim + d] = key_matrix[d * tokens + t];
                }
            }
#endif

            gemm_row_major(
                value_matrix,
                key_transposed,
                nullptr,
                scores,
                attention_head_dim + 1,
                tokens,
                attention_head_dim
            );
            gemm_row_major(
                scores,
                query_matrix,
                nullptr,
                hidden_matrix,
                attention_head_dim + 1,
                attention_head_dim,
                tokens
            );

            float * merged_group = merged_batch + g * attention_head_dim * spatial;
#if defined(__APPLE__)
            {
                constexpr float eps = 1e-15f;
                vDSP_vsadd(hidden_matrix + attention_head_dim * tokens, 1, &eps, denom, 1, tokens);
            }
            for (size_t d = 0; d < attention_head_dim; ++d) {
                vDSP_vdiv(denom, 1, hidden_matrix + d * tokens, 1, merged_group + d * spatial, 1, tokens);
            }
#else
            for (size_t t = 0; t < tokens; ++t) {
                denom[t] = hidden_matrix[attention_head_dim * tokens + t] + 1e-15f;
            }
            for (size_t t = 0; t < tokens; ++t) {
                for (size_t d = 0; d < attention_head_dim; ++d) {
                    merged_group[d * spatial + t] =
                        hidden_matrix[d * tokens + t] / denom[t];
                }
            }
#endif
        }
    }

    Tensor hidden = conv2d_1x1(merged, to_out_weight, nullptr);
    hidden = rms_norm_4d_channelwise(hidden, norm_out_weight, &norm_out_bias, 1e-5f);
    hidden.add_inplace(input);
    return hidden;
}

Tensor run_vae_upsample_block(
    const Tensor & input,
    const Tensor & conv_weight,
    const Tensor * conv_bias,
    bool shortcut
) {
    Tensor up = nearest_upsample_2x(input);
    up = conv2d_same(up, conv_weight, conv_bias);
    if (!shortcut) {
        return up;
    }
    add_upsample_shortcut_2x_inplace(up, input);
    return up;
}
