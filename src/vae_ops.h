#pragma once

#include "tensor.h"

Tensor conv2d_3x3_im2col_gemm(
    const Tensor & input,
    const Tensor & weight,
    const Tensor * bias
);
Tensor conv2d_3x3_tap_accumulate(
    const Tensor & input,
    const Tensor & weight,
    const Tensor * bias
);
Tensor conv2d_3x3_same_dispatch(
    const Tensor & input,
    const Tensor & weight,
    const Tensor * bias
);
Tensor conv2d_same(const Tensor & input, const Tensor & weight, const Tensor * bias);
Tensor depthwise_conv2d_same(const Tensor & input, const Tensor & weight, const Tensor * bias);
Tensor grouped_conv2d_1x1(
    const Tensor & input,
    const Tensor & weight,
    size_t groups
);
Tensor nearest_upsample_2x(const Tensor & input);
void add_upsample_shortcut_2x_inplace(
    Tensor & upsampled,
    const Tensor & input
);
void add_repeated_channels_inplace(
    Tensor & output,
    const Tensor & input,
    size_t repeats
);
Tensor rms_norm_4d_channelwise(
    const Tensor & input,
    const Tensor & weight,
    const Tensor * bias,
    float eps
);
Tensor linear_weight_to_conv1x1(const Tensor & weight);

Tensor run_vae_res_block(
    const Tensor & input,
    const Tensor & conv1_weight,
    const Tensor * conv1_bias,
    const Tensor & conv2_weight,
    const Tensor & norm_weight,
    const Tensor & norm_bias
);
Tensor run_vae_glumb_conv(
    const Tensor & input,
    const Tensor & conv_inverted_weight,
    const Tensor * conv_inverted_bias,
    const Tensor & conv_depth_weight,
    const Tensor * conv_depth_bias,
    const Tensor & conv_point_weight,
    const Tensor & norm_weight,
    const Tensor & norm_bias
);
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
);
Tensor run_vae_upsample_block(
    const Tensor & input,
    const Tensor & conv_weight,
    const Tensor * conv_bias,
    bool shortcut
);
