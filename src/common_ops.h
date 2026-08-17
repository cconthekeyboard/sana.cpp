#pragma once

#include "tensor.h"

void gemm_row_major(
    const float* A,     // [M, K]
    const float* B,     // [K, N], or [N, K] if b_transposed
    const float* bias,  // [N] or nullptr
    float* C,           // [M, N]
    size_t M,
    size_t K,
    size_t N,
    bool b_transposed = false  // if true, B is physically stored [N, K] and read as if transposed
);
void gemm_accumulate_row_major(
    const float* A,     // [M, K]
    const float* B,     // [K, N]
    float* C,           // [M, N], accumulated into
    size_t M,
    size_t K,
    size_t N
);
Tensor linear_2d(const Tensor & input, const Tensor & weight, const Tensor * bias);
Tensor mlp_2d(
    const Tensor & input,
    const Tensor & up_weight,
    const Tensor * up_bias,
    const Tensor & down_weight,
    const Tensor * down_bias
);
Tensor linear_3d_lastdim(const Tensor & input, const Tensor & weight, const Tensor * bias);
Tensor mlp_3d_lastdim(
    const Tensor & input,
    const Tensor & up_weight,
    const Tensor * up_bias,
    const Tensor & down_weight,
    const Tensor * down_bias
);
Tensor conv2d_1x1(const Tensor & input, const Tensor & weight, const Tensor * bias);
Tensor depthwise_conv2d_3x3(const Tensor & input, const Tensor & weight, const Tensor * bias);
Tensor depthwise_conv2d_5x5(const Tensor & input, const Tensor & weight, const Tensor * bias);
Tensor glumb_conv(
    const Tensor & input,
    const Tensor & conv_inverted_weight,
    const Tensor * conv_inverted_bias,
    const Tensor & conv_depth_weight,
    const Tensor * conv_depth_bias,
    const Tensor & conv_point_weight
);
