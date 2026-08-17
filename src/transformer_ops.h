#pragma once

#include "tensor.h"

Tensor project_caption_context(
    const Tensor & encoder_hidden_states,
    const Tensor & linear_1_weight,
    const Tensor * linear_1_bias,
    const Tensor & linear_2_weight,
    const Tensor * linear_2_bias,
    const Tensor & norm_weight
);
Tensor split_heads_3d(const Tensor & input, size_t num_heads);
Tensor merge_heads_4d(const Tensor & input);
Tensor attention_scores_4d(const Tensor & q, const Tensor & k);
Tensor attention_values_4d(const Tensor & weights, const Tensor & v);
Tensor softmax_lastdim_4d(const Tensor & input);
Tensor attention_4d(
    const Tensor & q,
    const Tensor & k,
    const Tensor & v,
    const Tensor * attention_bias = nullptr
);
Tensor linear_attention_4d(const Tensor & q, const Tensor & k, const Tensor & v);
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
);
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
);
