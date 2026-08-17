#include "common_ops.h"
#include "tensor.h"
#include "transformer.h"
#include "transformer_ops.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

SanaBlockWeights make_single_value_self_attention_block(float shift_msa, float gate_msa) {
    Tensor scale_shift_table({6, 1});
    scale_shift_table.fill(0.0f);
    scale_shift_table.at({0, 0}) = shift_msa;
    scale_shift_table.at({2, 0}) = gate_msa;

    Tensor one_by_one_weight({1, 1});
    one_by_one_weight.at({0, 0}) = 1.0f;
    Tensor zero_weight({1, 1});
    zero_weight.fill(0.0f);
    Tensor zero_bias({1});
    zero_bias.fill(0.0f);
    Tensor one_norm({1});
    one_norm.at(0) = 1.0f;
    Tensor zero_norm({1});
    zero_norm.at(0) = 0.0f;
    Tensor ff_inverted_weight({2, 1, 1, 1});
    ff_inverted_weight.fill(0.0f);
    Tensor ff_inverted_bias({2});
    ff_inverted_bias.fill(0.0f);
    Tensor ff_depth_weight({2, 1, 3, 3});
    ff_depth_weight.fill(0.0f);
    Tensor ff_depth_bias({2});
    ff_depth_bias.fill(0.0f);
    Tensor ff_point_weight({1, 1, 1, 1});
    ff_point_weight.fill(0.0f);

    return SanaBlockWeights{
        scale_shift_table,
        one_norm,
        zero_norm,
        one_by_one_weight,
        zero_bias,
        one_by_one_weight,
        zero_bias,
        one_by_one_weight,
        zero_bias,
        one_by_one_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        one_norm,
        zero_norm,
        ff_inverted_weight,
        ff_inverted_bias,
        ff_depth_weight,
        ff_depth_bias,
        ff_point_weight,
        1,
        1,
    };
}

}  // namespace

void run_transformer_tests() {
    bool threw = false;

    Tensor lin_input({2, 3});
    Tensor lin_weight({3, 2});
    Tensor lin_bias({2});
    lin_input.at({0, 0}) = 1.0f;
    lin_input.at({0, 1}) = 2.0f;
    lin_input.at({0, 2}) = 3.0f;
    lin_input.at({1, 0}) = 4.0f;
    lin_input.at({1, 1}) = 5.0f;
    lin_input.at({1, 2}) = 6.0f;
    lin_weight.at({0, 0}) = 7.0f;
    lin_weight.at({0, 1}) = 8.0f;
    lin_weight.at({1, 0}) = 9.0f;
    lin_weight.at({1, 1}) = 10.0f;
    lin_weight.at({2, 0}) = 11.0f;
    lin_weight.at({2, 1}) = 12.0f;
    lin_bias.at(0) = 1.0f;
    lin_bias.at(1) = -2.0f;

    Tensor lin_out = linear_2d(lin_input, lin_weight, &lin_bias);
    assert(lin_out.shape() == std::vector<size_t>({2, 2}));
    assert(lin_out.at({0, 0}) == 59.0f);
    assert(lin_out.at({0, 1}) == 62.0f);
    assert(lin_out.at({1, 0}) == 140.0f);
    assert(lin_out.at({1, 1}) == 152.0f);

    Tensor mlp_input({1, 2});
    Tensor up_weight({2, 2});
    Tensor up_bias({2});
    Tensor down_weight({2, 1});
    Tensor down_bias({1});
    mlp_input.at({0, 0}) = 1.0f;
    mlp_input.at({0, 1}) = -1.0f;
    up_weight.at({0, 0}) = 1.0f;
    up_weight.at({0, 1}) = 0.0f;
    up_weight.at({1, 0}) = 0.0f;
    up_weight.at({1, 1}) = 1.0f;
    up_bias.at(0) = 0.0f;
    up_bias.at(1) = 0.0f;
    down_weight.at({0, 0}) = 1.0f;
    down_weight.at({1, 0}) = 1.0f;
    down_bias.at(0) = 0.0f;

    Tensor mlp_out = mlp_2d(mlp_input, up_weight, &up_bias, down_weight, &down_bias);
    assert(mlp_out.shape() == std::vector<size_t>({1, 1}));
    assert(std::fabs(mlp_out.at({0, 0}) - 0.4621172f) < 1e-5f);

    Tensor seq_input({1, 2, 3});
    Tensor seq_weight({3, 2});
    Tensor seq_bias({2});
    seq_input.at({0, 0, 0}) = 1.0f;
    seq_input.at({0, 0, 1}) = 2.0f;
    seq_input.at({0, 0, 2}) = 3.0f;
    seq_input.at({0, 1, 0}) = 4.0f;
    seq_input.at({0, 1, 1}) = 5.0f;
    seq_input.at({0, 1, 2}) = 6.0f;
    seq_weight.at({0, 0}) = 7.0f;
    seq_weight.at({0, 1}) = 8.0f;
    seq_weight.at({1, 0}) = 9.0f;
    seq_weight.at({1, 1}) = 10.0f;
    seq_weight.at({2, 0}) = 11.0f;
    seq_weight.at({2, 1}) = 12.0f;
    seq_bias.at(0) = 1.0f;
    seq_bias.at(1) = -2.0f;

    Tensor seq_out = linear_3d_lastdim(seq_input, seq_weight, &seq_bias);
    assert(seq_out.shape() == std::vector<size_t>({1, 2, 2}));
    assert(seq_out.at({0, 0, 0}) == 59.0f);
    assert(seq_out.at({0, 0, 1}) == 62.0f);
    assert(seq_out.at({0, 1, 0}) == 140.0f);
    assert(seq_out.at({0, 1, 1}) == 152.0f);

    Tensor heads_input({1, 2, 4});
    heads_input.at({0, 0, 0}) = 1.0f;
    heads_input.at({0, 0, 1}) = 2.0f;
    heads_input.at({0, 0, 2}) = 3.0f;
    heads_input.at({0, 0, 3}) = 4.0f;
    heads_input.at({0, 1, 0}) = 5.0f;
    heads_input.at({0, 1, 1}) = 6.0f;
    heads_input.at({0, 1, 2}) = 7.0f;
    heads_input.at({0, 1, 3}) = 8.0f;

    Tensor heads_out = split_heads_3d(heads_input, 2);
    assert(heads_out.shape() == std::vector<size_t>({1, 2, 2, 2}));
    assert(heads_out.at({0, 0, 0, 0}) == 1.0f);
    assert(heads_out.at({0, 0, 0, 1}) == 2.0f);
    assert(heads_out.at({0, 1, 0, 0}) == 3.0f);
    assert(heads_out.at({0, 1, 0, 1}) == 4.0f);
    assert(heads_out.at({0, 0, 1, 0}) == 5.0f);
    assert(heads_out.at({0, 0, 1, 1}) == 6.0f);
    assert(heads_out.at({0, 1, 1, 0}) == 7.0f);
    assert(heads_out.at({0, 1, 1, 1}) == 8.0f);

    Tensor merged_out = merge_heads_4d(heads_out);
    assert(merged_out.shape() == std::vector<size_t>({1, 2, 4}));
    assert(merged_out.at({0, 0, 0}) == 1.0f);
    assert(merged_out.at({0, 0, 1}) == 2.0f);
    assert(merged_out.at({0, 0, 2}) == 3.0f);
    assert(merged_out.at({0, 0, 3}) == 4.0f);
    assert(merged_out.at({0, 1, 0}) == 5.0f);
    assert(merged_out.at({0, 1, 1}) == 6.0f);
    assert(merged_out.at({0, 1, 2}) == 7.0f);
    assert(merged_out.at({0, 1, 3}) == 8.0f);

    Tensor q({1, 1, 2, 2});
    Tensor k({1, 1, 2, 2});
    q.at({0, 0, 0, 0}) = 1.0f;
    q.at({0, 0, 0, 1}) = 0.0f;
    q.at({0, 0, 1, 0}) = 0.0f;
    q.at({0, 0, 1, 1}) = 1.0f;
    k.at({0, 0, 0, 0}) = 1.0f;
    k.at({0, 0, 0, 1}) = 2.0f;
    k.at({0, 0, 1, 0}) = 3.0f;
    k.at({0, 0, 1, 1}) = 4.0f;

    Tensor scores = attention_scores_4d(q, k);
    assert(scores.shape() == std::vector<size_t>({1, 1, 2, 2}));
    assert(scores.at({0, 0, 0, 0}) == 1.0f);
    assert(scores.at({0, 0, 0, 1}) == 3.0f);
    assert(scores.at({0, 0, 1, 0}) == 2.0f);
    assert(scores.at({0, 0, 1, 1}) == 4.0f);

    Tensor weights({1, 1, 2, 2});
    Tensor v({1, 1, 2, 2});
    weights.at({0, 0, 0, 0}) = 1.0f;
    weights.at({0, 0, 0, 1}) = 0.0f;
    weights.at({0, 0, 1, 0}) = 0.25f;
    weights.at({0, 0, 1, 1}) = 0.75f;
    v.at({0, 0, 0, 0}) = 10.0f;
    v.at({0, 0, 0, 1}) = 20.0f;
    v.at({0, 0, 1, 0}) = 30.0f;
    v.at({0, 0, 1, 1}) = 40.0f;

    Tensor values_out = attention_values_4d(weights, v);
    assert(values_out.shape() == std::vector<size_t>({1, 1, 2, 2}));
    assert(values_out.at({0, 0, 0, 0}) == 10.0f);
    assert(values_out.at({0, 0, 0, 1}) == 20.0f);
    assert(values_out.at({0, 0, 1, 0}) == 25.0f);
    assert(values_out.at({0, 0, 1, 1}) == 35.0f);

    Tensor attn_logits({1, 1, 2, 2});
    attn_logits.at({0, 0, 0, 0}) = 0.0f;
    attn_logits.at({0, 0, 0, 1}) = 0.0f;
    attn_logits.at({0, 0, 1, 0}) = 1.0f;
    attn_logits.at({0, 0, 1, 1}) = 2.0f;

    Tensor attn_probs = softmax_lastdim_4d(attn_logits);
    assert(attn_probs.shape() == std::vector<size_t>({1, 1, 2, 2}));
    assert(std::fabs(attn_probs.at({0, 0, 0, 0}) - 0.5f) < 1e-6f);
    assert(std::fabs(attn_probs.at({0, 0, 0, 1}) - 0.5f) < 1e-6f);
    assert(std::fabs(attn_probs.at({0, 0, 1, 0}) - 0.2689414f) < 1e-6f);
    assert(std::fabs(attn_probs.at({0, 0, 1, 1}) - 0.7310586f) < 1e-6f);

    Tensor attn_out = attention_4d(q, k, v);
    assert(attn_out.shape() == std::vector<size_t>({1, 1, 2, 2}));
    assert(std::fabs(attn_out.at({0, 0, 0, 0}) - 26.088594f) < 1e-5f);
    assert(std::fabs(attn_out.at({0, 0, 0, 1}) - 36.088594f) < 1e-5f);
    assert(std::fabs(attn_out.at({0, 0, 1, 0}) - 26.088594f) < 1e-5f);
    assert(std::fabs(attn_out.at({0, 0, 1, 1}) - 36.088594f) < 1e-5f);

    Tensor linear_attn_out = linear_attention_4d(q, k, v);
    assert(linear_attn_out.shape() == std::vector<size_t>({1, 1, 2, 2}));
    assert(std::fabs(linear_attn_out.at({0, 0, 0, 0}) - 25.0f) < 1e-5f);
    assert(std::fabs(linear_attn_out.at({0, 0, 0, 1}) - 35.0f) < 1e-5f);
    assert(std::fabs(linear_attn_out.at({0, 0, 1, 0}) - 23.333334f) < 1e-5f);
    assert(std::fabs(linear_attn_out.at({0, 0, 1, 1}) - 33.333332f) < 1e-5f);

    Tensor caption_input({1, 2, 4});
    Tensor caption_linear_1_weight({4, 2});
    Tensor caption_linear_1_bias({2});
    Tensor caption_linear_2_weight({2, 2});
    Tensor caption_linear_2_bias({2});
    caption_input.at({0, 0, 0}) = 1.0f;
    caption_input.at({0, 0, 1}) = 2.0f;
    caption_input.at({0, 0, 2}) = 3.0f;
    caption_input.at({0, 0, 3}) = 4.0f;
    caption_input.at({0, 1, 0}) = 5.0f;
    caption_input.at({0, 1, 1}) = 6.0f;
    caption_input.at({0, 1, 2}) = 7.0f;
    caption_input.at({0, 1, 3}) = 8.0f;
    caption_linear_1_weight.fill(0.0f);
    caption_linear_1_weight.at({0, 0}) = 1.0f;
    caption_linear_1_weight.at({1, 1}) = 1.0f;
    caption_linear_1_bias.fill(0.0f);
    caption_linear_2_weight.fill(0.0f);
    caption_linear_2_weight.at({0, 0}) = 1.0f;
    caption_linear_2_weight.at({1, 1}) = 1.0f;
    caption_linear_2_bias.fill(0.0f);

    Tensor caption_norm_weight({2});
    caption_norm_weight.at(0) = 1.0f;
    caption_norm_weight.at(1) = 2.0f;
    Tensor projected_caption = project_caption_context(
        caption_input,
        caption_linear_1_weight,
        &caption_linear_1_bias,
        caption_linear_2_weight,
        &caption_linear_2_bias,
        caption_norm_weight
    );
    assert(projected_caption.shape() == std::vector<size_t>({1, 2, 2}));
    assert(std::fabs(projected_caption.at({0, 0, 0}) - 0.5590534f) < 1e-5f);
    assert(std::fabs(projected_caption.at({0, 0, 1}) - 2.5980382f) < 1e-5f);
    assert(std::fabs(projected_caption.at({0, 1, 0}) - 0.9053573f) < 1e-5f);
    assert(std::fabs(projected_caption.at({0, 1, 1}) - 2.1728575f) < 1e-5f);

    Tensor block_context = project_caption_context(
        caption_input,
        caption_linear_1_weight,
        &caption_linear_1_bias,
        caption_linear_2_weight,
        &caption_linear_2_bias,
        caption_norm_weight
    );
    assert(block_context.shape() == std::vector<size_t>({1, 2, 2}));
    assert(std::fabs(block_context.at({0, 0, 0}) - projected_caption.at({0, 0, 0})) < 1e-6f);
    assert(std::fabs(block_context.at({0, 1, 1}) - projected_caption.at({0, 1, 1})) < 1e-6f);

    std::vector<SanaBlockWeights> blocks;
    blocks.push_back(make_single_value_self_attention_block(2.0f, 3.0f));
    blocks.push_back(make_single_value_self_attention_block(2.0f, 3.0f));
    validate_model_block_sequence(blocks);

    std::vector<SanaBlockWeights> bad_blocks = blocks;
    bad_blocks[1].scale_shift_table = Tensor({6, 2});
    bad_blocks[1].scale_shift_table.fill(0.0f);
    threw = false;
    try {
        validate_model_block_sequence(bad_blocks);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    Tensor block_input({1, 2, 2});
    Tensor proj_weight({2, 2});
    Tensor proj_bias({2});
    Tensor norm_gamma({2});
    Tensor norm_beta({2});
    block_input.at({0, 0, 0}) = 1.0f;
    block_input.at({0, 0, 1}) = 0.0f;
    block_input.at({0, 1, 0}) = 0.0f;
    block_input.at({0, 1, 1}) = 1.0f;
    proj_weight.at({0, 0}) = 1.0f;
    proj_weight.at({0, 1}) = 0.0f;
    proj_weight.at({1, 0}) = 0.0f;
    proj_weight.at({1, 1}) = 1.0f;
    proj_bias.at(0) = 0.0f;
    proj_bias.at(1) = 0.0f;
    norm_gamma.at(0) = 1.0f;
    norm_gamma.at(1) = 1.0f;
    norm_beta.at(0) = 0.0f;
    norm_beta.at(1) = 0.0f;
    Tensor zero_scale_shift_table({6, 2});
    zero_scale_shift_table.fill(0.0f);

    Tensor block_out = attention_block_3d(
        block_input,
        proj_weight, &proj_bias,
        proj_weight, &proj_bias,
        proj_weight, &proj_bias,
        proj_weight, &proj_bias,
        1
    );
    assert(block_out.shape() == std::vector<size_t>({1, 2, 2}));
    assert(std::fabs(block_out.at({0, 0, 0}) - 1.0f) < 1e-5f);
    assert(std::fabs(block_out.at({0, 0, 1}) - 0.0f) < 1e-5f);
    assert(std::fabs(block_out.at({0, 1, 0}) - 0.0f) < 1e-5f);
    assert(std::fabs(block_out.at({0, 1, 1}) - 1.0f) < 1e-5f);

    Tensor context_input({1, 2, 2});
    Tensor test_timestep({1, 12});
    context_input.at({0, 0, 0}) = 2.0f;
    context_input.at({0, 0, 1}) = 0.0f;
    context_input.at({0, 1, 0}) = 0.0f;
    context_input.at({0, 1, 1}) = 3.0f;
    test_timestep.fill(0.0f);
    test_timestep.at({0, 4}) = 1.0f;
    test_timestep.at({0, 5}) = 1.0f;
    test_timestep.at({0, 10}) = 1.0f;
    test_timestep.at({0, 11}) = 1.0f;
    Tensor cross_out = cross_attention_block_3d(
        block_input,
        context_input,
        nullptr,
        proj_weight, &proj_bias,
        proj_weight, &proj_bias,
        proj_weight, &proj_bias,
        proj_weight, &proj_bias,
        1
    );
    assert(cross_out.shape() == std::vector<size_t>({1, 2, 2}));
    assert(std::fabs(cross_out.at({0, 0, 0}) - 1.6088594f) < 1e-5f);
    assert(std::fabs(cross_out.at({0, 0, 1}) - 0.5867110f) < 1e-5f);
    assert(std::fabs(cross_out.at({0, 1, 0}) - 0.2140836f) < 1e-5f);
    assert(std::fabs(cross_out.at({0, 1, 1}) - 2.6788745f) < 1e-5f);

    Tensor block_forward_input({1, 1, 2});
    Tensor block_forward_context({1, 2, 2});
    Tensor ff_inverted_weight({4, 2, 1, 1});
    Tensor ff_inverted_bias({4});
    Tensor ff_depth_weight({4, 1, 3, 3});
    Tensor ff_depth_bias({4});
    Tensor ff_point_weight({2, 2, 1, 1});
    block_forward_input.at({0, 0, 0}) = 1.0f;
    block_forward_input.at({0, 0, 1}) = 0.0f;
    block_forward_context.at({0, 0, 0}) = 2.0f;
    block_forward_context.at({0, 0, 1}) = 0.0f;
    block_forward_context.at({0, 1, 0}) = 0.0f;
    block_forward_context.at({0, 1, 1}) = 3.0f;
    ff_inverted_weight.fill(0.0f);
    ff_inverted_weight.at({0, 0, 0, 0}) = 1.0f;
    ff_inverted_weight.at({1, 1, 0, 0}) = 1.0f;
    ff_inverted_weight.at({2, 0, 0, 0}) = 1.0f;
    ff_inverted_weight.at({3, 1, 0, 0}) = 1.0f;
    ff_inverted_bias.fill(0.0f);
    ff_depth_weight.fill(0.0f);
    ff_depth_weight.at({0, 0, 1, 1}) = 1.0f;
    ff_depth_weight.at({1, 0, 1, 1}) = 1.0f;
    ff_depth_weight.at({2, 0, 1, 1}) = 1.0f;
    ff_depth_weight.at({3, 0, 1, 1}) = 1.0f;
    ff_depth_bias.fill(0.0f);
    ff_point_weight.fill(0.0f);
    ff_point_weight.at({0, 0, 0, 0}) = 1.0f;
    ff_point_weight.at({1, 1, 0, 0}) = 1.0f;

    SanaBlockWeights block_weights{
        zero_scale_shift_table,
        norm_gamma,
        norm_beta,
        proj_weight,
        proj_bias,
        proj_weight,
        proj_bias,
        proj_weight,
        proj_bias,
        proj_weight,
        proj_bias,
        proj_weight,
        proj_bias,
        proj_weight,
        proj_bias,
        proj_weight,
        proj_bias,
        proj_weight,
        proj_bias,
        norm_gamma,
        norm_beta,
        ff_inverted_weight,
        ff_inverted_bias,
        ff_depth_weight,
        ff_depth_bias,
        ff_point_weight,
        1,
        1
    };
    Tensor model_block_out = block_forward(
        block_forward_input,
        block_forward_context,
        test_timestep,
        block_weights
    ).after_ff;
    assert(model_block_out.shape() == std::vector<size_t>({1, 1, 2}));
    assert(std::fabs(model_block_out.at({0, 0, 0}) - 4.3467007f) < 1e-5f);
    assert(std::fabs(model_block_out.at({0, 0, 1}) - (-0.9475632f)) < 1e-5f);

    Tensor zero_weight({2, 2});
    Tensor zero_bias({2});
    zero_weight.fill(0.0f);
    zero_bias.fill(0.0f);
    SanaBlockWeights residual_only_weights{
        zero_scale_shift_table,
        norm_gamma,
        norm_beta,
        zero_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        zero_weight,
        zero_bias,
        norm_gamma,
        norm_beta,
        ff_inverted_weight,
        ff_inverted_bias,
        ff_depth_weight,
        ff_depth_bias,
        ff_point_weight,
        1,
        1
    };
    Tensor residual_only_out = block_forward(
        block_forward_input,
        block_forward_context,
        test_timestep,
        residual_only_weights
    ).after_ff;
    assert(residual_only_out.shape() == std::vector<size_t>({1, 1, 2}));
    assert(std::fabs(residual_only_out.at({0, 0, 0}) - 1.3607715f) < 1e-5f);
    assert(std::fabs(residual_only_out.at({0, 0, 1}) - 0.0313307f) < 1e-5f);

    Tensor mlp3_input({1, 2, 2});
    Tensor mlp3_weight({2, 2});
    Tensor mlp3_bias({2});
    mlp3_input.at({0, 0, 0}) = 1.0f;
    mlp3_input.at({0, 0, 1}) = -1.0f;
    mlp3_input.at({0, 1, 0}) = 0.0f;
    mlp3_input.at({0, 1, 1}) = 2.0f;
    mlp3_weight.at({0, 0}) = 1.0f;
    mlp3_weight.at({0, 1}) = 0.0f;
    mlp3_weight.at({1, 0}) = 0.0f;
    mlp3_weight.at({1, 1}) = 1.0f;
    mlp3_bias.at(0) = 0.0f;
    mlp3_bias.at(1) = 0.0f;

    Tensor mlp3_out = mlp_3d_lastdim(
        mlp3_input,
        mlp3_weight, &mlp3_bias,
        mlp3_weight, &mlp3_bias
    );
    assert(mlp3_out.shape() == std::vector<size_t>({1, 2, 2}));
    assert(std::fabs(mlp3_out.at({0, 0, 0}) - 0.7310586f) < 1e-5f);
    assert(std::fabs(mlp3_out.at({0, 0, 1}) - (-0.2689414f)) < 1e-5f);
    assert(std::fabs(mlp3_out.at({0, 1, 0}) - 0.0f) < 1e-6f);
    assert(std::fabs(mlp3_out.at({0, 1, 1}) - 1.7615942f) < 1e-5f);

    Tensor bad_rank_input({2, 3, 4});
    Tensor bad_context({1, 2, 2});

    threw = false;
    try {
        linear_2d(bad_rank_input, lin_weight, &lin_bias);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        mlp_2d(bad_rank_input, up_weight, &up_bias, down_weight, &down_bias);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        linear_3d_lastdim(lin_input, seq_weight, &seq_bias);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        split_heads_3d(seq_input, 2);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        merge_heads_4d(heads_input);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        attention_scores_4d(heads_out, q);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        attention_values_4d(scores, heads_out);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        softmax_lastdim_4d(seq_input);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        attention_4d(heads_out, q, v);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        attention_block_3d(
            lin_input,
            proj_weight, &proj_bias,
            proj_weight, &proj_bias,
            proj_weight, &proj_bias,
            proj_weight, &proj_bias,
            1
        );
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        block_forward(lin_input, lin_input, lin_input, block_weights);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        mlp_3d_lastdim(
            lin_input,
            mlp3_weight, &mlp3_bias,
            mlp3_weight, &mlp3_bias
        );
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}
