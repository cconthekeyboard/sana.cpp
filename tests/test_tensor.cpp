#include "common_ops.h"
#include "tensor.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

int main() {
    Tensor image({2, 3, 4});
    assert(image.rank() == 3);
    assert(image.numel() == 24);
    assert(image.data().size() == 24);
    assert(image.dim_size(0) == 2);
    assert(image.dim_size(1) == 3);
    assert(image.dim_size(2) == 4);
    image.data()[0] = 1.0f;
    assert(image.data()[0] == 1.0f);
    image.at(1) = 2.5f;
    assert(image.data()[1] == 2.5f);
    const Tensor & image_const = image;
    assert(image_const.at(1) == 2.5f);
    for (size_t i = 1; i < image.data().size(); ++i) {
        if (i == 1) {
            continue;
        }
        assert(image.data()[i] == 0.0f);
    }

    Tensor scalar({});
    assert(scalar.rank() == 0);
    assert(scalar.numel() == 1);
    assert(scalar.data().size() == 1);
    scalar.fill(3.0f);
    assert(scalar.at(0) == 3.0f);

    image.fill(-1.25f);
    for (size_t i = 0; i < image.numel(); ++i) {
        assert(image.at(i) == -1.25f);
    }

    Tensor same({2, 3, 4});
    Tensor different_rank({2, 3});
    Tensor different_dim({2, 3, 5});
    assert(image.same_shape(same));
    assert(!image.same_shape(different_rank));
    assert(!image.same_shape(different_dim));
    assert(image.flat_index({0, 0, 0}) == 0);
    assert(image.flat_index({1, 2, 3}) == 23);
    assert(image.flat_index({1, 0, 0}) == 12);
    assert(image.flat_index({0, 1, 2}) == 6);
    image.at({1, 2, 3}) = 9.0f;
    assert(image.at(23) == 9.0f);
    assert(image_const.at({1, 2, 3}) == 9.0f);
    image.at({0, 0, 1}) = 7.0f;
    assert(image.at(1) == 7.0f);

    bool threw = false;

    Tensor lhs({2, 2});
    Tensor rhs({2, 2});
    lhs.fill(1.5f);
    rhs.fill(2.0f);
    lhs.add_inplace(rhs);
    for (size_t i = 0; i < lhs.numel(); ++i) {
        assert(lhs.at(i) == 3.5f);
    }
    lhs.mul_inplace_scalar(2.0f);
    for (size_t i = 0; i < lhs.numel(); ++i) {
        assert(lhs.at(i) == 7.0f);
    }

    scalar.mul_inplace_scalar(-2.0f);
    assert(scalar.at(0) == -6.0f);

    Tensor a({2, 3});
    a.at({0, 0}) = 1.0f;
    a.at({0, 1}) = 2.0f;
    a.at({0, 2}) = 3.0f;
    a.at({1, 0}) = 4.0f;
    a.at({1, 1}) = 5.0f;
    a.at({1, 2}) = 6.0f;

    Tensor t = a.transpose_2d();
    assert(t.shape() == std::vector<size_t>({3, 2}));
    assert(t.at({0, 0}) == 1.0f);
    assert(t.at({0, 1}) == 4.0f);
    assert(t.at({1, 0}) == 2.0f);
    assert(t.at({1, 1}) == 5.0f);
    assert(t.at({2, 0}) == 3.0f);
    assert(t.at({2, 1}) == 6.0f);

    Tensor norm3_input({1, 2, 2});
    norm3_input.at({0, 0, 0}) = 1.0f;
    norm3_input.at({0, 0, 1}) = 3.0f;
    norm3_input.at({0, 1, 0}) = 2.0f;
    norm3_input.at({0, 1, 1}) = 4.0f;
    Tensor norm3_out = norm3_input.layer_norm_3d_lastdim(0.0f);
    assert(norm3_out.shape() == std::vector<size_t>({1, 2, 2}));
    assert(norm3_out.at({0, 0, 0}) == -1.0f);
    assert(norm3_out.at({0, 0, 1}) == 1.0f);
    assert(norm3_out.at({0, 1, 0}) == -1.0f);
    assert(norm3_out.at({0, 1, 1}) == 1.0f);

    Tensor norm3_gamma({2});
    Tensor norm3_beta({2});
    norm3_gamma.at(0) = 2.0f;
    norm3_gamma.at(1) = 3.0f;
    norm3_beta.at(0) = 1.0f;
    norm3_beta.at(1) = -1.0f;
    Tensor norm3_affine = norm3_input.layer_norm_3d_lastdim(0.0f, &norm3_gamma, &norm3_beta);
    assert(norm3_affine.shape() == std::vector<size_t>({1, 2, 2}));
    assert(norm3_affine.at({0, 0, 0}) == -1.0f);
    assert(norm3_affine.at({0, 0, 1}) == 2.0f);
    assert(norm3_affine.at({0, 1, 0}) == -1.0f);
    assert(norm3_affine.at({0, 1, 1}) == 2.0f);

    Tensor rms_input({1, 2, 2});
    rms_input.at({0, 0, 0}) = 3.0f;
    rms_input.at({0, 0, 1}) = 4.0f;
    rms_input.at({0, 1, 0}) = 0.0f;
    rms_input.at({0, 1, 1}) = 5.0f;
    Tensor rms_weight({2});
    rms_weight.at(0) = 2.0f;
    rms_weight.at(1) = 3.0f;
    Tensor rms_out = rms_input.rms_norm_3d_lastdim(0.0f, &rms_weight);
    assert(rms_out.shape() == std::vector<size_t>({1, 2, 2}));
    assert(std::fabs(rms_out.at({0, 0, 0}) - 1.6970563f) < 1e-5f);
    assert(std::fabs(rms_out.at({0, 0, 1}) - 3.3941126f) < 1e-5f);
    assert(std::fabs(rms_out.at({0, 1, 0}) - 0.0f) < 1e-6f);
    assert(std::fabs(rms_out.at({0, 1, 1}) - 4.2426405f) < 1e-5f);

    Tensor mod_input({1, 2, 2});
    Tensor mod_shift({1, 1, 2});
    Tensor mod_scale({1, 1, 2});
    mod_input.at({0, 0, 0}) = 1.0f;
    mod_input.at({0, 0, 1}) = 2.0f;
    mod_input.at({0, 1, 0}) = 3.0f;
    mod_input.at({0, 1, 1}) = 4.0f;
    mod_shift.at({0, 0, 0}) = 10.0f;
    mod_shift.at({0, 0, 1}) = 20.0f;
    mod_scale.at({0, 0, 0}) = 0.5f;
    mod_scale.at({0, 0, 1}) = -0.5f;
    Tensor mod_out = mod_input.modulate_3d_lastdim(mod_shift, mod_scale);
    assert(mod_out.shape() == std::vector<size_t>({1, 2, 2}));
    assert(mod_out.at({0, 0, 0}) == 11.5f);
    assert(mod_out.at({0, 0, 1}) == 21.0f);
    assert(mod_out.at({0, 1, 0}) == 14.5f);
    assert(mod_out.at({0, 1, 1}) == 22.0f);

    Tensor grid_input({1, 4, 2});
    grid_input.at({0, 0, 0}) = 1.0f;
    grid_input.at({0, 0, 1}) = 2.0f;
    grid_input.at({0, 1, 0}) = 3.0f;
    grid_input.at({0, 1, 1}) = 4.0f;
    grid_input.at({0, 2, 0}) = 5.0f;
    grid_input.at({0, 2, 1}) = 6.0f;
    grid_input.at({0, 3, 0}) = 7.0f;
    grid_input.at({0, 3, 1}) = 8.0f;
    Tensor grid_out = grid_input.tokens_to_grid_4d(2, 2);
    assert(grid_out.shape() == std::vector<size_t>({1, 2, 2, 2}));
    assert(grid_out.at({0, 0, 0, 0}) == 1.0f);
    assert(grid_out.at({0, 0, 0, 1}) == 3.0f);
    assert(grid_out.at({0, 0, 1, 0}) == 5.0f);
    assert(grid_out.at({0, 0, 1, 1}) == 7.0f);
    assert(grid_out.at({0, 1, 0, 0}) == 2.0f);
    assert(grid_out.at({0, 1, 0, 1}) == 4.0f);
    assert(grid_out.at({0, 1, 1, 0}) == 6.0f);
    assert(grid_out.at({0, 1, 1, 1}) == 8.0f);
    Tensor tokens_roundtrip = grid_out.grid_to_tokens_3d();
    assert(tokens_roundtrip.shape() == std::vector<size_t>({1, 4, 2}));
    assert(tokens_roundtrip.at({0, 0, 0}) == 1.0f);
    assert(tokens_roundtrip.at({0, 0, 1}) == 2.0f);
    assert(tokens_roundtrip.at({0, 1, 0}) == 3.0f);
    assert(tokens_roundtrip.at({0, 1, 1}) == 4.0f);
    assert(tokens_roundtrip.at({0, 2, 0}) == 5.0f);
    assert(tokens_roundtrip.at({0, 2, 1}) == 6.0f);
    assert(tokens_roundtrip.at({0, 3, 0}) == 7.0f);
    assert(tokens_roundtrip.at({0, 3, 1}) == 8.0f);

    Tensor conv_input({1, 2, 2, 2});
    Tensor conv_weight({3, 2, 1, 1});
    Tensor conv_bias({3});
    conv_input.at({0, 0, 0, 0}) = 1.0f;
    conv_input.at({0, 0, 0, 1}) = 2.0f;
    conv_input.at({0, 0, 1, 0}) = 3.0f;
    conv_input.at({0, 0, 1, 1}) = 4.0f;
    conv_input.at({0, 1, 0, 0}) = 10.0f;
    conv_input.at({0, 1, 0, 1}) = 20.0f;
    conv_input.at({0, 1, 1, 0}) = 30.0f;
    conv_input.at({0, 1, 1, 1}) = 40.0f;

    conv_weight.at({0, 0, 0, 0}) = 1.0f;
    conv_weight.at({0, 1, 0, 0}) = 0.0f;
    conv_weight.at({1, 0, 0, 0}) = 0.0f;
    conv_weight.at({1, 1, 0, 0}) = 1.0f;
    conv_weight.at({2, 0, 0, 0}) = 1.0f;
    conv_weight.at({2, 1, 0, 0}) = 1.0f;

    conv_bias.at(0) = 0.0f;
    conv_bias.at(1) = 0.0f;
    conv_bias.at(2) = -1.0f;

    Tensor conv_out = conv2d_1x1(conv_input, conv_weight, &conv_bias);
    assert(conv_out.shape() == std::vector<size_t>({1, 3, 2, 2}));
    assert(conv_out.at({0, 0, 0, 0}) == 1.0f);
    assert(conv_out.at({0, 0, 1, 1}) == 4.0f);
    assert(conv_out.at({0, 1, 0, 0}) == 10.0f);
    assert(conv_out.at({0, 1, 1, 1}) == 40.0f);
    assert(conv_out.at({0, 2, 0, 0}) == 10.0f);
    assert(conv_out.at({0, 2, 0, 1}) == 21.0f);
    assert(conv_out.at({0, 2, 1, 0}) == 32.0f);
    assert(conv_out.at({0, 2, 1, 1}) == 43.0f);

    Tensor depth_input({1, 2, 2, 2});
    Tensor depth_weight({2, 1, 3, 3});
    Tensor depth_bias({2});
    depth_input.at({0, 0, 0, 0}) = 1.0f;
    depth_input.at({0, 0, 0, 1}) = 2.0f;
    depth_input.at({0, 0, 1, 0}) = 3.0f;
    depth_input.at({0, 0, 1, 1}) = 4.0f;
    depth_input.at({0, 1, 0, 0}) = 10.0f;
    depth_input.at({0, 1, 0, 1}) = 20.0f;
    depth_input.at({0, 1, 1, 0}) = 30.0f;
    depth_input.at({0, 1, 1, 1}) = 40.0f;

    depth_weight.fill(0.0f);
    depth_weight.at({0, 0, 1, 1}) = 1.0f;
    depth_weight.at({1, 0, 1, 1}) = 2.0f;

    depth_bias.at(0) = 0.5f;
    depth_bias.at(1) = -1.0f;

    Tensor depth_out = depthwise_conv2d_3x3(depth_input, depth_weight, &depth_bias);
    assert(depth_out.shape() == std::vector<size_t>({1, 2, 2, 2}));
    assert(depth_out.at({0, 0, 0, 0}) == 1.5f);
    assert(depth_out.at({0, 0, 0, 1}) == 2.5f);
    assert(depth_out.at({0, 0, 1, 0}) == 3.5f);
    assert(depth_out.at({0, 0, 1, 1}) == 4.5f);
    assert(depth_out.at({0, 1, 0, 0}) == 19.0f);
    assert(depth_out.at({0, 1, 0, 1}) == 39.0f);
    assert(depth_out.at({0, 1, 1, 0}) == 59.0f);
    assert(depth_out.at({0, 1, 1, 1}) == 79.0f);

    Tensor split_input({1, 4, 2, 2});
    for (size_t c = 0; c < 4; ++c) {
        for (size_t y = 0; y < 2; ++y) {
            for (size_t x = 0; x < 2; ++x) {
                split_input.at({0, c, y, x}) = static_cast<float>(c * 10 + y * 2 + x);
            }
        }
    }
    auto split_out = split_input.split_channels_4d();
    assert(split_out.first.shape() == std::vector<size_t>({1, 2, 2, 2}));
    assert(split_out.second.shape() == std::vector<size_t>({1, 2, 2, 2}));
    assert(split_out.first.at({0, 0, 0, 0}) == 0.0f);
    assert(split_out.first.at({0, 1, 1, 1}) == 13.0f);
    assert(split_out.second.at({0, 0, 0, 0}) == 20.0f);
    assert(split_out.second.at({0, 1, 1, 1}) == 33.0f);

    Tensor six_input({1, 6, 2});
    for (size_t part = 0; part < 6; ++part) {
        six_input.at({0, part, 0}) = static_cast<float>(part * 10 + 1);
        six_input.at({0, part, 1}) = static_cast<float>(part * 10 + 2);
    }
    auto six_out = six_input.split_six_way_3d();
    assert(std::get<0>(six_out).shape() == std::vector<size_t>({1, 1, 2}));
    assert(std::get<5>(six_out).shape() == std::vector<size_t>({1, 1, 2}));
    assert(std::get<0>(six_out).at({0, 0, 0}) == 1.0f);
    assert(std::get<0>(six_out).at({0, 0, 1}) == 2.0f);
    assert(std::get<5>(six_out).at({0, 0, 0}) == 51.0f);
    assert(std::get<5>(six_out).at({0, 0, 1}) == 52.0f);

    Tensor glumb_input({1, 2, 1, 1});
    Tensor glumb_inverted_weight({4, 2, 1, 1});
    Tensor glumb_inverted_bias({4});
    Tensor glumb_depth_weight({4, 1, 3, 3});
    Tensor glumb_depth_bias({4});
    Tensor glumb_point_weight({2, 2, 1, 1});

    glumb_input.at({0, 0, 0, 0}) = 1.0f;
    glumb_input.at({0, 1, 0, 0}) = 2.0f;

    glumb_inverted_weight.fill(0.0f);
    glumb_inverted_weight.at({0, 0, 0, 0}) = 1.0f;
    glumb_inverted_weight.at({1, 1, 0, 0}) = 1.0f;
    glumb_inverted_weight.at({2, 0, 0, 0}) = 1.0f;
    glumb_inverted_weight.at({3, 1, 0, 0}) = 1.0f;
    glumb_inverted_bias.fill(0.0f);

    glumb_depth_weight.fill(0.0f);
    glumb_depth_weight.at({0, 0, 1, 1}) = 1.0f;
    glumb_depth_weight.at({1, 0, 1, 1}) = 1.0f;
    glumb_depth_weight.at({2, 0, 1, 1}) = 1.0f;
    glumb_depth_weight.at({3, 0, 1, 1}) = 1.0f;
    glumb_depth_bias.fill(0.0f);

    glumb_point_weight.fill(0.0f);
    glumb_point_weight.at({0, 0, 0, 0}) = 1.0f;
    glumb_point_weight.at({1, 1, 0, 0}) = 1.0f;

    Tensor glumb_out = glumb_conv(
        glumb_input,
        glumb_inverted_weight, &glumb_inverted_bias,
        glumb_depth_weight, &glumb_depth_bias,
        glumb_point_weight
    );
    const float silu_1 = 1.0f / (1.0f + std::exp(-1.0f));
    const float silu_2 = 2.0f / (1.0f + std::exp(-2.0f));
    const float expected_0 = silu_1 * (silu_1 / (1.0f + std::exp(-silu_1)));
    const float expected_1 = silu_2 * (silu_2 / (1.0f + std::exp(-silu_2)));
    assert(glumb_out.shape() == std::vector<size_t>({1, 2, 1, 1}));
    assert(std::fabs(glumb_out.at({0, 0, 0, 0}) - expected_0) < 1e-6f);
    assert(std::fabs(glumb_out.at({0, 1, 0, 0}) - expected_1) < 1e-6f);

    Tensor act({3});
    act.at(0) = -1.0f;
    act.at(1) = 0.0f;
    act.at(2) = 1.0f;
    act.silu_inplace();
    assert(std::fabs(act.at(0) - (-0.2689414f)) < 1e-5f);
    assert(std::fabs(act.at(1) - 0.0f) < 1e-6f);
    assert(std::fabs(act.at(2) - 0.7310586f) < 1e-5f);

    Tensor diff_a({3});
    Tensor diff_b({3});
    diff_a.at(0) = 1.0f;
    diff_a.at(1) = -2.0f;
    diff_a.at(2) = 4.5f;
    diff_b.at(0) = 0.0f;
    diff_b.at(1) = -1.5f;
    diff_b.at(2) = 2.0f;
    assert(std::fabs(diff_a.max_abs_diff(diff_b) - 2.5f) < 1e-6f);
    assert(std::fabs(diff_a.mean_abs_diff(diff_b) - (4.0f / 3.0f)) < 1e-6f);

    Tensor mismatch({2, 3});
    threw = false;
    try {
        lhs.add_inplace(mismatch);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        diff_a.max_abs_diff(mismatch);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        diff_a.mean_abs_diff(mismatch);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        image.transpose_2d();
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        image.dim_size(3);
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        image.at(image.numel());
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        image_const.at(image_const.numel());
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        image.flat_index({1, 2});
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        image.flat_index({1, 3, 0});
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        image.at({2, 0, 0});
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        image_const.at({2, 0, 0});
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);

    return 0;
}
