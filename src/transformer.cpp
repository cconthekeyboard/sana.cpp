#include "transformer.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include "common_ops.h"
#include "transformer_ops.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace {

size_t infer_square_side(size_t tokens) {
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(tokens)));
    if (side * side != tokens) {
        throw std::invalid_argument("token count must form a square grid");
    }
    return side;
}

std::string block_weight_name(const std::string & block_prefix, const std::string & suffix) {
    if (block_prefix.empty()) {
        return suffix;
    }
    return block_prefix + "." + suffix;
}

bool parse_trailing_index(
    const std::string & prefix,
    std::string * stem,
    size_t * index
) {
    const std::size_t dot = prefix.rfind('.');
    if (dot == std::string::npos || dot + 1 >= prefix.size()) {
        return false;
    }

    const std::string tail = prefix.substr(dot + 1);
    for (char ch : tail) {
        if (ch < '0' || ch > '9') {
            return false;
        }
    }

    *stem = prefix.substr(0, dot);
    *index = static_cast<size_t>(std::stoull(tail));
    return true;
}

Tensor repeat_batch_4d_local(const Tensor & input, size_t repeats) {
    if (input.rank() != 4) {
        throw std::invalid_argument("repeat_batch_4d_local expects a rank-4 tensor");
    }
    if (repeats == 0) {
        throw std::invalid_argument("repeat count must be greater than 0");
    }

    const size_t batch = input.dim_size(0);
    const size_t channels = input.dim_size(1);
    const size_t height = input.dim_size(2);
    const size_t width = input.dim_size(3);
    const size_t input_size = batch * channels * height * width;
    Tensor out({batch * repeats, channels, height, width});
    const float * input_data = input.data().data();
    float * out_data = out.data().data();
    for (size_t r = 0; r < repeats; ++r) {
        std::copy(input_data, input_data + input_size, out_data + r * input_size);
    }
    return out;
}

Tensor timestep_embedding_2d(const Tensor & timesteps, size_t embedding_dim) {
    if (timesteps.rank() != 1) {
        throw std::invalid_argument("timesteps must be rank 1");
    }

    const size_t batch_size = timesteps.dim_size(0);
    const size_t half_dim = embedding_dim / 2;
    Tensor out({batch_size, embedding_dim});
    const float * timesteps_data = timesteps.data().data();
    float * out_data = out.data().data();

    std::vector<float> freq(half_dim);
    const float log_scale = -std::log(10000.0f) / static_cast<float>(half_dim);
#if defined(__APPLE__)
    for (size_t i = 0; i < half_dim; ++i) {
        freq[i] = static_cast<float>(i) * log_scale;
    }
    const int half_dim_count = static_cast<int>(half_dim);
    vvexpf(freq.data(), freq.data(), &half_dim_count);
#else
    for (size_t i = 0; i < half_dim; ++i) {
        freq[i] = std::exp(static_cast<float>(i) * log_scale);
    }
#endif

    std::vector<float> value(half_dim);
    for (size_t b = 0; b < batch_size; ++b) {
        const float t = timesteps_data[b];
        float * out_row = out_data + b * embedding_dim;
#if defined(__APPLE__)
        vDSP_vsmul(freq.data(), 1, &t, value.data(), 1, half_dim);
        vvcosf(out_row, value.data(), &half_dim_count);
        vvsinf(out_row + half_dim, value.data(), &half_dim_count);
#else
        for (size_t i = 0; i < half_dim; ++i) {
            value[i] = t * freq[i];
            out_row[i] = std::cos(value[i]);
            out_row[half_dim + i] = std::sin(value[i]);
        }
#endif
        if (embedding_dim % 2 == 1) {
            out_row[embedding_dim - 1] = 0.0f;
        }
    }
    return out;
}

// layer norm: shift = embedded_timestep + scale_shift_table[0], scale =
// embedded_timestep + scale_shift_table[1].
std::pair<Tensor, Tensor> compute_final_modulation(
    const Tensor & embedded_timestep,
    const Tensor & scale_shift_table
) {
    if (embedded_timestep.rank() != 2) {
        throw std::invalid_argument("embedded_timestep must be rank 2");
    }
    if (scale_shift_table.rank() != 2 || scale_shift_table.dim_size(0) != 2) {
        throw std::invalid_argument("final scale_shift_table must have shape {2, hidden}");
    }
    if (embedded_timestep.dim_size(1) != scale_shift_table.dim_size(1)) {
        throw std::invalid_argument("embedded_timestep hidden size must match final scale_shift_table");
    }

    const size_t batch_size = embedded_timestep.dim_size(0);
    const size_t hidden = embedded_timestep.dim_size(1);
    Tensor shift({batch_size, 1, hidden});
    Tensor scale({batch_size, 1, hidden});
    const float * embedded_data = embedded_timestep.data().data();
    const float * table_data = scale_shift_table.data().data();
    float * shift_data = shift.data().data();
    float * scale_data = scale.data().data();
    for (size_t b = 0; b < batch_size; ++b) {
        const float * embedded_row = embedded_data + b * hidden;
        float * shift_row = shift_data + b * hidden;
        float * scale_row = scale_data + b * hidden;
#if defined(__APPLE__)
        vDSP_vadd(embedded_row, 1, table_data, 1, shift_row, 1, hidden);
        vDSP_vadd(embedded_row, 1, table_data + hidden, 1, scale_row, 1, hidden);
#else
        for (size_t h = 0; h < hidden; ++h) {
            shift_row[h] = embedded_row[h] + table_data[h];
            scale_row[h] = embedded_row[h] + table_data[hidden + h];
        }
#endif
    }
    return {shift, scale};
}

}  // namespace

SanaCaptionWeights load_caption_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index
) {
    Tensor linear_1_weight = load_linear_weight_tensor(
        weights_dir,
        index,
        "caption_projection.linear_1.weight"
    );
    Tensor linear_1_bias = load_weight_tensor(
        weights_dir,
        index,
        "caption_projection.linear_1.bias"
    );
    Tensor linear_2_weight = load_linear_weight_tensor(
        weights_dir,
        index,
        "caption_projection.linear_2.weight"
    );
    Tensor linear_2_bias = load_weight_tensor(
        weights_dir,
        index,
        "caption_projection.linear_2.bias"
    );

    return SanaCaptionWeights{
        linear_1_weight,
        linear_1_bias,
        linear_2_weight,
        linear_2_bias
    };
}

SanaBlockWeights load_block_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    const std::string & block_prefix,
    size_t self_num_heads,
    size_t cross_num_heads
) {
    if (self_num_heads == 0) {
        throw std::invalid_argument("self_num_heads must be greater than 0");
    }
    if (cross_num_heads == 0) {
        throw std::invalid_argument("cross_num_heads must be greater than 0");
    }

    Tensor scale_shift_table = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "scale_shift_table")
    );
    Tensor q_weight = load_linear_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn1.to_q.weight")
    );
    Tensor k_weight = load_linear_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn1.to_k.weight")
    );
    Tensor v_weight = load_linear_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn1.to_v.weight")
    );
    Tensor out_weight = load_linear_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn1.to_out.0.weight")
    );
    Tensor out_bias = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn1.to_out.0.bias")
    );
    Tensor cross_q_weight = load_linear_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn2.to_q.weight")
    );
    Tensor cross_q_bias = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn2.to_q.bias")
    );
    Tensor cross_k_weight = load_linear_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn2.to_k.weight")
    );
    Tensor cross_k_bias = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn2.to_k.bias")
    );
    Tensor cross_v_weight = load_linear_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn2.to_v.weight")
    );
    Tensor cross_v_bias = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn2.to_v.bias")
    );
    Tensor cross_out_weight = load_linear_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn2.to_out.0.weight")
    );
    Tensor cross_out_bias = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "attn2.to_out.0.bias")
    );
    Tensor ff_inverted_weight = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "ff.conv_inverted.weight")
    );
    Tensor ff_inverted_bias = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "ff.conv_inverted.bias")
    );
    Tensor ff_depth_weight = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "ff.conv_depth.weight")
    );
    Tensor ff_depth_bias = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "ff.conv_depth.bias")
    );
    Tensor ff_point_weight = load_weight_tensor(
        weights_dir,
        index,
        block_weight_name(block_prefix, "ff.conv_point.weight")
    );

    if (scale_shift_table.rank() != 2 || scale_shift_table.dim_size(0) != 6) {
        throw std::runtime_error("scale_shift_table must have shape {6, hidden}");
    }
    if (q_weight.rank() != 2 || k_weight.rank() != 2 || v_weight.rank() != 2 || out_weight.rank() != 2) {
        throw std::runtime_error("attn1 projection weights must be rank 2");
    }
    if (out_bias.rank() != 1) {
        throw std::runtime_error("attn1 output bias must be rank 1");
    }
    if (cross_q_weight.rank() != 2 || cross_k_weight.rank() != 2 ||
        cross_v_weight.rank() != 2 || cross_out_weight.rank() != 2) {
        throw std::runtime_error("attn2 projection weights must be rank 2");
    }
    if (cross_q_bias.rank() != 1 || cross_k_bias.rank() != 1 ||
        cross_v_bias.rank() != 1 || cross_out_bias.rank() != 1) {
        throw std::runtime_error("attn2 biases must be rank 1");
    }
    if (ff_inverted_weight.rank() != 4 || ff_depth_weight.rank() != 4 || ff_point_weight.rank() != 4) {
        throw std::runtime_error("ff conv weights must be rank 4");
    }
    if (ff_inverted_bias.rank() != 1 || ff_depth_bias.rank() != 1) {
        throw std::runtime_error("ff conv biases must be rank 1");
    }

    const size_t hidden = q_weight.dim_size(1);
    if (scale_shift_table.dim_size(1) != hidden) {
        throw std::runtime_error("scale_shift_table hidden size must match attention hidden size");
    }
    if (hidden % self_num_heads != 0) {
        throw std::runtime_error("self attention hidden size must be divisible by self_num_heads");
    }
    if (hidden % cross_num_heads != 0) {
        throw std::runtime_error("cross attention hidden size must be divisible by cross_num_heads");
    }
    if (cross_q_weight.dim_size(1) != hidden || cross_out_weight.dim_size(0) != hidden) {
        throw std::runtime_error("attn2 hidden size must match block hidden size");
    }
    if (cross_q_bias.dim_size(0) != hidden || cross_out_bias.dim_size(0) != hidden) {
        throw std::runtime_error("attn2 q/out bias size must match block hidden size");
    }
    if (ff_inverted_weight.dim_size(1) != hidden || ff_point_weight.dim_size(0) != hidden) {
        throw std::runtime_error("ff hidden size must match block hidden size");
    }
    if (ff_inverted_bias.dim_size(0) != ff_inverted_weight.dim_size(0) ||
        ff_depth_weight.dim_size(0) != ff_inverted_weight.dim_size(0) ||
        ff_depth_bias.dim_size(0) != ff_inverted_weight.dim_size(0)) {
        throw std::runtime_error("ff expanded channel sizes must match");
    }
    if (ff_point_weight.dim_size(1) * 2 != ff_inverted_weight.dim_size(0)) {
        throw std::runtime_error("ff point conv input channels must match split GLUMB width");
    }
    Tensor zero_bias({hidden});
    zero_bias.fill(0.0f);

    Tensor ones_norm({hidden});
    ones_norm.fill(1.0f);
    Tensor zero_norm({hidden});
    zero_norm.fill(0.0f);

    return {
        scale_shift_table,
        ones_norm,
        zero_norm,
        q_weight,
        zero_bias,
        k_weight,
        zero_bias,
        v_weight,
        zero_bias,
        out_weight,
        out_bias,
        cross_q_weight,
        cross_q_bias,
        cross_k_weight,
        cross_k_bias,
        cross_v_weight,
        cross_v_bias,
        cross_out_weight,
        cross_out_bias,
        ones_norm,
        zero_norm,
        ff_inverted_weight,
        ff_inverted_bias,
        ff_depth_weight,
        ff_depth_bias,
        ff_point_weight,
        self_num_heads,
        cross_num_heads,
    };
}

std::vector<std::string> find_block_prefixes(
    const std::unordered_map<std::string, SavedWeightInfo> & index
) {
    const std::string suffix = ".scale_shift_table";
    std::set<std::string> unique_prefixes;
    for (const auto & entry : index) {
        if (ends_with(entry.first, suffix)) {
            unique_prefixes.insert(entry.first.substr(0, entry.first.size() - suffix.size()));
        }
    }

    std::vector<std::string> prefixes(unique_prefixes.begin(), unique_prefixes.end());
    std::sort(prefixes.begin(), prefixes.end(), [](const std::string & a, const std::string & b) {
        std::string a_stem;
        std::string b_stem;
        size_t a_index = 0;
        size_t b_index = 0;
        const bool a_has_index = parse_trailing_index(a, &a_stem, &a_index);
        const bool b_has_index = parse_trailing_index(b, &b_stem, &b_index);
        if (a_has_index && b_has_index && a_stem == b_stem) {
            return a_index < b_index;
        }
        return a < b;
    });
    return prefixes;
}

std::vector<SanaBlockWeights> load_model_block_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    size_t self_num_heads,
    size_t cross_num_heads
) {
    std::vector<std::string> block_prefixes = find_block_prefixes(index);
    if (block_prefixes.empty()) {
        throw std::invalid_argument("weight index does not contain any prefixed model blocks");
    }

    std::vector<SanaBlockWeights> blocks;
    blocks.reserve(block_prefixes.size());
    for (const std::string & block_prefix : block_prefixes) {
        blocks.push_back(load_block_weights(
            weights_dir,
            index,
            block_prefix,
            self_num_heads,
            cross_num_heads
        ));
    }
    return blocks;
}

SanaDenoiserWeights load_denoiser_weights(
    const std::string & block_weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & block_index,
    const std::string & caption_weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & caption_index,
    const std::string & denoiser_weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & denoiser_index,
    size_t self_num_heads,
    size_t cross_num_heads
) {
    SanaCaptionWeights caption = load_caption_weights(caption_weights_dir, caption_index);
    std::vector<SanaBlockWeights> blocks = load_model_block_weights(
        block_weights_dir,
        block_index,
        self_num_heads,
        cross_num_heads
    );
    SanaDenoiserWeights weights{
        caption,
        blocks,
        load_weight_tensor(denoiser_weights_dir, denoiser_index, "patch_embed.proj.weight"),
        load_weight_tensor(denoiser_weights_dir, denoiser_index, "patch_embed.proj.bias"),
        load_linear_weight_tensor(denoiser_weights_dir, denoiser_index, "time_embed.emb.timestep_embedder.linear_1.weight"),
        load_weight_tensor(denoiser_weights_dir, denoiser_index, "time_embed.emb.timestep_embedder.linear_1.bias"),
        load_linear_weight_tensor(denoiser_weights_dir, denoiser_index, "time_embed.emb.timestep_embedder.linear_2.weight"),
        load_weight_tensor(denoiser_weights_dir, denoiser_index, "time_embed.emb.timestep_embedder.linear_2.bias"),
        load_linear_weight_tensor(denoiser_weights_dir, denoiser_index, "time_embed.linear.weight"),
        load_weight_tensor(denoiser_weights_dir, denoiser_index, "time_embed.linear.bias"),
        load_weight_tensor(denoiser_weights_dir, denoiser_index, "caption_norm.weight"),
        load_weight_tensor(denoiser_weights_dir, denoiser_index, "scale_shift_table"),
        load_linear_weight_tensor(denoiser_weights_dir, denoiser_index, "proj_out.weight"),
        load_weight_tensor(denoiser_weights_dir, denoiser_index, "proj_out.bias"),
    };
    validate_denoiser_weights(weights);
    return weights;
}

// Computes one transformer block's 6 AdaLN modulation tensors (shift/scale/gate)
//   out[b, part, h] = timestep[b, part*hidden + h] + scale_shift_table[part, h]
//   timestep:          [batch, 6 * hidden]  (flat, 6 concatenated hidden-sized chunks)
//   scale_shift_table:  [6, hidden]          (fixed per-block weight)
//   returns: {shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp},
//   each [batch, 1, hidden], in that order (matching SanaBlockModulation's fields).
SanaBlockModulation compute_sana_block_modulation(
    const Tensor & timestep,
    const Tensor & scale_shift_table
) {
    if (timestep.rank() != 2) {
        throw std::invalid_argument("timestep must be rank 2");
    }
    if (scale_shift_table.rank() != 2) {
        throw std::invalid_argument("scale_shift_table must be rank 2");
    }
    if (scale_shift_table.dim_size(0) != 6) {
        throw std::invalid_argument("scale_shift_table must have shape {6, hidden}");
    }

    const size_t batch_size = timestep.dim_size(0);
    const size_t hidden = scale_shift_table.dim_size(1);
    if (timestep.dim_size(1) != 6 * hidden) {
        throw std::invalid_argument("timestep width must equal 6 * hidden");
    }

    const size_t row_width = 6 * hidden;
    Tensor combined({batch_size, 6, hidden});
    const float * timestep_data = timestep.data().data();
    const float * scale_shift_data = scale_shift_table.data().data();
    float * combined_data = combined.data().data();
    for (size_t b = 0; b < batch_size; ++b) {
        const float * timestep_row = timestep_data + b * row_width;
        float * combined_row = combined_data + b * row_width;
#if defined(__APPLE__)
        vDSP_vadd(timestep_row, 1, scale_shift_data, 1, combined_row, 1, row_width);
#else
        for (size_t i = 0; i < row_width; ++i) {
            combined_row[i] = timestep_row[i] + scale_shift_data[i];
        }
#endif
    }

    auto split = combined.split_six_way_3d();
    return {
        std::get<0>(split),
        std::get<1>(split),
        std::get<2>(split),
        std::get<3>(split),
        std::get<4>(split),
        std::get<5>(split),
    };
}

Tensor run_model_blocks(
    const Tensor & input,
    const Tensor & context,
    const Tensor & timestep,
    const SanaDenoiserWeights & weights,
    size_t block_begin,
    size_t block_end,
    const Tensor * encoder_attention_bias
) {
    if (block_begin > block_end || block_end > weights.blocks.size()) {
        throw std::invalid_argument("invalid model block range");
    }

    Tensor hidden = input;
    for (size_t block_index = block_begin; block_index < block_end; ++block_index) {
        hidden = block_forward(
            hidden,
            context,
            timestep,
            weights.blocks[block_index],
            encoder_attention_bias
        ).after_ff;
    }
    return hidden;
}

Tensor apply_classifier_free_guidance(
    const Tensor & noise_pred,
    float guidance_scale
) {
    if (noise_pred.rank() != 4) {
        throw std::invalid_argument("noise_pred must be rank 4");
    }
    if (noise_pred.dim_size(0) % 2 != 0) {
        throw std::invalid_argument("noise_pred batch size must be even for classifier-free guidance");
    }

    const size_t batch_size = noise_pred.dim_size(0) / 2;
    const size_t channels = noise_pred.dim_size(1);
    const size_t height = noise_pred.dim_size(2);
    const size_t width = noise_pred.dim_size(3);
    Tensor out({batch_size, channels, height, width});
    //out = uncond + guidance_scale * (text - uncond)
    const size_t n = batch_size * channels * height * width;
    const float * uncond = noise_pred.data().data();
    const float * text = uncond + n;
    float * out_data = out.data().data();

#if defined(__APPLE__)
    vDSP_vsub(uncond, 1, text, 1, out_data, 1, n);            // out = text - uncond
    vDSP_vsma(out_data, 1, &guidance_scale, uncond, 1, out_data, 1, n);  // out = out*scale + uncond
#else
    for (size_t i = 0; i < n; ++i) {
        out_data[i] = uncond[i] + guidance_scale * (text[i] - uncond[i]);
    }
#endif

    return out;
}

Tensor run_full_denoising_step_with_cfg(
    const Tensor & latents,
    const Tensor & context,
    float guidance_scale,
    SanaSchedulerState & scheduler_state,
    const SanaSchedulerConfig & scheduler_config,
    const SanaDenoiserWeights & denoiser_weights,
    const Tensor * encoder_attention_bias
) {
    if (scheduler_state.step_index >= scheduler_state.schedule.timesteps.size()) {
        throw std::invalid_argument("scheduler step_index is out of range");
    }
    if (context.rank() != 3) {
        throw std::invalid_argument("context must be rank 3");
    }
    if (context.dim_size(0) != latents.dim_size(0) * 2) {
        throw std::invalid_argument("context batch size must equal 2 * latent batch size for CFG");
    }
    if (encoder_attention_bias && encoder_attention_bias->dim_size(0) != latents.dim_size(0) * 2) {
        throw std::invalid_argument("encoder_attention_bias batch size must equal 2 * latent batch size for CFG");
    }
    validate_denoiser_weights(denoiser_weights);

    Tensor latent_model_input = repeat_batch_4d_local(latents, 2);
    Tensor timestep_values({latent_model_input.dim_size(0)});
    timestep_values.fill(scheduler_state.schedule.timesteps[scheduler_state.step_index]);

    Tensor hidden_grid = conv2d_1x1(
        latent_model_input,
        denoiser_weights.patch_embed_weight,
        &denoiser_weights.patch_embed_bias
    );
    Tensor hidden_tokens = hidden_grid.grid_to_tokens_3d();

    Tensor time_proj = timestep_embedding_2d(timestep_values, 256);
    Tensor embedded_timestep = linear_2d(
        time_proj,
        denoiser_weights.time_embed_linear_1_weight,
        &denoiser_weights.time_embed_linear_1_bias
    );
    embedded_timestep.silu_inplace();
    embedded_timestep = linear_2d(
        embedded_timestep,
        denoiser_weights.time_embed_linear_2_weight,
        &denoiser_weights.time_embed_linear_2_bias
    );
    Tensor timestep_hidden = embedded_timestep;
    timestep_hidden.silu_inplace();
    Tensor timestep_modulation = linear_2d(
        timestep_hidden,
        denoiser_weights.time_projection_weight,
        &denoiser_weights.time_projection_bias
    );

    Tensor hidden_states = run_model_blocks(
        hidden_tokens,
        context,
        timestep_modulation,
        denoiser_weights,
        0,
        denoiser_weights.blocks.size(),
        encoder_attention_bias
    );

    Tensor normalized = hidden_states.layer_norm_3d_lastdim(1e-6f);
    auto [shift, scale] = compute_final_modulation(
        embedded_timestep,
        denoiser_weights.final_scale_shift_table
    );
    Tensor modulated = normalized.modulate_3d_lastdim(shift, scale);
    Tensor projected = linear_3d_lastdim(
        modulated,
        denoiser_weights.proj_out_weight,
        &denoiser_weights.proj_out_bias
    );
    Tensor noise_pred = projected.tokens_to_grid_4d(
        latent_model_input.dim_size(2), latent_model_input.dim_size(3)
    );

    Tensor guided_noise_pred = apply_classifier_free_guidance(noise_pred, guidance_scale);
    return step_sana_scheduler(
        guided_noise_pred,
        latents,
        scheduler_state,
        scheduler_config
    );
}

Tensor run_full_denoising_loop_with_cfg(
    const Tensor & initial_latents,
    const Tensor & encoder_hidden_states,
    float guidance_scale,
    SanaSchedulerState & scheduler_state,
    const SanaSchedulerConfig & scheduler_config,
    const SanaDenoiserWeights & denoiser_weights,
    const Tensor * encoder_attention_bias
) {
    Tensor context = project_caption_context(
        encoder_hidden_states,
        denoiser_weights.caption.linear_1_weight,
        &denoiser_weights.caption.linear_1_bias,
        denoiser_weights.caption.linear_2_weight,
        &denoiser_weights.caption.linear_2_bias,
        denoiser_weights.caption_norm_weight
    );

    Tensor latents = initial_latents;
    const size_t num_steps = scheduler_state.schedule.timesteps.size();
    while (scheduler_state.step_index < num_steps) {
        latents = run_full_denoising_step_with_cfg(
            latents,
            context,
            guidance_scale,
            scheduler_state,
            scheduler_config,
            denoiser_weights,
            encoder_attention_bias
        );
    }
    return latents;
}

SanaBlockStageOutputs block_forward(
    const Tensor & input,
    const Tensor & context,
    const Tensor & timestep,
    const SanaBlockWeights & weights,
    const Tensor * encoder_attention_bias
) {
    const SanaBlockModulation modulation =
        compute_sana_block_modulation(timestep, weights.scale_shift_table);

    Tensor x = input.layer_norm_3d_lastdim(1e-6f, &weights.attn_norm_gamma, &weights.attn_norm_beta);
    Tensor attn_input = x.modulate_3d_lastdim(modulation.shift_msa, modulation.scale_msa);
    Tensor attn = attention_block_3d(
        attn_input,
        weights.q_weight, &weights.q_bias,
        weights.k_weight, &weights.k_bias,
        weights.v_weight, &weights.v_bias,
        weights.out_weight, &weights.out_bias,
        weights.self_num_heads
    );
    attn.mul_inplace_broadcast_lastdim(modulation.gate_msa);

    Tensor residual = input;
    residual.add_inplace(attn);
    Tensor after_self = residual;
    Tensor cross = cross_attention_block_3d(
        residual,
        context,
        encoder_attention_bias,
        weights.cross_q_weight, &weights.cross_q_bias,
        weights.cross_k_weight, &weights.cross_k_bias,
        weights.cross_v_weight, &weights.cross_v_bias,
        weights.cross_out_weight, &weights.cross_out_bias,
        weights.cross_num_heads
    );
    residual.add_inplace(cross);
    Tensor after_cross = residual;
    x = residual.layer_norm_3d_lastdim(1e-6f, &weights.mlp_norm_gamma, &weights.mlp_norm_beta);
    Tensor mlp_input = x.modulate_3d_lastdim(modulation.shift_mlp, modulation.scale_mlp);
    const size_t tokens = mlp_input.dim_size(1);
    const size_t side = infer_square_side(tokens);
    Tensor mlp_grid = mlp_input.tokens_to_grid_4d(side, side);
    Tensor mlp_grid_out = glumb_conv(
        mlp_grid,
        weights.ff_inverted_weight, &weights.ff_inverted_bias,
        weights.ff_depth_weight, &weights.ff_depth_bias,
        weights.ff_point_weight
    );
    Tensor mlp = mlp_grid_out.grid_to_tokens_3d();
    mlp.mul_inplace_broadcast_lastdim(modulation.gate_mlp);
    residual.add_inplace(mlp);
    return {after_self, after_cross, residual};
}

void validate_model_block_sequence(const std::vector<SanaBlockWeights> & blocks) {
    if (blocks.empty()) {
        throw std::invalid_argument("model must contain at least one block");
    }

    const size_t hidden = blocks[0].scale_shift_table.dim_size(1);
    for (size_t i = 0; i < blocks.size(); ++i) {
        const SanaBlockWeights & block = blocks[i];
        if (block.scale_shift_table.rank() != 2 || block.scale_shift_table.dim_size(0) != 6) {
            throw std::invalid_argument("each block scale_shift_table must have shape {6, hidden}");
        }
        if (block.scale_shift_table.dim_size(1) != hidden) {
            throw std::invalid_argument("all model blocks must use the same hidden size");
        }
        if (block.q_weight.rank() != 2 || block.q_weight.dim_size(0) != hidden ||
            block.q_weight.dim_size(1) != hidden) {
            throw std::invalid_argument("each block q_weight must have shape {hidden, hidden}");
        }
        if (block.cross_out_bias.rank() != 1 || block.cross_out_bias.dim_size(0) != hidden) {
            throw std::invalid_argument("each block cross_out_bias must match hidden size");
        }
        if (block.ff_point_weight.rank() != 4 || block.ff_point_weight.dim_size(0) != hidden) {
            throw std::invalid_argument("each block ff_point_weight output channels must match hidden size");
        }
    }
}

void validate_denoiser_weights(const SanaDenoiserWeights & denoiser_weights) {
    validate_model_block_sequence(denoiser_weights.blocks);

    if (denoiser_weights.caption.linear_1_weight.rank() != 2 ||
        denoiser_weights.caption.linear_2_weight.rank() != 2 ||
        denoiser_weights.caption.linear_1_bias.rank() != 1 ||
        denoiser_weights.caption.linear_2_bias.rank() != 1) {
        throw std::invalid_argument("caption weights must use expected linear tensor ranks");
    }

    const size_t caption_hidden = denoiser_weights.caption.linear_2_weight.dim_size(1);
    if (denoiser_weights.caption.linear_1_weight.dim_size(1) != denoiser_weights.caption.linear_1_bias.dim_size(0) ||
        denoiser_weights.caption.linear_2_weight.dim_size(0) != denoiser_weights.caption.linear_2_bias.dim_size(0)) {
        throw std::invalid_argument("caption weight dimensions must be internally consistent");
    }

    const size_t hidden = denoiser_weights.blocks[0].scale_shift_table.dim_size(1);
    if (caption_hidden != hidden) {
        throw std::invalid_argument("caption output hidden size must match model block hidden size");
    }

    const size_t latent_channels = denoiser_weights.patch_embed_weight.dim_size(1);
    if (denoiser_weights.patch_embed_weight.rank() != 4 ||
        denoiser_weights.patch_embed_weight.dim_size(0) != hidden ||
        denoiser_weights.patch_embed_weight.dim_size(2) != 1 ||
        denoiser_weights.patch_embed_weight.dim_size(3) != 1) {
        throw std::invalid_argument("patch_embed_weight must have shape {hidden, latent_channels, 1, 1}");
    }
    if (denoiser_weights.patch_embed_bias.rank() != 1 ||
        denoiser_weights.patch_embed_bias.dim_size(0) != hidden) {
        throw std::invalid_argument("patch_embed_bias must match hidden size");
    }
    if (denoiser_weights.time_embed_linear_1_weight.rank() != 2 ||
        denoiser_weights.time_embed_linear_1_weight.dim_size(0) != 256 ||
        denoiser_weights.time_embed_linear_1_weight.dim_size(1) != hidden) {
        throw std::invalid_argument("time_embed_linear_1_weight must have shape {256, hidden}");
    }
    if (denoiser_weights.time_embed_linear_1_bias.rank() != 1 ||
        denoiser_weights.time_embed_linear_1_bias.dim_size(0) != hidden) {
        throw std::invalid_argument("time_embed_linear_1_bias must match hidden size");
    }
    if (denoiser_weights.time_embed_linear_2_weight.rank() != 2 ||
        denoiser_weights.time_embed_linear_2_weight.dim_size(0) != hidden ||
        denoiser_weights.time_embed_linear_2_weight.dim_size(1) != hidden) {
        throw std::invalid_argument("time_embed_linear_2_weight must have shape {hidden, hidden}");
    }
    if (denoiser_weights.time_embed_linear_2_bias.rank() != 1 ||
        denoiser_weights.time_embed_linear_2_bias.dim_size(0) != hidden) {
        throw std::invalid_argument("time_embed_linear_2_bias must match hidden size");
    }
    if (denoiser_weights.time_projection_weight.rank() != 2 ||
        denoiser_weights.time_projection_weight.dim_size(0) != hidden ||
        denoiser_weights.time_projection_weight.dim_size(1) != 6 * hidden) {
        throw std::invalid_argument("time_projection_weight must have shape {hidden, 6 * hidden}");
    }
    if (denoiser_weights.time_projection_bias.rank() != 1 ||
        denoiser_weights.time_projection_bias.dim_size(0) != 6 * hidden) {
        throw std::invalid_argument("time_projection_bias must match 6 * hidden");
    }
    if (denoiser_weights.caption_norm_weight.rank() != 1 ||
        denoiser_weights.caption_norm_weight.dim_size(0) != hidden) {
        throw std::invalid_argument("caption_norm_weight must match hidden size");
    }
    if (denoiser_weights.final_scale_shift_table.rank() != 2 ||
        denoiser_weights.final_scale_shift_table.dim_size(0) != 2 ||
        denoiser_weights.final_scale_shift_table.dim_size(1) != hidden) {
        throw std::invalid_argument("final_scale_shift_table must have shape {2, hidden}");
    }
    if (denoiser_weights.proj_out_weight.rank() != 2 ||
        denoiser_weights.proj_out_weight.dim_size(0) != hidden ||
        denoiser_weights.proj_out_weight.dim_size(1) != latent_channels) {
        throw std::invalid_argument("proj_out_weight must have shape {hidden, latent_channels}");
    }
    if (denoiser_weights.proj_out_bias.rank() != 1 ||
        denoiser_weights.proj_out_bias.dim_size(0) != latent_channels) {
        throw std::invalid_argument("proj_out_bias must match latent channel count");
    }
}



