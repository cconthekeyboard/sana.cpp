#include "vae.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

#include "vae_ops.h"

namespace {

SanaVaeResBlockWeights load_vae_res_block_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    const std::string & layer_prefix
) {
    return SanaVaeResBlockWeights{
        load_weight_tensor(weights_dir, index, layer_prefix + ".conv1.weight"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".conv1.bias"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".conv2.weight"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".norm.weight"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".norm.bias"),
    };
}

SanaVaeAttentionWeights load_vae_attention_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    const std::string & layer_prefix
) {
    return SanaVaeAttentionWeights{
        linear_weight_to_conv1x1(load_weight_tensor(weights_dir, index, layer_prefix + ".attn.to_q.weight")),
        linear_weight_to_conv1x1(load_weight_tensor(weights_dir, index, layer_prefix + ".attn.to_k.weight")),
        linear_weight_to_conv1x1(load_weight_tensor(weights_dir, index, layer_prefix + ".attn.to_v.weight")),
        load_weight_tensor(weights_dir, index, layer_prefix + ".attn.to_qkv_multiscale.0.proj_in.weight"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".attn.to_qkv_multiscale.0.proj_out.weight"),
        linear_weight_to_conv1x1(load_weight_tensor(weights_dir, index, layer_prefix + ".attn.to_out.weight")),
        load_weight_tensor(weights_dir, index, layer_prefix + ".attn.norm_out.weight"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".attn.norm_out.bias"),
    };
}

SanaVaeGlumbConvWeights load_vae_glumb_conv_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    const std::string & layer_prefix
) {
    return SanaVaeGlumbConvWeights{
        load_weight_tensor(weights_dir, index, layer_prefix + ".conv_out.conv_inverted.weight"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".conv_out.conv_inverted.bias"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".conv_out.conv_depth.weight"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".conv_out.conv_depth.bias"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".conv_out.conv_point.weight"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".conv_out.norm.weight"),
        load_weight_tensor(weights_dir, index, layer_prefix + ".conv_out.norm.bias"),
    };
}

SanaVaeUpBlockWeights load_vae_up_block_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    const std::string & prefix
) {
    const bool has_upsample = index.find(prefix + ".0.conv.weight") != index.end();
    const size_t layer_base = has_upsample ? 1 : 0;

    SanaVaeUpBlockWeights up_block;
    if (has_upsample) {
        up_block.upsample_conv_weight = load_weight_tensor(weights_dir, index, prefix + ".0.conv.weight");
        up_block.upsample_conv_bias = load_weight_tensor(weights_dir, index, prefix + ".0.conv.bias");
        up_block.is_res_block =
            index.find(prefix + "." + std::to_string(layer_base) + ".conv1.weight") != index.end();
    }

    for (size_t layer = layer_base; layer < layer_base + 3; ++layer) {
        const std::string layer_prefix = prefix + "." + std::to_string(layer);
        if (up_block.is_res_block) {
            up_block.res_blocks.push_back(load_vae_res_block_weights(weights_dir, index, layer_prefix));
        } else {
            up_block.attn_glumb_layers.push_back(SanaVaeAttnGlumbLayerWeights{
                load_vae_attention_weights(weights_dir, index, layer_prefix),
                load_vae_glumb_conv_weights(weights_dir, index, layer_prefix)
            });
        }
    }

    return up_block;
}

}  // namespace

std::vector<std::string> find_vae_decoder_up_block_prefixes(
    const std::unordered_map<std::string, SavedWeightInfo> & index
) {
    std::set<std::pair<size_t, std::string>> ordered;
    for (const auto & entry : index) {
        const std::string & name = entry.first;
        const std::string prefix = "decoder.up_blocks.";
        if (name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }

        const std::size_t next_dot = name.find('.', prefix.size());
        if (next_dot == std::string::npos) {
            continue;
        }

        const std::string block_index_text =
            name.substr(prefix.size(), next_dot - prefix.size());
        if (block_index_text.empty()) {
            continue;
        }

        bool numeric = true;
        for (char ch : block_index_text) {
            if (ch < '0' || ch > '9') {
                numeric = false;
                break;
            }
        }
        if (!numeric) {
            continue;
        }

        const size_t block_index =
            static_cast<size_t>(std::stoull(block_index_text));
        ordered.insert({block_index, prefix + block_index_text});
    }

    std::vector<std::string> prefixes;
    prefixes.reserve(ordered.size());
    for (const auto & entry : ordered) {
        prefixes.push_back(entry.second);
    }
    return prefixes;
}

SanaVaeWeights load_vae_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    float scaling_factor,
    size_t spatial_compression_ratio
) {
    SanaVaeDecoderWeights decoder{
        load_weight_tensor(weights_dir, index, "decoder.conv_in.weight"),
        load_weight_tensor(weights_dir, index, "decoder.conv_in.bias"),
        load_weight_tensor(weights_dir, index, "decoder.norm_out.weight"),
        load_weight_tensor(weights_dir, index, "decoder.norm_out.bias"),
        load_weight_tensor(weights_dir, index, "decoder.conv_out.weight"),
        load_weight_tensor(weights_dir, index, "decoder.conv_out.bias"),
        {}
    };

    for (const std::string & prefix : find_vae_decoder_up_block_prefixes(index)) {
        decoder.up_blocks.push_back(load_vae_up_block_weights(weights_dir, index, prefix));
    }

    return SanaVaeWeights{
        std::move(decoder),
        scaling_factor,
        spatial_compression_ratio,
    };
}

void validate_vae_decoder_weights(const SanaVaeDecoderWeights & weights) {
    if (weights.conv_in_weight.rank() != 4) {
        throw std::invalid_argument("vae decoder conv_in.weight must be rank 4");
    }
    if (weights.conv_in_bias.rank() != 1) {
        throw std::invalid_argument("vae decoder conv_in.bias must be rank 1");
    }
    if (weights.norm_out_weight.rank() != 1) {
        throw std::invalid_argument("vae decoder norm_out.weight must be rank 1");
    }
    if (weights.norm_out_bias.rank() != 1) {
        throw std::invalid_argument("vae decoder norm_out.bias must be rank 1");
    }
    if (weights.conv_out_weight.rank() != 4) {
        throw std::invalid_argument("vae decoder conv_out.weight must be rank 4");
    }
    if (weights.conv_out_bias.rank() != 1) {
        throw std::invalid_argument("vae decoder conv_out.bias must be rank 1");
    }
    if (weights.up_blocks.empty()) {
        throw std::invalid_argument("vae decoder must contain at least one up block");
    }
    if (weights.conv_in_weight.dim_size(0) != weights.conv_in_bias.dim_size(0)) {
        throw std::invalid_argument("vae decoder conv_in bias shape must match output channels");
    }
    if (weights.norm_out_weight.dim_size(0) != weights.norm_out_bias.dim_size(0)) {
        throw std::invalid_argument("vae decoder norm_out bias shape must match");
    }
    if (weights.conv_out_weight.dim_size(0) != weights.conv_out_bias.dim_size(0)) {
        throw std::invalid_argument("vae decoder conv_out bias shape must match output channels");
    }
    if (weights.norm_out_weight.dim_size(0) != weights.conv_out_weight.dim_size(1)) {
        throw std::invalid_argument("vae decoder norm_out channels must match conv_out input channels");
    }
}

void validate_vae_weights(const SanaVaeWeights & weights) {
    validate_vae_decoder_weights(weights.decoder);
    if (!(weights.scaling_factor > 0.0f)) {
        throw std::invalid_argument("vae scaling_factor must be positive");
    }
    if (weights.spatial_compression_ratio == 0) {
        throw std::invalid_argument("vae spatial_compression_ratio must be positive");
    }
}

Tensor run_vae_decoder_input(
    const Tensor & latents,
    const SanaVaeDecoderWeights & weights,
    bool in_shortcut
) {
    validate_vae_decoder_weights(weights);
    if (latents.rank() != 4) {
        throw std::invalid_argument("vae decoder input latents must be rank 4");
    }
    if (latents.dim_size(1) != weights.conv_in_weight.dim_size(1)) {
        throw std::invalid_argument("vae decoder input channels must match conv_in.weight");
    }

    Tensor projected = conv2d_same(
        latents,
        weights.conv_in_weight,
        &weights.conv_in_bias
    );
    if (!in_shortcut) {
        return projected;
    }

    const size_t output_channels = weights.conv_in_weight.dim_size(0);
    const size_t input_channels = latents.dim_size(1);
    if (output_channels % input_channels != 0) {
        throw std::invalid_argument("vae decoder conv_in output channels must be divisible by latent channels");
    }

    add_repeated_channels_inplace(projected, latents, output_channels / input_channels);
    return projected;
}

Tensor run_vae_decode(
    const Tensor & latents,
    const SanaVaeWeights & weights
) {
    validate_vae_weights(weights);
    if (latents.rank() != 4) {
        throw std::invalid_argument("vae decode latents must be rank 4");
    }
    if (latents.dim_size(1) != weights.decoder.conv_in_weight.dim_size(1)) {
        throw std::invalid_argument("vae decode latent channels must match decoder conv_in input");
    }

    Tensor scaled = latents;
    scaled.mul_inplace_scalar(1.0f / weights.scaling_factor);
    Tensor hidden = run_vae_decoder_input(scaled, weights.decoder, true);

    const std::vector<SanaVaeUpBlockWeights> & up_blocks = weights.decoder.up_blocks;
    for (auto it = up_blocks.rbegin(); it != up_blocks.rend(); ++it) {
        const SanaVaeUpBlockWeights & up_block = *it;

        if (up_block.upsample_conv_weight.has_value()) {
            hidden = run_vae_upsample_block(
                hidden,
                *up_block.upsample_conv_weight,
                &*up_block.upsample_conv_bias,
                true
            );
        }

        if (up_block.is_res_block) {
            for (const SanaVaeResBlockWeights & res_block : up_block.res_blocks) {
                hidden = run_vae_res_block(
                    hidden,
                    res_block.conv1_weight,
                    &res_block.conv1_bias,
                    res_block.conv2_weight,
                    res_block.norm_weight,
                    res_block.norm_bias
                );
            }
        } else {
            for (const SanaVaeAttnGlumbLayerWeights & layer : up_block.attn_glumb_layers) {
                hidden = run_vae_multiscale_attention(
                    hidden,
                    layer.attention.to_q_weight_conv1x1,
                    layer.attention.to_k_weight_conv1x1,
                    layer.attention.to_v_weight_conv1x1,
                    layer.attention.proj_in_weight,
                    layer.attention.proj_out_weight,
                    layer.attention.to_out_weight_conv1x1,
                    layer.attention.norm_out_weight,
                    layer.attention.norm_out_bias,
                    32
                );

                hidden = run_vae_glumb_conv(
                    hidden,
                    layer.glumb.conv_inverted_weight,
                    &layer.glumb.conv_inverted_bias,
                    layer.glumb.conv_depth_weight,
                    &layer.glumb.conv_depth_bias,
                    layer.glumb.conv_point_weight,
                    layer.glumb.norm_weight,
                    layer.glumb.norm_bias
                );
            }
        }
    }

    hidden = rms_norm_4d_channelwise(
        hidden,
        weights.decoder.norm_out_weight,
        &weights.decoder.norm_out_bias,
        1e-5f
    );
    hidden.relu_inplace();

    return conv2d_same(
        hidden,
        weights.decoder.conv_out_weight,
        &weights.decoder.conv_out_bias
    );
}
