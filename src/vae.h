#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"
#include "weights_io.h"

struct SanaVaeResBlockWeights {
    Tensor conv1_weight;
    Tensor conv1_bias;
    Tensor conv2_weight;
    Tensor norm_weight;
    Tensor norm_bias;
};

struct SanaVaeAttentionWeights {
    Tensor to_q_weight_conv1x1;
    Tensor to_k_weight_conv1x1;
    Tensor to_v_weight_conv1x1;
    Tensor proj_in_weight;
    Tensor proj_out_weight;
    Tensor to_out_weight_conv1x1;
    Tensor norm_out_weight;
    Tensor norm_out_bias;
};

struct SanaVaeGlumbConvWeights {
    Tensor conv_inverted_weight;
    Tensor conv_inverted_bias;
    Tensor conv_depth_weight;
    Tensor conv_depth_bias;
    Tensor conv_point_weight;
    Tensor norm_weight;
    Tensor norm_bias;
};

struct SanaVaeAttnGlumbLayerWeights {
    SanaVaeAttentionWeights attention;
    SanaVaeGlumbConvWeights glumb;
};

// Most VAE up blocks start with an upsample conv, then either 3 res blocks
// (is_res_block) or 3 attention+glumb layers -- never both, never a mix. The
// innermost up block (highest index, first processed) has no upsample: it runs
// 3 attention+glumb layers directly at the decoder's native latent resolution.
struct SanaVaeUpBlockWeights {
    std::optional<Tensor> upsample_conv_weight;
    std::optional<Tensor> upsample_conv_bias;
    bool is_res_block = false;
    std::vector<SanaVaeResBlockWeights> res_blocks;
    std::vector<SanaVaeAttnGlumbLayerWeights> attn_glumb_layers;
};

struct SanaVaeDecoderWeights {
    Tensor conv_in_weight;
    Tensor conv_in_bias;
    Tensor norm_out_weight;
    Tensor norm_out_bias;
    Tensor conv_out_weight;
    Tensor conv_out_bias;
    std::vector<SanaVaeUpBlockWeights> up_blocks;
};

struct SanaVaeWeights {
    SanaVaeDecoderWeights decoder;
    float scaling_factor = 1.0f;
    size_t spatial_compression_ratio = 32;
};

std::vector<std::string> find_vae_decoder_up_block_prefixes(
    const std::unordered_map<std::string, SavedWeightInfo> & index
);
SanaVaeWeights load_vae_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    float scaling_factor,
    size_t spatial_compression_ratio
);
void validate_vae_decoder_weights(const SanaVaeDecoderWeights & weights);
void validate_vae_weights(const SanaVaeWeights & weights);
Tensor run_vae_decoder_input(
    const Tensor & latents,
    const SanaVaeDecoderWeights & weights,
    bool in_shortcut = true
);
Tensor run_vae_decode(
    const Tensor & latents,
    const SanaVaeWeights & weights
);
