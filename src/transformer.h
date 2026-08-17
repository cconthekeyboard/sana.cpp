#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "scheduler.h"
#include "tensor.h"
#include "weights_io.h"

struct SanaBlockModulation {
    Tensor shift_msa;
    Tensor scale_msa;
    Tensor gate_msa;
    Tensor shift_mlp;
    Tensor scale_mlp;
    Tensor gate_mlp;
};

struct SanaBlockStageOutputs {
    Tensor after_self;
    Tensor after_cross;
    Tensor after_ff;
};

struct SanaBlockWeights {
    Tensor scale_shift_table;
    Tensor attn_norm_gamma;
    Tensor attn_norm_beta;
    Tensor q_weight;
    Tensor q_bias;
    Tensor k_weight;
    Tensor k_bias;
    Tensor v_weight;
    Tensor v_bias;
    Tensor out_weight;
    Tensor out_bias;
    Tensor cross_q_weight;
    Tensor cross_q_bias;
    Tensor cross_k_weight;
    Tensor cross_k_bias;
    Tensor cross_v_weight;
    Tensor cross_v_bias;
    Tensor cross_out_weight;
    Tensor cross_out_bias;
    Tensor mlp_norm_gamma;
    Tensor mlp_norm_beta;
    Tensor ff_inverted_weight;
    Tensor ff_inverted_bias;
    Tensor ff_depth_weight;
    Tensor ff_depth_bias;
    Tensor ff_point_weight;
    size_t self_num_heads;
    size_t cross_num_heads;
};

struct SanaCaptionWeights {
    Tensor linear_1_weight;
    Tensor linear_1_bias;
    Tensor linear_2_weight;
    Tensor linear_2_bias;
};

struct SanaDenoiserWeights {
    SanaCaptionWeights caption;
    std::vector<SanaBlockWeights> blocks;
    Tensor patch_embed_weight;
    Tensor patch_embed_bias;
    Tensor time_embed_linear_1_weight;
    Tensor time_embed_linear_1_bias;
    Tensor time_embed_linear_2_weight;
    Tensor time_embed_linear_2_bias;
    Tensor time_projection_weight;
    Tensor time_projection_bias;
    Tensor caption_norm_weight;
    Tensor final_scale_shift_table;
    Tensor proj_out_weight;
    Tensor proj_out_bias;
};

SanaCaptionWeights load_caption_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index
);
SanaBlockWeights load_block_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    const std::string & block_prefix,
    size_t self_num_heads,
    size_t cross_num_heads
);
std::vector<std::string> find_block_prefixes(
    const std::unordered_map<std::string, SavedWeightInfo> & index
);
std::vector<SanaBlockWeights> load_model_block_weights(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    size_t self_num_heads,
    size_t cross_num_heads
);
SanaDenoiserWeights load_denoiser_weights(
    const std::string & block_weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & block_index,
    const std::string & caption_weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & caption_index,
    const std::string & denoiser_weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & denoiser_index,
    size_t self_num_heads,
    size_t cross_num_heads
);
void validate_denoiser_weights(const SanaDenoiserWeights & weights);
void validate_model_block_sequence(const std::vector<SanaBlockWeights> & blocks);

SanaBlockModulation compute_sana_block_modulation(
    const Tensor & timestep,
    const Tensor & scale_shift_table
);
Tensor run_model_blocks(
    const Tensor & input,
    const Tensor & context,
    const Tensor & timestep,
    const SanaDenoiserWeights & weights,
    size_t block_begin,
    size_t block_end,
    const Tensor * encoder_attention_bias = nullptr
);
Tensor apply_classifier_free_guidance(
    const Tensor & noise_pred,
    float guidance_scale
);
Tensor run_full_denoising_step_with_cfg(
    const Tensor & latents,
    const Tensor & context,
    float guidance_scale,
    SanaSchedulerState & scheduler_state,
    const SanaSchedulerConfig & scheduler_config,
    const SanaDenoiserWeights & denoiser_weights,
    const Tensor * encoder_attention_bias = nullptr
);
Tensor run_full_denoising_loop_with_cfg(
    const Tensor & initial_latents,
    const Tensor & encoder_hidden_states,
    float guidance_scale,
    SanaSchedulerState & scheduler_state,
    const SanaSchedulerConfig & scheduler_config,
    const SanaDenoiserWeights & denoiser_weights,
    const Tensor * encoder_attention_bias = nullptr
);
SanaBlockStageOutputs block_forward(
    const Tensor & input,
    const Tensor & context,
    const Tensor & timestep,
    const SanaBlockWeights & weights,
    const Tensor * encoder_attention_bias = nullptr
);
