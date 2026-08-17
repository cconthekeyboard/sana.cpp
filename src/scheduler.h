#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tensor.h"

struct SanaSchedulerConfig {
    size_t num_train_timesteps = 1000;
    float flow_shift = 3.0f;
    bool use_flow_sigmas = true;
    std::string timestep_spacing = "linspace";
    std::string final_sigmas_type = "zero";
    size_t solver_order = 2;
    bool lower_order_final = true;
    std::string prediction_type = "flow_prediction";
    std::string algorithm_type = "dpmsolver++";
    std::string solver_type = "midpoint";
};

struct SanaSchedulerTimesteps {
    std::vector<float> timesteps;
    std::vector<float> sigmas;
};

struct SanaSchedulerState {
    SanaSchedulerTimesteps schedule;
    size_t step_index = 0;
    size_t lower_order_step_count = 0;
    std::vector<Tensor> previous_model_outputs;
};

SanaSchedulerConfig make_sana_scheduler_config();
SanaSchedulerTimesteps build_sana_scheduler_timesteps(
    size_t num_inference_steps,
    const SanaSchedulerConfig & config
);
SanaSchedulerState init_sana_scheduler_state(
    size_t num_inference_steps,
    const SanaSchedulerConfig & config
);
Tensor convert_sana_scheduler_model_output(
    const Tensor & model_output,
    const Tensor & sample,
    const SanaSchedulerState & state,
    const SanaSchedulerConfig & config
);
Tensor sana_scheduler_first_order_update(
    const Tensor & converted_model_output,
    const Tensor & sample,
    const SanaSchedulerState & state,
    const SanaSchedulerConfig & config
);
Tensor sana_scheduler_second_order_update(
    const Tensor & sample,
    const SanaSchedulerState & state,
    const SanaSchedulerConfig & config
);
Tensor step_sana_scheduler(
    const Tensor & model_output,
    const Tensor & sample,
    SanaSchedulerState & state,
    const SanaSchedulerConfig & config
);
