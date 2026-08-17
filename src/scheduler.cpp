#include "scheduler.h"

#include <cmath>
#include <stdexcept>

namespace {

void validate_scheduler_config(const SanaSchedulerConfig & config) {
    if (config.solver_order == 0) {
        throw std::invalid_argument("solver_order must be positive");
    }
    if (config.solver_order > 2) {
        throw std::invalid_argument("only solver_order <= 2 is supported");
    }
    if (config.prediction_type != "flow_prediction") {
        throw std::invalid_argument("only flow_prediction is supported");
    }
    if (config.algorithm_type != "dpmsolver++") {
        throw std::invalid_argument("only dpmsolver++ is supported");
    }
    if (config.solver_type != "midpoint" && config.solver_type != "heun") {
        throw std::invalid_argument("solver_type must be midpoint or heun");
    }
}

void validate_scheduler_step_inputs(
    const Tensor & sample,
    const SanaSchedulerState & state
) {
    if (state.schedule.timesteps.empty()) {
        throw std::invalid_argument("scheduler state has no timesteps");
    }
    if (state.schedule.sigmas.size() != state.schedule.timesteps.size() + 1) {
        throw std::invalid_argument("scheduler sigmas must have one extra final entry");
    }
    if (state.step_index >= state.schedule.timesteps.size()) {
        throw std::invalid_argument("scheduler step_index is out of range");
    }
    (void)sample;
}

Tensor scaled_sum(const Tensor & a, float a_scale, const Tensor & b, float b_scale) {
    if (!a.same_shape(b)) {
        throw std::invalid_argument("tensor shapes must match");
    }
    Tensor out(a.shape());
    for (size_t i = 0; i < out.numel(); ++i) {
        out.at(i) = a_scale * a.at(i) + b_scale * b.at(i);
    }
    return out;
}

Tensor scaled_sum_three(
    const Tensor & a,
    float a_scale,
    const Tensor & b,
    float b_scale,
    const Tensor & c,
    float c_scale
) {
    if (!a.same_shape(b) || !a.same_shape(c)) {
        throw std::invalid_argument("tensor shapes must match");
    }
    Tensor out(a.shape());
    for (size_t i = 0; i < out.numel(); ++i) {
        out.at(i) = a_scale * a.at(i) + b_scale * b.at(i) + c_scale * c.at(i);
    }
    return out;
}

std::pair<float, float> sigma_to_alpha_sigma_t(float sigma, const SanaSchedulerConfig & config) {
    if (config.use_flow_sigmas) {
        return {1.0f - sigma, sigma};
    }
    const float alpha_t = 1.0f / std::sqrt(sigma * sigma + 1.0f);
    return {alpha_t, sigma * alpha_t};
}

float compute_lambda(float alpha_t, float sigma_t) {
    return std::log(alpha_t) - std::log(sigma_t);
}

}  // namespace

SanaSchedulerConfig make_sana_scheduler_config() {
    return SanaSchedulerConfig();
}

SanaSchedulerTimesteps build_sana_scheduler_timesteps(
    size_t num_inference_steps,
    const SanaSchedulerConfig & config
) {
    validate_scheduler_config(config);
    if (num_inference_steps == 0) {
        throw std::invalid_argument("num_inference_steps must be positive");
    }
    if (!config.use_flow_sigmas) {
        throw std::invalid_argument("only use_flow_sigmas=true is supported");
    }
    if (config.timestep_spacing != "linspace") {
        throw std::invalid_argument("only linspace timestep_spacing is supported");
    }
    if (config.final_sigmas_type != "zero") {
        throw std::invalid_argument("only zero final_sigmas_type is supported");
    }
    if (config.num_train_timesteps == 0) {
        throw std::invalid_argument("num_train_timesteps must be positive");
    }

    SanaSchedulerTimesteps schedule;
    schedule.timesteps.reserve(num_inference_steps);
    schedule.sigmas.reserve(num_inference_steps + 1);

    // Match diffusers DPMSolverMultistepScheduler.set_timesteps() for
    // use_flow_sigmas=True:
    //   alphas = linspace(1, 1 / num_train_timesteps, num_inference_steps + 1)
    //   sigmas = 1 - alphas
    //   sigmas = flip(flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas))[:-1]
    //   timesteps = (sigmas * num_train_timesteps).int()  -- diffusers truncates to
    //   an integer timestep here; confirmed against artifacts/reference/scheduler_timesteps.npy
    //   (real PyTorch output), which is integer-valued at every step.
    const float denom = static_cast<float>(config.num_train_timesteps);
    std::vector<float> shifted_sigmas(num_inference_steps + 1);
    for (size_t i = 0; i <= num_inference_steps; ++i) {
        const float alpha =
            1.0f + (1.0f / denom - 1.0f) *
                        (static_cast<float>(i) / static_cast<float>(num_inference_steps));
        const float sigma = 1.0f - alpha;
        shifted_sigmas[i] =
            config.flow_shift * sigma / (1.0f + (config.flow_shift - 1.0f) * sigma);
    }

    for (size_t i = 0; i < num_inference_steps; ++i) {
        const float shifted_sigma = shifted_sigmas[num_inference_steps - i];
        schedule.sigmas.push_back(shifted_sigma);
        schedule.timesteps.push_back(static_cast<float>(static_cast<int>(shifted_sigma * denom)));
    }

    schedule.sigmas.push_back(0.0f);
    return schedule;
}

SanaSchedulerState init_sana_scheduler_state(
    size_t num_inference_steps,
    const SanaSchedulerConfig & config
) {
    validate_scheduler_config(config);
    SanaSchedulerState state;
    state.schedule = build_sana_scheduler_timesteps(num_inference_steps, config);
    state.step_index = 0;
    state.lower_order_step_count = 0;
    return state;
}

Tensor convert_sana_scheduler_model_output(
    const Tensor & model_output,
    const Tensor & sample,
    const SanaSchedulerState & state,
    const SanaSchedulerConfig & config
) {
    validate_scheduler_config(config);
    validate_scheduler_step_inputs(sample, state);
    if (!model_output.same_shape(sample)) {
        throw std::invalid_argument("model_output and sample must have the same shape");
    }

    const float sigma_t = state.schedule.sigmas[state.step_index];
    Tensor out(sample.shape());
    for (size_t i = 0; i < out.numel(); ++i) {
        out.at(i) = sample.at(i) - sigma_t * model_output.at(i);
    }
    return out;
}

Tensor sana_scheduler_first_order_update(
    const Tensor & converted_model_output,
    const Tensor & sample,
    const SanaSchedulerState & state,
    const SanaSchedulerConfig & config
) {
    validate_scheduler_config(config);
    validate_scheduler_step_inputs(sample, state);
    if (!converted_model_output.same_shape(sample)) {
        throw std::invalid_argument("converted_model_output and sample must have the same shape");
    }

    const float sigma_t_value = state.schedule.sigmas[state.step_index + 1];
    const float sigma_s_value = state.schedule.sigmas[state.step_index];
    const auto alpha_sigma_t = sigma_to_alpha_sigma_t(sigma_t_value, config);
    const auto alpha_sigma_s = sigma_to_alpha_sigma_t(sigma_s_value, config);
    const float alpha_t = alpha_sigma_t.first;
    const float sigma_t = alpha_sigma_t.second;
    const float alpha_s = alpha_sigma_s.first;
    const float sigma_s = alpha_sigma_s.second;
    const float lambda_t = compute_lambda(alpha_t, sigma_t);
    const float lambda_s = compute_lambda(alpha_s, sigma_s);
    const float h = lambda_t - lambda_s;

    return scaled_sum(
        sample,
        sigma_t / sigma_s,
        converted_model_output,
        -alpha_t * (std::exp(-h) - 1.0f)
    );
}

Tensor sana_scheduler_second_order_update(
    const Tensor & sample,
    const SanaSchedulerState & state,
    const SanaSchedulerConfig & config
) {
    validate_scheduler_config(config);
    validate_scheduler_step_inputs(sample, state);
    if (state.step_index == 0) {
        throw std::invalid_argument("second-order update requires at least one previous step");
    }
    if (state.previous_model_outputs.size() < 2) {
        throw std::invalid_argument("second-order update requires two model outputs");
    }

    const float sigma_t_value = state.schedule.sigmas[state.step_index + 1];
    const float sigma_s0_value = state.schedule.sigmas[state.step_index];
    const float sigma_s1_value = state.schedule.sigmas[state.step_index - 1];
    const auto alpha_sigma_t = sigma_to_alpha_sigma_t(sigma_t_value, config);
    const auto alpha_sigma_s0 = sigma_to_alpha_sigma_t(sigma_s0_value, config);
    const auto alpha_sigma_s1 = sigma_to_alpha_sigma_t(sigma_s1_value, config);
    const float alpha_t = alpha_sigma_t.first;
    const float sigma_t = alpha_sigma_t.second;
    const float alpha_s0 = alpha_sigma_s0.first;
    const float sigma_s0 = alpha_sigma_s0.second;
    const float alpha_s1 = alpha_sigma_s1.first;
    const float sigma_s1 = alpha_sigma_s1.second;
    const float lambda_t = compute_lambda(alpha_t, sigma_t);
    const float lambda_s0 = compute_lambda(alpha_s0, sigma_s0);
    const float lambda_s1 = compute_lambda(alpha_s1, sigma_s1);

    const Tensor & m0 = state.previous_model_outputs.back();
    const Tensor & m1 = state.previous_model_outputs[state.previous_model_outputs.size() - 2];
    if (!m0.same_shape(sample) || !m1.same_shape(sample)) {
        throw std::invalid_argument("stored model outputs must match sample shape");
    }

    const float h = lambda_t - lambda_s0;
    const float h0 = lambda_s0 - lambda_s1;
    const float r0 = h0 / h;
    Tensor d1(sample.shape());
    for (size_t i = 0; i < d1.numel(); ++i) {
        d1.at(i) = (1.0f / r0) * (m0.at(i) - m1.at(i));
    }

    if (config.solver_type == "midpoint") {
        return scaled_sum_three(
            sample,
            sigma_t / sigma_s0,
            m0,
            -alpha_t * (std::exp(-h) - 1.0f),
            d1,
            -0.5f * alpha_t * (std::exp(-h) - 1.0f)
        );
    }

    return scaled_sum_three(
        sample,
        sigma_t / sigma_s0,
        m0,
        -alpha_t * (std::exp(-h) - 1.0f),
        d1,
        alpha_t * ((std::exp(-h) - 1.0f) / h + 1.0f)
    );
}

Tensor step_sana_scheduler(
    const Tensor & model_output,
    const Tensor & sample,
    SanaSchedulerState & state,
    const SanaSchedulerConfig & config
) {
    validate_scheduler_config(config);
    validate_scheduler_step_inputs(sample, state);
    if (!model_output.same_shape(sample)) {
        throw std::invalid_argument("model_output and sample must have the same shape");
    }

    Tensor converted_model_output =
        convert_sana_scheduler_model_output(model_output, sample, state, config);
    state.previous_model_outputs.push_back(converted_model_output);
    if (state.previous_model_outputs.size() > config.solver_order) {
        state.previous_model_outputs.erase(state.previous_model_outputs.begin());
    }

    const bool lower_order_final =
        (state.step_index == state.schedule.timesteps.size() - 1) &&
        (config.lower_order_final || config.final_sigmas_type == "zero");
    const bool lower_order_second =
        (state.step_index == state.schedule.timesteps.size() - 2) &&
        config.lower_order_final &&
        state.schedule.timesteps.size() < 15;

    Tensor prev_sample(sample.shape());
    if (config.solver_order == 1 || state.lower_order_step_count < 1 || lower_order_final) {
        prev_sample = sana_scheduler_first_order_update(
            converted_model_output,
            sample,
            state,
            config
        );
    } else if (config.solver_order == 2 || state.lower_order_step_count < 2 || lower_order_second) {
        prev_sample = sana_scheduler_second_order_update(sample, state, config);
    } else {
        throw std::invalid_argument("only solver_order <= 2 is supported");
    }

    if (state.lower_order_step_count < config.solver_order) {
        state.lower_order_step_count += 1;
    }
    state.step_index += 1;
    return prev_sample;
}
