#include "scheduler.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

Tensor make_scalar(float value) {
    Tensor tensor({1});
    tensor.at(0) = value;
    return tensor;
}

}  // namespace

int main() {
    SanaSchedulerConfig config = make_sana_scheduler_config();
    assert(config.num_train_timesteps == 1000);
    assert(std::fabs(config.flow_shift - 3.0f) < 1e-7f);
    assert(config.use_flow_sigmas);
    assert(config.timestep_spacing == "linspace");
    assert(config.final_sigmas_type == "zero");
    assert(config.solver_order == 2);
    assert(config.lower_order_final);

    SanaSchedulerTimesteps schedule = build_sana_scheduler_timesteps(4, config);
    assert(schedule.timesteps.size() == 4);
    assert(schedule.sigmas.size() == 5);
    assert(schedule.timesteps[0] > schedule.timesteps[1]);
    assert(schedule.timesteps[1] > schedule.timesteps[2]);
    assert(schedule.timesteps[2] > schedule.timesteps[3]);
    assert(std::fabs(schedule.timesteps[0] - 999.0f) < 1e-3f);
    assert(std::fabs(schedule.timesteps[1] - 899.0f) < 1e-3f);
    assert(std::fabs(schedule.timesteps[2] - 749.0f) < 1e-3f);
    assert(std::fabs(schedule.timesteps[3] - 499.0f) < 1e-3f);
    assert(std::fabs(schedule.sigmas[0] - 0.99966644f) < 1e-6f);
    assert(std::fabs(schedule.sigmas[1] - 0.89963979f) < 1e-6f);
    assert(std::fabs(schedule.sigmas[2] - 0.74962479f) < 1e-6f);
    assert(std::fabs(schedule.sigmas[3] - 0.49966654f) < 1e-6f);
    assert(std::fabs(schedule.sigmas[4] - 0.0f) < 1e-7f);

    bool threw = false;
    try {
        (void)build_sana_scheduler_timesteps(0, config);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        SanaSchedulerConfig unsupported = config;
        unsupported.use_flow_sigmas = false;
        (void)build_sana_scheduler_timesteps(4, unsupported);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    SanaSchedulerState state = init_sana_scheduler_state(4, config);
    assert(state.schedule.timesteps.size() == 4);
    assert(state.schedule.sigmas.size() == 5);
    assert(state.step_index == 0);
    assert(state.lower_order_step_count == 0);
    assert(state.previous_model_outputs.empty());
    assert(std::fabs(state.schedule.timesteps[0] - schedule.timesteps[0]) < 1e-6f);
    assert(std::fabs(state.schedule.sigmas[4] - schedule.sigmas[4]) < 1e-7f);

    threw = false;
    try {
        SanaSchedulerConfig bad_order = config;
        bad_order.solver_order = 0;
        (void)init_sana_scheduler_state(4, bad_order);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    SanaSchedulerState manual_state;
    manual_state.schedule.timesteps = {800.0f, 500.0f, 200.0f};
    manual_state.schedule.sigmas = {0.8f, 0.5f, 0.2f, 0.0f};
    manual_state.step_index = 0;

    Tensor sample = make_scalar(4.0f);
    Tensor model_output = make_scalar(2.0f);
    Tensor converted =
        convert_sana_scheduler_model_output(model_output, sample, manual_state, config);
    assert(std::fabs(converted.at(0) - 2.4f) < 1e-6f);

    Tensor first_order =
        sana_scheduler_first_order_update(converted, sample, manual_state, config);
    assert(std::fabs(first_order.at(0) - 3.4f) < 1e-5f);

    SanaSchedulerState stepped_state = manual_state;
    Tensor stepped_sample = step_sana_scheduler(model_output, sample, stepped_state, config);
    assert(std::fabs(stepped_sample.at(0) - 3.4f) < 1e-5f);
    assert(stepped_state.step_index == 1);
    assert(stepped_state.lower_order_step_count == 1);
    assert(stepped_state.previous_model_outputs.size() == 1);
    assert(std::fabs(stepped_state.previous_model_outputs[0].at(0) - 2.4f) < 1e-6f);

    SanaSchedulerState second_order_state;
    second_order_state.schedule.timesteps.resize(16);
    second_order_state.schedule.sigmas = {
        0.8f, 0.5f, 0.2f, 0.1f, 0.09f, 0.08f, 0.07f, 0.06f, 0.05f,
        0.04f, 0.03f, 0.02f, 0.015f, 0.01f, 0.005f, 0.001f, 0.0f
    };
    second_order_state.step_index = 1;
    second_order_state.lower_order_step_count = 1;
    second_order_state.previous_model_outputs.push_back(make_scalar(2.4f));
    second_order_state.previous_model_outputs.push_back(make_scalar(2.0f));
    Tensor second_order =
        sana_scheduler_second_order_update(make_scalar(3.4f), second_order_state, config);
    assert(std::fabs(second_order.at(0) - 2.44f) < 1e-5f);

    Tensor second_step_sample =
        step_sana_scheduler(make_scalar(3.1f), make_scalar(3.4f), second_order_state, config);
    assert(second_order_state.step_index == 2);
    assert(second_order_state.lower_order_step_count == 2);
    assert(second_order_state.previous_model_outputs.size() == 2);
    assert(second_step_sample.shape() == std::vector<size_t>({1}));

    threw = false;
    try {
        SanaSchedulerConfig unsupported_prediction = config;
        unsupported_prediction.prediction_type = "epsilon";
        (void)init_sana_scheduler_state(4, unsupported_prediction);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    return 0;
}
