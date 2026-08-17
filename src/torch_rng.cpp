#include "torch_rng.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace {

// std::numeric_limits<float>::digits == 24 (23 mantissa bits + implicit bit).
// Matches ATen's uniform_real<float>: u = (val & MASK) * (1 / 2^24), in float32.
constexpr std::uint32_t kUniformMask = (1u << 24) - 1;
constexpr float kUniformDivisor = 1.0f / static_cast<float>(1u << 24);

float next_uniform_float(std::mt19937 & engine) {
    const std::uint32_t val = static_cast<std::uint32_t>(engine());
    return static_cast<float>(val & kUniformMask) * kUniformDivisor;
}

// c10::pi<double>: the single correctly-rounded double representation of pi.
constexpr double kPiDouble = 3.14159265358979323846;

void normal_fill_16(float * data) {
    for (int j = 0; j < 8; ++j) {
        const float u1 = 1.0f - data[j];
        const float u2 = data[j + 8];
        const float radius = std::sqrt(-2.0f * std::log(u1));
        const float theta = static_cast<float>(2.0 * kPiDouble * static_cast<double>(u2));
        data[j] = radius * std::cos(theta);
        data[j + 8] = radius * std::sin(theta);
    }
}

}  // namespace

Tensor randn_torch_cpu(const std::vector<size_t> & shape, uint64_t seed) {
    Tensor out(shape);
    std::vector<float> & data = out.data();
    const size_t size = data.size();
    if (size < 16) {
        throw std::invalid_argument("randn_torch_cpu requires at least 16 elements");
    }

    // torch.Generator(device="cpu").manual_seed(seed) -> CPUGeneratorImpl::set_current_seed
    // -> engine_ = mt19937(seed)
    std::mt19937 engine(static_cast<std::mt19937::result_type>(seed & 0xffffffffULL));

    // ATen's normal_fill: fill the entire tensor with uniform(0,1) draws first, one
    // engine() call per element, left to right.
    for (size_t i = 0; i < size; ++i) {
        data[i] = next_uniform_float(engine);
    }

    // Then transform complete batches of 16 in place via Box-Muller.
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        normal_fill_16(data.data() + i);
    }

    if (size % 16 != 0) {
        float * tail = data.data() + size - 16;
        for (size_t j = 0; j < 16; ++j) {
            tail[j] = next_uniform_float(engine);
        }
        normal_fill_16(tail);
    }

    return out;
}
