#pragma once

#include "tensor.h"

#include <cstdint>
#include <vector>

// Bit-exact port of:
//   torch.Generator(device="cpu").manual_seed(seed)
//   torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
// See the approved plan for the derivation, verified directly against
// pytorch/pytorch's ATen source (MT19937RNGEngine.h, TransformationHelper.h,
// DistributionTemplates.h).
Tensor randn_torch_cpu(const std::vector<size_t> & shape, uint64_t seed);
