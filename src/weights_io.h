#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"

struct TensorComparison {
    float max_abs_diff;
    float mean_abs_diff;
};

enum class WeightFormat {
    Npy,
    Gguf,
};

struct SavedWeightInfo {
    std::string file;  // npy: relative filename within weights_dir. gguf: the .gguf file path.
    std::vector<size_t> shape;
    std::string dtype;
    WeightFormat format = WeightFormat::Npy;
    size_t offset = 0;  // gguf: byte offset of this tensor's raw data within `file`. unused for npy.
};

// True if `value` ends with `suffix`. Used to dispatch weight-index loading
// (".gguf" vs the legacy npy+json index) and to strip known suffixes off
// weight-tensor names when discovering block prefixes.
bool ends_with(const std::string & value, const std::string & suffix);

std::unordered_map<std::string, SavedWeightInfo> load_weight_index(const std::string & path);
Tensor load_weight_tensor(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    const std::string & name
);
Tensor load_linear_weight_tensor(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    const std::string & name
);
Tensor load_npy_tensor(const std::string & path);
void save_npy_tensor(const Tensor & tensor, const std::string & path);
TensorComparison compare_tensor_to_reference(const Tensor & actual, const std::string & path);
Tensor slice_batch_3d(const Tensor & input, size_t batch_index);
TensorComparison compare_batch_slice_to_reference(
    const Tensor & actual,
    const std::string & path,
    size_t batch_index
);
TensorComparison compare_block_output_to_reference(
    const Tensor & actual,
    const std::string & path,
    size_t batch_index
);
