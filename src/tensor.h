#pragma once

#include <cstddef>
#include <utility>
#include <tuple>
#include <vector>

class Tensor {
public:
    explicit Tensor(std::vector<size_t> shape);
    const std::vector<size_t> & shape() const;
    std::vector<float> & data();
    const std::vector<float> & data() const;

    // Readable indexing for tests/debugging, not production hot paths 
    float & at(size_t flat_index);
    const float & at(size_t flat_index) const;
    float & at(const std::vector<size_t> & indices);
    const float & at(const std::vector<size_t> & indices) const;
    
    void fill(float value);
    bool same_shape(const Tensor & other) const;
    size_t flat_index(const std::vector<size_t> & indices) const;  // see at() comment above
    size_t rank() const;
    size_t numel() const;
    size_t dim_size(size_t axis) const;
    void add_inplace(const Tensor & other);
    void mul_inplace_scalar(float scalar);
    void mul_inplace_broadcast_lastdim(const Tensor & gate);
    Tensor transpose_2d() const;
    Tensor layer_norm_3d_lastdim(float eps, const Tensor * gamma = nullptr, const Tensor * beta = nullptr) const;
    Tensor rms_norm_3d_lastdim(float eps, const Tensor * weight = nullptr) const;
    Tensor modulate_3d_lastdim(const Tensor & shift, const Tensor & scale) const;
    Tensor tokens_to_grid_4d(size_t height, size_t width) const;
    Tensor grid_to_tokens_3d() const;
    std::pair<Tensor, Tensor> split_channels_4d() const;
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> split_six_way_3d() const;
    void silu_inplace();
    void relu_inplace();
    float max_abs_diff(const Tensor & other) const;
    float mean_abs_diff(const Tensor & other) const;

private:
    std::vector<size_t> shape_;
    std::vector<float> data_;
};
