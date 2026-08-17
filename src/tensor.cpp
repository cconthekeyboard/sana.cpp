#include "tensor.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <cmath>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

Tensor::Tensor(std::vector<size_t> shape)
    : shape_(std::move(shape)) {
    data_.resize(numel());
}

const std::vector<size_t> & Tensor::shape() const {
    return shape_;
}

std::vector<float> & Tensor::data() {
    return data_;
}

const std::vector<float> & Tensor::data() const {
    return data_;
}

float & Tensor::at(size_t flat_index) {
    if (flat_index >= data_.size()) {
        throw std::out_of_range("index out of range");
    }
    return data_[flat_index];
}

const float & Tensor::at(size_t flat_index) const {
    if (flat_index >= data_.size()) {
        throw std::out_of_range("index out of range");
    }
    return data_[flat_index];
}

float & Tensor::at(const std::vector<size_t> & indices) {
    return at(flat_index(indices));
}

const float & Tensor::at(const std::vector<size_t> & indices) const {
    return at(flat_index(indices));
}

void Tensor::fill(float value) {
    for (size_t i=0;i<data_.size();i++) {
        data_[i] = value;
    }
}

bool Tensor::same_shape(const Tensor & other) const {
    if (shape_ == other.shape()){
        return true;
    } else{
        return false;
    }
}

size_t Tensor::flat_index(const std::vector<size_t> & indices) const {
    if (indices.size() != rank()) {
        throw std::out_of_range("indices size is greater than rank");
    }
    size_t idx = 0;
    size_t stride = 1;
    for (size_t k=rank(); k-- > 0;) {
        if (indices[k] >= dim_size(k)) {
            throw std::out_of_range("dim in indices cannot be greater than the dim in shape of the tensor");
        }
        idx += stride * indices[k];
        stride *= shape_[k];
    }

    return idx;
}

size_t Tensor::rank() const {
    return shape_.size();
}

size_t Tensor::numel() const {
    size_t res = 1;
    for (size_t dim : shape_) {
        res *= dim;
    }
    return res;
}

size_t Tensor::dim_size(size_t axis) const {
    if (axis >= shape_.size()) {
        throw std::out_of_range("tensor dimension axis out of range");
    }
    return shape_[axis];
}

void Tensor::add_inplace(const Tensor & other) {
    if (shape_ != other.shape()) {
        throw std::invalid_argument("cannot add two tensors with different shapes");
    }
    const size_t n = numel();
    const float * other_data = other.data().data();
#if defined(__APPLE__)
    vDSP_vadd(data_.data(), 1, other_data, 1, data_.data(), 1, n);
#else
    for (size_t i=0; i<n; i++) {
        data_[i] += other_data[i];
    }
#endif
}

void Tensor::mul_inplace_scalar(float scalar) {
    const size_t n = numel();
#if defined(__APPLE__)
    vDSP_vsmul(data_.data(), 1, &scalar, data_.data(), 1, n);
#else
    for (size_t i=0; i<n; i++) {
        data_[i] *= scalar;
    }
#endif
}

//   x[b, t, h] *= gate[b, h]
//   input (in-place):  [batch, tokens, hidden]
//   gate:              [batch, 1, hidden], broadcast over tokens
void Tensor::mul_inplace_broadcast_lastdim(const Tensor & gate) {
    if (rank() != 3) {
        throw std::invalid_argument("input must be rank 3");
    }
    if (gate.rank() != 3) {
        throw std::invalid_argument("gate must be rank 3");
    }
    const size_t batch_size = dim_size(0);
    const size_t tokens = dim_size(1);
    const size_t hidden = dim_size(2);
    if (gate.dim_size(0) != batch_size || gate.dim_size(1) != 1 || gate.dim_size(2) != hidden) {
        throw std::invalid_argument("gate must have shape {batch, 1, hidden}");
    }

    float * self_data = data_.data();
    const float * gate_data = gate.data().data();

    for (size_t b = 0; b < batch_size; ++b) {
        const float * gate_row = gate_data + b * hidden;
        for (size_t t = 0; t < tokens; ++t) {
            float * row = self_data + (b * tokens + t) * hidden;
#if defined(__APPLE__)
            vDSP_vmul(row, 1, gate_row, 1, row, 1, hidden);
#else
            for (size_t h = 0; h < hidden; ++h) {
                row[h] *= gate_row[h];
            }
#endif
        }
    }
}

Tensor Tensor::transpose_2d() const {
    if (rank() != 2) {
        throw std::invalid_argument("rank is not 2");
    }
    const size_t rows = shape_[0];
    const size_t cols = shape_[1];
    Tensor res({cols, rows});
    const float * src = data_.data();
    float * dst = res.data().data();

#if defined(__APPLE__)
    vDSP_mtrans(src, 1, dst, 1, cols, rows);
#else
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
#endif

    return res;
}

//   out = (x - mean) / sqrt(var + eps) * gamma + beta, mean/var reduced over hidden
//   input/output:  [batch, tokens, hidden]
//   gamma, beta:   [hidden] or nullptr
Tensor Tensor::layer_norm_3d_lastdim(float eps, const Tensor * gamma, const Tensor * beta) const {
    if (rank() != 3) {
        throw std::invalid_argument("tensor must be 3d");
    }

    const size_t batch_size = dim_size(0);
    const size_t tokens = dim_size(1);
    const size_t hidden = dim_size(2);
    if (gamma && (gamma->rank() != 1 || gamma->dim_size(0) != hidden)) {
        throw std::invalid_argument("gamma must be 1D and match hidden size");
    }
    if (beta && (beta->rank() != 1 || beta->dim_size(0) != hidden)) {
        throw std::invalid_argument("beta must be 1D and match hidden size");
    }

    Tensor out({batch_size, tokens, hidden});
    const float * gamma_data = gamma ? gamma->data().data() : nullptr;
    const float * beta_data = beta ? beta->data().data() : nullptr;
    float * out_data = out.data().data();

#if defined(__APPLE__)
    std::vector<float> centered(hidden);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < tokens; ++t) {
            const size_t row_base = (b * tokens + t) * hidden;
            const float * row = data_.data() + row_base;
            float * out_row = out_data + row_base;

            float mean = 0.0f;
            vDSP_meanv(row, 1, &mean, hidden);
            const float neg_mean = -mean;
            vDSP_vsadd(row, 1, &neg_mean, centered.data(), 1, hidden);

            float var = 0.0f;
            vDSP_measqv(centered.data(), 1, &var, hidden);
            const float inv_std = 1.0f / std::sqrt(var + eps);
            vDSP_vsmul(centered.data(), 1, &inv_std, out_row, 1, hidden);

            if (gamma_data && beta_data) {
                vDSP_vma(out_row, 1, gamma_data, 1, beta_data, 1, out_row, 1, hidden);
            } else if (gamma_data) {
                vDSP_vmul(out_row, 1, gamma_data, 1, out_row, 1, hidden);
            } else if (beta_data) {
                vDSP_vadd(out_row, 1, beta_data, 1, out_row, 1, hidden);
            }
        }
    }
#else
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < tokens; ++t) {
            const size_t row_base = (b * tokens + t) * hidden;
            const float * row = data_.data() + row_base;
            float * out_row = out_data + row_base;

            float mean = 0.0f;
            for (size_t h = 0; h < hidden; ++h) {
                mean += row[h];
            }
            mean /= static_cast<float>(hidden);

            float var = 0.0f;
            for (size_t h = 0; h < hidden; ++h) {
                const float diff = row[h] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(hidden);

            const float inv_std = 1.0f / std::sqrt(var + eps);
            for (size_t h = 0; h < hidden; ++h) {
                float value = (row[h] - mean) * inv_std;
                if (gamma_data) {
                    value *= gamma_data[h];
                }
                if (beta_data) {
                    value += beta_data[h];
                }
                out_row[h] = value;
            }
        }
    }
#endif

    return out;
}

//   out = x / sqrt(mean(x^2) + eps) * weight, mean reduced over hidden
//   input/output:  [batch, tokens, hidden]
//   weight:        [hidden] or nullptr
Tensor Tensor::rms_norm_3d_lastdim(float eps, const Tensor * weight) const {
    if (rank() != 3) {
        throw std::invalid_argument("tensor must be 3d");
    }

    const size_t batch_size = dim_size(0);
    const size_t tokens = dim_size(1);
    const size_t hidden = dim_size(2);
    if (weight && (weight->rank() != 1 || weight->dim_size(0) != hidden)) {
        throw std::invalid_argument("weight must be 1D and match hidden size");
    }

    Tensor out({batch_size, tokens, hidden});
    const float * weight_data = weight ? weight->data().data() : nullptr;
    float * out_data = out.data().data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < tokens; ++t) {
            const size_t row_base = (b * tokens + t) * hidden;
            const float * row = data_.data() + row_base;
            float * out_row = out_data + row_base;
            float mean_square = 0.0f;
            #pragma clang loop vectorize(disable)
            for (size_t h = 0; h < hidden; ++h) {
                mean_square += row[h] * row[h];
            }
            mean_square /= static_cast<float>(hidden);
            const float inv_rms = 1.0f / std::sqrt(mean_square + eps);

            for (size_t h = 0; h < hidden; ++h) {
                float value = row[h] * inv_rms;
                if (weight_data) {
                    value *= weight_data[h];
                }
                out_row[h] = value;
            }
        }
    }

    return out;
}

// Elementwise AdaLN-style modulation, no matmul: each token in a batch item is
// scaled/shifted by the same per-channel values (shift/scale have no token axis).
//   out[b, t, h] = x[b, t, h] * (1 + scale[b, h]) + shift[b, h]
//   input/output:  [batch, tokens, hidden]
//   shift, scale:  [batch, 1, hidden]
Tensor Tensor::modulate_3d_lastdim(const Tensor & shift, const Tensor & scale) const {
    if (rank() != 3) {
        throw std::invalid_argument("input must be rank 3");
    }
    if (shift.rank() != 3 || scale.rank() != 3) {
        throw std::invalid_argument("shift and scale must be rank 3");
    }
    const size_t batch_size = dim_size(0);
    const size_t tokens = dim_size(1);
    const size_t hidden = dim_size(2);
    if (shift.dim_size(0) != batch_size || shift.dim_size(1) != 1 || shift.dim_size(2) != hidden) {
        throw std::invalid_argument("shift must have shape {batch, 1, hidden}");
    }
    if (scale.dim_size(0) != batch_size || scale.dim_size(1) != 1 || scale.dim_size(2) != hidden) {
        throw std::invalid_argument("scale must have shape {batch, 1, hidden}");
    }
    Tensor out({batch_size, tokens, hidden});
    const float * shift_data = shift.data().data();
    const float * scale_data = scale.data().data();
    const float * in_data = data_.data();
    float * out_data = out.data().data();

#if defined(__APPLE__)
    std::vector<float> one_plus_scale(hidden);
    const float one = 1.0f;
    for (size_t b = 0; b < batch_size; ++b) {
        const float * scale_row = scale_data + b * hidden;
        const float * shift_row = shift_data + b * hidden;
        vDSP_vsadd(scale_row, 1, &one, one_plus_scale.data(), 1, hidden);
        for (size_t t = 0; t < tokens; ++t) {
            const size_t row_base = (b * tokens + t) * hidden;
            vDSP_vma(
                in_data + row_base, 1,
                one_plus_scale.data(), 1,
                shift_row, 1,
                out_data + row_base, 1,
                hidden
            );
        }
    }
#else
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t h = 0; h < hidden; ++h) {
                const size_t idx = (b * tokens + t) * hidden + h;
                const size_t mod_idx = b * hidden + h;
                out_data[idx] = in_data[idx] * (1.0f + scale_data[mod_idx]) + shift_data[mod_idx];
            }
        }
    }
#endif

    return out;
}

// Layout change only, no math -- unflattens each token index into (row, col).
//   out[b, h, row, col] = input[b, token, h], where token = row*width + col
//   input:  [batch, tokens, hidden]
//   out:    [batch, hidden, height, width]
Tensor Tensor::tokens_to_grid_4d(size_t height, size_t width) const {
    if (rank() != 3) {
        throw std::invalid_argument("input must be rank 3");
    }
    if (height == 0 || width == 0) {
        throw std::invalid_argument("height and width must be greater than 0");
    }

    const size_t batch_size = dim_size(0);
    const size_t tokens = dim_size(1);
    const size_t hidden = dim_size(2);
    if (tokens != height * width) {
        throw std::invalid_argument("tokens must equal height * width");
    }

    Tensor out({batch_size, hidden, height, width});
    const float * src = data_.data();
    float * dst = out.data().data();

#if defined(__APPLE__)
    // height*width == tokens, so this is exactly a per-batch [tokens, hidden] -> [hidden, tokens] transpose
    const size_t slice_size = tokens * hidden;
    for (size_t b = 0; b < batch_size; ++b) {
        vDSP_mtrans(src + b * slice_size, 1, dst + b * slice_size, 1, hidden, tokens);
    }
#else
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t token = 0; token < tokens; ++token) {
            const size_t row = token / width;
            const size_t col = token % width;
            for (size_t h = 0; h < hidden; ++h) {
                dst[((b * hidden + h) * height + row) * width + col] =
                    src[(b * tokens + token) * hidden + h];
            }
        }
    }
#endif

    return out;
}

// Inverse of tokens_to_grid_4d, same kind of operation: pure layout change, no math.
//   out[b, token, h] = input[b, h, row, col], where token = row*width + col
//   input:  [batch, hidden, height, width]
//   out:    [batch, tokens, hidden]
Tensor Tensor::grid_to_tokens_3d() const {
    const size_t batch_size = dim_size(0);
    const size_t hidden = dim_size(1);
    const size_t height = dim_size(2);
    const size_t width = dim_size(3);
    size_t tokens = height * width;
    Tensor out({batch_size, tokens, hidden});
    const float * src = data_.data();
    float * dst = out.data().data();

#if defined(__APPLE__)
    const size_t slice_size = hidden * tokens;
    for (size_t b = 0; b < batch_size; ++b) {
        vDSP_mtrans(src + b * slice_size, 1, dst + b * slice_size, 1, tokens, hidden);
    }
#else
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < hidden; ++h) {
            for (size_t row = 0; row < height; ++row) {
                for (size_t col = 0; col < width; ++col) {
                    size_t t = row * width + col;
                    dst[(b * tokens + t) * hidden + h] =
                        src[((b * hidden + h) * height + row) * width + col];
                }
            }
        }
    }
#endif
    return out;
}

// Splits the channel axis in half: no math, and since channels is the outer (slowest)
// spatial axis, each batch item's first/second half is already one contiguous block --
// a straight memcpy per half, not an element-by-element gather.
//   first[b, c, y, x]  = input[b, c, y, x]              for c in [0, channels/2)
//   second[b, c, y, x] = input[b, channels/2 + c, y, x]  for c in [0, channels/2)
//   input:          [batch, channels, height, width]
//   first, second:  [batch, channels/2, height, width]
std::pair<Tensor, Tensor> Tensor::split_channels_4d() const {
    if (rank() != 4) {
        throw std::invalid_argument("input must be rank 4");
    }

    const size_t batch_size = dim_size(0);
    const size_t channels = dim_size(1);
    const size_t height = dim_size(2);
    const size_t width = dim_size(3);
    if (channels % 2 != 0) {
        throw std::invalid_argument("channel count must be even");
    }

    const size_t half_channels = channels / 2;
    Tensor first({batch_size, half_channels, height, width});
    Tensor second({batch_size, half_channels, height, width});
    const float * src = data_.data();
    float * first_data = first.data().data();
    float * second_data = second.data().data();

    const size_t half_size = half_channels * height * width;
    for (size_t b = 0; b < batch_size; ++b) {
        const float * batch_src = src + b * channels * height * width;
        std::copy(batch_src, batch_src + half_size, first_data + b * half_size);
        std::copy(batch_src + half_size, batch_src + 2 * half_size, second_data + b * half_size);
    }

    return {first, second};
}

// No math -- each (b, part) slice is already a contiguous run of `hidden` floats.
//   output_i[b, 0, h] = input[b, i, h], for i = 0..5 (first, second, ..., sixth)
//   input:    [batch, 6, hidden]
//   outputs:  six tensors, each [batch, 1, hidden]
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> Tensor::split_six_way_3d() const {
    if (rank() != 3) {
        throw std::invalid_argument("input must be rank 3");
    }

    const size_t batch_size = dim_size(0);
    const size_t parts = dim_size(1);
    const size_t hidden = dim_size(2);
    if (parts != 6) {
        throw std::invalid_argument("second dimension must be exactly 6");
    }

    Tensor first({batch_size, 1, hidden});
    Tensor second({batch_size, 1, hidden});
    Tensor third({batch_size, 1, hidden});
    Tensor fourth({batch_size, 1, hidden});
    Tensor fifth({batch_size, 1, hidden});
    Tensor sixth({batch_size, 1, hidden});
    const float * src = data_.data();
    float * first_data = first.data().data();
    float * second_data = second.data().data();
    float * third_data = third.data().data();
    float * fourth_data = fourth.data().data();
    float * fifth_data = fifth.data().data();
    float * sixth_data = sixth.data().data();

    for (size_t b = 0; b < batch_size; ++b) {
        const float * batch_src = src + b * parts * hidden;
        std::copy(batch_src + 0 * hidden, batch_src + 1 * hidden, first_data + b * hidden);
        std::copy(batch_src + 1 * hidden, batch_src + 2 * hidden, second_data + b * hidden);
        std::copy(batch_src + 2 * hidden, batch_src + 3 * hidden, third_data + b * hidden);
        std::copy(batch_src + 3 * hidden, batch_src + 4 * hidden, fourth_data + b * hidden);
        std::copy(batch_src + 4 * hidden, batch_src + 5 * hidden, fifth_data + b * hidden);
        std::copy(batch_src + 5 * hidden, batch_src + 6 * hidden, sixth_data + b * hidden);
    }

    return {first, second, third, fourth, fifth, sixth};
}

void Tensor::silu_inplace() {
#if defined(__APPLE__)
    const size_t count = data_.size();
    static thread_local std::vector<float> denom;
    denom.resize(count);
    const float neg_one = -1.0f;
    vDSP_vsmul(data_.data(), 1, &neg_one, denom.data(), 1, count);  // denom = -x
    const int exp_count = static_cast<int>(count);
    vvexpf(denom.data(), denom.data(), &exp_count);                 // denom = exp(-x)
    const float one = 1.0f;
    vDSP_vsadd(denom.data(), 1, &one, denom.data(), 1, count);      // denom = 1 + exp(-x)
    // vDSP_vdiv computes C = B / A given (A, ..., B, ..., C, ...) -- divisor first.
    vDSP_vdiv(denom.data(), 1, data_.data(), 1, data_.data(), 1, count);
#else
    for (float & x : data_) {
        x = x / (1.0f + std::exp(-x));
    }
#endif
}

void Tensor::relu_inplace() {
#if defined(__APPLE__)
    const size_t count = data_.size();
    const float threshold = 0.0f;
    vDSP_vthres(data_.data(), 1, &threshold, data_.data(), 1, count);
#else
    for (float & x : data_) {
        x = std::max(0.0f, x);
    }
#endif
}

float Tensor::max_abs_diff(const Tensor & other) const {
    float maxDiff = 0.0f;
    if (shape_ != other.shape()) {
        throw std::invalid_argument("shape mismatch");
    }
    for (size_t i=0; i<numel(); i++) {
        float tmp = std::fabs(data_[i] - other.data()[i]);
        if (tmp > maxDiff) {
            maxDiff = tmp;
        }
    }
    return maxDiff;
}

float Tensor::mean_abs_diff(const Tensor & other) const {
    float meanDiff = 0.0f;
    if (shape_ != other.shape()) {
        throw std::invalid_argument("shape mismatch");
    }
    for (size_t i=0; i<numel(); i++) {
        meanDiff += std::fabs(data_[i] - other.data()[i]);
    }
    return meanDiff/static_cast<float> (numel());
}
