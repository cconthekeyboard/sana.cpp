#include "weights_io.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "ggml.h"
#include "gguf.h"

namespace {

std::string read_npy_header(std::ifstream & input) {
    char magic[6];
    input.read(magic, sizeof(magic));
    if (!input || std::string(magic, sizeof(magic)) != "\x93NUMPY") {
        throw std::runtime_error("invalid npy magic header");
    }

    std::uint8_t major = 0;
    std::uint8_t minor = 0;
    input.read(reinterpret_cast<char *>(&major), sizeof(major));
    input.read(reinterpret_cast<char *>(&minor), sizeof(minor));
    if (!input) {
        throw std::runtime_error("failed to read npy version");
    }

    std::size_t header_size = 0;
    if (major == 1) {
        std::uint16_t size16 = 0;
        input.read(reinterpret_cast<char *>(&size16), sizeof(size16));
        header_size = size16;
    } else if (major == 2 || major == 3) {
        std::uint32_t size32 = 0;
        input.read(reinterpret_cast<char *>(&size32), sizeof(size32));
        header_size = size32;
    } else {
        throw std::runtime_error("unsupported npy version");
    }

    if (!input) {
        throw std::runtime_error("failed to read npy header size");
    }

    std::string header(header_size, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!input) {
        throw std::runtime_error("failed to read npy header");
    }

    return header;
}

bool parse_fortran_order(const std::string & header) {
    std::size_t key = header.find("fortran_order");
    if (key == std::string::npos) {
        throw std::runtime_error("npy header missing fortran_order");
    }

    std::size_t colon = header.find(':', key);
    if (colon == std::string::npos) {
        throw std::runtime_error("invalid fortran_order entry");
    }

    std::size_t value = header.find_first_not_of(' ', colon + 1);
    if (value == std::string::npos) {
        throw std::runtime_error("invalid fortran_order entry");
    }

    if (header.compare(value, 4, "True") == 0) {
        return true;
    }
    if (header.compare(value, 5, "False") == 0) {
        return false;
    }

    throw std::runtime_error("invalid fortran_order value");
}

std::vector<size_t> parse_shape(const std::string & header) {
    std::size_t start = header.find('(');
    std::size_t end = header.find(')', start);
    if (start == std::string::npos || end == std::string::npos || end <= start) {
        throw std::runtime_error("invalid shape entry");
    }

    std::string inside = header.substr(start + 1, end - start - 1);
    std::stringstream stream(inside);
    std::vector<size_t> shape;
    while (stream.good()) {
        while (stream.peek() == ' ' || stream.peek() == ',') {
            stream.get();
        }
        if (!stream.good()) {
            break;
        }

        size_t dim = 0;
        stream >> dim;
        if (stream.fail()) {
            throw std::runtime_error("invalid shape dimension");
        }
        shape.push_back(dim);
    }

    if (shape.empty()) {
        throw std::runtime_error("empty shape is not supported in npy loader");
    }

    return shape;
}

void validate_dtype(const std::string & header) {
    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos) {
        throw std::runtime_error("only little-endian float32 npy tensors are supported");
    }
}

// ggml's ne[0] is the fastest-varying axis, i.e. the *last* numpy/PyTorch
// dimension -- reverse it to recover the original {out_channels, in_channels,
// ...}-style shape our Tensor class expects (verified directly against the
// Python gguf writer, which reverses shape metadata the same way on write).
std::vector<size_t> ggml_tensor_shape_local(const ggml_tensor * tensor) {
    const int n_dims = ggml_n_dims(tensor);
    std::vector<size_t> shape(static_cast<size_t>(n_dims));
    for (int i = 0; i < n_dims; ++i) {
        shape[static_cast<size_t>(i)] = static_cast<size_t>(tensor->ne[n_dims - 1 - i]);
    }
    return shape;
}

std::unordered_map<std::string, SavedWeightInfo> load_gguf_weight_index(const std::string & path) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx };
    struct gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (!gguf_ctx) {
        throw std::runtime_error("failed to open gguf weight file: " + path);
    }

    std::unordered_map<std::string, SavedWeightInfo> index;
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        const std::string name = ggml_get_name(cur);
        const int64_t tensor_idx = gguf_find_tensor(gguf_ctx, name.c_str());
        if (tensor_idx < 0) {
            ggml_free(ctx);
            gguf_free(gguf_ctx);
            throw std::runtime_error("tensor missing from gguf tensor-info section: " + name);
        }

        SavedWeightInfo info;
        info.file = path;
        info.shape = ggml_tensor_shape_local(cur);
        info.dtype = "float32";
        info.format = WeightFormat::Gguf;
        info.offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
        index[name] = info;
    }

    ggml_free(ctx);
    gguf_free(gguf_ctx);
    return index;
}

}  // namespace

bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::unordered_map<std::string, SavedWeightInfo> load_weight_index(const std::string & path) {
    if (ends_with(path, ".gguf")) {
        return load_gguf_weight_index(path);
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open weight index");
    }

    std::string json(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );

    std::unordered_map<std::string, SavedWeightInfo> index;
    std::size_t pos = 0;

    auto skip_ws = [&](std::size_t & i) {
        while (i < json.size() &&
               (json[i] == ' ' || json[i] == '\n' || json[i] == '\r' || json[i] == '\t')) {
            ++i;
        }
    };

    auto parse_quoted = [&](std::size_t & i) -> std::string {
        skip_ws(i);
        if (i >= json.size() || json[i] != '"') {
            throw std::runtime_error("expected quoted json string");
        }
        ++i;

        std::string value;
        while (i < json.size() && json[i] != '"') {
            if (json[i] == '\\') {
                ++i;
                if (i >= json.size()) {
                    throw std::runtime_error("invalid json escape");
                }
            }
            value.push_back(json[i]);
            ++i;
        }

        if (i >= json.size() || json[i] != '"') {
            throw std::runtime_error("unterminated json string");
        }
        ++i;
        return value;
    };

    auto expect = [&](std::size_t & i, char ch) {
        skip_ws(i);
        if (i >= json.size() || json[i] != ch) {
            throw std::runtime_error("invalid weight index json");
        }
        ++i;
    };

    auto parse_shape = [&](std::size_t & i) -> std::vector<size_t> {
        expect(i, '[');
        std::vector<size_t> shape;
        skip_ws(i);
        if (i < json.size() && json[i] == ']') {
            ++i;
            return shape;
        }

        while (true) {
            skip_ws(i);
            std::size_t consumed = 0;
            unsigned long long dim = std::stoull(json.substr(i), &consumed);
            if (consumed == 0 || dim > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("invalid shape dimension in weight index");
            }
            shape.push_back(static_cast<size_t>(dim));
            i += consumed;

            skip_ws(i);
            if (i >= json.size()) {
                throw std::runtime_error("unterminated shape array");
            }
            if (json[i] == ']') {
                ++i;
                return shape;
            }
            if (json[i] != ',') {
                throw std::runtime_error("invalid shape array separator");
            }
            ++i;
        }
    };

    expect(pos, '{');
    skip_ws(pos);
    if (pos < json.size() && json[pos] == '}') {
        return index;
    }

    while (true) {
        const std::string name = parse_quoted(pos);
        expect(pos, ':');
        expect(pos, '{');

        SavedWeightInfo info;
        while (true) {
            const std::string field = parse_quoted(pos);
            expect(pos, ':');

            if (field == "file") {
                info.file = parse_quoted(pos);
            } else if (field == "shape") {
                info.shape = parse_shape(pos);
            } else if (field == "dtype") {
                info.dtype = parse_quoted(pos);
            } else {
                throw std::runtime_error("unexpected field in weight index");
            }

            skip_ws(pos);
            if (pos >= json.size()) {
                throw std::runtime_error("unterminated weight info object");
            }
            if (json[pos] == '}') {
                ++pos;
                break;
            }
            if (json[pos] != ',') {
                throw std::runtime_error("invalid weight info separator");
            }
            ++pos;
        }

        index[name] = info;

        skip_ws(pos);
        if (pos >= json.size()) {
            throw std::runtime_error("unterminated weight index object");
        }
        if (json[pos] == '}') {
            ++pos;
            break;
        }
        if (json[pos] != ',') {
            throw std::runtime_error("invalid top-level weight index separator");
        }
        ++pos;
    }

    return index;
}

Tensor load_weight_tensor(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    const std::string & name
) {
    auto it = index.find(name);
    if (it == index.end()) {
        throw std::invalid_argument("weight name not found in weight index");
    }

    const SavedWeightInfo & info = it->second;
    if (info.dtype != "float32") {
        throw std::runtime_error("only float32 saved weights are supported");
    }

    if (info.format == WeightFormat::Gguf) {
        Tensor tensor(info.shape);
        std::ifstream input(info.file, std::ios::binary);
        if (!input) {
            throw std::runtime_error("failed to open gguf weight file: " + info.file);
        }
        input.seekg(static_cast<std::streamoff>(info.offset));
        std::vector<float> & data = tensor.data();
        input.read(
            reinterpret_cast<char *>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float))
        );
        if (!input) {
            throw std::runtime_error("failed to read gguf tensor payload for: " + name);
        }
        return tensor;
    }

    const std::string path = weights_dir + "/" + info.file;
    Tensor tensor = load_npy_tensor(path);
    if (tensor.shape() != info.shape) {
        throw std::runtime_error("loaded weight tensor shape does not match index metadata");
    }

    return tensor;
}

Tensor load_linear_weight_tensor(
    const std::string & weights_dir,
    const std::unordered_map<std::string, SavedWeightInfo> & index,
    const std::string & name
) {
    Tensor weight = load_weight_tensor(weights_dir, index, name);
    if (weight.rank() != 2) {
        throw std::invalid_argument("linear weight must be rank 2");
    }
    return weight.transpose_2d();
}

Tensor load_npy_tensor(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open npy file");
    }

    const std::string header = read_npy_header(input);
    validate_dtype(header);
    if (parse_fortran_order(header)) {
        throw std::runtime_error("fortran-order npy tensors are not supported");
    }

    const std::vector<size_t> shape = parse_shape(header);
    Tensor tensor(shape);

    std::vector<float> & data = tensor.data();
    input.read(reinterpret_cast<char *>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("failed to read tensor payload");
    }

    return tensor;
}

void save_npy_tensor(const Tensor & tensor, const std::string & path) {
    std::string shape_str = "(";
    const std::vector<size_t> & shape = tensor.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        shape_str += ",";
        if (i + 1 < shape.size()) {
            shape_str += " ";
        }
    }
    shape_str += ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";

    // Prefix is magic(6) + version(2) + header_size(2) = 10 bytes; pad the
    // header (including its trailing '\n') so the payload starts on a
    // 16-byte boundary, matching numpy's own .npy v1.0 writer.
    const size_t prefix_size = 10;
    const size_t unpadded_total = prefix_size + header.size() + 1;
    const size_t pad = (16 - (unpadded_total % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open npy file for writing: " + path);
    }

    output.write("\x93NUMPY", 6);
    const std::uint8_t major = 1;
    const std::uint8_t minor = 0;
    output.write(reinterpret_cast<const char *>(&major), sizeof(major));
    output.write(reinterpret_cast<const char *>(&minor), sizeof(minor));
    const std::uint16_t header_size = static_cast<std::uint16_t>(header.size());
    output.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    const std::vector<float> & data = tensor.data();
    output.write(
        reinterpret_cast<const char *>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(float))
    );
    if (!output) {
        throw std::runtime_error("failed to write npy payload: " + path);
    }
}

TensorComparison compare_tensor_to_reference(const Tensor & actual, const std::string & path) {
    Tensor ref = load_npy_tensor(path);
    if (ref.shape() != actual.shape()) {
        throw std::invalid_argument("reference tensor shape does not match the actual tensor shape");
    }
    return {actual.max_abs_diff(ref), actual.mean_abs_diff(ref)};
}

Tensor slice_batch_3d(const Tensor & input, size_t batch_index) {
    if (input.rank() != 3) {
        throw std::invalid_argument("input must be rank 3");
    }
    if (batch_index >= input.dim_size(0)) {
        throw std::invalid_argument("batch_index out of range");
    }

    const size_t tokens = input.dim_size(1);
    const size_t hidden = input.dim_size(2);
    Tensor out({1, tokens, hidden});
    const std::vector<float> & input_data = input.data();
    std::vector<float> & out_data = out.data();

    for (size_t t = 0; t < tokens; ++t) {
        for (size_t h = 0; h < hidden; ++h) {
            out_data[t * hidden + h] = input_data[(batch_index * tokens + t) * hidden + h];
        }
    }

    return out;
}

TensorComparison compare_batch_slice_to_reference(
    const Tensor & actual,
    const std::string & path,
    size_t batch_index
) {
    Tensor ref = load_npy_tensor(path);
    Tensor ref_slice = slice_batch_3d(ref, batch_index);
    if (ref_slice.shape() != actual.shape()) {
        throw std::invalid_argument("reference tensor slice shape does not match the actual tensor shape");
    }
    return {actual.max_abs_diff(ref_slice), actual.mean_abs_diff(ref_slice)};
}

TensorComparison compare_block_output_to_reference(
    const Tensor & actual,
    const std::string & path,
    size_t batch_index
) {
    Tensor ref = load_npy_tensor(path);
    Tensor ref_slice = slice_batch_3d(ref, batch_index);
    if (ref_slice.shape() != actual.shape()) {
        throw std::invalid_argument("reference block output slice shape does not match the actual tensor shape");
    }
    return {actual.max_abs_diff(ref_slice), actual.mean_abs_diff(ref_slice)};
}
