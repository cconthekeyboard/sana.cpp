#include "gemma_encoder.h"
#include "torch_rng.h"
#include "transformer.h"
#include "vae.h"
#include "weights_io.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#endif

namespace {

struct InferCliArgs {
    std::string weights_dir = "../weights";
    std::string conditioning_dir;
    std::string gemma_gguf_path = "../weights/gemma2_text_encoder/gemma-2-2b-it-f32.gguf";
    std::string output_path = "sana_output.png";
    std::string prompt;
    std::string negative_prompt;
    size_t steps = 20;
    size_t seed = 42;
    float guidance_scale = 4.5f;
};

void print_usage(const char * program_name) {
    std::cout
        << "Usage: " << program_name << " [options]\n"
        << "\n"
        << "Runs the full Sana pipeline natively in C++: text encoding (vendored\n"
        << "llama.cpp Gemma-2), the seeded initial latent, the denoising loop, and VAE\n"
        << "decode -- no Python involved.\n"
        << "\n"
        << "Options:\n"
        << "  --weights-dir PATH     Directory containing exported model/denoiser/VAE weights.\n"
        << "                         Default: ../weights\n"
        << "  --prompt TEXT          Prompt text (required).\n"
        << "  --negative-prompt TEXT Negative prompt for conditioning generation.\n"
        << "                         Default: empty string\n"
        << "  --gemma-gguf PATH      Path to the converted Gemma-2 GGUF file.\n"
        << "                         Default: ../weights/gemma2_text_encoder/gemma-2-2b-it-f32.gguf\n"
        << "  --output PATH          Output image path. Use .png on macOS, or .ppm anywhere.\n"
        << "                         Default: sana_output.png\n"
        << "  --steps N              Number of denoising steps.\n"
        << "                         Default: 20\n"
        << "  --seed N               Seed used for latent generation in conditioning prep.\n"
        << "                         Default: 42\n"
        << "  --guidance SCALE       CFG guidance scale.\n"
        << "                         Default: 4.5\n"
        << "  --help                 Show this message.\n";
}

InferCliArgs parse_args(int argc, char ** argv) {
    InferCliArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        auto require_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (option == "--weights-dir") {
            args.weights_dir = require_value("--weights-dir");
            continue;
        }
        if (option == "--output") {
            args.output_path = require_value("--output");
            continue;
        }
        if (option == "--prompt") {
            args.prompt = require_value("--prompt");
            continue;
        }
        if (option == "--negative-prompt") {
            args.negative_prompt = require_value("--negative-prompt");
            continue;
        }
        if (option == "--gemma-gguf") {
            args.gemma_gguf_path = require_value("--gemma-gguf");
            continue;
        }
        if (option == "--steps") {
            args.steps = static_cast<size_t>(std::stoull(require_value("--steps")));
            continue;
        }
        if (option == "--seed") {
            args.seed = static_cast<size_t>(std::stoull(require_value("--seed")));
            continue;
        }
        if (option == "--guidance") {
            args.guidance_scale = std::stof(require_value("--guidance"));
            continue;
        }
        if (!option.empty() && option[0] != '-') {
            args.output_path = option;
            continue;
        }
        throw std::invalid_argument("unknown option: " + option);
    }

    if (args.prompt.empty()) {
        throw std::invalid_argument("--prompt is required");
    }
    if (args.steps == 0) {
        throw std::invalid_argument("--steps must be positive");
    }
    if (!(args.guidance_scale >= 0.0f)) {
        throw std::invalid_argument("--guidance must be non-negative");
    }
    return args;
}

std::string build_runtime_conditioning_dir(
    const std::string & output_path,
    size_t seed
) {
    namespace fs = std::filesystem;

    fs::path output(output_path);
    std::string stem = output.stem().string();
    if (stem.empty()) {
        stem = "sana_output";
    }
    fs::path base = fs::path(".sana_conditioning") / (stem + "_seed" + std::to_string(seed));
    return base.string();
}

struct PreparedConditioning {
    Tensor text_embeddings;
    Tensor initial_latent;
    Tensor encoder_attention_mask;
};

PreparedConditioning prepare_conditioning(InferCliArgs & cli) {
    namespace fs = std::filesystem;
    cli.conditioning_dir = build_runtime_conditioning_dir(cli.output_path, cli.seed);
    fs::create_directories(cli.conditioning_dir);

    std::cout << "encoding prompt with native Gemma-2 (llama.cpp)\n" << std::flush;
    GemmaEncoderConfig gemma_config;
    gemma_config.gguf_path = cli.gemma_gguf_path;
    GemmaEncoder gemma_encoder(gemma_config);
    GemmaPromptPair conditioning = encode_prompt_pair(gemma_encoder, cli.prompt, cli.negative_prompt);
    save_npy_tensor(conditioning.embeddings, cli.conditioning_dir + "/caption_projection_input.npy");
    save_npy_tensor(conditioning.attention_bias, cli.conditioning_dir + "/encoder_attention_mask.npy");

    std::cout << "generating seeded initial latent (native RNG port)\n" << std::flush;
    constexpr size_t kNumChannelsLatents = 32;
    constexpr size_t kVaeScaleFactor = 32;
    constexpr size_t kHeight = 1024;
    constexpr size_t kWidth = 1024;
    Tensor initial_latent = randn_torch_cpu(
        {1, kNumChannelsLatents, kHeight / kVaeScaleFactor, kWidth / kVaeScaleFactor},
        cli.seed
    );
    save_npy_tensor(initial_latent, cli.conditioning_dir + "/initial_latent.npy");

    return PreparedConditioning{
        std::move(conditioning.embeddings),
        std::move(initial_latent),
        std::move(conditioning.attention_bias),
    };
}

std::uint8_t clamp_to_u8(float value) {
    const float scaled = std::round(std::clamp(value, 0.0f, 1.0f) * 255.0f);
    return static_cast<std::uint8_t>(scaled);
}

void validate_image_tensor(const Tensor & image) {
    if (image.rank() != 4) {
        throw std::invalid_argument("decoded image must be rank 4");
    }
    if (image.dim_size(0) != 1 || image.dim_size(1) != 3) {
        throw std::invalid_argument("decoded image must have shape {1, 3, H, W}");
    }
}

std::vector<std::uint8_t> tensor_to_rgb8(const Tensor & image) {
    validate_image_tensor(image);

    const size_t height = image.dim_size(2);
    const size_t width = image.dim_size(3);
    const size_t spatial = height * width;
    const float * image_data = image.data().data();
    const float * r = image_data;
    const float * g = image_data + spatial;
    const float * b = image_data + 2 * spatial;

    std::vector<std::uint8_t> rgb(spatial * 3);
    for (size_t i = 0; i < spatial; ++i) {
        rgb[i * 3 + 0] = clamp_to_u8(r[i]);
        rgb[i * 3 + 1] = clamp_to_u8(g[i]);
        rgb[i * 3 + 2] = clamp_to_u8(b[i]);
    }
    return rgb;
}

Tensor denormalize_image_for_output(const Tensor & image) {
    validate_image_tensor(image);
    Tensor out(image.shape());
    for (size_t i = 0; i < image.numel(); ++i) {
        out.at(i) = image.at(i) * 0.5f + 0.5f;
    }
    return out;
}

void save_tensor_as_ppm(const Tensor & image, const std::string & path) {
    Tensor display_image = denormalize_image_for_output(image);
    const size_t height = display_image.dim_size(2);
    const size_t width = display_image.dim_size(3);
    const std::vector<std::uint8_t> rgb = tensor_to_rgb8(display_image);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output image path");
    }

    output << "P6\n" << width << " " << height << "\n255\n";
    output.write(reinterpret_cast<const char *>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
}

#if defined(__APPLE__)
void save_tensor_as_png(const Tensor & image, const std::string & path) {
    Tensor display_image = denormalize_image_for_output(image);

    const size_t height = display_image.dim_size(2);
    const size_t width = display_image.dim_size(3);
    std::vector<std::uint8_t> rgb = tensor_to_rgb8(display_image);

    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    if (!color_space) {
        throw std::runtime_error("failed to create RGB color space");
    }

    CFDataRef data = CFDataCreate(kCFAllocatorDefault, rgb.data(), static_cast<CFIndex>(rgb.size()));
    if (!data) {
        CGColorSpaceRelease(color_space);
        throw std::runtime_error("failed to create PNG pixel buffer");
    }

    CGDataProviderRef provider = CGDataProviderCreateWithCFData(data);
    if (!provider) {
        CFRelease(data);
        CGColorSpaceRelease(color_space);
        throw std::runtime_error("failed to create PNG data provider");
    }

    CGImageRef cg_image = CGImageCreate(
        static_cast<size_t>(width),
        static_cast<size_t>(height),
        8,
        24,
        static_cast<size_t>(width * 3),
        color_space,
        kCGBitmapByteOrderDefault,
        provider,
        nullptr,
        false,
        kCGRenderingIntentDefault
    );
    if (!cg_image) {
        CGDataProviderRelease(provider);
        CFRelease(data);
        CGColorSpaceRelease(color_space);
        throw std::runtime_error("failed to create CGImage for PNG output");
    }

    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8 *>(path.data()),
        static_cast<CFIndex>(path.size()),
        false
    );
    if (!url) {
        CGImageRelease(cg_image);
        CGDataProviderRelease(provider);
        CFRelease(data);
        CGColorSpaceRelease(color_space);
        throw std::runtime_error("failed to create output URL for PNG");
    }

    CGImageDestinationRef destination =
        CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr);
    if (!destination) {
        CFRelease(url);
        CGImageRelease(cg_image);
        CGDataProviderRelease(provider);
        CFRelease(data);
        CGColorSpaceRelease(color_space);
        throw std::runtime_error("failed to create PNG destination");
    }

    CGImageDestinationAddImage(destination, cg_image, nullptr);
    const bool success = CGImageDestinationFinalize(destination);

    CFRelease(destination);
    CFRelease(url);
    CGImageRelease(cg_image);
    CGDataProviderRelease(provider);
    CFRelease(data);
    CGColorSpaceRelease(color_space);

    if (!success) {
        throw std::runtime_error("failed to write PNG image");
    }
}
#endif

void save_tensor_image(const Tensor & image, const std::string & path) {
    const size_t dot = path.find_last_of('.');
    const std::string ext = dot == std::string::npos ? "" : path.substr(dot);
#if defined(__APPLE__)
    if (ext == ".png") {
        save_tensor_as_png(image, path);
        return;
    }
#endif
    save_tensor_as_ppm(image, path);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        InferCliArgs cli = parse_args(argc, argv);
        PreparedConditioning conditioning = prepare_conditioning(cli);

        const std::string & weights_dir = cli.weights_dir;
        const std::string model_weight_path = weights_dir + "/model_weights.gguf";
        const std::string caption_weight_path = weights_dir + "/caption_weights.gguf";
        const std::string denoiser_weight_path = weights_dir + "/denoiser_weights.gguf";
        const std::string vae_decoder_weight_path = weights_dir + "/vae_decoder_weights.gguf";

        std::cout << "loading weight indices\n" << std::flush;
        auto model_weight_index = load_weight_index(model_weight_path);
        auto caption_weight_index = load_weight_index(caption_weight_path);
        auto denoiser_weight_index = load_weight_index(denoiser_weight_path);
        auto vae_decoder_weight_index = load_weight_index(vae_decoder_weight_path);

        std::cout << "loading runtime weights\n" << std::flush;
        SanaDenoiserWeights denoiser_weights = load_denoiser_weights(
            model_weight_path,
            model_weight_index,
            caption_weight_path,
            caption_weight_index,
            denoiser_weight_path,
            denoiser_weight_index,
            36,
            16
        );
        SanaVaeWeights vae_weights = load_vae_weights(
            vae_decoder_weight_path,
            vae_decoder_weight_index,
            0.41407f,
            32
        );
        validate_vae_weights(vae_weights);

        std::cout << "using weights from: " << weights_dir << "\n";
        std::cout << "using conditioning tensors from: " << cli.conditioning_dir << "\n";
        std::cout << "steps: " << cli.steps
                  << " guidance: " << cli.guidance_scale << "\n" << std::flush;
        std::cout << "running denoising loop\n" << std::flush;
        SanaSchedulerConfig scheduler_config = make_sana_scheduler_config();
        SanaSchedulerState scheduler_state = init_sana_scheduler_state(cli.steps, scheduler_config);
        Tensor final_latents = run_full_denoising_loop_with_cfg(
            conditioning.initial_latent,
            conditioning.text_embeddings,
            cli.guidance_scale,
            scheduler_state,
            scheduler_config,
            denoiser_weights,
            &conditioning.encoder_attention_mask
        );

        std::cout << "running vae decode\n" << std::flush;
        Tensor decoded = run_vae_decode(final_latents, vae_weights);

        std::cout << "writing image: " << cli.output_path << "\n" << std::flush;
        save_tensor_image(decoded, cli.output_path);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
