#include "gemma_encoder.h"
#include "torch_rng.h"
#include "transformer.h"
#include "vae.h"
#include "weights_io.h"

#include <chrono>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct PipelineRunResult {
    Tensor decoded_image;
    double encoder_ms = 0.0;
    double transformer_ms = 0.0;
    double vae_ms = 0.0;
    double total_ms = 0.0;
};

PipelineRunResult run_pipeline_once(
    GemmaEncoder & gemma_encoder,
    const std::string & prompt,
    const std::string & negative_prompt,
    float guidance_scale,
    const Tensor & initial_latent,
    size_t num_steps,
    const SanaSchedulerConfig & scheduler_config,
    const SanaDenoiserWeights & denoiser_weights,
    const SanaVaeWeights & vae_weights
) {
    const auto total_start = Clock::now();

    const auto encoder_start = Clock::now();
    GemmaPromptPair encoded = encode_prompt_pair(gemma_encoder, prompt, negative_prompt);
    const auto encoder_end = Clock::now();

    const auto transformer_start = Clock::now();
    SanaSchedulerState scheduler_state = init_sana_scheduler_state(num_steps, scheduler_config);
    Tensor final_latents = run_full_denoising_loop_with_cfg(
        initial_latent,
        encoded.embeddings,
        guidance_scale,
        scheduler_state,
        scheduler_config,
        denoiser_weights,
        &encoded.attention_bias
    );
    const auto transformer_end = Clock::now();

    const auto vae_start = Clock::now();
    Tensor decoded_image = run_vae_decode(final_latents, vae_weights);
    const auto vae_end = Clock::now();

    const auto total_end = Clock::now();

    // Aggregate-initialized (not default-constructed then assigned): Tensor
    // has no default constructor, only construct-with-value.
    return PipelineRunResult{
        std::move(decoded_image),
        elapsed_ms(encoder_start, encoder_end),
        elapsed_ms(transformer_start, transformer_end),
        elapsed_ms(vae_start, vae_end),
        elapsed_ms(total_start, total_end)
    };
}

void print_run(const char * label, int index, const PipelineRunResult & r) {
    std::cout << label << " " << index << ": " << r.total_ms << " ms"
               << " (encoder=" << r.encoder_ms
               << " transformer=" << r.transformer_ms
               << " vae=" << r.vae_ms << ")\n" << std::flush;
}

}  // namespace

int main(int argc, char ** argv) {
    int warmup = 0;
    int iters = 1;
    if (argc >= 2) {
        warmup = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        iters = std::stoi(argv[2]);
    }
    if (warmup < 0) {
        throw std::invalid_argument("warmup must be non-negative");
    }
    if (iters <= 0) {
        throw std::invalid_argument("iters must be positive");
    }

    // Matches tools/bench_reference_full_pipeline.py's defaults exactly.
    const std::string prompt = "a house by the lake";
    const std::string negative_prompt = "";
    constexpr float guidance_scale = 4.5f;
    constexpr size_t num_steps = 20;
    constexpr uint64_t seed = 42;
    constexpr size_t kNumChannelsLatents = 32;
    constexpr size_t kVaeScaleFactor = 32;
    constexpr size_t kHeight = 1024;
    constexpr size_t kWidth = 1024;

    // === Load model + weights once (excluded from all timing below) ===
    std::cout << "loading Gemma-2 model\n" << std::flush;
    const auto gemma_load_start = Clock::now();

    GemmaEncoderConfig gemma_config;
    gemma_config.gguf_path = "../weights/gemma2_text_encoder/gemma-2-2b-it-f32.gguf";
    GemmaEncoder gemma_encoder(gemma_config);

    const auto gemma_load_end = Clock::now();
    const double gemma_load_ms = elapsed_ms(gemma_load_start, gemma_load_end);

    std::cout << "loading transformer + vae weights\n" << std::flush;
    const auto weights_load_start = Clock::now();

    auto model_weight_index = load_weight_index("../weights/model_weights.gguf");
    auto caption_weight_index = load_weight_index("../weights/caption_weights.gguf");
    auto denoiser_weight_index = load_weight_index("../weights/denoiser_weights.gguf");
    auto vae_decoder_weight_index = load_weight_index("../weights/vae_decoder_weights.gguf");

    SanaDenoiserWeights denoiser_weights = load_denoiser_weights(
        "../weights/model_weights.gguf",
        model_weight_index,
        "../weights/caption_weights.gguf",
        caption_weight_index,
        "../weights/denoiser_weights.gguf",
        denoiser_weight_index,
        36,
        16
    );
    SanaVaeWeights vae_weights = load_vae_weights(
        "../weights/vae_decoder_weights.gguf",
        vae_decoder_weight_index,
        0.41407f,
        32
    );
    validate_vae_weights(vae_weights);

    const auto weights_load_end = Clock::now();
    const double weights_load_ms = elapsed_ms(weights_load_start, weights_load_end);

    Tensor initial_latent = randn_torch_cpu(
        {1, kNumChannelsLatents, kHeight / kVaeScaleFactor, kWidth / kVaeScaleFactor},
        seed
    );
    SanaSchedulerConfig scheduler_config = make_sana_scheduler_config();

    // === Warmup runs (timed but not counted toward the averages below) ===
    std::cout << "warming up full pipeline (" << warmup << " runs, " << num_steps << " steps each)\n"
               << std::flush;
    for (int i = 0; i < warmup; ++i) {
        PipelineRunResult r = run_pipeline_once(
            gemma_encoder, prompt, negative_prompt, guidance_scale, initial_latent,
            num_steps, scheduler_config, denoiser_weights, vae_weights
        );
        print_run("warmup", i + 1, r);
    }

    // === Benchmark runs ===
    std::cout << "benchmarking full pipeline (" << iters << " runs)\n" << std::flush;
    double total_encoder_ms = 0.0;
    double total_transformer_ms = 0.0;
    double total_vae_ms = 0.0;
    double total_total_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        PipelineRunResult r = run_pipeline_once(
            gemma_encoder, prompt, negative_prompt, guidance_scale, initial_latent,
            num_steps, scheduler_config, denoiser_weights, vae_weights
        );
        total_encoder_ms += r.encoder_ms;
        total_transformer_ms += r.transformer_ms;
        total_vae_ms += r.vae_ms;
        total_total_ms += r.total_ms;
        print_run("run", i + 1, r);
    }

    const double count = static_cast<double>(iters);
    std::cout << "\n=== timing summary ===\n";
    std::cout << "gemma_load_ms:      " << gemma_load_ms << "\n";
    std::cout << "weights_load_ms:    " << weights_load_ms << "\n";
    std::cout << "warmup_iters:       " << warmup << "\n";
    std::cout << "iters:              " << iters << "\n";
    std::cout << "total_ms:           " << total_total_ms << "\n";
    std::cout << "avg_ms:             " << (total_total_ms / count) << "\n";
    std::cout << "encoder_ms_avg:     " << (total_encoder_ms / count) << "\n";
    std::cout << "transformer_ms_avg: " << (total_transformer_ms / count)
               << " (" << num_steps << " steps)\n";
    std::cout << "vae_ms_avg:         " << (total_vae_ms / count) << "\n";

    return 0;
}
