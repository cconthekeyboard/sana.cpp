#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the full PyTorch/diffusers Sana pipeline (encode + denoise + VAE decode) end to end.",
    )
    parser.add_argument(
        "--prompt",
        default="a house by the lake",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--model-id",
        default="Efficient-Large-Model/Sana_600M_1024px_diffusers",
        help="Diffusers model id to load.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Execution device for the benchmark.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Weight/activation dtype.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of PyTorch CPU threads to use. Defaults to this machine's CPU count "
        "(os.cpu_count()), matching sana_infer's use of every core via Accelerate/GCD "
        "rather than artificially restricting Python to one thread.",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--benchmark-runs", type=int, default=1)
    parser.add_argument("--output", default=None, help="If set, save the last run's image to this path.")
    return parser.parse_args()


def main() -> None:
    import torch

    try:
        from diffusers import SanaPipeline
    except ImportError as exc:
        raise RuntimeError(
            "This script requires diffusers with Sana support. "
            "Install the current diffusers main branch if needed."
        ) from exc

    args = parse_args()
    device = args.device
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS device requested but not available.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    if device == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    pipe = SanaPipeline.from_pretrained(
        args.model_id,
        variant="fp16",
        torch_dtype=dtype,
    )
    pipe.to(device)
    pipe.text_encoder.to(dtype=dtype)
    pipe.vae.to(dtype=dtype)
    pipe.transformer.to(dtype=dtype)

    def run_once():
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        return pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=generator,
            use_resolution_binning=False,
        )

    print("warming up full pipeline", flush=True)
    output = None
    for i in range(args.warmup_runs):
        start = time.perf_counter()
        output = run_once()
        end = time.perf_counter()
        print(f"warmup {i + 1}: {(end - start) * 1000.0:.3f} ms", flush=True)

    print("benchmarking full pipeline", flush=True)
    total_ms = 0.0
    for i in range(args.benchmark_runs):
        start = time.perf_counter()
        output = run_once()
        end = time.perf_counter()
        run_ms = (end - start) * 1000.0
        total_ms += run_ms
        print(f"run {i + 1}: {run_ms:.3f} ms", flush=True)

    print(f"average: {total_ms / float(args.benchmark_runs):.3f} ms", flush=True)

    if args.output and output is not None:
        output.images[0].save(args.output)
        print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
