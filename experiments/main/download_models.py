import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM


def main():
    # Default models
    default_models = [
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "deepseek-ai/deepseek-coder-33b-instruct",
        "codellama/CodeLlama-34b-Instruct-hf",
        "Qwen/Qwen2.5-32B-Instruct",
    ]

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Download and load models for causal language modeling."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(default_models),
        help=f"Comma-separated list of model names to load.\nDefault: {','.join(default_models)}",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device to load models on temporarily (e.g., 'cpu', 'auto', 'cuda:0').\nDefault: 'cpu'",
    )
    args = parser.parse_args()

    # Load datasets
    dataset_name = "nuprl/MultiPL-E"
    load_dataset(dataset_name, "humaneval-ts")["test"]
    load_dataset(dataset_name, "mbpp-ts")["test"]

    # Parse models
    models = [x.strip() for x in args.models.split(",")]

    # Load models
    for model in models:
        x = AutoModelForCausalLM.from_pretrained(model, device_map=args.device_map)
        del x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
