import pathlib
import subprocess
import time
from math import ceil

import fire

parent = pathlib.Path(__file__).parent.absolute()
GPUS = list(range(8))  # Example list of GPUs, adjust based on your system
N = 1  # Set the max number of allowed processes per GPU
GPUSIZE = 80
MODEL_SIZE_MAP = {
    "google/gemma-2-2b-it": 2,
    "google/gemma-2-9b-it": 9,
    "google/gemma-2-27b-it": 27,
    "deepseek-ai/deepseek-coder-33b-instruct": 33,
    "codellama/CodeLlama-34b-Instruct-hf": 34,
    "Qwen/Qwen2.5-32B-Instruct": 32,
}


def compute_needed_gpus(size_model, size_gpu):
    return (size_model * 2 * 1.15) / size_gpu


SUBSETS = ["humaneval", "mbpp"]
TEMPS = ["1"]
SEEDS = [0, 1, 2, 3]
CONFIGS = [
    ("", "_synth"),
    (
        "--input_file '../translation/{}/dataset.json' --translate True --translation_source_lang Python",
        "_translate",
    ),
]
CONSTRAINEDS = [False, True]
TIMEOUT = 300
MAX_TOKENS = 1000
TRY_TOP_K = 10000000000000000


def find_available_gpus(gpus, n):
    found_gpus = []
    for gpu in gpus:
        process_count = int(
            subprocess.check_output(
                [
                    "/bin/bash",
                    "-c",
                    f"nvidia-smi -i {gpu} --query-compute-apps=pid --format=csv,noheader | wc -l",
                ],
            ).strip()
        )
        if process_count < n:
            found_gpus.append(gpu)
    return found_gpus


def main(
    subsets=SUBSETS,
    seeds=SEEDS,
    temps=TEMPS,
    constraineds=CONSTRAINEDS,
    models=list(MODEL_SIZE_MAP.keys()),
    timeout=TIMEOUT,
    max_tokens=MAX_TOKENS,
    try_top_k=TRY_TOP_K,
    tasks=["synth", "translate"],
    gpu_size=GPUSIZE,
    gpus=GPUS,
    n_process_per_gpu=N,
):
    if isinstance(models, str):
        models = models.split(",")
    if isinstance(subsets, str):
        subsets = subsets.split(",")
    if isinstance(temps, str):
        temps = [float(x) for x in temps.split(",")]
    elif isinstance(temps, int):
        temps = [temps]
    if isinstance(seeds, str):
        seeds = [int(x) for x in seeds.split(",")]
    elif isinstance(seeds, int):
        seeds = [seeds]
    if isinstance(constraineds, str):
        constraineds = [constraineds == "True"]
    elif isinstance(constraineds, int):
        constraineds = [constraineds != 0]
    if isinstance(tasks, str):
        tasks = tasks.split(",")
    if isinstance(gpus, str):
        gpus = [int(x) for x in gpus.split(",")]
    elif isinstance(gpus, int):
        gpus = [gpus]

    assert all(subset in ["humaneval", "mbpp"] for subset in subsets)
    assert all(model in MODEL_SIZE_MAP for model in models)

    configs = [(config, name) for config, name in CONFIGS if name[1:] in tasks]
    total_configs = []
    for constrained in constraineds:
        for subset in subsets:
            for seed in seeds:
                for temp in temps:
                    for config, name in configs:
                        for model in models:
                            total_configs.append(
                                (
                                    seed,
                                    temp,
                                    config,
                                    name,
                                    constrained,
                                    model,
                                    subset,
                                )
                            )

    remaining_configs = total_configs.copy()
    running_configs = list()
    while remaining_configs or running_configs:
        # reinsert crashed programs
        for config, pipe in running_configs:
            if pipe.poll() is not None:
                running_configs.remove((config, pipe))
                if pipe.returncode != 0:
                    remaining_configs.append(config)
        cuda_devices, needed_gpus = [], 1
        cuda_devices = find_available_gpus(gpus, n_process_per_gpu)
        total_config = None
        for total_config in remaining_configs:
            (
                seed,
                temp,
                config,
                name,
                constrained,
                model,
                subset,
            ) = total_config
            needed_gpus = compute_needed_gpus(MODEL_SIZE_MAP[model], gpu_size)
            if needed_gpus > len(gpus):
                print(f"model {model} is too large, skipping")
                remaining_configs.remove(total_config)
                continue
            if len(cuda_devices) >= needed_gpus:
                break
        if len(cuda_devices) < needed_gpus or total_config is None:
            print("No available GPU found or all configs running. Waiting...")
            time.sleep(60)
            continue
        remaining_configs.remove(total_config)
        if subset == "mbpp" and seed != 0:
            continue
        cuda_devices = cuda_devices[: int(ceil(needed_gpus))]
        if "translate True" in config:
            config = config.format("humaneval-x" if subset == "humaneval" else subset)

        if constrained:
            suffix = "c"
        else:
            suffix = "nc"
        command = (
            f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in cuda_devices)} python3 inference_multiple.py "
            f"--max-tokens {max_tokens} --timeout {timeout} --model_name {model} --seed {seed} --temp {temp} --subset {subset}  --try_top_k {try_top_k} "
            f"--constrained {constrained} --output_file 'results/{subset}_{model.replace('/', '_')}_s={seed}_t={temp}{name}_{suffix}.jsonl' {config}"
        )
        print("+ " + command)
        pipe = subprocess.Popen(["/bin/bash", "-c", command], cwd=parent)
        running_configs.append((total_config, pipe))
        time.sleep(20)


if __name__ == "__main__":
    fire.Fire(main)
