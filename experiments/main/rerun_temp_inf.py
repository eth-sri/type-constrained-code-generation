import subprocess
import time
from math import ceil
# rerun the failed instances with ASI

GPUS = list(range(8))  # Example list of GPUs, adjust based on your system
N = 1  # Set the max number of allowed processes per GPU
GPUSIZE = 80
models = [
    # ("microsoft/Phi-3.5-mini-instruct", 4),
    # ("codellama/CodeLlama-7b-Instruct-hf", 7),
    # ("meta-llama/Llama-3.1-8B-Instruct", 8),
    #     ("google/gemma-2b-it", 2),
    ("google/gemma-2-2b-it", 2),
    ("google/gemma-2-9b-it", 9),
    ("google/gemma-2-27b-it", 27),
    ("deepseek-ai/deepseek-coder-33b-instruct", 33),
    # ("deepseek-ai/deepseek-coder-7b-instruct-v1.5", 7),
    # ("deepseek-ai/deepseek-coder-1.3b-instruct", 1.3),
    # ("google/codegemma-2b", 2),
    # ("meta-llama/Llama-3.1-70B-Instruct", 70),
    # ("codellama/CodeLlama-70b-Instruct-hf", 70),
    ("codellama/CodeLlama-34b-Instruct-hf", 34),
    # ("codellama/CodeLlama-13b-Instruct-hf", 13),
    # ("google/codegemma-7b-it", 7),
    # ("bigcode/octocoder", 14),
    ("Qwen/Qwen2.5-32B-Instruct", 32),
]


def compute_needed_gpus(size_model, size_gpu):
    return (size_model * 2 * 1.15) / size_gpu


# subsets = ["main", "mbpp"]
# temps = ["1"]  # , "0", "0.5"]
# seeds = [0, 1, 2, 3]
configs = [
    ("", "_synth"),
    (
        "--input_file '../translation/{}/dataset.json' --translate True --translation_source_lang Python",
        "_translate",
    ),
]
# constraineds = [True]
timeout = 300
max_tokens = 1000


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


constrained = True
with open("failed_without_ASI") as f:
    failed = f.readlines()
total_configs = []
subset, model, seed, temp, suffix = None, None, None, None, None
current_tasks = []
config_to_tasks = {}
for entry in failed:
    if entry.startswith("params: "):
        config_to_tasks[(subset, model, seed, temp, suffix)] = current_tasks
        current_tasks = []
        if "repair-all" in entry:
            subset, model, seed, temp, suffix = None, None, None, None, None
            continue
        subset, model, seed, temp, suffix, _, _ = entry[len("params: ") :].split(",")
        subset = subset.strip()
        model = model.strip()
        seed = seed.strip()
        temp = temp.strip()
        suffix = suffix.strip()
        for config, name in configs:
            if name == suffix:
                total_configs.append(
                    (
                        seed,
                        temp,
                        config,
                        name,
                        constrained,
                        model,
                        next(
                            size for model_name, size in models if model_name == model
                        ),
                        subset,
                    )
                )
    else:
        current_task = entry.strip().split("_")[1]
        current_tasks.append(current_task)


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
    cuda_devices = find_available_gpus(GPUS, N)
    total_config = None
    for total_config in remaining_configs:
        (
            seed,
            temp,
            config,
            name,
            constrained,
            model,
            model_size,
            subset,
        ) = total_config
        needed_gpus = compute_needed_gpus(model_size, GPUSIZE)
        if needed_gpus > len(GPUS):
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
        config = config.format("main-x" if subset == "main" else subset)

    if constrained:
        suffix = "c"
    else:
        suffix = "nc"

    task_ids = config_to_tasks[(subset, model, seed, temp, name)]
    command = (
        f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in cuda_devices)} python3 inference_multiple.py "
        f"--max-tokens {max_tokens} --timeout {timeout} --model_name {model} --seed {seed} --temp {temp} --subset {subset} --task_id {','.join(task_ids)} "
        f"--constrained {constrained} --output_file 'results/rerun_{subset}_{model.replace('/', '_')}_s={seed}_t={temp}{name}_{suffix}.jsonl' {config}"
    )
    print("+ " + command)
    pipe = subprocess.Popen(["/bin/bash", "-c", command])
    running_configs.append((total_config, pipe))
    time.sleep(20)
