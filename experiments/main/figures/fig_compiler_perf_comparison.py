import csv
import sys
from collections import defaultdict

import fire
from tabulate import tabulate

from experiments.main.analyze_inf_res import main as inf_res

SUFFIXES = ["_synth", "_translate"]
SUBSETS = ["main", "mbpp"]
SUBSET_SIZE = {
    "main": 1,
    "mbpp": 1,
}
MODEL_NAME_MAP = {
    "google/gemma-2-2b-it": "Gemma 2 2B",
    "google/gemma-2-9b-it": "Gemma 2 9B",
    "google/gemma-2-27b-it": "Gemma 2 27B",
    "deepseek-ai/deepseek-coder-33b-instruct": "DeepSeek Coder 33B",
    "codellama/CodeLlama-34b-Instruct-hf": "CodeLlama 34B",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5 32B",
}


def main(format="github", field="compiler_output", subset="main"):
    models = (
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "deepseek-ai/deepseek-coder-33b-instruct",
        "codellama/CodeLlama-34b-Instruct-hf",
        "Qwen/Qwen2.5-32B-Instruct",
    )
    temp = 1
    condition = "compiler_output"
    unconstrained = []
    constrained = []
    ideal_syntax = []
    totals = []
    for model in models:
        du = defaultdict(float)
        dc = defaultdict(float)
        id = defaultdict(float)
        total = defaultdict(int)
        for suffix in SUFFIXES:
            seeds = [0] if subset == "mbpp" else [1, 2, 3, 4]
            for seed in seeds:
                res, n, _ = inf_res(
                    [
                        f"results_bak_04122024/{subset}_{model.replace('/','_')}_s={seed}_t={temp}{suffix}_nc.jsonl",
                        f"results/{subset}_{model.replace('/', '_')}_s={seed}_t={temp}{suffix}_c.jsonl",
                    ],
                    field=field,
                    condition=condition,
                    show_list=False,
                    intersection=True,
                )
                total[suffix] += n
                du[suffix] += res * n if res != "incomplete" else -float("inf")
                res, n, _ = inf_res(
                    [
                        f"results_bak_04122024/{subset}_{model.replace('/','_')}_s={seed}_t={temp}{suffix}_nc.jsonl",
                        f"results/{subset}_{model.replace('/', '_')}_s={seed}_t={temp}{suffix}_c.jsonl",
                    ],
                    field=field,
                    condition=condition,
                    show_list=False,
                    intersection=True,
                )
                dc[suffix] += res * n if res != "incomplete" else -float("inf")
        unconstrained.append(du)
        constrained.append(dc)
        ideal_syntax.append(id)
        totals.append(total)

    headers = ["Limited", "Unlimited"] * len(SUFFIXES)
    rows = []
    for model, unc, con, id, total in zip(
        models, unconstrained, constrained, ideal_syntax, totals
    ):
        row = [MODEL_NAME_MAP[model]]
        for suffix in SUFFIXES:
            row.extend(
                (
                    "{:.1f}".format(unc[suffix] * 100 / total[suffix])
                    if total[suffix]
                    else -1,
                    "{:.1f}".format(con[suffix] * 100 / total[suffix])
                    if total[suffix]
                    else -1,
                )
            )
        rows.append(row)
    if format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow([""] + headers)
        writer.writerows(rows)
    else:
        print(tabulate(rows, headers=headers, tablefmt=format))


if __name__ == "__main__":
    fire.Fire(main)
