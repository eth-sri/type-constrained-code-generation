import csv
import sys
from collections import defaultdict

import fire
from tabulate import tabulate

from experiments.main.analyze_inf_res import main as inf_res

SUFFIXES = ["_synth", "_translate"]
SUBSETS = ["humaneval", "mbpp"]
SUBSET_SIZE = {
    "humaneval": 1,
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


def main(
    format="github", field="compiler_output", subset="humaneval", directory="results"
):
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
    for model in models:
        du = defaultdict(float)
        dc = defaultdict(float)
        id = defaultdict(float)
        for suffix in SUFFIXES:
            seeds = [0] if subset == "mbpp" else [0, 1, 2, 3]
            for seed in seeds:
                res, n, _ = inf_res(
                    [
                        f"{directory}/{subset}_{model.replace('/','_')}_s={seed}_t={temp}{suffix}_nc.jsonl"
                    ],
                    field=field,
                    condition=condition,
                    show_list=False,
                    intersection=field == "compiler_output",
                )
                du[suffix] += res * n if res != "incomplete" else 0
                res, n, _ = inf_res(
                    [
                        f"{directory}/{subset}_{model.replace('/','_')}_s={seed}_t={temp}{suffix}_nc.jsonl",
                        f"{directory}/{subset}_{model.replace('/', '_')}_s={seed}_t={temp}{suffix}_c.jsonl",
                    ],
                    field=field,
                    condition=condition,
                    show_list=False,
                    intersection=field == "compiler_output",
                )
                dc[suffix] += res * n if res != "incomplete" else 0
                res, n, _ = inf_res(
                    [
                        f"{directory}/{subset}_{model.replace('/', '_')}_s={seed}_t={temp}{suffix}_nc.jsonl",
                    ],
                    field="syntax_error",
                    condition="none",
                    show_list=False,
                    intersection=True,
                )
                id[suffix] += res * n if res != "incomplete" else 0
        unconstrained.append(du)
        constrained.append(dc)
        ideal_syntax.append(id)

    headers = ["", "Model"] + ["Standard", "Syntax", "Types", ""] * len(SUFFIXES)
    rows = []
    for model, unc, con, id in zip(models, unconstrained, constrained, ideal_syntax):
        row = ["", MODEL_NAME_MAP[model]]
        for suffix in SUFFIXES:
            row.extend(
                (
                    int(unc[suffix] * SUBSET_SIZE[subset]),
                    "${}$&$_{{\downarrow {:.1f}\\%}}$".format(
                        int((unc[suffix] - id[suffix]) * SUBSET_SIZE[subset]),
                        id[suffix] * 100 / unc[suffix] if unc[suffix] != 0 else 0,
                    ),
                    "$\\textbf{{{}}}$&$_{{\downarrow {:.1f}\\%}}$".format(
                        int(con[suffix] * SUBSET_SIZE[subset]),
                        (unc[suffix] - con[suffix]) * 100 / unc[suffix]
                        if unc[suffix] != 0
                        else 0,
                    ),
                    "",
                )
            )
        row.pop(-1)
        rows.append(row)
    if format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow([""] + headers)
        writer.writerows(rows)
    else:
        print(tabulate(rows, headers=headers, tablefmt=format))


if __name__ == "__main__":
    fire.Fire(main)
