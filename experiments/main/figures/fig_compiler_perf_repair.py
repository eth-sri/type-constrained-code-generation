import csv
import sys
from collections import defaultdict

import fire
from tabulate import tabulate

from experiments.main.analyze_inf_res import main as inf_res

SUFFIXES = ["_repair-all"]
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


def main(format="github", field="compiler_output", directory="results"):
    models = (
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "deepseek-ai/deepseek-coder-33b-instruct",
        "codellama/CodeLlama-34b-Instruct-hf",
        "Qwen/Qwen2.5-32B-Instruct",
    )
    suffix = "_repair-all"
    temp = 1
    condition = "compiler_output"
    unconstrained = []
    constrained = []
    ideal_syntax = []
    totals = []
    for model in models:
        total = defaultdict(float)
        du = defaultdict(float)
        dc = defaultdict(float)
        id = defaultdict(float)
        for subset in SUBSETS:
            seeds = [0]
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
                total[subset] += n if res != "incomplete" else 0
                du[subset] += res * n if res != "incomplete" else 0
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
                dc[subset] += res * n if res != "incomplete" else 0
                res, n, _ = inf_res(
                    [
                        f"{directory}/{subset}_{model.replace('/', '_')}_s={seed}_t={temp}{suffix}_nc.jsonl",
                    ],
                    field="syntax_error",
                    condition="none",
                    show_list=False,
                    intersection=True,
                )
                id[subset] += res * n if res != "incomplete" else 0
        unconstrained.append(du)
        constrained.append(dc)
        ideal_syntax.append(id)
        totals.append(total)

    headers = ["Model"] + ["Standard", "Types"] * len(SUBSETS)
    rows = []
    for model, unc, con, id, total in zip(
        models, unconstrained, constrained, ideal_syntax, totals
    ):
        row = [MODEL_NAME_MAP[model]]
        for subset in SUBSETS:
            row.extend(
                (
                    # int(total[subset]),
                    "${}$&$_{{\\downarrow {:.1f}\\%}}$".format(
                        int(unc[subset] * SUBSET_SIZE[subset]),
                        (total[subset] - unc[subset]) * 100 / total[subset]
                        if unc[subset] != 0
                        else 0,
                    ),
                    "$\\textbf{{{}}}$&$_{{\\downarrow {:.1f}\\%}}$".format(
                        int(con[subset] * SUBSET_SIZE[subset]),
                        (total[subset] - con[subset]) * 100 / total[subset]
                        if total[subset] != 0
                        else 0,
                    ),
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
