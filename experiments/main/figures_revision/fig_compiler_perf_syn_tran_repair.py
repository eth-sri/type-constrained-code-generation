import csv
import sys
from collections import defaultdict

import fire
from tabulate import tabulate

from experiments.main.analyze_inf_res import main as inf_res

SUFFIXES = ["_synth", "_translate", "_repair-all"]
SUBSETS = ["humaneval", "mbpp"]
SUBSET_SIZE_REPAIR = {
    "humaneval": 309,
    "mbpp": 317,
}
MODEL_NAME_MAP = {
    "google/gemma-2-2b-it": "Gemma 2 2B",
    "google/gemma-2-9b-it": "Gemma 2 9B",
    "google/gemma-2-27b-it": "Gemma 2 27B",
    "deepseek-ai/deepseek-coder-33b-instruct": "DS Coder 33B",
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
            seeds = [0] if subset == "mbpp" or suffix == "_repair-all" else [0, 1, 2, 3]
            discard_unconstrained, discard_constrained = False, False
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
                if res == "incomplete":
                    discard_unconstrained = True
                    discard_constrained = True
                    break
                du[suffix] += res * n
                res, n, _ = inf_res(
                    [
                        f"{directory}/{subset}_{model.replace('/', '_')}_s={seed}_t={temp}{suffix}_nc.jsonl",
                    ],
                    field="syntax_error",
                    condition="none",
                    show_list=False,
                    intersection=True,
                )
                if res == "incomplete":
                    discard_unconstrained = True
                    break
                id[suffix] += res * n
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
                if res == "incomplete":
                    discard_constrained = True
                    break
                dc[suffix] += res * n if res != "incomplete" else 0
            if discard_unconstrained:
                du[suffix] = -1
                id[suffix] = -1
            if discard_constrained:
                dc[suffix] = -1
        unconstrained.append(du)
        constrained.append(dc)
        ideal_syntax.append(id)

    # print("Average improvement compiler errors (over Models)")
    # for suffix in SUFFIXES:
    #     improvements_syntax = []
    #     improvements_constrained = []
    #     for model, unc, con, id in zip(
    #         models, unconstrained, constrained, ideal_syntax
    #     ):
    #         improvements_syntax.append(
    #             id[suffix] * 100 / unc[suffix] if unc[suffix] != 0 else 0
    #         )
    #         improvements_constrained.append(
    #             (unc[suffix] - con[suffix]) * 100 / unc[suffix]
    #             if unc[suffix] != 0
    #             else 0
    #         )
    #     print(
    #         f"{suffix}, Syntax: {sum(improvements_syntax) / len(improvements_syntax):.1f}%"
    #     )
    #     print(
    #         f"{suffix}, Types: {sum(improvements_constrained) / len(improvements_constrained):.1f}%"
    #     )

    # print("Repair improvements")
    # suffix = "_repair-all"
    # subset_size = SUBSET_SIZE_REPAIR[subset]
    # for model, unc, con, id in zip(models, unconstrained, constrained, ideal_syntax):
    #     print(
    #         f"{model}, Standard: {(subset_size - unc[suffix]) * 100 / subset_size:.1f}%, Syntax: {(subset_size - (unc[suffix] - id[suffix])) * 100 / subset_size:.1f}%, Types: {(subset_size - con[suffix]) * 100 / subset_size:.1f}%"
    #     )

    headers = ["", "Model"] + ["Standard", "Syntax", r"Types", ""] * len(SUFFIXES)
    rows = []
    for model, unc, con, id in zip(models, unconstrained, constrained, ideal_syntax):
        row = ["", MODEL_NAME_MAP[model]]
        for suffix in SUFFIXES:
            row.extend(
                (
                    int(unc[suffix]) if unc[suffix] != 0 else -1,
                    "${}$&$_{{\downarrow {:.1f}\\%}}$".format(
                        int((unc[suffix] - id[suffix])),
                        id[suffix] * 100 / unc[suffix] if unc[suffix] != 0 else 0,
                    )
                    if unc[suffix] != -1 and id[suffix] != -1
                    else -1,
                    "$\\textbf{{{}}}$&$_{{\downarrow {:.1f}\\%}}$".format(
                        int(con[suffix]),
                        (unc[suffix] - con[suffix]) * 100 / unc[suffix]
                        if unc[suffix] != 0
                        else 0,
                    )
                    if unc[suffix] != -1 and con[suffix] != -1
                    else -1,
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
