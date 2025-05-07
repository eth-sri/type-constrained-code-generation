import csv
import sys
from collections import defaultdict
from statistics import median

import fire
from tabulate import tabulate

from experiments.main.analyze_avg_time import main as inf_res

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


def main(format="github", field="time_taken", suffix="_synth", directory="results"):
    models = (
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "deepseek-ai/deepseek-coder-33b-instruct",
        "codellama/CodeLlama-34b-Instruct-hf",
        "Qwen/Qwen2.5-32B-Instruct",
    )
    temp = 1
    condition = "none"
    unconstrained = []
    constrained = []
    totals = []
    for model in models:
        du = defaultdict(list)
        dc = defaultdict(list)
        total = defaultdict(int)
        for subset in SUBSETS:
            seeds = [0] if subset != "humaneval" or "repair" in suffix else [0, 1, 2, 3]
            for seed in seeds:
                res, n, _ = inf_res(
                    [
                        f"{directory}/{subset}_{model.replace('/','_')}_s={seed}_t={temp}{suffix}_nc.jsonl"
                    ],
                    field=field,
                    condition=condition,
                )
                if res == "incomplete":
                    total[subset] = -1
                    break
                total[subset] += n
                du[subset] += res if res != "incomplete" else []
                res, n, _ = inf_res(
                    [
                        f"{directory}/{subset}_{model.replace('/','_')}_s={seed}_t={temp}{suffix}_nc.jsonl",
                        f"{directory}/{subset}_{model.replace('/', '_')}_s={seed}_t={temp}{suffix}_c.jsonl",
                    ],
                    field=field,
                    condition=condition,
                )
                if res == "incomplete":
                    total[subset] = -1
                    break
                dc[subset] += res if res != "incomplete" else []
        unconstrained.append(du)
        constrained.append(dc)
        totals.append(total)

    # print(
    #     "Average additional time taken for constrained synthesis compared to unconstrained synthesis"
    # )
    # for subset in SUBSETS:
    #     adds = []
    #     for model, unc, con, total in zip(models, unconstrained, constrained, totals):
    #         if con[subset] and unc[subset]:
    #             adds.append(median(con[subset]) * 100 / median(unc[subset]) - 100)
    #         else:
    #             adds.append(-1)
    #     print(f"{subset}: {sum(adds)/len(adds):.1f}%")

    headers = ["Model", "HumanEval", "MBPP"]
    rows = []
    for model, unc, con, total in zip(models, unconstrained, constrained, totals):
        row = [MODEL_NAME_MAP[model]]
        for subset in SUBSETS:
            row.extend(
                (
                    # "{:.1f}".format(median(unc[suffix])) if total[suffix] else -1,
                    "${:.1f}$&$_{{\\uparrow {:.1f}\\%}}$".format(
                        median(con[subset]),
                        median(con[subset]) * 100 / median(unc[subset]) - 100,
                    )
                    if total[subset] != -1
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
