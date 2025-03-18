import os
import json
import argparse
import numpy as np
from tabulate import tabulate

K = [1, 5, 10]


def compute(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_translation")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.output_name)

    return args


if __name__ == "__main__":
    args = get_args()

    pass_at_ks = [[] for _ in range(len(K))]
    compile_at_ks = [[] for _ in range(len(K))]
    for tgt_progs_dir in os.listdir(args.output_dir):
        tgt_progs_dir = os.path.join(args.output_dir, tgt_progs_dir)
        n, c_pass, c_compile = 0, 0, 0
        for res_name in os.listdir(tgt_progs_dir):
            if not res_name.endswith(".json"):
                continue

            n += 1
            with open(os.path.join(tgt_progs_dir, res_name)) as f:
                j = json.load(f)
            if j["compile"]:
                c_compile += 1
                if j["test"] == "Correct":
                    c_pass += 1
            else:
                print(j["tgt_prog"])
            for i, k in enumerate(K):
                pass_at_ks[i].append(compute(n, c_pass, k))
                compile_at_ks[i].append(compute(n, c_compile, k))
    for i, k in enumerate(K):
        pass_at_ks[i] = np.mean(pass_at_ks[i]) * 100
        compile_at_ks[i] = np.mean(compile_at_ks[i]) * 100

    header, row = [], []
    for i, k in enumerate(K):
        header.append(f"pass@{k}")
        row.append("{:.1f}".format(pass_at_ks[i]))
    print(tabulate([row], headers=header, stralign="right", tablefmt="orgtbl"))
    print()

    header, row = [], []
    for i, k in enumerate(K):
        header.append(f"compile@{k}")
        row.append("{:.1f}".format(compile_at_ks[i]))
    print(tabulate([row], headers=header, stralign="right", tablefmt="orgtbl"))
