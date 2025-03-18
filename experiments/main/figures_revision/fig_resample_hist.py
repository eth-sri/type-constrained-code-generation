import argparse
import functools
import json
from math import ceil, log2
from statistics import median
from typing import Literal

import numpy as np
from transformers import AutoTokenizer

from experiments.main.util import cutoff, invalid_mbpp_instances, extract_code


@functools.lru_cache(maxsize=None)
def load_output_file(outputs_file):
    res_dict = {}
    try:
        with open(outputs_file, "r") as f:
            outputs = []
            for i, line in enumerate(f):
                # print(i)
                outputs.append(json.loads(line))
    except Exception:
        outputs = []
    for output in outputs:
        if output["instance_id"] in invalid_mbpp_instances:
            continue
        res_dict[output["instance_id"]] = output
    return res_dict


def main(
    outputs_files,
    mode: Literal["resample", "correction"] = "resample",
    style: Literal["ascii", "latex", "plain", "matplotlib"] = "latex",
):
    print(outputs_files)
    outputs_by_instance = {}
    instances = set()
    for outputs_file in outputs_files:
        outputs = load_output_file(outputs_file)
        outputs_by_instance[outputs_file] = outputs
        instances.update(outputs.keys())
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    counted_instances = 0
    instances_needing_resample = 0
    amounts_of_resamples = []
    for file_name in outputs_files:
        if file_name.endswith("_nc.jsonl"):
            continue
        for instance_id in sorted(instances):
            outputs_by_instance_f = outputs_by_instance.get(file_name, {})
            instance = outputs_by_instance_f.get(instance_id, {})
            if not instance:
                continue
            resamples = instance.get("resamples", None)
            cutoffed_code = cutoff(extract_code(instance["code"], "TypeScript", 0))
            pos_of_cutoff = instance["code"].find(cutoffed_code)
            len_of_cutoffed = pos_of_cutoff + len(cutoffed_code)
            num_tokens_in_cutoffed = len(
                tokenizer.encode(cutoffed_code, add_special_tokens=False)
            )
            if resamples is None:
                amounts_of_resamples.extend([0] * num_tokens_in_cutoffed)
                continue
            counted_instances += 1
            # check if unconstrained version resolved correctly -> doesnt need resample
            unconstrained_file_version = file_name.replace("_c.jsonl", "_nc.jsonl")
            if (
                not load_output_file(unconstrained_file_version)
                .get(instance_id, {})
                .get("compiler_output", True)
            ):
                amounts_of_resamples.extend([0] * num_tokens_in_cutoffed)
                continue
            # otherwise determine if resample occurred in relevant part of code
            resamples = [x for x in resamples if x[0] <= len_of_cutoffed]
            amounts_of_resamples.extend(
                [x[1] for x in resamples]
                + [0] * (num_tokens_in_cutoffed - len(resamples))
            )
            if resamples:
                instances_needing_resample += len(resamples) > 0

    print(
        f"Instances needing {mode}: {instances_needing_resample/counted_instances*100:.2f}% ({instances_needing_resample}/{counted_instances})"
    )
    print(
        f"Tokens needing {mode}: {len(tuple(x for x in amounts_of_resamples if x > 0))/len(amounts_of_resamples)*100:.2f}% ({len(tuple(x for x in amounts_of_resamples if x > 0))}/{len(amounts_of_resamples)})"
    )
    print(
        f"Average amount of {mode}s: {sum(amounts_of_resamples)/len(amounts_of_resamples):.2f}"
    )
    print(f"Median amount of {mode}s: {median(amounts_of_resamples):.2f}")
    print(f"Max amount of {mode}s: {max(amounts_of_resamples)}")
    print(f"Histogram of {mode} amounts:")

    if style == "ascii":
        asciihist(
            amounts_of_resamples,
            bins=10,
            minmax="auto",
            str_tag="Resample",
            scale_output=30,
        )
    elif style == "latex":
        latex_hist(amounts_of_resamples)
    elif style == "plain":
        notsosimplehist(amounts_of_resamples)
    elif style == "matplotlib":
        matplotlib_hist(amounts_of_resamples)


def matplotlib_hist(amounts_of_resamples):
    import matplotlib.pyplot as plt

    # show as histogram with logarithmic scale

    plt.hist(amounts_of_resamples, bins=30)
    plt.yscale("log")
    plt.show()


def latex_hist(amounts_of_resamples):
    hist_template = r"""
\begin{figure}
    \centering
    \resizebox{\textwidth}{!}{
    \begin{tikzpicture}
        \begin{axis}[
            ybar,
            width=7cm,
            height=6cm,
            ylabel={Count},
            xlabel={Amount of Resamples},
            ymin=0,
            ymax=1300,
            bar width=0.1cm,
            x=0.1cm,
        ]
        \addplot coordinates {%s};
        \end{axis}
    \end{tikzpicture}
    }
    \caption{Histogram of Resample Amounts}
    \label{fig:resample_histogram}
\end{figure}
"""
    hist = {}
    bucket_size = 1
    for amount in amounts_of_resamples:
        hist[amount // bucket_size] = hist.get(amount // bucket_size, 0) + 1
    min_key = min(hist.keys())
    max_key = max(hist.keys())
    for i in range(min_key, max_key):
        hist[i] = hist.get(i, 0)
    # x_coords = ", ".join(str(k) for k in sorted(hist))
    y_coords = " ".join(
        str((k + 1, v if v > 1 else 1.1)) for k, v in sorted(hist.items()) if v > 0
    )
    print(hist_template % (y_coords))


def simplehist(amounts_of_resamples):
    hist = {}
    bucket_size = 1
    for amount in amounts_of_resamples:
        hist[amount // bucket_size] = hist.get(amount // bucket_size, 0) + 1
    for k, v in sorted(hist.items()):
        print(f"| {k} | {v} | `{'-' * v}` |")


def notsosimplehist(
    amounts_of_resamples,
    scale=(lambda x: log2(x)),
    ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    max_key=30,
):
    hist = {}
    bucket_size = 1
    for amount in amounts_of_resamples:
        hist[amount // bucket_size] = hist.get(amount // bucket_size, 0) + 1
    max_height = int(ceil(scale(max(hist.values()))))
    min_key = min(hist.keys())
    actual_max_key = max(hist.keys())
    for i in range(min_key, max_key):
        hist[i] = hist.get(i, 0.0001)
    for i in range(max_key, actual_max_key + 1):
        if i in hist:
            del hist[i]
    max_x_width = max(max(len(str(k)) for k in hist), max(len(str(t)) for t in ticks))
    lines = []
    # draw lines
    for i in range(max_height, 0, -1):
        line = ""
        if i in ticks:
            line += "2^{:<{}d}-".format(i, max_x_width)
        else:
            line += " " * max_x_width + "   "
        line += "| "
        plot = []
        for k, v in sorted(hist.items()):
            if int(scale(v)) >= i:
                plot.append("❚")
            elif int(ceil(scale(v))) >= i or (v == 1 and i == 1):
                plot.append(".")
            else:
                plot.append(" ")
        lines.append(line + ((" ") * max_x_width).join(plot))
    # draw x-axis
    lines.append("-" * (len(hist) * (max_x_width + 1)) + "------")
    # draw x-ticks
    lines.append(
        (" ") * max_x_width
        + "   "
        + " ".join("{:{}d}".format(k, max_x_width) for k in sorted(hist))
    )

    for line in lines:
        print(line)


def gnuplothist(amounts_of_resamples, mode):
    print(amounts_of_resamples)
    import gnuplotlib as gp
    import numpy as np

    gp.plot(
        (
            np.array(amounts_of_resamples),
            {
                "histogram": True,
                "binwidth": 1,
            },
        ),
        _with="boxes",
        unset=["grid"],
        terminal="dumb 180,20",
        set=["boxwidth 0.25", "style fill solid"],
        _xmin=0,
        _xmax=60,
        title=f"Histogram of {mode} amounts",
        xlabel="Amount",
        ylabel="Count",
    )


def asciihist(
    it,
    bins=10,
    minmax=None,
    str_tag="",
    scale_output=30,
    generate_only=False,
    print_function=print,
):
    """Create an ASCII histogram from an interable of numbers.
    Author: Boris Gorelik boris@gorelik.net. based on  http://econpy.googlecode.com/svn/trunk/pytrix/pytrix.py
    License: MIT
    """
    ret = []
    itarray = np.asanyarray(it)
    if minmax == "auto":
        minmax = np.percentile(it, [5, 95])
        if minmax[0] == minmax[1]:
            # for very ugly distributions
            minmax = None
    if minmax is not None:
        # discard values that are outside minmax range
        mn = minmax[0]
        mx = minmax[1]
        itarray = itarray[itarray >= mn]
        itarray = itarray[itarray <= mx]
    if itarray.size:
        total = len(itarray)
        counts, cutoffs = np.histogram(itarray, bins=bins)
        cutoffs = cutoffs[1:]
        if str_tag:
            str_tag = "%s " % str_tag
        else:
            str_tag = ""
        if scale_output is not None:
            scaled_counts = counts.astype(float) / counts.sum() * scale_output
        else:
            scaled_counts = counts

        if minmax is not None:
            ret.append("Trimmed to range (%s - %s)" % (str(minmax[0]), str(minmax[1])))
        for cutoff, original_count, scaled_count in zip(cutoffs, counts, scaled_counts):
            ret.append(
                "{:s}{:>8.2f} |{:<7,d} | {:s}".format(
                    str_tag, cutoff, original_count, "*" * int(scaled_count)
                )
            )
        ret.append("{:s}{:s} |{:s} | {:s}".format(str_tag, "-" * 8, "-" * 7, "-" * 7))
        ret.append("{:s}{:>8s} |{:<7,d}".format(str_tag, "N=", total))
    else:
        ret = []
    if not generate_only:
        for line in ret:
            print_function(line)
    ret = "\n".join(ret)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument(
        "--mode", choices=["resample", "correction"], default="resample"
    )
    parser.add_argument(
        "--style",
        choices=["ascii", "latex", "plain", "matplotlib"],
        default="matplotlib",
    )
    args = parser.parse_args()
    main(
        args.files,
        args.mode,
        args.style,
    )
