import argparse
import json
from collections import defaultdict

from experiments.main.util import cutoff, invalid_mbpp_instances, extract_code


def main(
    outputs_files,
):
    outputs_by_instance = {}
    instances = set()
    for outputs_file in outputs_files:
        outputs_by_instance[outputs_file] = {}
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
            outputs_by_instance[outputs_file][output["instance_id"]] = output
            instances.add(output["instance_id"])
    chart_max_resample_vs_syntax_error_total = defaultdict(int)
    chart_max_resample_vs_syntax_error_se = defaultdict(int)
    for file_name in outputs_files:
        for instance_id in sorted(instances):
            outputs_by_instance_f = outputs_by_instance.get(file_name, {})
            instance = outputs_by_instance_f.get(instance_id, {})
            if not instance:
                continue
            resamples = instance.get("resamples", None)
            if resamples is None:
                continue
            did_not_terminate = "SyntaxError" in instance["compiler_output"]
            cutoffed_code = cutoff(extract_code(instance["code"], "TypeScript", 0))
            len_of_cutoffed = len(cutoffed_code)
            resamples = [x for x in resamples if x[0] < len_of_cutoffed]
            if not resamples:
                max_resample = 0
            else:
                max_resample = max(x[1] for x in resamples)
            if max_resample > 15:
                max_resample -= max_resample % 10
            chart_max_resample_vs_syntax_error_total[max_resample] += 1
            chart_max_resample_vs_syntax_error_se[max_resample] += did_not_terminate
    for max_resample, total in sorted(chart_max_resample_vs_syntax_error_total.items()):
        print(
            f"({max_resample}, {chart_max_resample_vs_syntax_error_se[max_resample]*100/total})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    main(
        args.files,
    )
