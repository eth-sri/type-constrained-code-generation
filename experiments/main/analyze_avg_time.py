import argparse
import json

from experiments.main.util import invalid_mbpp_instances


def main(
    outputs_files,
    field,
    condition,
):
    # mbpp or main size
    target_size = (
        (
            159
            if any("humaneval" in s for s in outputs_files)
            else (390 - len(invalid_mbpp_instances))
        )
        if not any("repair" in s for s in outputs_files)
        else (292 if any("humaneval" in s for s in outputs_files) else 248)
    )
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
    if (
        len(instances) < target_size
        and not any("repair" in s for s in outputs_files)
        or any(
            len(outputs_by_instance_f) < target_size
            for filename, outputs_by_instance_f in outputs_by_instance.items()
        )
    ):
        res = "incomplete"
    else:
        i = []
        for instance_id in sorted(instances):
            res = 0
            for file_name in outputs_files:
                outputs_by_instance_f = outputs_by_instance.get(file_name, {})
                cond = (
                    outputs_by_instance_f.get(instance_id, {}).get(condition)
                    if condition != "none"
                    else False
                )
                if outputs_by_instance_f.get(instance_id, {}) and not cond:
                    res = max(outputs_by_instance_f[instance_id][field], res)
            i.append(res)
        res = i
    return res, len(instances), outputs_by_instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument(
        "-f",
        "--field",
        choices=["time_taken"],
        default="time_taken",
    )
    parser.add_argument(
        "-c",
        "--condition",
        choices=["none"],
        default="none",
    )
    args = parser.parse_args()
    res, n, outputs_by_instance = main(
        args.files,
        args.field,
        args.condition,
    )
    print(res)
