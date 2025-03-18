import argparse
import json

from experiments.main.util import cutoff, invalid_mbpp_instances, extract_code


def main(
    outputs_files,
    field,
    condition,
    show_list,
    intersection,
):
    # mbpp or main size
    target_size = (
        159
        if any("humaneval" in s for s in outputs_files)
        else (390 - len(invalid_mbpp_instances))
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
        except Exception as e:
            print("Error", e)
        for output in outputs:
            if output["instance_id"] in invalid_mbpp_instances:
                continue
            outputs_by_instance[outputs_file][output["instance_id"]] = output
            instances.add(output["instance_id"])
    if (
        len(instances) == 0
        or (
            len(instances) < target_size
            and not any("repair" in s for s in outputs_files)
        )
        or any(
            len(outputs_by_instance_f) < target_size
            for filename, outputs_by_instance_f in outputs_by_instance.items()
            if "repair" not in filename
        )
    ):
        res = "incomplete"
    else:
        # print(len(instances))
        i = 0
        for instance_id in sorted(instances):
            res = True if intersection else False
            for file_name in outputs_files:
                outputs_by_instance_f = outputs_by_instance.get(file_name, {})
                cond = (
                    outputs_by_instance_f.get(instance_id, {}).get(condition)
                    if condition != "none"
                    else False
                )
                if outputs_by_instance_f.get(instance_id, {}) and not cond:
                    if field == "syntax_error":
                        res = (
                            "SyntaxError"
                            in outputs_by_instance_f[instance_id]["compiler_output"]
                            or "syntax error"
                            in outputs_by_instance_f[instance_id]["compiler_output"]
                        )
                    else:
                        res = bool(outputs_by_instance_f[instance_id].get(field, True))
                    break
            res_pos = res
            if res_pos and show_list:
                print(instance_id)
            i += res_pos
        res = i / len(instances)
    return res, len(instances), outputs_by_instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("-n", "--non_interactive", action="store_true")
    parser.add_argument(
        "-f",
        "--field",
        choices=[
            "syntax_error",
            "compiler_output",
            "tests_passed",
            "compiled",
            "syntax_ok",
        ],
        default="compiler_output",
    )
    parser.add_argument(
        "-c",
        "--condition",
        choices=["compiler_output", "tests_passed", "none", "compiled", "syntax_ok"],
        default="compiler_output",
    )
    parser.add_argument("-l", action="store_true", help="list all negative cases")
    parser.add_argument(
        "-i",
        action="store_true",
        help="intersection instead of union  (use for compiler_output)",
    )
    args = parser.parse_args()
    res, n, outputs_by_instance = main(
        args.files,
        args.field,
        args.condition,
        args.l,
        args.i,
    )
    print(res)
    if args.non_interactive:
        exit()
    input()
    for file_name, outputs_by_instance_f in outputs_by_instance.items():
        for instance_id, output in outputs_by_instance_f.items():
            if not output[args.field]:
                code = cutoff(extract_code(output["code"], "Go", 0)).splitlines()
                code = "\n".join([f"{i+1:03d} {s}" for i, s in enumerate(code)])
                print(code)
                print(output["crashed"])
                print(output["compiler_output"])
                print(output["instance_id"])
                input()
