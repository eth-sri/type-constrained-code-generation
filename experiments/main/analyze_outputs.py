import json
import time

import fire

from typesafe_llm.parser.parser_ts import parse_ts_program
from experiments.main.util import cutoff, tsx_compiles, passes_tests_js


def main(outputs_file, recompile=False):
    with open(outputs_file, "r") as f:
        outputs = [json.loads(line) for line in f]
    outputs_by_instance_constrained = {}
    for output in outputs:
        outputs_by_instance_constrained[
            (output["instance_id"], output["constrained"])
        ] = output
    nums = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]  # unconstrained: total, compiled, compiled in sublang, passed, time
    solved = [set(), set()]
    for (instance_id, constrained), output in outputs_by_instance_constrained.items():
        if "initial prompt" in output["crashed"]:
            continue
        compiled, err = (
            tsx_compiles(output["tsc_code"])
            if recompile
            else (
                output["compiled"],
                False,
            )
        )
        compiled_sub = output["compiled_in_sublang"] if not constrained else compiled
        start = time.time()
        tests_passed = (
            (passes_tests_js(compiled) if recompile else output["tests_passed"])
            if compiled
            else False
        )
        end = time.time()
        nums[constrained][0] += 1
        nums[constrained][1] += bool(compiled)
        nums[constrained][2] += bool(compiled_sub)
        nums[constrained][3] += bool(tests_passed) and bool(compiled_sub)
        nums[constrained][4] += end - start if recompile else output["time_taken"]
        if compiled_sub and compiled:
            solved[constrained].add(output["instance_id"])
        if (
            False
            and not compiled
            and constrained
            and outputs_by_instance_constrained[(instance_id, False)]["compiled"]
        ):
            new_sub_compiled = parse_ts_program(
                cutoff(outputs_by_instance_constrained[(instance_id, False)]["code"])
            )
            if new_sub_compiled:
                continue
            print(instance_id)
            print(cutoff(outputs_by_instance_constrained[(instance_id, False)]["code"]))
            print("-" * 80)
            print(cutoff(output["code"]))
            print(output["compiled_in_sublang"])
            print(output["tests_passed"])
            print(output["crashed"])
            print(output["time_taken"])
            print("-" * 80)
            input()
    print(nums)
    print([[x / total for x in num for total in [num[0]]] for num in nums])
    print(solved[0] & solved[1])
    print(solved[0] - solved[1])
    print(solved[1] - solved[0])


if __name__ == "__main__":
    fire.Fire(main)
