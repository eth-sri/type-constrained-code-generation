import json
import sys

from experiments.main.util import cutoff
from typesafe_llm.parser.parser_ts import parse_ts_program

outputs_file = sys.argv[1]
with open(outputs_file, "r") as f:
    outputs = [json.loads(line) for line in f]
outputs_by_instance_constrained = {}
for output in outputs:
    outputs_by_instance_constrained[(output["instance_id"], output["constrained"])] = (
        output
    )
i = 0
for (instance_id, constrained), output in outputs_by_instance_constrained.items():
    if output["tests_passed"]:
        i += 1
print(i / len(outputs))
for (instance_id, constrained), output in outputs_by_instance_constrained.items():
    if output["tests_passed"] and not any(
        s.accept for s in parse_ts_program(cutoff(output["compilable"]))
    ):
        print(output["instance_id"])
