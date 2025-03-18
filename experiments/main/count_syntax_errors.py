import json
import subprocess
import sys
import tempfile

from experiments.main.util import cutoff

path_to_ts_parser = "../../ts_parser/target/release/ts_parser"
outputs_file = sys.argv[1]
try:
    with open(outputs_file, "r") as f:
        outputs = []
        for i, line in enumerate(f):
            # print(i)
            outputs.append(json.loads(line))
except Exception:
    outputs = []
outputs_by_instance_constrained = {}
for output in outputs:
    outputs_by_instance_constrained[(output["instance_id"], output["constrained"])] = (
        output
    )
total = len(outputs)
syntax = 0
other = 0
for (instance_id, constrained), output in outputs_by_instance_constrained.items():
    with tempfile.NamedTemporaryFile(suffix=".ts") as f:
        code = cutoff(output["compilable"]).encode()
        f.write(code)
        f.flush()
        res = subprocess.run([path_to_ts_parser, f.name], capture_output=True)
    if res.returncode != 0:
        syntax += 1
    elif not output["compiled"]:
        other += 1
print((other + syntax) / total, other / (syntax + other) if syntax + other != 0 else 1)
