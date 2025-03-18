import json
import pathlib
import sys

multiple_e_instance_file = (
    pathlib.Path(__file__).parent
    / "openai_openai_humaneval_ts_gpt-4o-2024-05-13_filtered.jsonl"
)
instance_num = int(sys.argv[1])

with open(multiple_e_instance_file, "r") as f:
    for i, line in enumerate(f):
        if i == int(instance_num):
            print(json.loads(line)["translation"])
            break
