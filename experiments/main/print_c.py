import sys
import json

nc_file, problem_name = sys.argv[1:]
c_file = nc_file[:-8] + "c.jsonl"
# results/humaneval_google_codegemma-7b-it_s\=0_t\=1_nc.jsonl
with open(c_file) as f:
    for line in f:
        output = json.loads(line)
        if problem_name in output["instance_id"]:
            # print('\n'.join([f"{i+1:03d} {s}" for i, s in enumerate(output["code"].strip().split('\n'))]))
            print(output["code"].strip())
            print(output["crashed"])
            print(output["compiler_output"])
            print(output["instance_id"])
            print(output["tests_passed"])
