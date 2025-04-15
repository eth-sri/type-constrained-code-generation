import json
import re
import fire


def main(
    outputs_files,
):
    if isinstance(outputs_files, str):
        outputs_files = [outputs_files]
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
            raise e
            outputs = []
        for output in outputs:
            outputs_by_instance[outputs_file][output["instance_id"]] = output
            instances.add(output["instance_id"])
    i = 0
    for instance_id in instances:
        res_pos = False
        for file_name in outputs_files:
            outputs_by_instance_f = outputs_by_instance.get(file_name, {})
            output = outputs_by_instance_f[instance_id]
            res = re.findall(
                r"error TS2304: Cannot find name '(.+)'", output["compiler_output"]
            )
            if res:
                for r in res:
                    if r in output["code"]:
                        res_pos = True
                        break
            if res_pos:
                break
        i += res_pos
    print(i)


fire.Fire(main)
