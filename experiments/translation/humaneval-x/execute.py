import os
import json
import argparse
import subprocess
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_name", type=str, required=True)

    parser.add_argument("--data_path", type=str, default="dataset.json")
    parser.add_argument("--output_dir", type=str, default="output_translation")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.output_name)

    return args


if __name__ == "__main__":
    args = get_args()

    with open(args.data_path) as f:
        dataset = json.load(f)
    tgt_dataset = dataset["ts"]

    for task_id in tqdm(list(sorted(os.listdir(args.output_dir)))):
        tests = tgt_dataset[task_id]["tests"]
        tgt_progs_dir = os.path.join(args.output_dir, task_id)
        for tgt_prog_name in os.listdir(tgt_progs_dir):
            if not tgt_prog_name.endswith(".ts"):
                continue

            tgt_prog_name = tgt_prog_name[:-3]
            exec_res = {
                "tgt_prog": os.path.join(tgt_progs_dir, tgt_prog_name + ".ts"),
                "compile": False,
                "test": False,
            }

            returncode = subprocess.call(
                ["tsc", os.path.join(tgt_progs_dir, tgt_prog_name + ".ts")]
            )
            if returncode != 0:
                with open(
                    os.path.join(tgt_progs_dir, tgt_prog_name + ".json"), "w"
                ) as f:
                    f.write(json.dumps(exec_res, indent=2))
                continue

            exec_res["compile"] = True
            try:
                returncode = subprocess.call(
                    ["node", os.path.join(tgt_progs_dir, tgt_prog_name + ".js")],
                    timeout=5,
                )
                if returncode != 0:
                    test_res = "Incorrect"
                else:
                    test_res = "Correct"
            except subprocess.TimeoutExpired:
                test_res = "Timeout"
            except Exception as e:
                test_res = e.__class__.__name__
            exec_res["test"] = test_res

            with open(os.path.join(tgt_progs_dir, tgt_prog_name + ".json"), "w") as f:
                json.dump(exec_res, f, indent=2)
