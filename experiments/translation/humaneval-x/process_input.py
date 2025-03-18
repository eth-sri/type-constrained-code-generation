import json
import gzip
import argparse
import datasets


def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    res_dataset = dict()
    task_ids = set()

    hf_dataset = datasets.load_dataset("nuprl/MultiPL-E", "main-ts")["test"]
    ts_dataset = dict()
    for d in hf_dataset:
        j = dict()
        task_id = d["name"][10:]
        task_id = int(task_id[: task_id.find("_")])
        prompt = d["prompt"].strip().split("\n")[-1]
        task_ids.add(task_id)
        j["task_id"] = task_id
        j["prompt"] = prompt
        if task_id == 10:
            j["prompt"] = "function is_palindrome(string: string): boolean {"
        j["tests"] = d["tests"].replace("declare var require: any;", "").strip()
        ts_dataset[task_id] = j
    res_dataset["ts"] = ts_dataset

    for lang in ["python", "cpp", "go", "java", "js"]:
        with gzip.open(f"humaneval_{lang}.jsonl.gz") as f:
            lines = f.readlines()
        lang_dataset = dict()
        for line in lines:
            j = dict()
            d = json.loads(line)
            task_id = d["task_id"]
            task_id = int(task_id[task_id.find("/") + 1 :])
            if task_id not in task_ids:
                continue
            j["task_id"] = task_id
            j["prompt"] = d["declaration"] + d["canonical_solution"]
            lang_dataset[task_id] = j
        res_dataset[lang] = lang_dataset

    with open("dataset.json", "w") as f:
        json.dump(res_dataset, f, indent=2)
