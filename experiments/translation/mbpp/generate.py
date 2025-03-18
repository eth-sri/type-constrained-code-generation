import datasets
import json

ds = datasets.load_dataset("google-research-datasets/mbpp", "full")
dataset = {}
for split in ds:
    for instance in ds[split]:
        dataset[instance["task_id"]] = {"prompt": instance["code"]}
json.dump({"python": dataset}, open("dataset.json", "w"), indent=2)
