import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset

from experiments.main.util import (
    extract_code,
    cutoff,
    go_compiles_passes,
)


def process_file(file_path, tests_by_inst):
    try:
        if str(file_path).endswith(".bak"):
            return
        backup_path = file_path + ".bak"
        shutil.copy(file_path, backup_path)

        with open(file_path, "r") as f:
            lines = f.readlines()

        results = []
        for line in lines:
            data = json.loads(line)
            if not data["compiled"]:
                extracted = cutoff(extract_code(data["code"], "Go", 0))
                tests = tests_by_inst[data["instance_id"]]
                test_result = go_compiles_passes(extracted, tests, timeout=300)
                data["syntax_ok"] = test_result.syntax_ok
                data["compiled"] = test_result.compiled
                data["compiler_output"] = (
                    test_result.error_message if not test_result.compiled else None
                )
                data["tests_passed"] = test_result.passed
                data["test_output"] = (
                    test_result.error_message if not test_result.passed else None
                )
            results.append(json.dumps(data))

        with open(file_path, "w") as f:
            f.write("\n".join(results))
    except Exception as e:
        print(e)


def main():
    files = sys.argv[1:]
    dataset_name = "THUDM/humaneval-x"
    dataset = load_dataset(dataset_name, "go")["test"]
    tests_by_inst = {x["task_id"]: x["test"] for x in dataset}

    with ThreadPoolExecutor() as executor:
        executor.map(process_file, files, [tests_by_inst] * len(files))


if __name__ == "__main__":
    main()
