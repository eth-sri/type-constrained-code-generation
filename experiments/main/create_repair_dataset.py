import argparse
from pathlib import Path
import json


def main(input_files: list[Path]):
    for file in input_files:
        with open(file) as f:
            for line in f:
                line = line.strip()
                instance = json.loads(line)
                instance["repair_id"] = instance["instance_id"] + str(file)
                if instance["compiler_output"]:
                    print(json.dumps(instance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    main(args.files)
