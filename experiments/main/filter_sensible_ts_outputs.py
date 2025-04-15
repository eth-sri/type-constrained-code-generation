import json
import subprocess
import sys
from tempfile import NamedTemporaryFile

from tqdm import tqdm

target_file = sys.argv[1]

with open(target_file, "r") as f:
    lines = f.readlines()
lines = [json.loads(line) for line in lines]

stats = {
    "total": 0,
    "found_code": 0,
    "compiled": 0,
}
for line in tqdm(lines):
    stats["total"] += 1
    # extract code from code block
    code = line["translation"]
    code = code.split("```typescript")[1].strip()
    code = code.split("```")[0].strip()
    if not code:
        continue
    stats["found_code"] += 1
    # try to compile the code
    with NamedTemporaryFile("w", suffix=".ts") as f:
        f.write(code)
        try:
            subprocess.run(
                ["npx", "tsc", "--noEmit", f.name],
                text=True,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            continue
        stats["compiled"] += 1
        print(json.dumps({"task_id": line["task_id"], "translation": code}))
