import os
import pathlib
import subprocess
from pathlib import Path
from subprocess import TimeoutExpired
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Tuple

from attr import dataclass


def extract_code(output: str, humanreadable_target_language: str, nth: int):
    prefix = f"```{humanreadable_target_language.lower()}\n"
    pos = 0
    for _ in range(nth + 1):
        pos = output.find(prefix, pos) + len(prefix)
    code = output[pos:]
    code = code[: code.find("```")]
    return code.strip().strip("`") + "\n"


def cutoff(str_program: str):
    """
    Cutoff after the last outermost function is closed
    """
    curly_open = 0
    # default to just returning the entire string
    last_balanced_pos = len(str_program)
    for i, char in enumerate(str_program):
        if char == "{":
            curly_open += 1
        if char == "}":
            if curly_open <= 0:
                break
            curly_open -= 1
            if curly_open == 0 and str_program[i + 1] in ("\n", ";"):
                last_balanced_pos = i
    return str_program[: last_balanced_pos + 1]


def test_code_extract_2():
    code = """
```typescript
function words_string(s: string): string[] {                
    return s.replace(/,/g, ').split(' ).trim().split(' ').filter(word => word!== '');          
}               
``"""
    extracted = extract_code(code, "TypeScript", 0)
    cutoffed = cutoff(extracted)
    print(cutoffed)


def test_cutoff():
    code = """
function words_string(s: string): string[] {                
    return s.replace(/,/g, ').split(' ).trim().split(' ').filter(word => word!== '');          
};
"""
    extracted = extract_code(code, "TypeScript", 0)
    cutoffed = cutoff(extracted)
    print(cutoffed)


def test_cutoff2():
    code = """
function words_string(s: string): string[] {                
    return s.replace(/,/g, ').split(' ).trim().split(' ').filter(word => word!== '');          
}
function abc(s: string): string[] {                
    for (x in bla){
     y.map(x => {
     })
     }
}
"""
    extracted = extract_code(code, "TypeScript", 0)
    cutoffed = cutoff(extracted)
    print(cutoffed)


def test_double_cutoff():
    code = """
```typescript
function prod_signs(arr: number[]): number | undefined {
  if (arr.length === 0) {
    return undefined;
  }
  let product = 1;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === 0) {
      continue;
    }
    product *= arr[i];
  }
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === 0) {
      continue;
    }
    sum += Math.abs(arr[i]) * product;
  }
  return sum;
}
``` 


**Explanation:**
* **Function Definition:**
```typescript
function prod_signs(arr: number[]): number | undefined {
  // Function code goes here
}
``` 
"""
    res = extract_code(code, "TypeScript", 0)
    print(res)


def test_cutoff3():
    code = """
function words_string(s: string): string[] {                
    return s.replace(/,/g, ').split(' ).trim().split(' ').filter(word => word!== '');          
"""
    extracted = extract_code(code, "TypeScript", 0)
    cutoffed = cutoff(extracted)
    print(cutoffed)


def test_cutoff4():
    code = """
function words_string(s: string): string[] {                
    return s.replace(/,/g, ').split(' ).trim().split(' ').filter(word => word!== '');          
}
// example use
console.log(`${words_string("blabla")}`);
"""
    extracted = extract_code(code, "TypeScript", 0)
    cutoffed = cutoff(extracted)
    print(cutoffed)


def test_code_extract():
    code = """
<bos><start_of_turn>user
You are an expert in TypeScript programming. Solve the given problem by writing solution code in TypeScript.
When answering, insert the solution code in a ```typescript...``` block.


Check if in given array of numbers, are any two numbers closer to each other than
given threshold.
>>> has_close_elements([1.0, 2.0, 3.0], 0.5)
false
>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
true<end_of_turn>
<start_of_turn>model
```typescript
function has_close_elements(numbers: number[], threshold: number): boolean {
  // Sort the array in ascending order
  numbers.sort();

  // Iterate over the array
  for (let i = 0; i < numbers.length - 1; i++) {
    // Skip the last element, as it is already considered close to the previous element
    if (i > 0 && Math.abs(numbers[i] - numbers[i - 1]) <= threshold) {
      return true;
    }
  }

  // If no elements are close to the threshold, return false
  return false;
}
```
"""
    extracted = extract_code(code, "TypeScript", 0)
    cutoffed = cutoff(extracted)
    print(cutoffed)


def tsx_compiles(ts_program, timeout=300) -> Tuple[str | None, str]:
    with NamedTemporaryFile(suffix=".ts") as f:
        f.write(ts_program.encode())
        f.flush()
        if has_syntax_error(f.name, timeout):
            return (None, "SyntaxError: Abort compilation")
        try:
            res = subprocess.run(
                [
                    "npx",
                    "tsc",
                    "--lib",
                    "es2024",
                    "--target",
                    "es2024",
                    "--strict",
                    f.name,
                    "--outFile",
                    "/dev/stderr",
                ],
                capture_output=True,
                timeout=timeout,
            )
            if res.returncode != 0:
                return res.stderr.decode(), res.stdout.decode()
            return res.stderr.decode(), res.stdout.decode()
        except TimeoutExpired:
            return None, "Timeout"


def rust_compiles(rust_program, timeout=300, path="a.out") -> Tuple[str | None, str]:
    with NamedTemporaryFile(suffix=".rs") as f:
        f.write(rust_program.encode())
        f.flush()
        if os.path.exists(path):
            os.unlink(path)
        try:
            res = subprocess.run(
                ["rustc", f.name, "-o", path], capture_output=True, timeout=timeout
            )
            if res.returncode != 0:
                return None, res.stderr.decode()
            return path, res.stderr.decode()
        except TimeoutExpired:
            return None, "Timeout"


def setup_go_env(go_program: str, go_tests: str, dir: pathlib.Path, timeout):
    """
    Write program into file (adding relevant imports) and setup go package stuff
    """
    main_prefix = """
package main
"""
    main_suffix = (
        """
func main(){
}
"""
        if "func main(" not in go_program
        else ""
    )
    main_file = dir / "main.go"
    with open(main_file, "w") as f:
        f.write(main_prefix + go_program + main_suffix)
    test_file = dir / "main_test.go"
    tests_prefix = """
package main
import (
	"testing"
	"github.com/stretchr/testify/assert"
)
"""
    with open(test_file, "w") as f:
        f.write(tests_prefix + go_tests)
    subprocess.run(
        ["go", "mod", "init", "sample"], timeout=timeout, check=True, cwd=dir
    )
    subprocess.run(["go", "mod", "tidy"], timeout=timeout, check=True, cwd=dir)
    return [main_file, test_file]


@dataclass
class TestResult:
    setup_ok: bool
    syntax_ok: bool
    compiled: bool
    passed: bool
    error_message: str


def go_compiles_passes(program, tests, timeout=300) -> TestResult:
    with TemporaryDirectory() as tmpdir:
        try:
            files = setup_go_env(program, tests, pathlib.Path(tmpdir), timeout)
        except TimeoutExpired:
            return TestResult(False, False, False, False, "Setup Timeout")
        try:
            res = subprocess.run(
                [
                    "goimports",
                    "-w",
                    *files,
                ],
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            if res.returncode != 0:
                return TestResult(
                    True,
                    False,
                    False,
                    False,
                    res.stderr.decode() + "\n\n" + res.stdout.decode(),
                )
        except TimeoutExpired:
            return TestResult(True, False, False, False, "Format/Import Timeout")
        try:
            res = subprocess.run(
                [
                    "go",
                    "build",
                ],
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            if res.returncode != 0:
                return TestResult(True, True, False, False, res.stderr.decode())
        except TimeoutExpired:
            return TestResult(True, True, False, False, "Compilation Timeout")
        try:
            res = subprocess.run(
                ["go", "test", f"-timeout={timeout}s"],
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            if res.returncode != 0:
                return TestResult(
                    True,
                    True,
                    True,
                    False,
                    res.stdout.decode() + "\n\n" + res.stderr.decode(),
                )
        except TimeoutExpired:
            return TestResult(True, True, True, False, "Test Timeout")
        return TestResult(True, True, True, True, "")


def go_passes_tests(ts_program, timeout=300) -> Tuple[str | None, str]:
    with TemporaryDirectory() as tmpdir:
        file = "main_test.go"
        tmpfile = pathlib.Path(tmpdir) / file
        with open(tmpfile, "w") as f:
            f.write(ts_program.encode())
        if has_syntax_error(f.name, timeout):
            return (None, "SyntaxError: Abort compilation")
        try:
            res = subprocess.run(
                [
                    "go",
                    "build",
                    f"-timeout={timeout}s",
                    "main.go",
                ],
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            res = subprocess.run(
                [
                    "go",
                    "build",
                    f"-timeout={timeout}s",
                    file,
                ],
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            if res.returncode != 0:
                return res.stderr.decode(), res.stdout.decode()
            return res.stderr.decode(), res.stdout.decode()
        except TimeoutExpired:
            return None, "Timeout"


def passes_tests_js(js_program, timeout=300) -> Tuple[bool, str]:
    try:
        res = subprocess.run(
            ["node", "-e", js_program],
            check=False,
            capture_output=True,
            timeout=timeout,
        )
        return res.returncode == 0, (
            {"stdout": res.stdout.decode(), "stderr": res.stderr.decode()}
        )
    except TimeoutExpired:
        return False, "Timeout"


path_to_ts_parser = "../../ts_parser/target/release/ts_parser"


def has_syntax_error(ts_program_location, timeout=300) -> bool:
    res = subprocess.run([path_to_ts_parser, ts_program_location], capture_output=True)
    return res.returncode != 0


with open(Path(__file__).parent / "invalid_mbpp") as f:
    invalid_mbpp_instances = {line.strip() for line in f.readlines()}
