"""
Sample from the model constrained or unconstrained

Can also simulate a repair setting
"""

import json
import multiprocessing
import os
import time
import traceback

import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.main.util import (
    extract_code,
    cutoff,
    tsx_compiles,
    passes_tests_js,
)
from typesafe_llm.parser.parser_ts import parse_ts_program
from typesafe_llm.sampling import sample_constrained
from datasets import load_dataset

LANGUAGE_SUBSET_MAP = {
    ("typescript", "humaneval"): "humaneval-ts",
    ("rust", "humaneval"): "humaneval-rs",
    ("typescript", "mbpp"): "mbpp-ts",
    ("rust", "mbpp"): "mbpp-rs",
}
TRANSLATION_SUBSET_MAP = {
    "python": "python",
    "typescript": "ts",
    "rust": "rs",
    "c++": "cpp",
    "cpp": "cpp",
}

LANGUAGE_PARSER_MAP = {
    "typescript": parse_ts_program,
    "rust": None,
}
LANGUAGE_COMPILER_MAP = {
    "typescript": tsx_compiles,
}
LANGUAGE_TEST_MAP = {
    "typescript": passes_tests_js,
}
LANGUAGE_PREFIX_MAP = {
    "typescript": "",
    "rust": "",
}


def TRANSLATION_SYSTEM_PROMPT(
    human_readable_source_lang: str, human_readable_target_lang: str
):
    return f"""
You are a helpful and expert programmer in {human_readable_source_lang} and {human_readable_target_lang}. You will be given an input program in {human_readable_source_lang} and your task is to translate this program into {human_readable_target_lang}. You may assume that the input program is correct and that the translation should be semantically equivalent.
When answering, insert the solution code in a ```{human_readable_target_lang.lower()}...``` block.
"""


def TRANSLATION_PROMPT(
    human_readable_source_lang, src_prog, human_readable_target_lang
):
    return f"The following is the source program in {human_readable_source_lang}:\n```{human_readable_source_lang.lower()}\n{src_prog}\n```\n\nPlease translate the source program to {human_readable_target_lang}."


def SYNTHESIS_SYSTEM_PROMPT(human_readable_target_lang: str):
    return f"""
You are an expert in {human_readable_target_lang} programming. Solve the given problem by writing solution code in {human_readable_target_lang}.
When answering, insert the solution code in a ```{human_readable_target_lang.lower()}...``` block.
"""


def format_prompt_to_question(prompt: str):
    user_input = []
    split = prompt.splitlines()
    for i, line in enumerate(split):
        if line.startswith("//"):
            user_input.append(line[len("//") :].strip())
        else:
            break
    first_code_line = "\n".join(split[i:])
    return "\n".join(user_input), first_code_line


def main(
    model_name="microsoft/Phi-3.5-mini-instruct",
    device="cuda",
    language="TypeScript",
    subset="humaneval",
    split="test",
    temp=0,
    seed=0,
    max_tokens=1000,
    timeout=300,
    output_file="multiple_outputs.jsonl",
    trace=False,
    constrained=False,
    limit=1000,
    input_file=None,
    task_id=None,
    reraise=False,
):
    try_top_k = 10000000000000
    if isinstance(task_id, str):
        task_id = (task_id,)
    dataset_name = "nuprl/MultiPL-E"
    human_readable_target_lang = language
    language = language.lower()
    orig_dataset = load_dataset(dataset_name, LANGUAGE_SUBSET_MAP[language, subset])[
        split
    ]
    dataset_by_instance_id = {instance["name"]: instance for instance in orig_dataset}

    # load code to repair in repair setting
    assert os.path.exists(input_file), "Must provide an input file for repair"
    dataset = []
    if os.path.exists(input_file):
        with open(input_file, "r") as f:
            for line in f:
                output = json.loads(line)
                dataset.append(output)

    # load already inferred stuff
    already_done = set()
    if os.path.exists(output_file) and output_file not in ("/dev/stdout", "-"):
        with open(output_file, "r") as f:
            for i, line in enumerate(f):
                output = json.loads(line)
                already_done.add(output["instance_id"])

    tokenizer = None
    model = None
    system_messages = [
        {
            "role": "system",
            "content": (
                SYNTHESIS_SYSTEM_PROMPT(
                    human_readable_target_lang=human_readable_target_lang,
                )
            ),
        },
    ]
    subset_prefix = "HumanEval" if subset == "humaneval" else "mbpp"
    # run through all instances
    with multiprocessing.Pool(1) as pool:
        for repair_instance in tqdm(
            sorted(dataset, key=lambda x: x["instance_id"])[:limit]
        ):
            orig_instance = dataset_by_instance_id[repair_instance["instance_id"]]
            if repair_instance["repair_id"] in already_done and task_id is None:
                continue
            if task_id is not None and not any(
                f"{subset_prefix}_{tid}_" in repair_instance["repair_id"]
                for tid in task_id
            ):
                continue
            if tokenizer is None or model is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                kwargs = (
                    {
                        "device_map": "auto",
                        "torch_dtype": torch.bfloat16,
                        "attn_implementation": "flash_attention_2",
                    }
                    if device == "cuda"
                    else {"device_map": device}
                )

                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            user, first_code_line = format_prompt_to_question(orig_instance["prompt"])
            messages = system_messages + [
                {"role": "user", "content": user},
            ]
            if "octocoder" in model_name:
                chat_template = """\
    {%- for message in messages %}
        {%- if message['role'] == 'user' %}
            {{- 'Question: ' + message['content'].strip() + '\\n\\n' }}
        {%- elif message['role'] == 'system' %}
            {{- 'System: ' + message['content'].strip() + '\\n\\n' }}
        {%- elif message['role'] == 'assistant' %}
            {{- 'Answer: '  + message['content'] + '\\n\\n' }}
        {%- endif %}
    {%- endfor %}"""
                tokenizer.chat_template = chat_template
            try:
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                messages[1]["content"] = (
                    messages[0]["content"] + "\n\n" + messages[1]["content"]
                )
                messages.pop(0)
            formatted_last_iteration_res = "\n".join(
                f"{i+1:03d}: {s}"
                for i, s in enumerate(
                    cutoff(
                        extract_code(
                            repair_instance["code"], human_readable_target_lang, 0
                        )
                    ).split("\n")
                )
            )
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": f"```\n{formatted_last_iteration_res}\n```",
                    },
                    {
                        "role": "user",
                        "content": f"Compiling this code produced an error:\n{repair_instance['compiler_output']}\n\nWrite the program again, and make sure to fix the error this time.",
                    },
                ]
            )
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            suffix = f"```{human_readable_target_lang.lower()}\n"
            prompt += suffix + first_code_line
            start = time.time()
            with torch.no_grad():
                code, eos, crashed, resamples = sample_constrained(
                    device=device,
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    do_sample=temp != 0,
                    seed=seed,
                    constrain_from=(suffix if constrained else None),
                    constrain_until="```",
                    trace=trace,
                    model=model,
                    tokenizer=tokenizer,
                    timeout=timeout,
                    try_top_k=try_top_k,
                    reraise=reraise,
                )
            end = time.time()
            time_taken = end - start
            extracted = extract_code(code, human_readable_target_lang, 0)
            extracted = cutoff(extracted)
            tests: str = orig_instance["tests"]
            if tests.strip().startswith("}") and extracted.strip().endswith("}"):
                tests = tests[tests.find("}") + 1 :]
            compilable = extracted + "\n\n" + tests
            pool.apply_async(
                compile_test_and_dump,
                (
                    output_file,
                    {
                        "dataset": dataset_name,
                        "language": language,
                        "split": split,
                        "instance_id": repair_instance["repair_id"],
                        "orig_instance_id": orig_instance["name"],
                        "prompt": prompt,
                        "constrained": constrained,
                        "eos": eos,
                        "crashed": str(crashed),
                        "model_name": model_name,
                        "temp": temp,
                        "max_tokens": max_tokens,
                        "time_taken": time_taken,
                        "code": code,
                        "compilable": compilable,
                        "trace": trace,
                        "resamples": resamples,
                        "timeout": timeout,
                    },
                ),
            )
        pool.close()
        pool.join()


def compile_test_and_dump(output_file: str, specs: dict):
    try:
        compiled, compiler_output = LANGUAGE_COMPILER_MAP[specs["language"]](
            specs["compilable"], specs["timeout"]
        )
        if compiled is not None:
            tests_passed, test_output = LANGUAGE_TEST_MAP[specs["language"]](
                compiled, specs["timeout"]
            )
        else:
            tests_passed, test_output = None, None
        specs["compiled"] = compiled
        specs["compiler_output"] = compiler_output
        specs["tests_passed"] = tests_passed
        specs["test_output"] = test_output
        with open(output_file, "a") as f:
            print(
                json.dumps(
                    specs,
                ),
                flush=True,
                file=f,
            )
        if specs["trace"]:
            print("compiler_output:", compiler_output)
            print("tests_passed:", tests_passed)
    except Exception:
        print("WARNING CATASROPHIC FAILURE")
        print("RESULTS ARE NOT WRITTEN TO FILE")
        traceback.print_exc()
        print("", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
