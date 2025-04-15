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
import re

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
You are a helpful and expert programmer in {human_readable_source_lang} and {human_readable_target_lang}. You will be given an input program in {human_readable_source_lang} and your task is to translate this program into {human_readable_target_lang}. You may assume that the input program is correct and that the translation should be semantically equivalent. Do not translate word by word and be careful about difference of language features between {human_readable_source_lang} and {human_readable_target_lang}.
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
    repair=False,
    task_id=None,
    translate=False,
    translation_source_lang=None,
    reraise=False,
):
    try_top_k = 10000000000000
    if isinstance(task_id, int):
        task_id = str(task_id)
    if isinstance(task_id, str):
        # task ids always converted to a tuple of taskids
        task_id = (task_id,)
    dataset_name = "nuprl/MultiPL-E"
    human_readable_target_lang = language
    language = language.lower()
    dataset = load_dataset(dataset_name, LANGUAGE_SUBSET_MAP[language, subset])[split]
    assert not (repair and translate), "Must either choose translate or repair"

    # load code to repair in repair setting
    last_iteration = dict()
    if repair:
        assert os.path.exists(input_file), "Must provide an input file for repair"
        if os.path.exists(input_file):
            with open(input_file, "r") as f:
                for line in f:
                    output = json.loads(line)
                    last_iteration[output["instance_id"]] = output
    # load original code for translation in translate setting
    if translate:
        assert os.path.exists(
            input_file
        ), "Must provide an input file for translation (contains source language)"
        assert (
            translation_source_lang is not None
        ), "Must provide a source language for translation"
        with open(input_file) as f:
            raw_translation_dataset = json.load(f)
        translation_src_dataset = raw_translation_dataset[
            TRANSLATION_SUBSET_MAP[translation_source_lang.lower()]
        ]

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
                if not translate
                else TRANSLATION_SYSTEM_PROMPT(
                    human_readable_target_lang=human_readable_target_lang,
                    human_readable_source_lang=translation_source_lang,
                )
            )
            + (
                "\nDo not include test cases in the code."
                if "Qwen" in model_name
                else ""
            ),
        },
    ]
    subset_prefix = "HumanEval" if subset == "humaneval" else "mbpp"
    with multiprocessing.Pool(1) as pool:
        # run through all instances
        for instance in tqdm(sorted(dataset, key=lambda x: x["name"])[:limit]):
            if instance["name"] in already_done and task_id is None:
                continue
            if task_id is not None and not any(
                f"{subset_prefix}_{tid}_" in instance["name"] for tid in task_id
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
                model.eval()
            user, first_code_line = format_prompt_to_question(instance["prompt"])
            if translate:
                instance_num = re.findall(
                    rf"{subset_prefix}_(\d+)_*", instance["name"]
                )[0]
                user = TRANSLATION_PROMPT(
                    human_readable_source_lang=translation_source_lang,
                    src_prog=translation_src_dataset[instance_num]["prompt"],
                    human_readable_target_lang=human_readable_target_lang,
                )
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
            if repair:
                last_iteration_res = last_iteration[instance["name"]]
                if not last_iteration_res["compiler_output"]:
                    continue
                formatted_last_iteration_res = "\n".join(
                    f"{i+1:03d}: {s}"
                    for i, s in enumerate(
                        cutoff(last_iteration_res["compilable"]).split("\n")
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
                            "content": f"This output produced an error:\n{last_iteration_res['compiler_output']}\n\nWrite the program again, and make sure to fix the error this time.",
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
            tests: str = instance["tests"]
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
                        "instance_id": instance["name"],
                        "prompt": instance["prompt"],
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
