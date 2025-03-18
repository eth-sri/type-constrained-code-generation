import json
import pathlib
import sys

import fire
from datasets import load_dataset
import argparse

from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

import os

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Translate canonical solutions from the humaneval dataset"
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="ts",
        help="The target language to translate to",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data",
        help="The target directory to save the translations",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-2024-05-13",
        help="The model name to use for translation",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="openai/openai_humaneval",
        help="The dataset of which to translate the canonical solution",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name)

    target_lang = args.target_lang
    target_dir = args.target_dir
    if not target_dir:
        print("Please provide the target directory")
        sys.exit(1)
    model_name = args.model_name
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-2024-05-13")

    TARGET_LANG_MAP = {
        "ts": "TypeScript",
        "rs": "Rust",
    }
    if target_lang not in TARGET_LANG_MAP:
        print(f"Unsupported target language: {target_lang}")
        sys.exit(1)

    human_readable_target_lang = TARGET_LANG_MAP[target_lang]
    prompt_template = f"""
    You are an expert in {human_readable_target_lang} and Python programming.
    Translate the following Python code to {human_readable_target_lang}.
    You may assume that the input code is correct and that the translation should be semantically equivalent.
    Just answer with the translated code in a ```{human_readable_target_lang.lower()}...``` block.

    This is the program to translate:
    ```python
    {{code}}
    ```
    """

    def parse_code(ai_message: AIMessage) -> str:
        return ai_message.content

    chain = PromptTemplate.from_template(prompt_template) | llm | parse_code

    translations_resolved = {}
    target_file = f"{target_dir}/{dataset_name.replace('/', '_')}_{target_lang}_{model_name}.jsonl"
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    if os.path.exists(target_file):
        with open(target_file, "r") as f:
            for line in f:
                json_line = json.loads(line)
                translations_resolved[json_line["task_id"]] = json_line["translation"]

    with open(target_file, "a") as f:
        for instance in tqdm(dataset["test"]):
            task_id = instance["task_id"]
            if task_id in translations_resolved:
                continue
            code = instance["prompt"] + "\n" + instance["canonical_solution"]
            translation = chain.invoke(code)
            f.write(json.dumps({"task_id": task_id, "translation": translation}) + "\n")
            f.flush()


if __name__ == "__main__":
    fire.Fire(main)
