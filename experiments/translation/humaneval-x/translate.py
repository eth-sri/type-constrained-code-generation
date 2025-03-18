import re
import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typesafe_llm.sampling import sample_constrained


def FULL_LANG(lang):
    if lang == "cpp":
        return "C++"
    elif lang == "go":
        return "Go"
    elif lang == "java":
        return "Java"
    elif lang == "python":
        return "Python"
    elif lang == "ts":
        return "TypeScript"
    else:
        assert False


def SYSTEM_PROMPT(src_lang):
    full_lang = FULL_LANG(src_lang)
    return f"You are a helpful and expert programmer in {full_lang} and TypeScript. You will be given an input program in {full_lang} and your task is to translate this program into TypeScript. You may assume that the input program is correct and that the translation should be semantically equivalent. You will NOT return anything except for the program."


def TRANSLATION_PROMPT(src_lang, src_prog):
    full_lang = FULL_LANG(src_lang)
    src_prog = src_prog.strip()
    return f"The following is the source program in {full_lang}:\n```{src_lang}\n{src_prog}\n```\n\nPlease translate the source program to TypeScript."


REGEX = "^```typescript\n([\s\S]*?)```$"


def extract_tgt_code(output):
    res = re.search(r"^```typescript\n([\s\S]*?)```$", output, re.M)
    if res is not None:
        span = res.span()
        output = output[span[0] : span[1]]
        output = output[14:]
        output = output[:-3]
        return output.strip() + "\n"
    else:
        output = output[output.find("```typescript\n") + 14 :]
        return output.strip() + "\n"


class HFChatModel:
    def __init__(self, args, **kwargs):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
        self.model.eval()

    def translate(self, src_prog, signature):
        src_lang = self.args.src_lang
        system_prompt = SYSTEM_PROMPT(src_lang)
        translation_prompt = TRANSLATION_PROMPT(src_lang, src_prog)
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT(src_lang)},
                {"role": "user", "content": TRANSLATION_PROMPT(src_lang, src_prog)},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            messages = [
                {"role": "user", "content": system_prompt + "\n\n" + translation_prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompt += "```typescript\n"
        prompt += signature

        # input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # outputs = self.model.generate(**input_ids, max_new_tokens=self.args.max_tokens, do_sample=False)
        # output = self.tokenizer.decode(outputs[0])
        # print(output)
        # return [extract_tgt_code(output)]

        with torch.no_grad():
            outputs = sample_constrained(
                prompt=prompt,
                tokenizer=self.tokenizer,
                model=self.model,
                constrain_from=None if self.args.no_constrain else "```typescript\n",
                constrain_until="\n```",
                max_tokens=self.args.max_tokens,
                try_top_k=self.args.try_top_k,
                device="cuda",
            )
        return [extract_tgt_code(outputs[0])]


def create_model(args):
    if args.model_name == "gemma-2b-it":
        args.model_name = "google/gemma-2b-it"
        kwargs = {
            "device_map": "cuda",
            "torch_dtype": torch.float16,
            "attn_implementation": "flash_attention_2",
        }
    elif args.model_name == "phi-3.5-it":
        args.model_name = "microsoft/Phi-3.5-mini-instruct"
        kwargs = {
            "device_map": "cuda",
            "torch_dtype": torch.float16,
            "attn_implementation": "flash_attention_2",
        }
    model = HFChatModel(args, **kwargs)
    return model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="phi-3.5-it")
    parser.add_argument("--no_constrain", default=False, action="store_true")
    parser.add_argument("--num_gen", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=250)
    parser.add_argument("--try_top_k", type=int, default=100)
    parser.add_argument("--task_id", type=str, default=None)

    parser.add_argument("--src_lang", type=str, default="python")
    parser.add_argument("--data_path", type=str, default="dataset.json")
    parser.add_argument("--output_dir", type=str, default="output_translation")

    args = parser.parse_args()

    if args.output_name is not None:
        args.output_dir = os.path.join(args.output_dir, args.output_name)

    return args


if __name__ == "__main__":
    args = get_args()
    model = create_model(args)

    with open(args.data_path) as f:
        dataset = json.load(f)
    src_dataset = dataset[args.src_lang]
    tgt_dataset = dataset["ts"]

    task_ids = [args.task_id] if args.task_id is not None else list(sorted(src_dataset))

    if args.output_name is not None:
        os.makedirs(os.path.join(args.output_dir), exist_ok=False)

    for task_id in tqdm(task_ids):
        src_prog = src_dataset[task_id]["prompt"]
        tgt_progs = model.translate(src_prog, tgt_dataset[task_id]["prompt"])

        if args.output_name is not None:
            os.makedirs(os.path.join(args.output_dir, str(task_id)))
            for i in range(args.num_gen):
                with open(
                    os.path.join(args.output_dir, str(task_id), f"{i}.ts"), "w"
                ) as f:
                    f.write(tgt_progs[i] + "\n\n" + tgt_dataset[task_id]["tests"])
