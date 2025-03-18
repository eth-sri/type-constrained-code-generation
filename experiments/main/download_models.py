from datasets import load_dataset
from transformers import AutoModelForCausalLM

dataset_name = "nuprl/MultiPL-E"
dataset = load_dataset(dataset_name, "humaneval-ts")["test"]
dataset = load_dataset(dataset_name, "mbpp-ts")["test"]

for model in [
    ("google/gemma-2-2b-it", 2),
    ("google/gemma-2-9b-it", 9),
    ("google/gemma-2-27b-it", 27),
    ("deepseek-ai/deepseek-coder-33b-instruct", 33),
    ("codellama/CodeLlama-34b-Instruct-hf", 34),
    ("Qwen/Qwen2.5-32B-Instruct", 32),
]:
    AutoModelForCausalLM.from_pretrained(model[0])
