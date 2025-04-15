params=("compiler_output -i" "tests_passed")
temp="1"
suffixs=("" "_translate" "_repair")
models=(
    # "microsoft/Phi-3.5-mini-instruct"
    # "codellama/CodeLlama-7b-Instruct-hf"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "google/gemma-2b-it"
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
    "deepseek-ai/deepseek-coder-33b-instruct"
    # "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    # "deepseek-ai/deepseek-coder-1.3b-instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "codellama/CodeLlama-70b-Instruct-hf"
    "codellama/CodeLlama-34b-Instruct-hf"
    # "codellama/CodeLlama-13b-Instruct-hf"
    # "google/codegemma-7b-it"
    # "bigcode/octocoder"
    "google/gemma-2-27b-it"
    "Qwen/Qwen2.5-32B-Instruct"
)
subset="humaneval"
for param in "${params[@]}"
do
for suffix in "${suffixs[@]}"
do
	echo "$param unconstrained $suffix"
	for model in "${models[@]}"
	do
		python3 analyze_inf_res.py -f ${param} "results/${subset}_${model//\//_}_s=0_t=${temp}${suffix}_nc.jsonl" --non_interactive
	done
	echo "$param constrained $suffix"
	for model in "${models[@]}"
	do
		python3 analyze_inf_res.py -f ${param} "results/${subset}_${model//\//_}_s=0_t=${temp}${suffix}_c.jsonl" --non_interactive
	done
	echo "$param union $suffix"
	for model in "${models[@]}"
	do
		python3 analyze_inf_res.py -f ${param} "results/${subset}_${model//\//_}_s=0_t=${temp}${suffix}_nc.jsonl" "results/${subset}_${model//\//_}_s=0_t=${temp}${suffix}_c.jsonl" --non_interactive
	done
done
done
