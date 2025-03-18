models=(
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
    "deepseek-ai/deepseek-coder-33b-instruct"
    "codellama/CodeLlama-34b-Instruct-hf"
    "Qwen/Qwen2.5-32B-Instruct"
)
seeds=(0 )
temp=1
suffix="_synth"
subset=mbpp
ds_name="repair_datasets/${subset}_repair_dataset.jsonl"
# empty
: > $ds_name
# append outputs for models and seeds
for seed in "${seeds[@]}"
do
  for model in "${models[@]}"
  do
    python3 create_repair_dataset.py "results/${subset}_${model//\//_}_s=${seed}_t=${temp}${suffix}_nc.jsonl" >> $ds_name
  done
done
