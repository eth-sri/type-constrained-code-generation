Type-Constrained Code Generation with Language Models
=====================================================
[![arXiv](https://img.shields.io/badge/arXiv-2504.09246-b31b1b.svg)](https://arxiv.org/abs/2504.09246)
[![QA & Tests](https://github.com/eth-sri/type-constrained-code-generation/actions/workflows/tests.yml/badge.svg)](https://github.com/eth-sri/type-constrained-code-generation/actions/workflows/tests.yml)


This is an implementation of a completion engine that parses type safe programs incrementally, guaranteeing that intermediate outputs can be completed to type-safe programs.
The completion enginge can be used to constrain the sampling from an LLM model to only type-safe programs.
The implementation currently only handles TypeScript.

More details on the properties of the completion engine and supported features can be found in the paper [Type-Constrained Code Generation with Language Models](https://arxiv.org/abs/2504.09246).

### Overview
When set-up correctly, the package can be used to sample type-safe TypeScript programs from a language model.
The following will incrementally generate the code for a TypeScript merge sort function, while ensuring that the generated code is type-safe:

```python
from typesafe_llm.sampling import sample_constrained

sample_constrained(
    prompt="function merge_sort(x:number[]):number[] {",
    max_tokens=100,
    device="cuda",
    model_name = "google/gemma-2-2b-it",
    temperature=0,
    do_sample=False,
    trace=True,
)
print("Generation completed")
```

The project contains two main parts:
- The sampling algorithm, which is used to sample type-safe TypeScript programs from a language model.
- The parser, which is used to parse TypeScript programs and check their completability to type-safe programs.

### Setup

To install the package, we recommend setting up a conda environment using NVIDIA GPUs.

```bash
git clone https://github.com/eth-sri/type-constrained-code-generation.git
cd type-constrained-code-generation  
conda create -n typesafe_llm python=3.11
conda activate typesafe_llm

# for LLM inference
# set up torch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
# install flash-attention
pip install flash-attn==2.7.3 --no-build-isolation

# install package
pip install -e .
```

If you only want to use the parser and do not want to sample from a language model, you can skip the installation of `torch` and `flash-attention`.

### Programmatic Usage

#### LLM Sampling

To sample type-safe TypeScript programs from a language model, you can use the `sample_constrained` function from the `typesafe_llm.sampling` module.

```python
from typesafe_llm.sampling import sample_constrained

sample = sample_constrained(
    prompt="function merge_sort(x:number[]):number[] {",
    max_tokens=100,
    device="cuda",
    model_name = "google/gemma-2-2b-it",
    temperature=0.1,
    do_sample=True,
)
print(sample)
```

If GPU is available, set device to "cuda", on MacBook Pro set device to "mps" (when pytorch nightly is installed).
Setting the device to "cpu" always works.
`trace` controls a debugging output for live debugging of the generation process.
Set to False for programmatic use.

#### Incremental TypeScript parsing

You can also independently use the parser to parse TypeScript programs and check their completability.

```python
from typesafe_llm.parser.parser_ts import parse_ts_program

states = parse_ts_program("let x:number = 1;x;")
print(list(states))
# only one accepting state

states = parse_ts_program('let x:number = "he')
print(list(states))
# some accepting states, could be saved by y".length

states = parse_ts_program('let x:boolean = 1 < "hey" +')
print(list(states))
# no states, can not turn "hey" + ... into a number, but need number for < operator

states = parse_ts_program('let x:number = 1;let y')
print(list(states))
# two partial states, one where the second variable has name "y" and one where it is not completed yet
```

### Tests

To run the tests, you can use the following command:

```bash
pytest test
```

## Reproducing experiments

In this section we provide an overview on how to reproduce the experiments presented in our [paper](https://arxiv.org/abs/2504.09246).

### Requirements

To reproduce our experiments locally, it is required to have higher-end GPUs, e.g. NVIDIA A100 80GB. The package includes setup scripts for all software requirements using miniconda. Required Hardware / Software:

- x86/64 architecture CPUs
- 80GB GPU VRAM
- CUDA 12.4 or newer

Further the Gemma 2 model family requires accepting an EULA. Please create a huggingface account and visit the model websites to accept the EULA.
- https://huggingface.co/google/gemma-2b-it
- https://huggingface.co/google/gemma-9b-it
- https://huggingface.co/google/gemma-27b-it

You will later be requested for a Hugginface Access Token. Log in with the account with which you accepted the EULA and visit [the Access Token page](https://huggingface.co/settings/tokens) to generate an access token: https://huggingface.co/settings/tokens

### Setup

Follow the installation instructions to install conda and all dependencies for the experiments:

```bash
bash ./setup_conda.sh
# Restart your shell
bash ./setup_env.sh 
# NOTE: Some models are guarded on huggingface, so you will need to visit their model page, accept the EULA and enter the huggingface Access Token to your account when prompted. See section "Requirements" for more details.
```

Before running the experiments, you need to download the models and datasets.
To download the models and datasets, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/main/download_models.py
```

### Warming up

To warm up, we start by reproducing the result for synthesis of the smallest model (Gemma 2 2B) and the MBPP dataset. To avoid using busy GPUs in a shared setting, use command `nvidia-smi` to check which GPUs are free. Then specify the IDs of GPUs you want to use by setting the `CUDA_VISIBLE_DEVICES` environment variable.  If you want to use GPU 0 and 1, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 experiments/main/run_experiments_syn_tran.py --models google/gemma-2-2b-it --tasks synth --subsets mbpp
```

This reproduces the results for Gemma-2B on the synthesis task on MBPP.
The experiment should finish within approximately 4 hours on a single GPU.
The results of the experiment (and all other results) will be stored in `experiments/main/results` in an appropriately named `jsonl` file, in this concrete example `experiments/main/results/mbpp_google_gemma-2-2b-it_s=0_t=1_synth_nc.jsonl` and `..._c.jsonl` for the unconstrained and type-constrained variants respectively.

> The experiment runs can be cancelled at any time, intermediate results are stored in the `results` folder. Upon restarting, the script will automatically pick up the last completed instance and continue from there. It may happen that running tasks daemonize and continue running (check `nvidia-smi`). Make sure to kill them manually before restarting.

Our experiment script automatically distributes jobs over indicated GPUs.
The script then repeatedly queries whether running jobs are completed and new GPUs are available. You will therefore see something like the following ouput:
```
+ CUDA_VISIBLE_DEVICES=0 python3 inference_multiple.py --max-tokens 1000 --timeout 300 --model_name google/gemma-2-2b-it --seed 0 --temp 1 --subset mbpp  --try_top_k 10000000000000000 --constrained False --output_file 'results/mbpp_google_gemma-2-2b-it_s=0_t=1_synth_nc.jsonl' 
+ CUDA_VISIBLE_DEVICES=1 python3 inference_multiple.py --max-tokens 1000 --timeout 300 --model_name google/gemma-2-2b-it --seed 0 --temp 1 --subset mbpp  --try_top_k 10000000000000000 --constrained True --output_file 'results/mbpp_google_gemma-2-2b-it_s=0_t=1_synth_c.jsonl' 
Total jobs: 2, Running jobs: 2, Remaining jobs: 0. Waiting for running jobs to finish...
```

To reproduce other tasks, the following commands reproduce the results for the translation task and the repair task on MBPP, and should take around 4 hours each:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 experiments/main/run_experiments_syn_tran.py --models google/gemma-2-2b-it --tasks translate --subsets mbpp
CUDA_VISIBLE_DEVICES=0,1 python3 experiments/main/run_experiments_repair.py --models google/gemma-2-2b-it --subsets mbpp
```


### Running more experiments

Then you can run more experiments for synthesis and translation by providing different models (`--models`), tasks (`--tasks`), and benchmarks (`--subsets`). Remember to use `CUDA_VISIBLE_DEVICES`.
Note that a single 80 GB GPU provides sufficient VRAM to host any model used in our experiments.

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/main/run_experiments_syn_tran.py --models google/gemma-2-2b-it,google/gemma-2-9b-it --tasks synth --subsets mbpp,humaneval
CUDA_VISIBLE_DEVICES=0 python3 experiments/main/run_experiments_syn_tran.py --models Qwen/Qwen2.5-32B-Instruct --tasks translate --subsets mbpp
```

You can similarly start the repair task:

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/main/run_experiments_repair.py --models google/gemma-2-2b-it,google/gemma-2-9b-it --subsets mbpp,humaneval
CUDA_VISIBLE_DEVICES=0 python3 experiments/main/run_experiments_repair.py --models Qwen/Qwen2.5-32B-Instruct --subsets mbpp
```

Below is the list of all options for these parameters. Running all these options will cover all our experiments but can take several days. For the sake of time, reviewers may check a subset that they are interested in.

```bash
FLAGS
    --models=MODELS
        Default: google/gemma-2-2b-it,google/gemma-2-9b-it,google/gemma-2-27b-it,deepseek-ai/deepseek-coder-33b-instruct,codellama/CodeLlama-34b-Instruct-hf,Qwen/Qwen2.5-32B-Instruct
    --tasks=TASKS (only for experiments/main/run_experiments_syn_tran.py)
        Default: synth,translate 
    --subsets=SUBSETS
        Default: humaneval,mbpp
```

You can also deep dive into obtaining the list of all available parameters:

```bash
python3 experiments/main/run_experiments_syn_tran.py --help
python3 experiments/main/run_experiments_repair.py --help
```

### Execution Time of Benchmarks

The runtime of our main experiments depends on the choice of datasets and tasks and the choice of models. Generally, larger datasets and larger models result in longer execution times.

Our benchmark features the MBPP and HumanEval datasets, adapted for three tasks: synthesis, translate, and repair.
Taking into account additional instances due to running on several seeds, the experiments can be ordered in increasing order of runtime as:  MBPP-repair, HumanEval-repair, MBPP-{synthesis,translate}, and HumanEval-{synthesis,translate}.

Our evaluation further features 6 models, in order of increasing parameter size, Gemma 2 2B, Gemma 2 9B, Gemma 2 27B, Qwen 2.5 32B, DeepSeek Coder 33B, and CodeLlama 34B. 

Thus, the quickest experiment is computing the performance of Gemma 2 2B synthesis on MBPP, taking approximately 4h on a single GPU. The longest experiment is computing performance of CodeLlama 34B synthesis on HumanEval.

### Recreating Figures

You can run the following command to produce the figures for the paper. You may run this script with partial results, in which case you will receive a print out of missing results and its positions in the table will be substituted with "-1".

```bash
bash experiments/main/figures.sh
```

The results map to the corresponding figures in the paper as follows:
- Table 2 and 3: all models, all tasks, all datasets, i.e., `[mbpp|humaneval]_*_s=[0|1|2|3]_t=1_[synth|translate|repair-all]_[c|nc].jsonl`. Vanilla and Syntax can be computed based on non-constrained (`nc`) variants.
- Table 4: all models, synthesis, all datasets, i.e., `[mbpp|humaneval]_*_s=[0|1|2|3]_t=1_synth_[c|nc].jsonl`
- Figure 8: Gemma 2 2B, synthesis, HumanEval, i.e., `humaneval_google_gemma-2-2b-it_s=[0|1|2|3]_t=1_synth_[c|nc].jsonl`

Since running the entire pipeline takes several days using 8 GPUs, we have included our raw data in the `experiments/main/results_paper` directory. You can directly run the figures script without running the experiments for the submitted results like this:

```bash
bash experiments/main/figures.sh results_paper
```

> Note: Table 4 is a runtime table. You should expect the runtime per instance to differ based on the CPU and GPU used, however the *runtime increase* should be consistent with our findings.

## Project Structure

The core part of our work is the implementation of a completion engine that incrementally parses type-safe TypeScript programs.
The completion engine can then be used to constrain the sampling from an LLM model to only generate type-safe programs.

This project is organized as a Python package.
The relevant code for the implementation of type-constrained decoding and sampling is located in the `typesafe_llm` directory.
The experiments are located in the `experiments` directory.
We further provide a test suite in the `tests` directory.
The usage of the latter two is described above.
In the following sections we describe the structure of the `typesafe_llm` package.

### (Constrained) Sampling (Algorithm 1)

The sampling algorithm presented in Section 2.1 of the paper is located in `typesafe_llm/sampling.py`.
It uses the `transformers` library to infer predictions from a language model, sample from it and, if constraining is enabled, runs a parser in parallel to reject invalid programs (`sample_constrained`).

### Prefix Automaton Definition and Base Automata (Section 3.2)

The prefix automaton is defined in `typesafe_llm/automata/parser_base.py`.
The automaton is implicitely defined by defining the transition function and acceptance status in each state, subclassing from `IncrementalParserState`.
A state indicates that it is an accepting state by setting the field `accept` to True.
The transition function is invoked by the method `parse_char` and returns a list of new states that can be reached by parsing the given character.
The file further contains the definitions of concatenation, union, kleene plus and terminal automata.

### Identifiers, Literals and Types (Section 3.3)

The automaton for identifiers (`ExistingIdentifierParserState`) is the first automaton defined in `typesafe_llm/automata/parser_ts.py`.
The following automata parse literals (`LiteralParserState` and its subclasses), including more advanced literals such as regular expressions and template strings.

The automaton for types is defined seperately in `typesafe_llm/automata/parser_ts_types.py`.

### Expressions (Section 3.4)

The expression automaton is defined in `typesafe_llm/automata/parser_ts.py` in the class `ExpressionParserState`.
It implements the extension logic and the pruning of invalid transitions due to operator precedence and type constraints.
The derivability algorithm is implemented for each state individually in the method `derivable`. It determines the directly derivable types and call the reachability algorithm with them.
The type reachability algorithm is implemented in `typesafe_llm/parser/types_ts.py` in the method `reachable`, leveraging `_reachable_bfs` - a straightforward breadth-first search translation of the presented reachability algorithm.

### Statements and the entire Program (Section 3.5)

The automaton for statements is defined in `typesafe_llm/automata/parser_ts.py` in the class `StatementParserState`.
It handles the constraining for valid return types.
The automaton for the entire program is defined in `typesafe_llm/automata/parser_ts.py` in the class `ProgramParserState`.