Replication package for "Type-Aware Constraining for LLM Code Generation"
================================================================================

This is an implementation of a completion engine for TypeScript that parses type safe programs incrementally.
The completion engine can then be used to constrain the sampling from an LLM model to only type-safe programs.

More details on the properties of the completion engine and supported features can be found in the paper and supplementary material.

> Note: This version of the codebase matches the revised version of the paper, in which we added more features to the completion engine and fixed some minor issues in the original version.
> The results presented in the experimental section have thus improved compared to the original submission and the new results are included in this codebase in `results_revision.pdf`.

## Reproducing experiments

### Setup

Follow the installation instructions to install conda and all dependencies for the experiments:

```bash
bash ./setup_conda.sh
# restart your shell
bash ./setup_env.sh 
# NOTE: some models are gated on huggingface, so you will be prompted to enter a huggingface API token
```

### Running Experiments
Here we describe how to reproduce the results reported in the paper.

#### Entire Pipeline

The following command executes the entire pipeline, including downloading the required LLMs and datasets, running synthesis and translation, creating the repair dataset, and running the repair experiments.
Note that the experiments take around 1h per model/setting/GPU, so overall may run over 1-2 days.

```bash
nohup bash experiments/main/run_experiments.sh &
```

This spawns the experiments in the background, so you can close the terminal and check the progress later.
You can check the progress by looking at the `nohup.out` file or by running the following command:

```bash
tail nohup.out -f
```

#### Individual Steps

Before running the experiments, you need to download the models and datasets.
To download the models and datasets, run the following command:

```bash
python3 experiments/main/download_models.py
```

Then you can run the experiments for synthesis and translation by model and task:

```bash
python3 experiments/main/run_experiments_syn_tran.py --models google/gemma-2-2b-it,google/gemma-2-9b-it --tasks synth --subsets mbpp,humaneval
python3 experiments/main/run_experiments_syn_tran.py --models Qwen/Qwen2.5-32B-Instruct --tasks translate --subsets mbpp
```

You can also run the experiments for the repair task by model and subset:

```bash
python3 experiments/main/run_experiments_repair.py --models google/gemma-2-2b-it,google/gemma-2-9b-it --subsets mbpp,humaneval
python3 experiments/main/run_experiments_repair.py --models Qwen/Qwen2.5-32B-Instruct --subsets mbpp
```

### Recreating Figures

After the experiments are done, you can run the following command to produce the figures for the paper.

```bash
bash experiments/main/figures.sh
```

We have included our raw data in the `experiments/main/results_submitted` directory, so you can also directly run the figures script without running the experiments like this:

```bash
bash experiments/main/figures.sh results_submitted
```

### Recreating the repair dataset
We provide the repair dataset in the `experiments/main/repair_dataset` directory. However, we also provide the tooling to reproduce the repair dataset.
First, run the generation tasks for all models on synthesis, as detailed above. Then recreate the repair dataset by running the following command:

```bash
cd experiments/main/
bash create_humaneval_repair_dataset.sh
bash create_mbpp_repair_dataset.sh
```

### Tests

To optionally run the test suite and check that the installation was successful, you can use the following command:

```bash
pytest test
```

## Project Structure


This project is organized as a Python package.
The relevant code for the implementation of Type-Aware Constraining for LLM Code Generation is located in the `typesafe_llm` directory.
The experiments are located in the `experiments` directory.
We further provide a test suite in the `tests` directory.
The usage of the latter two is described above.
In the following sections we describe the structure of the `typesafe_llm` package.

### (Constrained) Sampling

The sampling algorithm presented in Section 2.1 of the paper is located in `typesafe_llm/sampling.py`.
It uses the `transformers` library to infer predictions from a language model, sample from it and, if constraining is enabled, runs a parser in parallel to reject invalid programs (`sample_constrained`).

### Prefix Automaton Definition and Base Automata

The prefix automaton is defined in `typesafe_llm/automata/parser_base.py`.
The automaton is implicitely defined by defining the transition function and acceptance status in each state, subclassing from `IncrementalParserState`.
A state indicates that it is an accepting state by setting the field `accept` to True.
The transition function is invoked by the method `parse_char` and returns a list of new states that can be reached by parsing the given character.
The file further contains the definitions of concatenation, union, kleene plus and terminal automata.

### Identifiers, Literals and Types

The automaton for identifiers (`ExistingIdentifierParserState`) is the first automaton defined in `typesafe_llm/automata/parser_ts.py`.
The following automata parse literals (`LiteralParserState` and its subclasses), including more advanced literals such as regular expressions and template strings.

The automaton for types is defined seperately in `typesafe_llm/automata/parser_ts_types.py`.

### Expressions

The expression automaton is defined in `typesafe_llm/automata/parser_ts.py` in the class `ExpressionParserState`.
It implements the extension logic and the pruning of invalid transitions due to operator precedence and type constraints.
The derivability algorithm is implemented for each state individually in the method `derivable`. It determines the directly derivable types and call the reachability algorithm with them.
The type reachability algorithm is implemented in `typesafe_llm/parser/types_ts.py` in the method `reachable`, leveraging `_reachable_bfs` - a straightforward breadth-first search translation of the presented reachability algorithm.

### Statements and the entire Program

The automaton for statements is defined in `typesafe_llm/automata/parser_ts.py` in the class `StatementParserState`.
It handles the constraining for valid return types.
The automaton for the entire program is defined in `typesafe_llm/automata/parser_ts.py` in the class `ProgramParserState`.