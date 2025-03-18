#!/usr/bin/env bash
# Description: Run the experiment for generating samples from the model constrained and unconstrained
set -e
# cd into the directory of this file
cd "$(dirname "${BASH_SOURCE[0]}")"
# DIR is either the parameter passed to this script or the current directory
DIR="${1:-results}"
pip install tabulate

echo "Table 1"
echo "Humaneval"
python3 figures/fig_compiler_perf_syn_tran.py --subset humaneval --directory "$DIR"
echo "MBPP"
python3 figures/fig_compiler_perf_syn_tran.py --subset mbpp --directory "$DIR"

echo "Table 2"
python3 figures/fig_compiler_perf_repair.py --directory "$DIR"

echo "Table 3"
echo "Humaneval"
python3 figures/fig_compiler_perf_fc.py --subset humaneval --directory "$DIR"
echo "MBPP"
python3 figures/fig_compiler_perf_fc.py --subset mbpp --directory "$DIR"

echo "Figure 9b"
python3 figures/fig_compiler_time.py --directory "$DIR"