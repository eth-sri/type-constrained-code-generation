#!/usr/bin/env bash
# Description: Run the experiment for generating samples from the model constrained and unconstrained
set -e
# cd into the directory of this file
cd "$(dirname "${BASH_SOURCE[0]}")"
# DIR is either the parameter passed to this script or the result directory
DIR="${1:-results}"
python3 -m pip install tabulate
echo ""
echo "The following Table/Figure references refer to the revised paper, as attached to the comment thread of the discussion."
echo ""
echo "Table 2"
echo "Humaneval"
python3 figures_revision/fig_compiler_perf_syn_tran_repair.py --subset humaneval --directory "$DIR"
echo "MBPP"
python3 figures_revision/fig_compiler_perf_syn_tran_repair.py --subset mbpp --directory "$DIR"
echo ""
echo "Table 3"
echo "Humaneval"
python3 figures/fig_compiler_perf_fc.py --subset humaneval --directory "$DIR"
echo "MBPP"
python3 figures/fig_compiler_perf_fc.py --subset mbpp --directory "$DIR"
echo ""
echo "Table 4"
python3 figures/fig_compiler_time.py --directory "$DIR"
echo ""
echo "Figure 8"
bash figures_revision/fig_resample_hist.sh "$DIR"
