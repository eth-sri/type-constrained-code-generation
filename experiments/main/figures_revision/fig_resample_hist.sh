#!/usr/bin/env bash
# DIR is either the parameter passed to this script or the result directory
DIR="${1:-results}"
python3 figures_revision/fig_resample_hist.py "${DIR}"/humaneval_google_gemma-2-2b-it_s=*_t=1_synth_c.jsonl --style latex
