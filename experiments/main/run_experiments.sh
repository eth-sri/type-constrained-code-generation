#!/usr/bin/env bash
# Description: Run the experiment for generating samples from the model constrained and unconstrained
set -ex
# cd into the directory of this file
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 run_experiments_syn_tran.py
python3 run_experiments_repair.py
