# A simple script to stop all experiments synchronously
pkill -f run_experiments_repair.py
pkill -f run_experiments_syn_tran.py
pkill -f run_experiments.sh
pkill -f inference_multiple.py
pkill -f inference_multiple_repair.py
