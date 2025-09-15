#!/bin/bash

#BSUB -J policy_stats_500m           # job name
#BSUB -q serial                 # run in the serial queue
#BSUB -n 1                      # number of CPU cores
#BSUB -W 24:00                  # wall-time HH:MM
#BSUB -o logs/%J.out            # stdout → logs/<jobid>.out
#BSUB -e logs/%J.err            # stderr → log

mkdir -p logs

python run_champsim.py

