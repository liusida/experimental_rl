#!/bin/bash

# run any job on bluemoon:
# sbatch bluemoon.sh <command> ...
# e.g.
# sbatch bluemoon.sh python run.py --total_timesteps=1e8

#SBATCH --partition bluemoon
#SBATCH --mem 2G
#SBATCH -c 1
#SBATCH --output /gpfs2/scratch/sliu1/slurm.out/%j
#SBATCH --cores-per-socket 7

cd ${SLURM_SUBMIT_DIR}

source activate erl

time $@
