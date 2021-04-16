#!/bin/bash

# run any job on deep green:
# sbatch deepgreen.sh <command> ...
# e.g.
# sbatch deepgreen.sh python run.py --total_timesteps=1e8

#SBATCH --partition dggpu,dg-jup
#SBATCH --mem 24G
#SBATCH --gres gpu:1
#SBATCH -c 1
#SBATCH --output /gpfs2/scratch/sliu1/slurm.out/%j

cd ${SLURM_SUBMIT_DIR}

source activate erl

time $@
