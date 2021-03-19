#!/bin/bash

#SBATCH --partition dggpu
#SBATCH --mem 24G
#SBATCH --gres gpu:1
#SBATCH -c 1
#SBATCH --output slurm.out/erl-%j

cd ${SLURM_SUBMIT_DIR}

source activate erl

time python $@
