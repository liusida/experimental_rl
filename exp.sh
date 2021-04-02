#!/bin/sh
set -x

# 2021-04-02
if true
then
    for seed in 0 1 2
    do
        sbatch -J A deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=MyFlattenExtractor --total_timesteps=2e6 --seed=$seed 
        sbatch -J A deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=OneMlpExtractor --total_timesteps=2e6 --seed=$seed 
        sbatch -J A deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=TwoMlpExtractor --total_timesteps=2e6 --seed=$seed 

        sbatch -J A deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=MyFlattenExtractor --total_timesteps=2e6 --seed=$seed 
        sbatch -J A deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=OneMlpExtractor --total_timesteps=2e6 --seed=$seed 
        sbatch -J A deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=TwoMlpExtractor --total_timesteps=2e6 --seed=$seed 
    done
fi
