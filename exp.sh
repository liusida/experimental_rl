#!/bin/sh
set -x

# 2021-04-02
if true
then
    exp_name="TestMultiMlpsAgain"
    for seed in 0 1 2
    do
        for m in 1 2 4 8 16
        do
            sbatch -J $exp_name deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=MultiMlpExtractor:m=$m --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
            sbatch -J $exp_name deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=MultiMlpExtractor:m=$m --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
        done
    done
fi

if false
then
    exp_name="FirstTryAgain"
    for seed in 0 1 2
    do
        sbatch -J $exp_name deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=MyFlattenExtractor --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
        sbatch -J $exp_name deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=OneMlpExtractor --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
        sbatch -J $exp_name deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=TwoMlpExtractor --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name

        sbatch -J $exp_name deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=MyFlattenExtractor --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
        sbatch -J $exp_name deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=OneMlpExtractor --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
        sbatch -J $exp_name deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=TwoMlpExtractor --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
    done
fi
