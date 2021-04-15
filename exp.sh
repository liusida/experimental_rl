#!/bin/sh
set -x

git pull

# 2021-04-15
if true
then
    exp_name="ImplementingRNN"
    for seed in 0 1 2
    do
        sbatch -J $exp_name deepgreen.sh python run.py --cuda --exp_name=$exp_name --exp_group=rnns --env_id=HopperBulletEnv-v0 --extractor=MultiLSTMExtractor:m=1 --num_envs=16 --total_timesteps=3e6 --seed=$seed
    done
fi


# 2021-04-14
if true
then
    exp_name="ImplementingRNN"
    for seed in 0 1 2
    do
        sbatch -J $exp_name bluemoon.sh python run.py --exp_name=$exp_name --exp_group=rnns --env_id=HopperBulletEnv-v0 --extractor=MultiLSTMExtractor:m=1 --num_envs=16 --total_timesteps=3e6 --seed=$seed
    done
fi

if false
then
    exp_name="CompareBaselineAndMlpsOnHopper"
    for seed in 0 1 2
    do
        sbatch -J $exp_name bluemoon.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=FlattenExtractor --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
        sbatch -J $exp_name bluemoon.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=MultiMlpExtractor:m=2 --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
    done
fi

# 2021-04-05
if false
then
    exp_name="SameOutputDim"
    for seed in 0 1 2
    do
        for m in 0 1 2 4 8
        do
            sbatch -J $exp_name deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=MultiMlpExtractor:m=$m --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
            sbatch -J $exp_name deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=MultiMlpExtractor:m=$m --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
        done
    done
fi

# 2021-04-03
if false
then
    exp_name="MultiMlps"
    for seed in 0 1 2
    do
        for m in 0 1 2 4 8
        do
            sbatch -J $exp_name deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=MultiMlpExtractor:m=$m --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
            sbatch -J $exp_name deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=MultiMlpExtractor:m=$m --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
        done
    done
fi

# 2021-04-02
if false
then
    exp_name="Debug"
    sbatch -J $exp_name deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=FlattenExtractor --num_envs=4 --total_timesteps=2e6 --seed=0 --exp_name=$exp_name
    sbatch -J $exp_name deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=FlattenExtractor --num_envs=4 --total_timesteps=2e6 --seed=0 --exp_name=$exp_name
fi

if false
then
    exp_name="NumOfEnvs"
    m=4
    for seed in 0 1 2
    do
        for env in 2 4 6 8 10
        do
            sbatch -J $exp_name deepgreen.sh python run.py --env_id=HopperBulletEnv-v0 --extractor=MultiMlpExtractor:m=$m --num_envs=$env --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
            sbatch -J $exp_name deepgreen.sh python run.py --env_id=Walker2DwithVisionEnv-v0 --extractor=MultiMlpExtractor:m=$m --num_envs=$env --total_timesteps=2e6 --seed=$seed --exp_name=$exp_name
        done
    done
fi

if false
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
