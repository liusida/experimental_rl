#!/bin/sh
set -x

git pull


# 2021-4-20
if true
then
    exp_name="MaskEnv2"
    exp_description="Test RNN on Masked Hopper, see if RNN has advantages."
    for seed in 0 1 2 3 4
    do
        common_cmd="python run.py --exp_name=$exp_name --n_epochs=10 --rnn_sequence_length=16 --rnn_move_window_step=1 --rnn_layer_size=64 --vec_normalize --env_id=HopperMaskEnv-v0 --total_timesteps=1e7 --seed=$seed"
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --num_rnns=1
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --flatten
    done
fi

if false
then
    exp_name="ThirdPerson"
    exp_description="Test CNN RNN with only third person vision observation."
    for seed in 0 1 2 3 4
    do
        common_cmd="python run.py --exp_name=$exp_name --cuda --cnn --n_epochs=10 --rnn_sequence_length=16 --rnn_move_window_step=1 --rnn_layer_size=64 --vec_normalize --env_id=Walker2DOnlyVisionEnv-v0 --total_timesteps=1e7 --seed=$seed"
        sbatch -J $exp_name ~/bin/deepgreen.sh erl $common_cmd --num_rnns=1
        sbatch -J $exp_name ~/bin/deepgreen.sh erl $common_cmd --flatten
    done
fi

if false
then
    exp_name="Omiga"
    exp_description="sweep the hyperparameters II, for more n_epochs"
    for seed in 0 1 2 3 4
    do
        common_cmd="python run.py --exp_name=$exp_name --n_epochs=10 --rnn_sequence_length=16 --rnn_move_window_step=1 --rnn_layer_size=64 --vec_normalize --env_id=HopperBulletEnv-v0 --total_timesteps=1e7 --seed=$seed"
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --flatten
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --flatten --num_rnns=1
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --flatten --num_mlps=1
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --num_rnns=1
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --num_mlps=1
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --num_rnns=1 --num_mlps=1
    done
fi

if false
then
    exp_name="Zeta"
    exp_description="sweep the hyperparameters II, for more n_epochs"
    for seed in 0 1 2 3 4
    do
        for n_epochs in 20 40 80
        do
            for moving_step in 1
            do
                for seqlen in 16
                do
                    for rnn_layer_size in 64
                    do
                        common_cmd="python run.py --exp_name=$exp_name --n_epochs=$n_epochs --rnn_sequence_length=$seqlen --rnn_move_window_step=$moving_step --rnn_layer_size=$rnn_layer_size --vec_normalize --env_id=HopperBulletEnv-v0 --total_timesteps=1e7 --num_rnns=1 --seed=$seed"
                        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd
                    done
                done
            done
        done
    done
fi


# 2021-04-19
if false
then
    exp_name="VisionCNN"
    exp_description="use walker2d with vision environment, use CNN extractor before RNN"
    for seed in 0 1 2 3 4
    do
        common_cmd="python run.py --exp_name=$exp_name --n_epochs=10 --rnn_sequence_length=16 --rnn_move_window_step=1 --rnn_layer_size=64 --vec_normalize --env_id=Walker2DwithVisionEnv-v0 --total_timesteps=1e7 --seed=$seed"
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --num_rnns=1
        # sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --flatten
    done
fi

if false
then
    exp_name="Baseline"
    exp_description="flatten and mlp"
    for seed in 0 1 2 3 4
    do
        common_cmd="python run.py --exp_name=$exp_name --vec_normalize --env_id=HopperBulletEnv-v0 --total_timesteps=1e7 --seed=$seed"
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --num_mlp=1
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --flatten
    done
fi

if false
then
    exp_name="Zeta"
    exp_description="sweep the hyperparameters"
    for seed in 0 1 2 3 4
    do
        for n_epochs in 1 4 10
        do
            for moving_step in 1 8
            do
                for seqlen in 16 64 256
                do
                    for rnn_layer_size in 16 64
                    do
                        common_cmd="python run.py --exp_name=$exp_name --n_epochs=$n_epochs --rnn_sequence_length=$seqlen --rnn_move_window_step=$moving_step --rnn_layer_size=$rnn_layer_size --vec_normalize --env_id=HopperBulletEnv-v0 --total_timesteps=1e7 --num_rnns=1 --seed=$seed"
                        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd
                    done
                done
            done
        done
    done
fi

# 2021-04-18
if false
then
    exp_name="SeqLen256"
    exp_description="stack sequences to speedup training rnn"
    for seed in 0 1 2
    do
        common_cmd="python run.py --exp_name=$exp_name --n_epochs=10 --rnn_sequence_length=256 --rnn_move_window_step=1 --vec_normalize --env_id=HopperBulletEnv-v0 --total_timesteps=1e7 --num_rnns=1 --seed=$seed"
        # sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --sde 
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd
    done
fi

if false
then
    exp_name="StackSigma"
    exp_description="stack sequences to speedup training rnn"
    for seed in 0 1 2
    do
        common_cmd="python run.py --exp_name=$exp_name --n_epochs=10 --rnn_sequence_length=16 --rnn_move_window_step=1 --vec_normalize --env_id=HopperBulletEnv-v0 --total_timesteps=1e7 --num_rnns=1 --seed=$seed"
        # sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --sde 
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd
    done
fi

if false
then
    exp_name="SmallSigma"
    exp_description="change RNN size of hidden layer to 16, smaller RNN should be easier to train"
    for seed in 0 1 2
    do
        common_cmd="python run.py --exp_name=$exp_name --n_epochs=4 --rnn_sequence_length=16 --rnn_move_window_step=1 --vec_normalize --env_id=HopperBulletEnv-v0 --total_timesteps=1e7 --num_rnns=1 --seed=$seed"
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd --sde 
        sbatch -J $exp_name ~/bin/bluemoon.sh erl $common_cmd
    done
fi

if false
then
    exp_name="Delta"
    for seed in 0 1 2
    do
        common_args="--exp_name=$exp_name --sde --n_epochs=1 --vec_normalize --env_id=HopperBulletEnv-v0 --total_timesteps=3e6 --num_rnns=1 --seed=$seed"
        for seqlen in 8 16 32
        do
            sbatch -J $exp_name ~/bin/bluemoon.sh erl python run.py $common_args --rnn_sequence_length=$seqlen --rnn_move_window_step=1
            sbatch -J $exp_name ~/bin/bluemoon.sh erl python run.py $common_args --rnn_sequence_length=$seqlen --rnn_move_window_step=4
            sbatch -J $exp_name ~/bin/bluemoon.sh erl python run.py $common_args --rnn_sequence_length=$seqlen --rnn_move_window_step=$seqlen
        done

        common_args="--exp_name=$exp_name --sde --n_epochs=10 --vec_normalize --env_id=HopperBulletEnv-v0 --total_timesteps=3e6 --num_rnns=1 --seed=$seed"
        for seqlen in 8 16 32
        do
            sbatch -J $exp_name ~/bin/bluemoon.sh erl python run.py $common_args --rnn_sequence_length=$seqlen --rnn_move_window_step=1
            sbatch -J $exp_name ~/bin/bluemoon.sh erl python run.py $common_args --rnn_sequence_length=$seqlen --rnn_move_window_step=4
            sbatch -J $exp_name ~/bin/bluemoon.sh erl python run.py $common_args --rnn_sequence_length=$seqlen --rnn_move_window_step=$seqlen
        done

        common_args="--exp_name=$exp_name --n_epochs=10 --vec_normalize --env_id=HopperBulletEnv-v0 --total_timesteps=3e6 --num_rnns=1 --seed=$seed"
        for seqlen in 8 16 32
        do
            sbatch -J $exp_name ~/bin/bluemoon.sh erl python run.py $common_args --rnn_sequence_length=$seqlen --rnn_move_window_step=1
            sbatch -J $exp_name ~/bin/bluemoon.sh erl python run.py $common_args --rnn_sequence_length=$seqlen --rnn_move_window_step=4
            sbatch -J $exp_name ~/bin/bluemoon.sh erl python run.py $common_args --rnn_sequence_length=$seqlen --rnn_move_window_step=$seqlen
        done
    done
fi


# 2021-04-15
if false
then
    exp_name="Gamma1"
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        common_args="--cuda --exp_name=$exp_name --vec_normalize --env_id=HopperBulletEnv-v0 --num_envs=16 --total_timesteps=3e6 --seed=$seed"
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --implementation_check
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --flatten
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --num_rnns=1
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --num_rnns=2
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --num_mlps=1
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --num_mlps=2
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --flatten --num_rnns=1
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --flatten --num_rnns=2
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --flatten --num_mlps=1
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --flatten --num_mlps=2
        sbatch -J $exp_name deepgreen.sh python run.py $common_args --flatten --num_rnns=1 --num_mlps=1
    done
fi

if false
then
    exp_name="OnlyRNN"
    for seed in 0 1 2
    do
        sbatch -J $exp_name deepgreen.sh python run.py --cuda --exp_name=$exp_name --exp_group=rnns --env_id=HopperBulletEnv-v0 --extractor=MultiLSTMExtractor:m=1 --num_envs=16 --total_timesteps=3e6 --seed=$seed
    done
fi

if false
then
    exp_name="ImplementingRNN"
    for seed in 0 1 2
    do
        sbatch -J $exp_name deepgreen.sh python run.py --cuda --exp_name=$exp_name --exp_group=rnns --env_id=HopperBulletEnv-v0 --extractor=MultiLSTMExtractor:m=1 --num_envs=16 --total_timesteps=3e6 --seed=$seed
    done
fi


# 2021-04-14
if false
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
