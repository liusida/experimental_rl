import torch

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import erl.envs  # need this to register the bullet envs
from erl.tools.wandb_logger import WandbCallback
from erl.tools.gym_helper import make_env
from erl.tools.adjust_camera_callback import AdjustCameraCallback

from erl.customized_agents.customized_ppo import CustomizedPPO
from erl.customized_agents.customized_callback import CustomizedEvalCallback
from erl.customized_agents.customized_policy import CustomizedPolicy
from erl.tools.debug_callback import DebugCallback
from erl.customized_agents.multi_extractor import MultiExtractor


class MultiModuleExp:
    """ 
    A whole experiment.
    It should contain: (1) environments, (2) policies, (3) training, (4) testing.
    The results should be able to compare with other experiments.

    The Multi-RNN experiment.
    """

    def __init__(self,
                 args,
                 env_id="HopperBulletEnv-v0",
                 features_extractor_kwargs={},
                 ) -> None:
        print("Starting MultiModuleExp")
        """ Init with parameters to control the training process """
        self.args = args
        self.env_id = env_id
        self.use_cuda = torch.cuda.is_available() and args.cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Make Environments
        print("Making train environments...")
        venv = DummyVecEnv([make_env(env_id=env_id, rank=i, seed=args.seed, render=args.render) for i in range(args.num_envs)])
        self.eval_env = DummyVecEnv([make_env(env_id=env_id, rank=99, seed=args.seed, render=False)])
        if args.vec_normalize:
            venv = VecNormalize(venv)
            self.eval_env = VecNormalize(self.eval_env, norm_reward=False)

        features_extractor_kwargs["num_envs"] = args.num_envs
        policy_kwargs = {
            "features_extractor_class": MultiExtractor,
            "features_extractor_kwargs": features_extractor_kwargs,
            # Note: net_arch must be specified, because sb3 won't set the default network architecture if we change the features_extractor.
            # pi: Actor (policy-function); vf: Critic (value-function)
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        }

        self.model = CustomizedPPO(
            CustomizedPolicy, venv, n_steps=args.rollout_n_steps, tensorboard_log="tb", policy_kwargs=policy_kwargs, device=self.device, verbose=1,
            rnn_move_window_step=args.rnn_move_window_step, rnn_sequence_length=args.rnn_sequence_length, use_sde=args.sde)

    def train(self) -> None:
        """ Start training """
        print(f"train using {self.model.device.type}")

        callback = [
            DebugCallback("Customized"),
            AdjustCameraCallback(),
            WandbCallback(self.args),
            CustomizedEvalCallback(
                self.eval_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=self.args.eval_freq,
                n_eval_episodes=3,
                verbose=0,
            )
        ]
        self.model.learn(self.args.total_timesteps, callback=callback)
