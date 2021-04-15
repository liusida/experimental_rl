import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import erl.envs  # need this to register the bullet envs
from erl.tools.wandb_logger import WandbCallback
from erl.tools.gym_helper import make_env
from erl.tools.adjust_camera_callback import AdjustCameraCallback

class BaselineExp:
    """ 
    default flatten version, no customization.
    """

    def __init__(self,
                 args,
                 env_id="HopperBulletEnv-v0",
                 ) -> None:
        """ Init with parameters to control the training process """
        self.args = args
        self.env_id = env_id
        self.use_cuda = torch.cuda.is_available() and args.cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Make Environments
        print("Making train environments...")
        venv = DummyVecEnv([make_env(env_id=env_id, rank=i, seed=args.seed, render=args.render) for i in range(args.num_envs)])
        self.eval_env = make_env(env_id=env_id, rank=99, seed=args.seed, render=False)()
        if args.vec_normalize:
            venv = VecNormalize(venv)
            self.eval_env = VecNormalize(self.eval_env, norm_reward=False)
        
        self.model = PPO("MlpPolicy", venv, tensorboard_log="tb", device=self.device, verbose=1)
        self.model.experiment = self  # pass the experiment handle into the model, and then into the TrainVAECallback
        

    def train(self) -> None:
        """ Start training """
        print(f"train using {self.model.device.type}")

        callback = [
            AdjustCameraCallback(),
            WandbCallback(self.args),
            EvalCallback(
                self.eval_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=self.args.eval_freq,
                n_eval_episodes=3,
                verbose=0,
            )
        ]
        self.model.learn(self.args.total_timesteps, callback=callback)
