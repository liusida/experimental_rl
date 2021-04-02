from typing import *
import datetime
from collections import defaultdict

import wandb
import socket

from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback, EventCallback

class WeightsAndBiasesOutputFormat(logger.KVWriter):
    project = "ERL"
    enabled = True
    def __init__(self, args={}) -> None:
        """
        Dumps key/value pairs onto Weights and Biases.
        To enable this logger, please include this in the header of your code:
        ```
        from stable_baselines3.common.logger import WeightsAndBiasesOutputFormat
        WeightsAndBiasesOutputFormat.project = "cartpole"
        WeightsAndBiasesOutputFormat.enabled = True
        ```
        """
        current_hostname = socket.gethostname()
        if current_hostname.startswith("dg-"):
            current_hostname = "DeepGreen"
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        wandb.init(project=f"{self.project}[{current_hostname}]{current_date}", config=vars(args), tags=[args.extractor, args.env_id])
    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        key_values.update({'step': step})
        wandb.log(key_values)
    def close(self):
        pass


class WandbCallback(EventCallback):
    """ Watch the models, so the architecture can be uploaded to WandB """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.log_interval = 1000

        self.last_time_length = defaultdict(lambda: 0)
        self.last_distance_x = defaultdict(lambda: 0)
        self.reset_episodic_average()

    def _init_callback(self) -> None:
        # inject WandB to stable-baselines3.logger!
        logger.Logger.CURRENT.output_formats.append(WeightsAndBiasesOutputFormat(self.args))
        return super()._init_callback()

    def _on_training_start(self) -> None:
        wandb.watch([self.model.policy], log="all", log_freq=100)
        return True

    def episodic_log(self):
        for env_i in range(self.training_env.num_envs):
            if self.locals['dones'][env_i]:
                self.average_episodic_distance_N += 1
                self.average_episodic_distance_G += (self.last_distance_x - self.average_episodic_distance_G) / self.average_episodic_distance_N

                # self.average_episodic_length_N += 1
                # self.average_episodic_length_G += (self.last_time_length[env_i] - self.average_episodic_length_G) / self.average_episodic_length_N

            self.last_distance_x = self.training_env.envs[env_i].robot.body_xyz[0]
            # self.last_time_length[env_i] = self.training_env.envs[env_i].episodic_steps
        if self.n_calls % self.log_interval != 0:
            # Skip
            return
        wandb.log({
            f'episodes/distance': self.average_episodic_distance_G,
            f'episodes/time_length': self.average_episodic_length_G,
            'step': self.num_timesteps,
        })
        self.reset_episodic_average()

    def reset_episodic_average(self):
        self.average_episodic_distance_G = 0
        self.average_episodic_distance_N = 0
        self.average_episodic_length_G = 0
        self.average_episodic_length_N = 0

    def detailed_log(self):
        if self.n_calls % self.log_interval != 0:
            # Skip
            return

        for env_i in range(self.training_env.num_envs):
            relative_height = self.training_env.buf_obs[None][env_i][0]
            velocity = self.training_env.buf_obs[None][env_i][3]
            distance_x = self.training_env.envs[env_i].robot.body_xyz[0]
            wandb.log({
                f'raw_relative_height/env_{env_i}': relative_height,
                f'raw_velocity/env_{env_i}': velocity,
                f'raw_distance/env_{env_i}': distance_x,
                'step': self.num_timesteps,
            })

        wandb.log({
            'network/values': self.locals['values'].detach().mean().cpu().numpy(),
            'step': self.num_timesteps,
        })

    def _on_step(self):
        self.episodic_log()
        self.detailed_log()
        return True
