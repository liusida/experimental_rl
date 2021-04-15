from typing import *
import datetime, os, copy
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
        elif current_hostname.startswith("node") or current_hostname.startswith("shared"):
            current_hostname = "BlueMoon"
            
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        wandb.init(project=f"{self.project}[{current_hostname}]", config=vars(args), tags=[current_date, args.exp_name, args.extractor, args.env_id])
        wandb.run.name = f"{args.exp_name}-{args.extractor}-{args.env_id}-{args.seed}"
        wandb.save()

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
        self.model_save_interval = 10000

        self.last_time_length = defaultdict(lambda: 0)
        self.last_distance_x = defaultdict(lambda: 0)
        self.reset_episodic_average()

    def _init_callback(self) -> None:
        # inject WandB to stable-baselines3.logger!
        logger.Logger.CURRENT.output_formats.append(WeightsAndBiasesOutputFormat(self.args))
        return super()._init_callback()

    def _on_training_start(self) -> None:
        wandb.watch([self.model.policy.features_extractor], log="all", log_freq=10)
        # wandb.watch([self.model.policy], log="parameters", log_freq=10)
        return True

    def episodic_log(self):
        for env_i in range(self.training_env.num_envs):
            if self.locals['dones'][env_i]:
                self.average_episodic_distance_N += 1
                self.average_episodic_distance_G += (self.last_distance_x - self.average_episodic_distance_G) / self.average_episodic_distance_N

                self.average_episodic_length_N += 1
                self.average_episodic_length_G += (self.last_time_length[env_i] - self.average_episodic_length_G) / self.average_episodic_length_N

            self.last_distance_x = self.training_env.envs[env_i].robot.body_xyz[0]
            self.last_time_length[env_i] = self.training_env.envs[env_i]._elapsed_steps
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

    def save_model(self):
        if self.n_calls % self.model_save_interval == 0:
            filename = f"checkpoints/model_at_{self.num_timesteps}_steps.zip"
            policy_filename = f"checkpoints/policy_at_{self.num_timesteps}_steps.h5"
            
            # Note: if we set the `experiment` member variable, the model will fail to save to local file system
            # So make a copy of the model and change the `experiment` member variable.
            _model = copy.copy(self.model)
            _model.experiment = None
            _model.save(os.path.join(wandb.run.dir, filename))
            _model.policy.save(os.path.join(wandb.run.dir, policy_filename))
            # After save to local file system, upload it to wandb
            wandb.save(filename)
            wandb.save(policy_filename)


    def camera_simpy_follow_robot(self, p, robot, rotate=True):
        if not hasattr(self, "camera_angle"): # lazy init
            self.camera_angle = 0.0
        self.camera_angle += 5
        distance = 3
        pitch = -30
        if rotate:
            # rotate at 60 degree.
            yaw = (self.camera_angle//60)*60
        else:
            yaw = 0

        # Why I need to '*1.1' here?
        _current_x = robot.body_xyz[0] * 1.1
        _current_y = robot.body_xyz[1] * 1.1

        lookat = [_current_x, _current_y, 0.7]
        p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

    def save_camera_img(self):
        if self.n_calls % self.log_interval == 0:
            width = 640
            height = 480
            env = self.model.env.envs[0]
            p = env.env._p
            self.camera_simpy_follow_robot(p, env.robot)
            _, _, rgbPixels, _, _ = p.getCameraImage(width=width, height=height)
            camera_img = rgbPixels[:,:,:3] / 255.0
            # crop
            camera_img = camera_img[ int(height/5):int(height*4/5), int(width/5):int(width*4/5) ]
            wandb.log({"images/screenshots": [wandb.Image(camera_img)]})

    def _on_step(self):
        if wandb.run.dir=='/': # wandb disabled
            return
        self.episodic_log()
        self.detailed_log()
        self.save_model()
        # self.save_camera_img()
        return True
