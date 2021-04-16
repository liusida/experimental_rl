import numpy as np

from stable_baselines3.common.callbacks import EventCallback

import wandb

class DebugCallback(EventCallback):
    def _on_step(self) -> bool:
        s = np.sum(self.locals["actions"])
        wandb.log({
            "debug/sum_actions": s,
            "step": self.num_timesteps
        })
        print("debug> actions> ", s)
        return super()._on_step()