import numpy as np

from stable_baselines3.common.callbacks import EventCallback

# import wandb

class DebugCallback(EventCallback):
    def __init__(self, name=""):
        self.name = name
        super().__init__()

    def _on_step(self) -> bool:
        s = np.sum(self.locals["actions"])
        # wandb.log({
        #     "debug/sum_actions": s,
        #     "step": self.num_timesteps
        # })
        if self.n_calls%100==0:
            print(f"{self.name} debug> [{self.num_timesteps}] actions> {s}")
        return super()._on_step()