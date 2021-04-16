import numpy as np

from stable_baselines3.common.callbacks import EventCallback

class DebugCallback(EventCallback):
    def __init__(self, name=""):
        self.name = name
        self.enabled = True
        super().__init__()

    def _on_step(self) -> bool:
        if self.enabled:
            s = np.sum(self.locals["actions"])
            if self.n_calls%100==0:
                print(f"{self.name} debug> [{self.num_timesteps}] actions> {s}")
        return super()._on_step()