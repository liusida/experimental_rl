import numpy as np

from stable_baselines3.common.callbacks import EventCallback

class DebugCallback(EventCallback):
    def _on_step(self) -> bool:
        s = np.sum(self.locals["actions"])
        print("debug> actions> ", s)
        return super()._on_step()