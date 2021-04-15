from stable_baselines3.common.callbacks import EvalCallback

class CustomizedEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        ret = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            with self.model.policy.features_extractor.start_testing():
                ret = super()._on_step()
        return ret
