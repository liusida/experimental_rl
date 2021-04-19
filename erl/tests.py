import time

from erl.customized_agents.customized_ppo import CustomizedPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from erl.tools.gym_helper import make_env

def test_current_exp(args):
    env_id="HopperBulletEnv-v0"
    env = DummyVecEnv([make_env(env_id=env_id, rank=0, seed=0, render=True)])
    env = VecNormalize.load(args.vnorm_filename, env)
    model = CustomizedPPO.load(args.model_filename, env=env)

    obs = env.reset()
    with model.policy.features_extractor.start_testing():
        for i in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                break
            time.sleep(0.01)
    time.sleep(1)
    env.close()