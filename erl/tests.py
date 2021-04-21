import time, os, glob
import cv2
import numpy as np

from erl.customized_agents.customized_ppo import CustomizedPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from erl.tools.gym_helper import make_env
from erl.tools.adjust_camera_callback import AdjustCameraCallback

img_path = f"videos/images/"

def test_current_exp(args):
    if args.save_img:
        all_folders = glob.glob(os.path.join(img_path,"*"))
        all_folders = [os.path.basename(x) for x in all_folders]
        all_folders = [int(x) if x.isnumeric() else -1 for x in all_folders] + [0]
        current_folder = max(all_folders) + 1
        current_folder = os.path.join(img_path, str(current_folder))
        os.makedirs(current_folder, exist_ok=True)
        print(f"Writing into {current_folder}")
        input("Press Enter...")

    env = DummyVecEnv([make_env(env_id=args.env_id, rank=0, seed=0, render=True)])
    env = VecNormalize.load(args.vnorm_filename, env)
    model = CustomizedPPO.load(args.model_filename, env=env)
    callback = AdjustCameraCallback()
    
    obs = env.reset()
    callback.reset_lights(env.envs[0].env._p) # once window is opened, change the lighting

    if args.save_img:
        time.sleep(1) # please use this time to maximize the window, so that the image recorded will be full size

    with model.policy.features_extractor.start_testing():
        while True:
            for i in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                callback.camera_simpy_follow_robot(target_env=env.envs[0])
                if args.save_img:
                    callback.write_a_image(current_folder=current_folder, step=i, target_env=env.envs[0])
                    if obs.shape[1]>100: # With Vision I guess
                        image = np.rollaxis(obs[:, -3*8*8:].reshape([3,8,8]), 0, start=3) * 255.0
                        print(image.shape)
                        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(f"{current_folder}/vision_{i:05}.png", image)
                if done:
                    break
                time.sleep(0.01)
            break
        time.sleep(0.1)
    env.close()