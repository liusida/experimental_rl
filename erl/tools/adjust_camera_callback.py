import cv2
from stable_baselines3.common.callbacks import EventCallback

class AdjustCameraCallback(EventCallback):
    def _on_step(self) -> bool:
        if self.num_timesteps-1 % 100 == 0:
            self.camera_simpy_follow_robot(rotate=False)
        return super()._on_step()

    def reset_lights(self, pybullet):
        # to change lighting condition, need an arbitrary flag.
        pybullet.configureDebugVisualizer(flag=pybullet.COV_ENABLE_MOUSE_PICKING, enable=1,lightPosition=[10,-10,10])

    def camera_simpy_follow_robot(self, rotate=True, target_env=None):
        if target_env is None:
            target_env = self.model.env.envs[0]
        if not target_env.env.isRender:
            return
        p = target_env.env._p
        robot = target_env.robot
        if not hasattr(self, "camera_angle"): # lazy init
            self.camera_angle = 0.0
        self.camera_angle += 0.1
        distance = 1
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

    def write_a_image(self, current_folder, step, target_env=None):
        if target_env is None:
            target_env = self.model.env.envs[0]
        (width, height, rgbPixels, _, _) = target_env.env._p.getCameraImage(1920,1080, renderer=target_env.env._p.ER_BULLET_HARDWARE_OPENGL)
        image = rgbPixels[:,:,:3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{current_folder}/{step:05}.png", image)
