from stable_baselines3.common.callbacks import EventCallback

class AdjustCameraCallback(EventCallback):
    def _on_step(self) -> bool:
        self.camera_simpy_follow_robot()
        return super()._on_step()


    def camera_simpy_follow_robot(self, rotate=True):
        p = self.model.env.envs[0].env._p
        robot = self.model.env.envs[0].robot
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
