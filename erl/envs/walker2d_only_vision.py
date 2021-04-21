import os
import gym
import numpy as np
import pybullet_envs
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv, WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import Walker2D, WalkerBase

def debug_imshow(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

class Walker2DOnlyVision(Walker2D):
    """ a third person view. like controling a game. """
    def __init__(self):
        self.p = None
        self.camera_width = 8
        self.camera_height = 8
        self.camera_channel = 3
        self.gravity = -9.8
        super().__init__()
        
        # Reset the robot XML file:
        total_joints = 8
        WalkerBase.__init__(self,
                            f'{os.getcwd()}/erl/envs/xmls/walk2d_with_vision.xml',
                            'torso',
                            action_dim=total_joints,
                            # 10 real numbers of basic information
                            # each joint has 2 real numbers (position, velocity)
                            # WxHxC from camera image
                            # obs_dim=10 + total_joints*2 + self.camera_width*self.camera_height*self.camera_channel, 
                            obs_dim=self.camera_width*self.camera_height*self.camera_channel, 
                            power=0.40)

    def get_camera_image(self):
        """ 
        Get a RGB image from the camera mounted on the robot's camera lens.
        From an example: https://github.com/bulletphysics/bullet3/issues/1616
        """
        if self.p is None and self.scene._p:
            # Lazy load pybullet
            self.p = self.scene._p
            # Turn on the Debug GUI
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1)
            self.p.setGravity(0,0,self.gravity)

            distance = 3
            pitch = -30
            yaw = 0

            # Precalculate the projection matrix
            fov, aspect, nearplane, farplane = 128, 1.0, 0.01, 100
            self.projection_matrix = self.p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
            
            # Get Index
            self.robot_id = self.parts['camera_lens'].bodies[0]
            self.camera_lens_id = self.parts['camera_lens'].bodyPartIndex
            # Change the camera_lens to white, just to make sure the camera is mounted on the right body part
            self.p.changeVisualShape(self.robot_id, self.camera_lens_id,rgbaColor=[1,1,1,1])

        # Why I need to '*1.1' here?
        _current_x = self.body_xyz[0] * 1.1
        _current_y = self.body_xyz[1] * 1.1

        lookat = [_current_x, _current_y, 0.7]
        
        ret = self.p.getDebugVisalizerCamera()
        view_matrix,projection_matrix = ret[1],ret[2]
        _, _, rgbPixels, depthPixels, segmentationMaskBuffer = self.p.getCameraImage(self.camera_width, self.camera_height, view_matrix, projection_matrix)
        return rgbPixels

    def calc_state(self):
        # basic_state: 46 dimensions
        basic_state = super().calc_state()

        # camera_img: 262144 dimensions (256x256x4)
        camera_img = self.get_camera_image()
        # remove the Alpha channel to save space
        # camera_img: 196608 dimensions (256x256x3)
        camera_img = camera_img[:,:,:3] / 255.0
        #(WxHxC) => (CxHxW)
        # camera_img = np.swapaxes(camera_img, 0, 2)
        camera_img = np.rollaxis(camera_img, -1)
        
        # Flatten the observation and reconstruct them in the feature extractor.
        # Hard coded warning. see vae_extractor.py
        # basic_state = [:46] and camera_img = obs[46:].view(3,256,256)
        camera_img = camera_img.flatten()
        # obs = np.concatenate([basic_state, camera_img])
        obs = camera_img
        return obs

class Walker2DOnlyVisionEnv(Walker2DBulletEnv):
    def __init__(self, render=False):
        self.robot = Walker2DOnlyVision()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
    
    def reset(self):
        self.episodic_steps = 0
        return super().reset()
    
    def step(self, action):
        self.episodic_steps += 1
        return super().step(action)