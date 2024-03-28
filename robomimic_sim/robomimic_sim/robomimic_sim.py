import argparse
import json
import h5py
import imageio
import numpy as np
import os
from copy import deepcopy
import asyncio
from PIL import Image
import io

import torch

import robosuite
from robosuite import load_controller_config

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

import urllib.request

class RobomimicSim:
    def __init__(self):
        self.rollout_visualizing = False
        self.rollout_task = None
        self.render_task = None
        self.close_renderer_flag = asyncio.Event()  # Use an asyncio Event for better coordination


    def setup(self):
        # Get pretrained checkpooint from the model zoo

        ckpt_path = "models/lift_ph_low_dim_epoch_1000_succ_100.pth"
        # Lift (Proficient Human)
        urllib.request.urlretrieve(
            "http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/lift/bc_rnn/lift_ph_low_dim_epoch_1000_succ_100.pth",
            filename=ckpt_path
        )

        assert os.path.exists(ckpt_path)

        device = TorchUtils.get_torch_device(try_to_use_cuda=True)

        # restore policy
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

        # create environment from saved checkpoint
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            render=True, 
            render_offscreen=True, # render to RGB images for video
            verbose=True,
        )

        self.policy = policy
        self.env = env
    
    def custom_env(self):
        config = load_controller_config(default_controller="OSC_POSE") # load default controller config
        # create environment from scratch
        env = robosuite.make(
            env_name="Lift", # try with other tasks like "Stack" and "Door"
            robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
            gripper_types="default",
            controller_configs=None,
            control_freq=20,
            has_renderer=True,
            render_camera="frontview",
            camera_names=["frontview", "agentview"],
            has_offscreen_renderer=True,
            use_object_obs=False,                  
            use_camera_obs=True,                       
        )
        self.env = env
    
    async def start_rollout(self):
        if self.rollout_task is None or self.rollout_task.done():
            self.close_renderer_flag.clear()
            await self.start_renderer()
            self.rollout_task = asyncio.create_task(self.run())
    
    async def close_renderer(self):
        self.close_renderer_flag.set()  # Signal to stop the tasks
        if self.render_task and not self.render_task.done():
            await self.render_task  # Await the task to ensure it completes
        if self.rollout_task and not self.rollout_task.done():
            self.rollout_task.cancel()  # Cancel rollout task as it might be waiting for external input
            try:
                await self.rollout_task  # Attempt to await the task to handle any cleanup
            except asyncio.CancelledError:
                pass  # Expected if the task was cancelled
        self.env.base_env.close_renderer()
    
    async def render(self):
        hz = 5
        while not self.close_renderer_flag.is_set():  # Use the Event for checking
            if not self.rollout_visualizing:
                self.env.render(mode="human", camera_name="frontview")
            await asyncio.sleep(1/hz)
    
    async def start_renderer(self):
        if self.render_task is None or self.render_task.done():
            self.close_renderer_flag.clear()
            self.env.reset()
            print("Now starting renderer...")
            self.render_task = asyncio.create_task(self.render())
        return True
    
    async def reset(self):
        self.env.reset()
        return True

    async def rollout(self, policy, env, horizon, render=False, video_writer=None, video_skip=5, camera_names=None):
        """
        Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
        and returns the rollout trajectory.
        Args:
            policy (instance of RolloutPolicy): policy loaded from a checkpoint
            env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
            horizon (int): maximum horizon for the rollout
            render (bool): whether to render rollout on-screen
            video_writer (imageio writer): if provided, use to write rollout to video
            video_skip (int): how often to write video frames
            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.
        Returns:
            stats (dict): some statistics for the rollout - such as return, horizon, and task success
        """
        print("Rolling out policy...")

        assert isinstance(env, EnvBase)
        assert isinstance(policy, RolloutPolicy)
        assert not (render and (video_writer is not None))

        policy.start_episode()
        # obs = env.reset()
        # state_dict = env.get_state()

        # # hack that is necessary for robosuite tasks for deterministic action playback
        # obs = env.reset_to(state_dict)

        obs = env.get_observation()

        results = {}
        video_count = 0  # video frame counter
        total_reward = 0.
        
        self.rollout_visualizing = True
        try:
            for step_i in range(horizon):
                await asyncio.sleep(0)  # Allow other tasks to run
                if self.close_renderer_flag.is_set():
                    print("Stopping rollout due to renderer close request...")
                    break

                # get action from policy
                act = policy(ob=obs)

                # play action
                next_obs, r, done, _ = env.step(act)

                # compute reward
                total_reward += r
                success = env.is_success()["task"]

                # visualization
                if render:
                    env.render(mode="human", camera_name=camera_names[0])
                if video_writer is not None:
                    if video_count % video_skip == 0:
                        video_img = []
                        for cam_name in camera_names:
                            video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                        video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                        video_writer.append_data(video_img)
                    video_count += 1

                # break if done or if success
                if done or success:
                    break

                # update for next iter
                obs = deepcopy(next_obs)
                state_dict = env.get_state()

        except env.rollout_exceptions as e:
            print("WARNING: got rollout exception {}".format(e))
        
        self.rollout_visualizing = False

        stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

        return stats

    async def run(self):
        rollout_horizon = 400
        np.random.seed(0)
        torch.manual_seed(0)
        video_path = "output/rollout.mp4"
        video_writer = imageio.get_writer(video_path, fps=20)

        policy = self.policy
        env = self.env

        stats = await self.rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=True,
            # render=False, 
            # video_writer=video_writer, 
            # video_skip=5, 
            camera_names=["frontview", "agentview"]
        )
        print(stats)
        video_writer.close()
    
    def get_image(self):
        img = self.env.render(mode="rgb_array", height=512, width=512, camera_name="frontview")
        img = Image.fromarray(img)
        return img


