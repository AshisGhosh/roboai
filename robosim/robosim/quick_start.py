import numpy as np
from dataclasses import dataclass
from enum import Enum
import robosuite as suite
from robosuite import load_controller_config

class ControllerType(Enum):
    JOINT_VELOCITY = 1
    OSC_POSE = 2

@dataclass
class OSCControlStep:
    dx: float
    dy: float
    dz: float
    droll: float
    dpitch: float
    dyaw: float
    gripper: float

    def to_list(self):
        return [self.dx, self.dy, self.dz, self.droll, self.dpitch, self.dyaw, self.gripper]


def dummy_joint_vel_control(action, env, step):
    '''
    Dummy control function for joint velocity control
    '''
    if action is None:
        action = np.random.randn(env.robots[0].dof)  # sample random action
    for i, a in enumerate(action):
        action[i] += 0.1 * np.sin(step / 100)  # add some oscillation to the action
    print(f"Action {i}: {action}")
    return action


def dummy_osc_control(action, env, step):
    '''
    Dummy control function for OSC control
    dx, dy, dz, droll, dpitch, dyaw, gripper
    '''
    if action is None:
        action = OSCControlStep(0, 0, 0, 0, 0, 0, 0)
    else:
        action = OSCControlStep(*action)
    
    action.dx = 0.1 * np.sin(step / 100)
    action.dy = 0.1 * np.cos(step / 100)
    action.dz = 0.1 * np.sin(step / 100)
    action.droll = 0.1 * np.cos(step / 100)
    action.dpitch = 0.1 * np.sin(step / 100)
    action.dyaw = 0.1 * np.cos(step / 100)
    action.gripper = 0.1 * np.sin(step / 100)
    print(f"Action: {action.to_list()}")
    return action.to_list()


class robosim:
    def __init__(self, controller_type=ControllerType.OSC_POSE):
        self.controller_type = controller_type
        self.update_action = self.get_action_func()
    
    def get_action_func(self):
        if self.controller_type == ControllerType.JOINT_VELOCITY:
            return dummy_joint_vel_control
        elif self.controller_type == ControllerType.OSC_POSE:
            return dummy_osc_control
        else:
            raise ValueError("Invalid controller type")

    def start(self):
        print("Starting Robosuite Simulation...")

        config = load_controller_config(default_controller=self.controller_type.name) # load default controller config

        # create environment instance
        env = suite.make(
            env_name="Lift", # try with other tasks like "Stack" and "Door"
            robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
            gripper_types="default",
            controller_configs=config,
            control_freq=20,
            has_renderer=True,
            render_camera="frontview",
            camera_names=["frontview", "agentview"],
            has_offscreen_renderer=True,
            use_object_obs=False,                  
            use_camera_obs=True,                       
        )

        # reset the environment
        env.reset()

        action = None
        for i in range(1000):
            action = self.update_action(action, env, i)
            obs, reward, done, info = env.step(action)  # take action in the environment
            env.render()  # render on display



if __name__ == "__main__":
    sim = robosim()
    sim.start()
