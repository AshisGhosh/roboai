import numpy as np
from dataclasses import dataclass
from enum import Enum
import asyncio

from PIL import Image

import robosuite as suite
from robosuite import load_controller_config

import logging
logging.basicConfig(level=logging.INFO)

from robosim.task import TaskFactory, TaskClass
from robosim.robot import Robot


class ControllerType(Enum):
    JOINT_VELOCITY = 1
    OSC_POSE = 2

@dataclass
class OSCControlStep:
    dx: float = 0
    dy: float = 0
    dz: float = 0
    droll: float = 0
    dpitch: float = 0
    dyaw: float = 0
    gripper: float = 0

    def to_list(self):
        return [self.dx, self.dy, self.dz, self.droll, self.dpitch, self.dyaw, self.gripper]
    

class RoboSim:
    def __init__(self, controller_type=ControllerType.OSC_POSE):
        self.controller_type = controller_type
        self.env = None

        self.task_factory = TaskFactory()
        self.tasks = []
        self.current_task = None
        # self.__goal_position = None

        self.render_task = None
        self.execute_async_task = None
        self.__close_renderer_flag = asyncio.Event()
        self.__executing_async = asyncio.Event()
        self.__pause_execution = asyncio.Event()
        self.__getting_image = asyncio.Event()
        self.__getting_image_ts = None

    
    def setup(self):
        self.env = self.setup_env()
        self.robot = Robot(self)
        self.register_tasks()
    
    def register_tasks(self):
        self.task_factory = TaskFactory()
        self.task_factory.register_task(self.robot.go_to_position)
        self.task_factory.register_task(self.robot.go_to_relative_position)
        self.task_factory.register_task(self.robot.go_to_pick_center)
        self.task_factory.register_task(self.robot.go_to_object)
        self.task_factory.register_task(self.robot.get_grasp, TaskClass.DATA_TASK)
    
    def setup_env(self):
        config = load_controller_config(default_controller=self.controller_type.name) # load default controller config

        # create environment instance
        env = suite.make(
            env_name="PickPlace", # try with other tasks like "Stack" and "Door"
            robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
            gripper_types="default",
            controller_configs=config,
            control_freq=20,
            has_renderer=True,
            render_camera="frontview",
            camera_names=["frontview", "agentview", "robot0_eye_in_hand"],
            has_offscreen_renderer=True,
            use_object_obs=False,                  
            use_camera_obs=True,                       
        )

        # reset the environment
        env.reset()
        return env

    def start(self):
        logging.info("Starting Robosuite Simulation...")

        self.env.reset()
        self.env.render()
        action = None
        for i in range(1000):
            action = self.check_for_action()
            if action is None:
                action = OSCControlStep().to_list()
            obs, reward, done, info = self.env.step(action)  # take action in the environment
            self.env.render()  # render on display
    
    async def start_async(self):
        if self.render_task is None or self.render_task.done():
            self.__close_renderer_flag.clear()
            self.env.reset()
            logging.debug("Now starting renderer...")
            self.render_task = asyncio.create_task(self.render())
        return True
    
    async def render(self):
        hz = 5
        while not self.__close_renderer_flag.is_set():  # Use the Event for checking
            if not self.__executing_async.is_set():
                self.env.render()
            await asyncio.sleep(1/hz)
    
    async def close_renderer(self):
        self.__close_renderer_flag.set()
        if self.render_task and not self.render_task.done():
            await self.render_task
        self.env.close_renderer()
        return True
    
    async def start_execution(self):
        self.execute_async_task = asyncio.create_task(self.execute_async())
        return True
    
    async def execute_async(self):
        if not self.render_task or self.render_task.done():
            await self.start_async()

        self.__pause_execution.clear()
        self.__executing_async.set()
        while self.tasks or self.current_task:            
            if self.__pause_execution.is_set():
                await self.manage_execution_delay() 
                continue

            action = await self.check_for_action()
            if action is None:
                action = OSCControlStep().to_list()
            obs, reward, done, info = self.env.step(action)
            if self.__getting_image.is_set():
                continue
            else:
                self.env.render()
            await self.manage_execution_delay()
        
        self.__executing_async.clear()
        return "All tasks executed."
    
    async def manage_execution_delay(self):
        delay = 0.0
        if self.__getting_image.is_set():
            delay = 0.1
        else:
            if self.__getting_image_ts is not None:
                current_time = asyncio.get_event_loop().time()
                if current_time - self.__getting_image_ts < 1:
                    delay = 0.1
                else:
                    self.__getting_image_ts = None
        await asyncio.sleep(delay)  
    
    async def pause_execution(self):
        logging.info("Pausing execution...")
        self.__pause_execution.set()
        return True
    
    async def resume_execution(self):
        logging.info("Resuming execution...")
        self.__pause_execution.clear()
        self.__executing_async.set()
        return True
    
    async def check_for_action(self):
        '''
        Check if there is a task in the queue. If there is, execute it.
        '''
        if self.current_task == None and self.tasks:
            self.current_task = self.tasks.pop(0)
            logging.info(f"Current Task: {self.current_task.name}")

        if self.current_task:
            if self.current_task.task_class != TaskClass.CONTROL_TASK:
                data = await self.current_task.execute()
                logging.info(f"Data: {data}")
                self.finish_current_task()
                return OSCControlStep().to_list()
            return await self.do_current_task()
        
        return None
    
    async def do_current_task(self):
        '''
        Execute the current task in the queue.
        '''
        action = self.current_task.execute()
        logging.debug(f"Action: {action}")
        if action == OSCControlStep().to_list():
            self.finish_current_task()
        return action
    
    def finish_current_task(self):
        '''
        Finish the current task in the queue.
        '''
        logging.info(f"Task finished: {self.current_task.name}")
        self.current_task = None
        self.robot.__goal_position = None
    
    def add_task(self, name, function, *args, **kwargs):
        '''
        Add a task to the queue.
        '''
        task = self.task_factory.create_task(function, name, *args, **kwargs)
        self.tasks.append(task)
        logging.info(f"Task added: {task}")
    
    def get_tasks(self):
        return self.tasks
    
    async def get_image(self, camera_name="agentview", width=512, height=512) -> Image:
        self.__getting_image.set()
        self.__getting_image_ts = asyncio.get_event_loop().time()
        im = self.env.sim.render(width=width, height=height, camera_name=camera_name)
        img = Image.fromarray(im[::-1])
        self.__getting_image.clear()
        return img
    
    async def get_grasp_image(self) -> Image:
        return await self.get_image("robot0_eye_in_hand", width=640, height=480)

if __name__ == "__main__":
    sim = RoboSim()
    sim.setup()
    available_tasks = sim.task_factory.get_task_types()
    logging.info(f"Available Tasks: {available_tasks}")
    sim.add_task('Position Check', 'go_to_position', [-0.3, -0.3, 1])
    sim.add_task('Relative Position Check', 'go_to_relative_position', [0.3, 0.1, 0.1])
    sim.add_task('Go to can', 'go_to_object', 'Can')
    sim.start()


