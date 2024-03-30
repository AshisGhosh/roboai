import numpy as np
from dataclasses import dataclass
from enum import Enum
import asyncio

from PIL import Image

import robosuite as suite
from robosuite import load_controller_config

import logging
logging.basicConfig(level=logging.DEBUG)


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
    
class TaskFactory:
    def __init__(self):
        self._creators = {}

    def register_task(self, creator):
        self.register_task_type(
            creator.__name__,
            lambda name, *args, **kwargs: Task(name, creator, *args, **kwargs)
        )

    def register_task_type(self, task_type, creator):
        self._creators[task_type] = creator

    def create_task(self, task_type, task_name=None, *args, **kwargs):
        creator = self._creators.get(task_type)
        if not creator:
            raise ValueError(f"Task type {task_type} not registered.")
        if task_name is not None:
            # Use the provided task_name or fallback to a default naming convention
            return creator(task_name, *args, **kwargs)
        else:
            return creator(task_type, *args, **kwargs)
    
    def get_task_types(self):
        return self._creators.keys()
    

class Task:
    def __init__(self, name, function, *args, **kwargs):
        self.name = name
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def execute(self):
        try:
            return self.function(*self.args, **self.kwargs)
        except Exception as e:
            logging.error(f"Error executing task {self.name}: {e}")
    
    def __str__(self):
        return f"Task: {self.name}\n    Function: {self.function}\n    Args: {self.args}\n    Kwargs: {self.kwargs}"


class RoboSim:
    def __init__(self, controller_type=ControllerType.OSC_POSE):
        self.controller_type = controller_type
        self.env = None

        self.task_factory = TaskFactory()
        self.tasks = []
        self.current_task = None
        self.__goal_position = None

        self.render_task = None
        self.execute_async_task = None
        self.__close_renderer_flag = asyncio.Event()
        self.__executing_async = asyncio.Event()
        self.__pause_execution = asyncio.Event()
        self.__getting_image = asyncio.Event()
        self.__getting_image_ts = None

    
    def setup(self):
        self.env = self.setup_env()
        self.register_tasks()
    
    def register_tasks(self):
        self.task_factory = TaskFactory()
        self.task_factory.register_task(self.go_to_position)
        self.task_factory.register_task(self.go_to_relative_position)
        self.task_factory.register_task(self.go_to_object)
    
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
            camera_names=["frontview", "agentview"],
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
                action = OSCControlStep(0, 0, 0, 0, 0, 0, 0).to_list()
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

            action = self.check_for_action()
            if action is None:
                action = OSCControlStep(0, 0, 0, 0, 0, 0, 0).to_list()
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
    
    
    def check_for_action(self):
        '''
        Check if there is a task in the queue. If there is, execute it.
        '''
        if self.current_task:
            return self.do_current_task()
        else:
            if self.tasks:
                self.current_task = self.tasks.pop(0)
                logging.info(f"Current Task: {self.current_task.name}")
                return self.do_current_task()
        
        return None
    
    def do_current_task(self):
        '''
        Execute the current task in the queue.
        '''
        action = self.current_task.execute()
        logging.debug(f"Action: {action}")
        if action == OSCControlStep(0, 0, 0, 0, 0, 0, 0).to_list():
            self.finish_current_task()
        return action
    
    def finish_current_task(self):
        '''
        Finish the current task in the queue.
        '''
        logging.info(f"Task finished: {self.current_task.name}")
        self.current_task = None
        self.__goal_position = None
    
    def add_task(self, name, function, *args, **kwargs):
        '''
        Add a task to the queue.
        '''
        task = self.task_factory.create_task(function, name, *args, **kwargs)
        self.tasks.append(task)
        logging.info(f"Task added: {task}")
    
    def get_tasks(self):
        return self.tasks
    
    def go_to_position(self, position, frame="gripper"):
        if len(position) != 3:
            raise ValueError("Position must be a 3D point.")
        dist = self.distance_to_position(position)
        return self.simple_velocity_control(dist)
    
    def go_to_relative_position(self, position, frame="gripper"):
        if len(position) != 3:
            raise ValueError("Position must be a 3D point.")
        if frame != "gripper":
            raise NotImplementedError("Only gripper frame is supported for now.")
        
        if self.__goal_position is None:
            self.__goal_position = self.get_gripper_position() + np.array(position)
        dist = self.distance_to_position(self.__goal_position)
        return self.simple_velocity_control(dist)
    
    def get_gripper_position(self):
        gripper=self.env.robots[0].gripper
        gripper_pos = self.env.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        return gripper_pos
    
    def distance_to_position(self, position):
        logging.debug(f"Position: {position}")
        gripper_pos = self.get_gripper_position()
        logging.debug(f"Gripper Position: {gripper_pos}")
        dist = position - gripper_pos
        logging.debug(f"Distance: {dist}")
        return dist

    def go_to_object(self, target_obj_name="Can"):
        obj = self.env.objects[self.env.object_to_id[target_obj_name.lower()]]
        dist = self.env._gripper_to_target(
                        gripper=self.env.robots[0].gripper,
                        target=obj.root_body,
                        target_type="body",
                    )
        return self.simple_velocity_control(dist)

    def simple_velocity_control(self, dist):
        euclidean_dist = np.linalg.norm(dist)
        if euclidean_dist < 0.03:
            return [0, 0, 0, 0, 0, 0, 0]
        cartesian_velocities = dist / euclidean_dist
        action = OSCControlStep(*cartesian_velocities, 0, 0, 0, 0)
        return action.to_list()
    
    def get_object_names(self):
        return [obj.name for obj in self.env.objects]

    def get_object_pose(self):
        for obj in self.env.objects:
            dist = self.env._gripper_to_target(
                    gripper=self.env.robots[0].gripper,
                    target=obj.root_body,
                    target_type="body",
                    return_distance=True,
                )
            logging.info(f"Object {obj.name}: {dist}")
    
    async def get_image(self):
        self.__getting_image.set()
        self.__getting_image_ts = asyncio.get_event_loop().time()
        im = self.env.sim.render(width=512, height=512, camera_name="agentview")
        img = Image.fromarray(im[::-1])
        self.__getting_image.clear()
        return img

if __name__ == "__main__":
    sim = RoboSim()
    sim.setup()
    available_tasks = sim.task_factory.get_task_types()
    logging.info(f"Available Tasks: {available_tasks}")
    sim.add_task('Position Check', 'go_to_position', [-0.3, -0.3, 1])
    sim.add_task('Relative Position Check', 'go_to_relative_position', [0.3, 0.1, 0.1])
    sim.add_task('Go to can', 'go_to_object', 'Can')
    sim.start()


