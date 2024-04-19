import numpy as np
from dataclasses import dataclass
from enum import Enum
import asyncio
import copy
from PIL import Image
import httpx

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.utils.transform_utils import mat2quat, euler2mat
from robosuite.utils.camera_utils import CameraMover
from robosuite.utils.mjcf_utils import new_body, new_site

import logging
log = logging.getLogger("robosim")
log.setLevel(logging.INFO)

from robosim.task import TaskFactory, TaskClass, TaskStatus
from robosim.robot import Robot
from robosim.grasp_handler import Camera

import shared.utils.gradio_client as gradio
from shared.utils.model_server_client import _answer_question_from_image
import shared.utils.replicate_client as replicate

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
        self._last_task = None
        self._last_task_finish_status = None

        self.render_task = None
        self.execute_async_task = None
        self.__close_renderer_flag = asyncio.Event()
        self.__executing_async = asyncio.Event()
        self.__pause_execution = asyncio.Event()
        self.__stop_execution = asyncio.Event()
        self.__getting_image = asyncio.Event()
        self.__getting_image_ts = None

    
    def setup(self):
        self.env = self.setup_env()
        self.setup_markers()
        self.setup_cameras()
        self.robot = Robot(self)
        self.register_tasks()
        # self.test_action([0,0,0,0,0,0,0,0])
    
    def register_tasks(self):
        self.task_factory = TaskFactory()
        self.task_factory.register_task(self.robot.go_to_position)
        self.task_factory.register_task(self.robot.go_to_relative_position)
        self.task_factory.register_task(self.robot.go_to_orientation)
        self.task_factory.register_task(self.robot.go_to_pick_center)
        self.task_factory.register_task(self.robot.go_to_object)
        self.task_factory.register_task(self.robot.get_grasp, TaskClass.DATA_TASK)
        self.task_factory.register_task(self.robot.go_to_grasp_orientation)
        self.task_factory.register_task(self.robot.go_to_grasp_position)
        self.task_factory.register_task(self.robot.go_to_pose)
        self.task_factory.register_task(self.robot.go_to_pre_grasp)
        self.task_factory.register_task(self.add_grasp_marker, TaskClass.DATA_TASK)
        self.task_factory.register_task(self.robot.close_gripper)
        self.task_factory.register_task(self.robot.open_gripper)
        self.task_factory.register_task(self.robot.go_to_drop)
    
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
            # render_camera="robot0_eye_in_hand",
            camera_names=["frontview", "agentview", "robot0_eye_in_hand"],
            camera_heights=[672, 672, 480],
            camera_widths=[672, 672, 640],
            camera_depths=[False, False, True],  # set to true for using depth sensor
            has_offscreen_renderer=True,
            use_object_obs=False,                  
            use_camera_obs=True,                       
        )

        # reset the environment
        env.reset()
        return env

    def setup_cameras(self):
        self.camera_mover = CameraMover(self.env, "agentview")
        self.camera_mover.set_camera_pose(pos=[0.65, -0.25, 1.4])
        log.info(f"Camera Pose: {self.camera_mover.get_camera_pose()}")
        self.env.sim.forward()
        self.env.sim.step()
        self.env.step(np.zeros(self.env.action_dim))
    
    def setup_markers(self):
        self.markers = []
        # self.add_marker([0.5, 0, 1.0], size=0.3, name="indicator_ball")
        self.add_marker([0.5, 0, 2.0], size=0.05, name="grasp_marker", rgba=[0, 0, 1, 0.65])
        self.add_marker([0.5, 0, 1.0], type="box", size=(0.03, 0.05, 0.1), name="gripper_goal")

    def test_action(self, action, *args):
        obs, reward, done, info = self.env.step(action)

    def add_marker(self, pos, type = "sphere", size = 0.03, name = "indicator_ball", rgba = [1, 0, 0, 0.65]):
        indicator_config = {
            "name": name,
            "type": type,
            "size": size,
            "rgba": rgba,
            "pos": pos
        }
        self.markers.append(indicator_config)
        self.env = VisualizationWrapper(self.env, self.markers)
        self.env.sim.forward()
        self.env.sim.step()
        self.env.set_xml_processor(processor=None)
    
    def _add_indicators(self, xml):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        worldbody = root.find("worldbody")
        for indicator_config in self.markers:
            config = copy.deepcopy(indicator_config)
            indicator_body = new_body(name=config["name"] + "_body", pos=config.pop("pos", (0, 0, 0)))
            indicator_body.append(new_site(**config))
            worldbody.append(indicator_body)

            xml = ET.tostring(root, encoding="utf8").decode("utf8")
        
        return xml
        
    async def add_grasp_marker(self, *args):
        grasp_pos = self.robot.get_grasp_pose()[0]
        self.add_marker(grasp_pos, name="grasp_marker")
        self.env.render()
        return f"Marker added at {grasp_pos}."
    
    def reset(self):
        self.env.reset()
        self.setup_markers()
        self.setup_cameras()
    
    def move_gripper_goal_to_gripper(self):
        gripper_pos = self.robot.get_gripper_position()
        gripper_ori = mat2quat(self.robot.get_gripper_orientation())
        self.move_marker(gripper_pos, gripper_ori, name="gripper_goal")
        return f"Marker moved to gripper position: {gripper_pos}."

    def move_marker(self, position=None, orientation=None, name="grasp_marker", *args):
        if position is None and orientation is None:
            raise ValueError("Either position or orientation must be provided.")
        if position is not None:
            self.env.sim.model.body_pos[self.env.sim.model.body_name2id(name + "_body")] = position
        if orientation is not None:
            if len(orientation) == 3:
                base_orientation = np.array([np.pi, 0, np.pi/2])
                o = copy.deepcopy(orientation)
                o = np.array(o) - base_orientation
                orientation = base_orientation + [-o[1], o[2], -o[0]]
                orientation = euler2mat(orientation)
                orientation = mat2quat(orientation)
            self.env.sim.model.body_quat[self.env.sim.model.body_name2id(name + "_body")] = orientation
        self.env.sim.forward()
        self.env.sim.step()
        self.env.render()
        resp = f"Marker {name} moved to {position} with orientation {orientation}."
        log.debug(resp)
        return resp
    
    def pixel_to_marker(self, pixel, camera_name="robot0_eye_in_hand"):
        if camera_name != "robot0_eye_in_hand":
            raise NotImplementedError(f"pixel_to_marker only supports robot0_eye_in_hand currently.")
        
        camera = Camera(self.env, camera_name)
        marker_pose = camera.pixel_to_world(pixel)
        log.debug(f"Marker Pose: {marker_pose}")
        self.move_marker(marker_pose)
        return str(marker_pose)       
        

    def start(self):
        log.info("Starting Robosuite Simulation...")

        # self.env.reset()
        self.reset()
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
            # self.env.reset()
            self.reset()

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
            if self.__stop_execution.is_set():
                self.__executing_async.clear()
                return "Execution stopped."
                    
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
    
    async def stop_execution(self):
        log.info("Stopping execution...")
        self.__stop_execution.set()
        return True
    
    async def pause_execution(self):
        log.info("Pausing execution...")
        self.__pause_execution.set()
        return True
    
    async def resume_execution(self):
        log.info("Resuming execution...")
        self.__pause_execution.clear()
        self.__executing_async.set()
        return True
    
    async def check_for_action(self):
        '''
        Check if there is a task in the queue. If there is, execute it.
        '''
        if self.current_task == None and self.tasks:
            self.current_task = self.tasks.pop(0)
            log.info(f"Current Task: {self.current_task.name}")

        if self.current_task:
            if self.current_task.task_class != TaskClass.CONTROL_TASK:
                log.info(f"Executing Task: {self.current_task.name}")
                data = await self.current_task.execute()
                log.info(f"Data: {data}")
                if data is None:
                    self.finish_current_task(status=TaskStatus.FAILED, status_msg="Task failed.")
                self.finish_current_task()
                return OSCControlStep().to_list()
            return await self.do_current_task()
        
        return None
    
    async def do_current_task(self):
        '''
        Execute the current task in the queue.
        '''
        action = self.current_task.execute()
        log.debug(f"Action: {action}")

        self.check_if_task_finished(action)
        
        return action
    
    def check_if_task_finished(self, action):
        if action == OSCControlStep().to_list():
            self.finish_current_task()
        if not self.robot.is_gripper_moving(action):
            self.finish_current_task(status=TaskStatus.FAILED, status_msg="Gripper not moving.")

    
    def finish_current_task(self, status=TaskStatus.COMPLETED, status_msg=None):
        '''
        Finish the current task in the queue.
        '''
        log.info(f"Task finished: {self.current_task.name} with status {status} and message {status_msg}.")
        self._last_task = self.current_task
        self._last_task_finish_status = { "status": status, "status_msg": status_msg}
        self.current_task = None
        self.robot.__goal_position = None
        self.robot.__goal_orientation = None
    
    def add_task(self, name, function, *args, **kwargs):
        '''
        Add a task to the queue.
        '''
        task = self.task_factory.create_task(function, name, *args, **kwargs)
        self.tasks.append(task)
        log.info(f"Task added: {task}")
    
    def get_tasks(self):
        return self.tasks
    
    def clear_tasks(self):
        self.tasks = []
        if self.current_task:
            self.finish_current_task()
    
    async def get_image_realtime(self, camera_name="agentview", width=512, height=512) -> Image:
        self.__getting_image.set()
        self.__getting_image_ts = asyncio.get_event_loop().time()
        im = self.env.sim.render(width=width, height=height, camera_name=camera_name)
        img = Image.fromarray(im[::-1])
        self.__getting_image.clear()
        return img
    
    async def get_image(self, camera_name="agentview") -> Image:
        markers = ["gripper0_grip_site", "gripper0_grip_site_cylinder", "gripper_goal", "grasp_marker"]
        for marker in markers:
            self.env.sim.model.site_rgba[self.env.sim.model.site_name2id(marker)][3] = 0

        self.env.step(np.zeros(self.env.action_dim))
        im = self.env._get_observations()[camera_name + "_image"]
        img = Image.fromarray(im[::-1])

        # turn on marker visualization
        for marker in markers:
            self.env.sim.model.site_rgba[self.env.sim.model.site_name2id(marker)][3] = 0.25
        
        return img
    
    async def get_image_with_markers(self, camera_name="agentview") -> Image:
        
        self.env.step(np.zeros(self.env.action_dim))
        im = self.env._get_observations()[camera_name + "_image"]
        img = Image.fromarray(im[::-1])
        
        return img
    
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
            log.info(f"Object {obj.name}: {dist}")
    
    async def pick(self, object_name):
        self.clear_tasks()
        await self.start_async()
        # self.add_task("go to object", "go_to_object", object_name)
        self.add_task("go to pick center", "go_to_pick_center", "")
        self.add_task("get grasp", "get_grasp", object_name)
        await self.execute_async()
        if self._last_task_finish_status["status"] == TaskStatus.FAILED:
            retry_attempts = 3
            for i in range(retry_attempts):
                log.info(f"Retrying pick attempt {i+1}...")
                self.add_task("go to pick center", "go_to_pick_center", "")
                self.add_task("get grasp", "get_grasp", object_name)
                if self._last_task_finish_status["status"] == TaskStatus.COMPLETED:
                    break
        success, _ = await self.get_feedback("grasp-selection-feedback", object_name)
        if not success:
            log.info("Grasp selection feedback failed.")
            return False
        self.add_task("move to pre-grasp", "go_to_pre_grasp", "")
        self.add_task("open gripper", "open_gripper", "")
        self.add_task("go to grasp position", "go_to_grasp_position", "")
        self.add_task("close gripper", "close_gripper", "")
        await self.execute_async()
        if self._last_task_finish_status["status"] != TaskStatus.COMPLETED:
            log.info("Pick failed.")
            return False
        success, _ = await self.get_feedback("grasp-feedback", object_name)

    async def get_feedback(self, feedback_type, object_name):
        if feedback_type == "grasp-selection-feedback":
            image = await self.get_image_with_markers()
            question = f"Is the the blue sphere marker over the {object_name}?"
        elif feedback_type == "grasp-feedback":
            image = await self.get_image()
            question = f"Is the object {object_name} grasped by the robot?"
        log.info(f"Giving feedback for {feedback_type}...")
        log.info(f"Question: {question}")
        # output = await _answer_question_from_image(image, question)
        try:
            # output = gradio.moondream_answer_question_from_image(image, question)
            # output = replicate.moondream_answer_question_from_image(image, question)
            output = gradio.qwen_vl_max_answer_question_from_image(image, question)

        except httpx.ConnectError as e:
            log.error(f"Error connecting to the model server: {e}")
            output = await _answer_question_from_image(image, question)
        log.warn(output)
        if "yes" in output["result"].lower():
            return True, output
        return False, output


if __name__ == "__main__":
    sim = RoboSim()
    sim.setup()
    available_tasks = sim.task_factory.get_task_types()
    log.info(f"Available Tasks: {available_tasks}")
    sim.add_task('Position Check', 'go_to_position', [-0.3, -0.3, 1])
    sim.add_task('Relative Position Check', 'go_to_relative_position', [0.3, 0.1, 0.1])
    sim.add_task('Go to can', 'go_to_object', 'Can')
    sim.start()


