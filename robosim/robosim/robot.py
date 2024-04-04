import numpy as np
from PIL import Image
import asyncio
import copy

from robosim.grasp_handler import GraspHandler
from robosuite.utils.transform_utils import mat2euler

import logging
logging.basicConfig(level=logging.WARN)

log = logging.getLogger("robosim robot")
log.setLevel(logging.DEBUG)

class Robot:
    def __init__(self, robosim):
        self.robosim = robosim
        self.env = robosim.env
        self.grasp = GraspHandler(self)
        self.__goal_position = None
        self.__grasp_sequence = None
        
    def go_to_position(self, position):
        if len(position) != 3:
            raise ValueError("Position must be a 3D point.")
        dist = self.distance_to_position(position)
        log.debug(f"Distance: {dist}")
        return self.simple_velocity_control(dist)
    
    def go_to_relative_position(self, position, frame="gripper"):
        if len(position) != 3:
            raise ValueError("Position must be a 3D point.")
        if frame != "gripper":
            raise NotImplementedError("Only gripper frame is supported for now.")
        
        if self.__goal_position is None:
            self.__goal_position = self.get_gripper_position() + np.array(position)
        dist = self.distance_to_position(self.__goal_position)
        log.debug(f"Distance: {dist}")
        return self.simple_velocity_control(dist)
    
    def go_to_pick_center(self, *args):
        return self.go_to_position(position=[-0.02, -0.27, 1.05])
    
    def go_to_orientation(self, orientation):
        if len(orientation) != 3:
            raise ValueError("Orientation must be a 3D rotation.")
        dist = self.delta_to_orientation(orientation)
        log.debug(f"Distance (orientation): {dist}")
        return self.simple_orientation_control(dist)
    
    def go_to_pose(self, pose, gripper=0):
        position = pose[:3]
        orientation = pose[3:]
        position_action = self.go_to_position(position)
        orientation_action = self.go_to_orientation(orientation)
        action = [*position_action, *orientation_action, gripper]
        log.debug(f"Action: {action}")
        return action
        
    async def get_grasp(self, *args):
        grasp, grasp_sequence = await self.grasp.get_grasp()
        self.__grasp_sequence = grasp_sequence
        return grasp

    async def do_grasp(self, *args):
        pos, ori = await self.grasp.get_grasp()
        # convert to gripper orientation
        desired_gripper_ori = [0, 0, ori]
        pass

    def get_grasp_sequence(self):
        return self.__grasp_sequence
    
    def get_grasp_pose(self):
        return self.__grasp_sequence[1]

    def go_to_grasp_orientation(self, *args):
        grasp_pose = self.__grasp_sequence[1]
        grasp_ori = [0,0,grasp_pose[1] - np.pi/2]
        return self.go_to_orientation(grasp_ori)
    
    def go_to_grasp_position(self, *args):
        grasp_pose = copy.deepcopy(self.__grasp_sequence[1])
        grasp_position = grasp_pose[0]
        grasp_position[2] += 0.05
        return self.go_to_position(grasp_position)
    
    def go_to_pre_grasp(self, *args):
        gripper_pos = self.get_gripper_position()
        grasp_pose = self.__grasp_sequence[1]
        pre_grasp_pos = [grasp_pose[0][0], grasp_pose[0][1], gripper_pos[2]]
        pre_grasp_ori = [0, 0, grasp_pose[1] - np.pi/2]
        return self.go_to_pose([pre_grasp_pos, pre_grasp_ori])

    def get_gripper_position(self):
        gripper=self.env.robots[0].gripper
        gripper_pos = self.env.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        return gripper_pos
    
    def get_gripper_orientation(self):
        gripper_ori = self.robosim.env._eef_xmat
        log.debug(f"Gripper Orientation [mat]: {gripper_ori}")
        gripper_ori = mat2euler(gripper_ori)
        log.debug(f"Gripper Orientation [euler]: {gripper_ori}")
        return gripper_ori
    
    def distance_to_position(self, position):
        log.debug(f"Position: {position}")
        gripper_pos = self.get_gripper_position()
        log.debug(f"Gripper Position: {gripper_pos}")
        dist = position - gripper_pos
        log.debug(f"Distance: {dist}")
        return dist
    
    def delta_to_orientation(self, orientation):
        log.debug(f"Orientation: {orientation}")
        gripper_ori = self.get_gripper_orientation()
        dist = orientation - gripper_ori
        # keep the orientation within 0 to 2pi
        dist = (dist + np.pi) % (2 * np.pi) - np.pi
        # dist = (dist + np.pi) % (2 * np.pi) - np.pi
        dist[0] = dist[1] = 0
        log.debug(f"Distance (orientation): {dist}")
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
        log.debug(f"Cartesian Velocities: {cartesian_velocities}")
        action =[*cartesian_velocities, 0, 0, 0, 0]
        log.debug(f"Action: {action}")
        return action
    
    def simple_orientation_control(self, orientation):
        euclidean_dist = np.linalg.norm(orientation)
        if euclidean_dist < 0.003:
            return [0, 0, 0, 0, 0, 0, 0]
        vel_scaler = 1/max(0.1, euclidean_dist)
        cartesian_velocities = orientation * vel_scaler
        action =[0, 0, 0, *cartesian_velocities, 0]
        return action
    
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
   