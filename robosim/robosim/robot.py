import numpy as np
from PIL import Image
import asyncio
import copy

from robosim.grasp_handler import GraspHandler
from robosuite.utils.transform_utils import euler2mat, mat2euler, mat2quat, _AXES2TUPLE
from robosuite.utils.sim_utils import get_contacts

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
        self.__goal_orientation = None
        self.__grasp_sequence = None
        
    def go_to_position(self, position):
        if self.__goal_position is None:
            self.__goal_position = position

        marker_orientation = self.__goal_orientation if self.__goal_orientation is not None else self.get_gripper_orientation_in_world_as_euler()
        self.robosim.move_marker(name="gripper_goal", position=self.__goal_position, orientation=marker_orientation)

        if len(position) != 3:
            raise ValueError("Position must be a 3D point.")
        dist = self.distance_to_position(position)
        # log.debug(f"Distance: {dist}")
        action = self.simple_velocity_control(dist)
        if (action[:-1] == [0, 0, 0, 0, 0, 0]):
            self.__goal_position = None
        return action
    
    def go_to_relative_position(self, position, frame="gripper"):
        if len(position) != 3:
            raise ValueError("Position must be a 3D point.")
        if frame != "gripper":
            raise NotImplementedError("Only gripper frame is supported for now.")
        
        if self.__goal_position is None:
            self.__goal_position = self.get_gripper_position() + np.array(position)
        self.robosim.move_marker(name="gripper_goal", position=self.__goal_position, orientation=mat2quat(self.get_gripper_orientation_in_world()))
        dist = self.distance_to_position(self.__goal_position)
        # log.debug(f"Distance: {dist}")
        return self.simple_velocity_control(dist)
    
    def go_to_pick_center(self, *args):
        # return self.go_to_position(position=[-0.02, -0.27, 1.05])
        return self.go_to_pose(pose=[-0.02, -0.27, 1.05, -3.0, 0.005, -1.57])
    
    def go_to_orientation(self, orientation, roll_only=False):
        if len(orientation) != 3:
            raise ValueError("Orientation must be a 3D rotation.")

        dist = self.delta_to_orientation(orientation)
        log.debug(f"Distance (orientation): {dist}")
        if roll_only:
            dist[0] = dist[1] = 0
            log.debug(f"Distance (roll only): {dist}")
        
        action = self.simple_orientation_control(dist)
        if (action[:-1] == [0, 0, 0, 0, 0, 0]):
            self.__goal_orientation = None
        return action
    
    def go_to_pose(self, pose, gripper=0):
        position = pose[:3]
        orientation = pose[3:]
        
        if self.__goal_position is None:
            self.__goal_position = position

        dist = self.distance_to_position(position)
        position_action = self.simple_velocity_control(dist)[:3]

        dist = self.delta_to_orientation(orientation)
        orientation_action = self.simple_orientation_control(dist)[3:-1]

        if (position_action == [0, 0, 0]) and (orientation_action == [0, 0, 0]):
            self.__goal_position = None
            self.__goal_orientation = None
        
        action = [*position_action, *orientation_action, gripper]
        log.debug(f"ACTION: {action}")
        return action
        
    async def get_grasp(self, *args):
        grasp, grasp_sequence = await self.grasp.get_grasp()
        self.__grasp_sequence = grasp_sequence
        self.robosim.move_marker(grasp_sequence[1][0])
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
        current_ori = self.get_gripper_orientation_as_euler()
        grasp_ori = [current_ori[0], current_ori[1], grasp_pose[1] - np.pi/2]
        return self.go_to_orientation(grasp_ori)
    
    def go_to_grasp_position(self, *args):
        grasp_pose = copy.deepcopy(self.__grasp_sequence[1])
        grasp_position = grasp_pose[0]
        grasp_position[2] -= 0.02
        return self.go_to_position(grasp_position)
    
    def go_to_pre_grasp(self, *args):
        # gripper_pos = self.get_gripper_position()
        # eef_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        grasp_pose = self.__grasp_sequence[1]
        pre_grasp_pos = [grasp_pose[0][0], grasp_pose[0][1], 1.05]
        pre_grasp_ori = [-3.0, 0.005, grasp_pose[1]]
        return self.go_to_pose([*pre_grasp_pos, *pre_grasp_ori])

    def get_gripper_position(self):
        gripper=self.env.robots[0].gripper
        gripper_pos = self.env.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        return gripper_pos
    
    def get_gripper_orientation_in_world(self):
        gripper_ori = self.robosim.env._eef_xmat
        # log.debug(f"Gripper Orientation [mat]: {gripper_ori}")
        # gripper_ori = mat2euler(gripper_ori)
        return gripper_ori
    
    def get_gripper_orientation_in_world_as_euler(self):
        gripper_ori = self.get_gripper_orientation_in_world()
        gripper_ori = mat2euler(gripper_ori, axes="rxyz")
        return gripper_ori
    
    def get_gripper_orientation(self):
        gripper_ori = self.get_gripper_orientation_in_world()
        gripper_ori = np.transpose(gripper_ori)
        return gripper_ori
    
    def get_gripper_orientation_as_euler(self):
        gripper_ori = self.get_gripper_orientation()
        gripper_ori = mat2euler(gripper_ori, axes="rxyz")
        return gripper_ori
        
    def distance_to_position(self, position):
        log.debug(f"    Goal Position: {position}")
        gripper_pos = self.get_gripper_position()
        log.debug(f"        Gripper Position: {gripper_pos}")
        dist = position - gripper_pos
        log.debug(f"    Distance: {dist}")
        return dist
    
    def delta_to_orientation(self, orientation):
    
        axis = "rxyz"
        gripper_ori_world_mat = self.get_gripper_orientation_in_world()
        gripper_ori_world_euler = mat2euler(gripper_ori_world_mat, axes=axis)
        
        # distances flip from -pi to pi, so we need to correct that
        def get_closest_distance(a, b):
            dist = np.remainder(a - b, 2*np.pi)
            if dist > np.pi:
                dist -= 2*np.pi
            elif dist < -np.pi:
                dist += 2*np.pi            
            return dist

        # Calculate the rotation matrix for the desired yaw change
        # goal_change_euler = [0, 0, desired_yaw_change]
        log.debug(f"-----")
        log.debug(f"    Goal Orientation: {orientation}")

        gripper_ori_gripper_mat = np.transpose(gripper_ori_world_mat)
        gripper_ori_gripper_euler = mat2euler(gripper_ori_gripper_mat, axes=axis)
        goal_change_in_gripper_euler = np.array([get_closest_distance(a, b) for a, b in zip(orientation, gripper_ori_gripper_euler)])
        # log.debug(f"    Goal in gripper:")
        # log.debug(f"        Gripper in gripper [euler]:     {gripper_ori_gripper_euler}")
        # log.debug(f"        Goal in gripper [euler]:        {orientation}")
        # log.debug(f"        dist:                           {goal_change_in_gripper_euler}")

        goal_change_in_gripper_mat = euler2mat(goal_change_in_gripper_euler)
        goal_in_world_mat = np.dot(gripper_ori_world_mat, goal_change_in_gripper_mat)

        if self.__goal_orientation is None:
            goal_in_world_euler = mat2euler(goal_in_world_mat, axes=axis)
        else:
            goal_in_world_euler = self.__goal_orientation 

        goal_change_in_world_euler = np.array([get_closest_distance(a, b) for a, b in zip(goal_in_world_euler, gripper_ori_world_euler)])
        log.debug(f"    Goal in world:")        
        log.debug(f"        Gripper in world [euler]:       {gripper_ori_world_euler}")
        log.debug(f"        Goal in world [euler]:          {goal_in_world_euler}")
        log.debug(f"        dist:                           {goal_change_in_world_euler}")


        direct_change_in_world_euler = np.array([get_closest_distance(a, b) for a, b in zip(orientation, gripper_ori_world_euler)])
        # log.debug(f"    Direct goal in world:")
        # log.debug(f"        Gripper in world [euler]:       {gripper_ori_world_euler}")
        # log.debug(f"        Goal in world [euler]:          {orientation}")
        # log.debug(f"        dist:                           {direct_change_in_world_euler}")

        marker_orientation = goal_in_world_euler
        if self.__goal_orientation is None:
            self.__goal_orientation = goal_in_world_euler
        marker_position = self.__goal_position if self.__goal_position is not None else self.get_gripper_position()
        self.robosim.move_marker(name="gripper_goal", orientation=marker_orientation, position=marker_position)

        dist = goal_change_in_world_euler
        dist[1] *= -1
        dist[2] *= -1
        log.debug(f"    Distance (orientation): {dist}")
        log.debug(f"------")
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
        log.debug(f"    Cartesian Velocities: {cartesian_velocities}")
        action =[*cartesian_velocities, 0, 0, 0, 0]
        log.debug(f"XYZ Action: {action}")
        return action
    
    def simple_orientation_control(self, orientation):
        euclidean_dist = np.linalg.norm(orientation)
        if euclidean_dist < 0.02:
            return [0, 0, 0, 0, 0, 0, 0]
        
        max_vel = 0.4
        if euclidean_dist < 0.4:
            max_vel = 0.1
        if euclidean_dist < 0.2:
            max_vel = 0.05
        if euclidean_dist < 0.05:
            max_vel = 0.02
        cartesian_velocities = orientation / euclidean_dist
        cartesian_velocities = np.clip(cartesian_velocities, -max_vel, max_vel)
        for i in range(3):
            if abs(orientation[i]) < 0.02: # ~ 1 degree threshold
                cartesian_velocities[i] = 0
        action =[0, 0, 0, *cartesian_velocities, 0]
        log.debug(f"RPY Action: {action} (euclidean_dist: {euclidean_dist})")
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
        
    
    def close_gripper(self, *args):
        # get current gripper position
        gripper = self.env.robots[0].gripper
        gripper_contacts = get_contacts(self.robosim.env.sim, gripper)
        log.info(f"Gripper contacts: {gripper_contacts}")

        right_fingerpad_pos = self.env.sim.data.get_geom_xpos(gripper.important_geoms["right_fingerpad"][0])
        left_fingerpad_pos = self.env.sim.data.get_geom_xpos(gripper.important_geoms["left_fingerpad"][0])
        log.debug(f"     Right fingerpad position: {right_fingerpad_pos}")
        log.debug(f"     Left fingerpad position: {left_fingerpad_pos}")
        distance = np.linalg.norm(right_fingerpad_pos - left_fingerpad_pos)
        log.debug(f"     Distance: {distance}")

        if gripper_contacts or distance < 0.01:
            return [0, 0, 0, 0, 0, 0, 0]
        return [0, 0, 0, 0, 0, 0, 1]
    
    def open_gripper(self, *args):
        gripper = self.env.robots[0].gripper
        right_fingerpad_pos = self.env.sim.data.get_geom_xpos(gripper.important_geoms["right_fingerpad"][0])
        left_fingerpad_pos = self.env.sim.data.get_geom_xpos(gripper.important_geoms["left_fingerpad"][0])
        log.debug(f"     Right fingerpad position: {right_fingerpad_pos}")
        log.debug(f"     Left fingerpad position: {left_fingerpad_pos}")
        distance = np.linalg.norm(right_fingerpad_pos - left_fingerpad_pos)
        log.debug(f"     Distance: {distance}")

        if distance > 0.08:
            return [0, 0, 0, 0, 0, 0, 0]
        return [0, 0, 0, 0, 0, 0, -1]