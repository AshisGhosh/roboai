import numpy as np
import copy

from robosim.grasp_handler import GraspHandler
from robosuite.utils.transform_utils import euler2mat, mat2euler, mat2quat
from robosuite.utils.sim_utils import get_contacts

import logging
logging.basicConfig(level=logging.WARN)

log = logging.getLogger("robosim robot")
log.setLevel(logging.INFO)

class Robot:
    def __init__(self, robosim):
        self.robosim = robosim
        self.env = robosim.env
        self.grasp = GraspHandler(self)
        self._last_position = None
        self._last_orientation = None
        self.__goal_position = None
        self.__goal_orientation = None
        self.__grasp_sequence = None
        self.__gripper_contact_started = None
        
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
        return self.simple_velocity_control(dist)
    
    def go_to_pick_center(self, *args):
        return self.go_to_pose(pose=[-0.02, -0.27, 1.1, 0, 0, 0])
    
    def go_to_drop(self, *args):
        return self.go_to_pose(pose=[ 0.1, -0.57, 1.1, 0, 0, 0])
    
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
        
    async def get_grasp(self, object_name="Cereal", *args):
        log.debug(f"Getting grasp for object: {object_name}")
        grasp, grasp_sequence = await self.grasp.get_grasp(obj_name=object_name)
        self.__grasp_sequence = grasp_sequence
        self.robosim.move_marker(grasp_sequence[1][0])
        return grasp

    def get_grasp_sequence(self):
        return self.__grasp_sequence
    
    def get_grasp_pose(self):
        return self.__grasp_sequence[1]

    def go_to_grasp_orientation(self, *args):
        grasp_pose = self.__grasp_sequence[1]
        grasp_ori = [0,0, grasp_pose[1]]
        return self.go_to_orientation(grasp_ori)
    
    def go_to_grasp_position(self, *args):
        grasp_pose = copy.deepcopy(self.__grasp_sequence[1])
        grasp_position = grasp_pose[0]
        grasp_position[2] -= 0.01
        return self.go_to_position(grasp_position)
    
    def go_to_pre_grasp(self, *args):
        grasp_pose = self.__grasp_sequence[1]
        pre_grasp_pos = [grasp_pose[0][0], grasp_pose[0][1], 1.05]
        pre_grasp_ori = [0, 0, grasp_pose[1]]
        return self.go_to_pose([*pre_grasp_pos, *pre_grasp_ori])

    def get_gripper_position(self):
        gripper=self.env.robots[0].gripper
        gripper_pos = copy.deepcopy(self.env.sim.data.get_site_xpos(gripper.important_sites["grip_site"]))
        return gripper_pos
    
    def get_gripper_orientation_in_world(self):
        gripper_ori = self.robosim.env._eef_xmat
        return gripper_ori
    
    def get_gripper_orientation_in_world_as_euler(self):
        gripper_ori = self.get_gripper_orientation_in_world()
        gripper_ori = mat2euler(gripper_ori, axes="rxyz")
        return gripper_ori
    
    def is_gripper_moving(self, action):
        if action[-1]:
            return True

        if self._last_position is None:
            self._last_position = [self.get_gripper_position()]
            return True
        if self._last_orientation is None:
            self._last_orientation = [self.get_gripper_orientation_in_world_as_euler()]
            return True
        
        if len(self._last_position) < 10:
            self._last_position.append(self.get_gripper_position())
            self._last_position = None
            self._last_orientation = None
            return True
        if len(self._last_orientation) < 10:
            self._last_orientation.append(self.get_gripper_orientation_in_world_as_euler())
            self._last_position = None
            self._last_orientation = None
            return True
        
        if len(self._last_position) > 10:
            self._last_position.pop(0)
        if len(self._last_orientation) > 10:
            self._last_orientation.pop(0)
        
        current_pos = self.get_gripper_position()
        current_ori = self.get_gripper_orientation_in_world_as_euler()
        delta_pos = np.linalg.norm(self._last_position[-1] - current_pos)
        delta_ori = np.linalg.norm(np.array([self._get_closest_distance(a,b) for a, b in zip(self._last_orientation[-1], current_ori)]))
        self._last_position.append(current_pos)
        self._last_orientation.append(current_ori)

        log.info(f"Delta Position: {delta_pos}, Delta Orientation: {delta_ori}")

        if delta_pos < 0.001 and delta_ori < 0.01:
            return False

        return True
        
    def distance_to_position(self, position):
        log.debug(f"    Goal Position: {position}")
        gripper_pos = self.get_gripper_position()
        log.debug(f"        Gripper Position: {gripper_pos}")
        dist = position - gripper_pos
        log.debug(f"    Distance: {dist}")
        return dist
    
    def _get_closest_distance(self, a, b):
            dist = np.remainder(a - b, 2*np.pi)
            if dist > np.pi:
                dist -= 2*np.pi
            elif dist < -np.pi:
                dist += 2*np.pi            
            return dist
    
    def delta_to_orientation(self, orientation):
        gripper_calibration_euler = [ 3.13,  0.14, -1.56 ]
        gripper_calibration = euler2mat(gripper_calibration_euler)
        gripper_calibration_quat = mat2quat(gripper_calibration)
        
        log.debug("-----")
        log.debug(f"    request:    {orientation}")
        goal_mat = euler2mat(orientation)
        goal_in_world_mat = np.dot(gripper_calibration, goal_mat)
        goal_in_world_euler = mat2euler(goal_in_world_mat, axes="rxyz")
        goal_in_world_quat = mat2quat(goal_in_world_mat)
        current_gripper_ori_mat = self.robosim.env._eef_xmat
        current_ori = mat2euler(current_gripper_ori_mat, axes="rxyz")
        current_ori_quat = mat2quat(current_gripper_ori_mat)
        
        actual_dist =np.array([self._get_closest_distance(a,b) for a, b in zip(goal_in_world_euler, current_ori)])
        
        dist = actual_dist
        dist[1] *= -1
        dist[2] *= -1

        if self.__goal_orientation is None:
            self.__goal_orientation = goal_in_world_euler
            marker_position = self.__goal_position if self.__goal_position is not None else self.get_gripper_position()
            self.robosim.move_marker(name="gripper_goal", orientation=goal_in_world_euler, position=marker_position)

        log.debug(f"    Gripper Calibration: {gripper_calibration_euler}")
        log.debug(f"    Goal in world: {goal_in_world_euler}")
        log.debug(f"    Current in world: {current_ori}")
        log.debug(" ")
        log.debug(f"    Gripper Calibration [quat]:   {gripper_calibration_quat}")
        log.debug(f"    Goal in world [quat]:         {goal_in_world_quat}")
        log.debug(f"    Current in world [quat]:      {current_ori_quat}")

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
        if euclidean_dist < 0.01:
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
        # if euclidean_dist < 0.05:
        #     max_vel = 0.02
        cartesian_velocities = orientation / euclidean_dist
        cartesian_velocities = np.clip(cartesian_velocities, -max_vel, max_vel)
        for i in range(3):
            if abs(orientation[i]) < 0.02: # ~ 1 degree threshold
                cartesian_velocities[i] = 0
        action =[0, 0, 0, *cartesian_velocities, 0]
        log.debug(f"RPY Action: {action} (euclidean_dist: {euclidean_dist})")
        return action        
    
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

        if self._is_gripper_closed():
            return [0, 0, 0, 0, 0, 0, 0]
        return [0, 0, 0, 0, 0, 0, 1]

    def _is_gripper_closed(self):
        gripper = self.env.robots[0].gripper
        gripper_contacts = get_contacts(self.robosim.env.sim, gripper)
        log.info(f"Gripper contacts: {gripper_contacts}")

        right_fingerpad_pos = self.env.sim.data.get_geom_xpos(gripper.important_geoms["right_fingerpad"][0])
        left_fingerpad_pos = self.env.sim.data.get_geom_xpos(gripper.important_geoms["left_fingerpad"][0])
        log.debug(f"     Right fingerpad position: {right_fingerpad_pos}")
        log.debug(f"     Left fingerpad position: {left_fingerpad_pos}")
        distance = np.linalg.norm(right_fingerpad_pos - left_fingerpad_pos)
        log.debug(f"     Distance: {distance}")

        if gripper_contacts:
            if self.__gripper_contact_started is None:
                self.__gripper_contact_started = [left_fingerpad_pos, right_fingerpad_pos]
            else:
                if np.linalg.norm(self.__gripper_contact_started[0] - left_fingerpad_pos) > 0.01 and np.linalg.norm(self.__gripper_contact_started[1] - right_fingerpad_pos) > 0.01:
                    return False
                return True
        else:
            self.__gripper_contact_started = None
        
        if distance < 0.01:
            return True
        return False
        
    
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
