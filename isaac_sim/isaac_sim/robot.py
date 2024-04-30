from enum import Enum
import numpy as np
import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def distance_between_quaternions(q1, q2):
    q1 = np.array(q1)
    q2 = np.array(q2)
        
    dot_product = np.dot(q1, q2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle = 2 * np.arccos(np.abs(dot_product))

    angle = min(angle, np.pi - angle)

    return angle

class RobotStatus(Enum):
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3

class RobotActor:
    POS_TOLERANCE = 0.01
    ORI_TOLERANCE = 0.021 # radians, ~1.2deg
    def __init__(self, world, robot, controller, articulator):
        self.world = world
        self.robot = robot
        self.controller = controller
        self.articulator = articulator
        self.__goal_pos = None
        self.__goal_ori = None
        self.__pose_history = []
        self.__moving_timeout = 3.0
    
    def _get_end_effector_pose(self):
        return self.robot.end_effector.get_world_pose()
    
    def _goal_marker(self):
        marker_cube = self.world.scene.get_object("marker_cube")
        return marker_cube
    
    def _check_if_goal_reached(self, pos, ori):
        current_pos, current_ori = self._get_end_effector_pose()
        pos_diff = np.linalg.norm(current_pos - pos)
        ori_diff = distance_between_quaternions(current_ori, ori)
        logger.warn(f"Position difference: {pos_diff}, Orientation difference: {ori_diff}")
        return pos_diff < self.POS_TOLERANCE and ori_diff < self.ORI_TOLERANCE
    
    def _is_moving(self):
        if len(self.__pose_history) == 0:
            self.__pose_history = [time.time(), *self._get_end_effector_pose()]
            return True
        
        dt = time.time() - self.__pose_history[0]
        if dt < self.__moving_timeout:
            return True
        
        current_pos, current_ori = self._get_end_effector_pose()
        delta_pos = np.linalg.norm(current_pos - self.__pose_history[1])
        logger.warn(f"  delta_pos: {delta_pos}")
        
        delta_ori = distance_between_quaternions(current_ori, self.__pose_history[2])
        logger.warn(f"  delta_ori: {delta_ori}")
        
        if delta_pos < self.POS_TOLERANCE and delta_ori < self.ORI_TOLERANCE:
            self.__pose_history = []
            return False

        self.__pose_history = [time.time(), *self._get_end_effector_pose()]
        return True      

    def move(self, pos, ori):
        if self.__goal_pos is None:
            self.__goal_pos = pos
        if self.__goal_ori is None:
            self.__goal_ori = ori

        goal_marker = self._goal_marker()
        goal_marker.set_world_pose(position=pos, orientation=ori)

        if self._check_if_goal_reached(pos, ori):
            logger.warn("Goal reached")
            self.__goal_pos = None
            self.__goal_ori = None
            return RobotStatus.COMPLETED

        if not self._is_moving():
            logger.error("Robot not moving")
            self.__goal_pos = None
            self.__goal_ori = None
            return RobotStatus.FAILED

        actions = self.controller.forward(
            target_end_effector_position=pos,
            target_end_effector_orientation=ori,
        )
        logger.debug(f"Actions: {actions}")
        self.articulator.apply_action(actions)
        return RobotStatus.RUNNING

    def move_pos(self, pos):
        if self.__goal_ori is None:
            self.__goal_ori = self._get_end_effector_pose()[1]
        return self.move(pos, self.__goal_ori)

    def move_pos_relative(self, rel_pos = np.array([0.01, 0.01, 0.01])):
        pos, ori = self._get_end_effector_pose()
        if self.__goal_pos is None:
            self.__goal_pos = pos + rel_pos
        return self.move_pos(self.__goal_pos)
    
    def move_to_preset(self, preset_name="pick_center"):
        robot_presets = {
            "pick_center": (np.array([0.0, -0.25, 0.85]), np.array([0.0, -0.707, 0.707, 0.0])),
        }
        pos, ori = robot_presets[preset_name]
        return self.move(pos, ori)
