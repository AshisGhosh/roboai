import numpy as np
from PIL import Image
import asyncio

from robosim.grasp_handler import GraspHandler

import logging
logging.basicConfig(level=logging.INFO)

class Robot:
    def __init__(self, robosim):
        self.robosim = robosim
        self.env = robosim.env
        self.grasp = GraspHandler(self)
        self.__goal_position = None
        
    def go_to_position(self, position):
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
    
    def go_to_pick_center(self, *args):
        return self.go_to_position(position=[-0.02, -0.27, 1.1])
    
    async def get_grasp_from_image(self, image: Image):
        return await self.grasp.get_grasp_from_image(image)
    
    async def get_grasp_image(self):
        return await self.robosim.get_grasp_image()
    
    async def get_grasp(self, *args):
        img = await self.get_grasp_image()
        return await self.get_grasp_from_image(img)
        # return await self.grasp.check_server()
    
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
        action =[*cartesian_velocities, 0, 0, 0, 0]
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
            logging.info(f"Object {obj.name}: {dist}")
   