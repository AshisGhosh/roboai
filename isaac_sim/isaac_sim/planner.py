import numpy as np
from PIL import Image
from roboai.enums import CameraMode
from roboai.shared.utils.grasp_client import get_grasp_from_image
from roboai.shared.utils.robotic_grasping_client import get_grasps_from_rgb_and_depth

class Planner:
    def __init__(self, sim_manager, robot_actor):
        self.sim_manager = sim_manager
        self.robot_actor = robot_actor
    
    def clean_plan(self):
        '''
        Get and image of the scene and get a plan to clean the environment
        '''
        pass
    
    def grasp_plan(self):
        '''
        Get an image of the scene and get a plan to grasp an object
        '''
        img = self.sim_manager.get_image(camera_name='realsense', mode=CameraMode.RGB)
        image = Image.fromarray(img)

        depth = self.sim_manager.get_image(camera_name='realsense', mode=CameraMode.DEPTH)
        squeezed_depth = np.squeeze(depth)
        normalized_depth = (squeezed_depth - np.min(squeezed_depth)) / (np.max(squeezed_depth) - np.min(squeezed_depth)) * 255
        depth_uint8 = normalized_depth.astype(np.uint8)
        depth_image = Image.fromarray(depth_uint8)
        grasps = get_grasps_from_rgb_and_depth(image, depth_image)
        grasp = grasps[0]
        print(grasp)
        # grasp = get_grasp_from_image(image)
        return grasp


        

class GraspHandler:
    def __init__(self, sim_manager, robot_actor):
        self.sim_manager = sim_manager
        self.robot_actor = robot_actor
    
    def grasp_object(self, object_name):
        '''
        Grasp an object in the scene
        '''
        pass
    
    def release_object(self, object_name):
        '''
        Release an object in the scene
        '''
        pass