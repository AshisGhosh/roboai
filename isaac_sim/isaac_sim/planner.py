from PIL import Image
from roboai.shared.utils.grasp_client import get_grasp_from_image

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
        img = self.sim_manager.get_image(camera_name='realsense')
        image = Image.fromarray(img)
        grasp = get_grasp_from_image(image)
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