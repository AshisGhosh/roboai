import asyncio
import base64
import io
import numpy as np
from PIL import Image
import cv2

from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world

from shared.utils.grasp_client import _check_server, _get_grasp_from_image
import shared.utils.llm_utils as llm_utils

import logging
log = logging.getLogger("robosim robot grasp")
log.setLevel(logging.DEBUG)

class Camera:
    def __init__(self, sim, name, camera_height=480, camera_width=640):
        self.sim = sim
        self.name = name
        self.camera_height = camera_height
        self.camera_width = camera_width
        log.debug(f"Getting intrinsic matrix for {name}")
        self.intrinsic_matrix = get_camera_intrinsic_matrix(sim, name, camera_height, camera_width)
        log.debug(f"Getting extrinsic matrix for {name}")
        self.extrinsic_matrix = get_camera_extrinsic_matrix(sim, name)
        log.debug(f"Getting transform matrix for {name}")
        self.transform_matrix = get_camera_transform_matrix(sim, name, camera_height, camera_width)
        log.debug(f"Getting camera to world transform for {name}")
        self.camera_to_world_transform = np.linalg.inv(self.transform_matrix)
        log.debug(f"Camera initialized for {name}")

    def get_world_coords_from_pixels(self, pixels, depth):
        log.debug(f"Getting world coordinates from pixels {pixels} and depth {depth.shape}")
        real_depth_map = get_real_depth_map(self.sim, depth)
        log.debug(f"Real depth map: {real_depth_map.shape}")
        log.debug(f"pixels leading shape: depth map leading shape -- {pixels.shape[:-1]} -- {real_depth_map.shape[:-3]}")
        return transform_from_pixels_to_world(pixels, real_depth_map, self.camera_to_world_transform)

class Grasp:
    def __init__(self, cls, cls_name, score, bbox, r_bbox, image, depth, env):
        self.cls = cls
        self.cls_name = cls_name
        self.score = score
        self.bbox = bbox
        self.r_bbox = r_bbox
        self.image = image
        self.depth = depth
        self.env = env
        log.debug(f"Initializing camera for {self}")
        self.camera = Camera(self.env.sim, "robot0_eye_in_hand")

        self.appoach_poses = []
        self.grasp_pose = None
        self.retract_poses = []
        log.debug(f"Generated grasp for {self}")
    
    def generate_grasp_sequence(self):
        log.info(f"Generating grasp sequence for {self}")
        self.grasp_pose = self.get_grasp_pose_from_r_bbox()
        return self.appoach_poses, self.grasp_pose, self.retract_poses
    
    def get_grasp_pose_from_r_bbox(self):
        # Get the center of the bounding box
        log.debug(f"Getting grasp pose from r_bbox: {self.r_bbox}")
        center = int(np.mean([coord[0] for coord in self.r_bbox])), int(np.mean([coord[1] for coord in self.r_bbox]))
        log.debug(f"Center of the bounding box: {center}")
        # Get the world coordinates of the center
        log.debug(f"{np.array(center).shape} -- {np.array(self.depth).shape}")
        world_coords = self.camera.get_world_coords_from_pixels(np.array(center), np.array(self.depth))
        log.debug(f"World coordinates of the center: {world_coords}")
        self.grasp_postion = world_coords
        # self.visualize_grasp(world_coords)

        # Get grasp orientation
        # Get the angle from the bounding box
        pt1 = self.r_bbox[0]
        pt2 = self.r_bbox[1]

        angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        self.grasp_orientation = angle
        log.debug(f"Grasp orientation: {angle}")

        return world_coords, angle
    
    def visualize_grasp(self, world_coords):
        target_pos = world_coords
        gripper = self.env.robots[0].gripper
        log.debug(f"Getting gripper position for {gripper}")
        gripper_pos = self.env.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        log.debug(f"Gripper position: {gripper_pos}")
        
        # color the gripper site appropriately based on (squared) distance to target
        dist = np.sum(np.square((target_pos - gripper_pos)))
        max_dist = 0.1
        scaled = (1.0 - min(dist / max_dist, 1.0)) ** 15
        rgba = np.zeros(3)
        rgba[0] = 1 - scaled
        rgba[1] = scaled
        # rgba = [0, 1, 0]
        self.env.sim.model.site_rgba[self.env.sim.model.site_name2id(gripper.important_sites["grip_site"])][:3] = rgba
        log.debug(f"Visualizing grasp for {self}")
        self.env.render()

    def __str__(self):
        return f"Grasp: {self.cls_name} with score {self.score} at bbox {self.bbox}"

class GraspHandler:
    def __init__(self, robot):
        self.robot = robot

    async def get_grasp_from_image(self, image: Image, visualize=True):
        res = await _get_grasp_from_image(image)
        if visualize:
            self.show_image(res["image"])
        return res["result"]
    
    async def get_grasp_image(self) -> Image:
        im = self.robot.robosim.env._get_observations()["robot0_eye_in_hand_image"]
        img = Image.fromarray(im[::-1])
        return img
    
    async def get_grasp_image_and_depth(self):
        im = self.robot.robosim.env._get_observations()
        img = Image.fromarray(im["robot0_eye_in_hand_image"][::-1])
        depth = im["robot0_eye_in_hand_depth"][::-1]
        return img, depth

    async def get_grasp(self):
        # return await self.get_grasps()
        img, depth = await self.get_grasp_image_and_depth()
        grasps = await self.get_grasp_from_image(img)

        obj_name = "cereal"
        candidate_objs = [obj["cls_name"].replace("_", " ") for obj in grasps]
        log.info(f"Getting closest object to {obj_name} from {candidate_objs}")
        closest_obj = llm_utils.get_closest_text(obj_name, candidate_objs)
        log.info(f"Closest object: {closest_obj}") 
        grasp = grasps[candidate_objs.index(closest_obj)]  

        g_obj = Grasp(
            cls=grasp["cls"],
            cls_name=grasp["cls_name"],
            score=grasp["obj"],
            bbox=grasp["bbox"],
            r_bbox=grasp["r_bbox"],
            image=img,
            depth=depth,
            env = self.robot.robosim.env
        )
        return grasp, g_obj.generate_grasp_sequence()
    
    async def check_server(self):
        return await _check_server()
    
    def show_image(self, base64_image):
        image_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Display the image using OpenCV
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

