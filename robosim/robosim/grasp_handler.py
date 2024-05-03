import base64
import numpy as np
from PIL import Image
import cv2
from enum import Enum

from robosim.camera import Camera

from shared.utils.grasp_client import _check_server, _get_grasp_from_image
from shared.utils.robotic_grasping_client import _get_grasps_from_rgb_and_depth
import shared.utils.llm_utils as llm_utils

import logging
log = logging.getLogger("robosim robot grasp")
log.setLevel(logging.DEBUG)

class GraspMethod(Enum):
    GRASP_DET_SEG = "grasp_det_seg"
    GR_CONVNET = "gr_convnet"

class Grasp:
    def __init__(self, r_bbox, image, depth, env, bbox=None, cls=None, cls_name=None, score=None):
        log.debug("Initializing Grasp object.")
        self.r_bbox = r_bbox
        self.image = image
        self.depth = depth
        self.env = env
        log.debug("Initializing camera")
        self.camera = Camera(self.env, "robot0_eye_in_hand")

        self.cls = cls
        self.cls_name = cls_name
        self.score = score
        self.bbox = bbox

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
        # pixels work in y, x not x, y
        center = int(np.mean([coord[1] for coord in self.r_bbox])), int(np.mean([coord[0] for coord in self.r_bbox]))
        log.debug(f"Center of the bounding box: {center}")
        # Get the world coordinates of the center
        log.debug(f"{np.array(center).shape} -- {np.array(self.depth).shape}")
        world_coords = self.camera.get_world_coords_from_pixels(np.array(center), np.array(self.depth))
        log.debug(f"World coordinates of the center: {world_coords}")
        self.grasp_postion = world_coords

        # Get grasp orientation
        # Get the angle from the bounding box
        pt1 = self.r_bbox[0]
        pt2 = self.r_bbox[1]

        angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) + np.pi / 2
        self.grasp_orientation = angle
        log.debug(f"Grasp orientation: {angle}")

        return world_coords, angle

    def __str__(self):
        return f"Grasp: {self.cls_name} with score {self.score} at r_bbox {self.r_bbox}"

class GraspHandler:
    def __init__(self, robot):
        self.robot = robot
        self.env = robot.robosim.env

    async def get_grasps_from_image(self, image: Image, visualize=True):
        res = await _get_grasp_from_image(image)
        if visualize:
            self.show_image(res["image"])
        return res["result"]
    
    async def get_grasp_image(self) -> Image:
        # turn off marker visualization
        markers = ["gripper0_grip_site", "gripper0_grip_site_cylinder", "gripper_goal", "grasp_marker"]
        for marker in markers:
            self.env.sim.model.site_rgba[self.env.sim.model.site_name2id(marker)][3] = 0

        im = self.env._get_observations()["robot0_eye_in_hand_image"]
        img = Image.fromarray(im[::-1])

        # turn on marker visualization
        for marker in markers:
            self.env.sim.model.site_rgba[self.env.sim.model.site_name2id(marker)][3] = 0.25

        return img
    
    async def get_grasp_image_and_depth(self):
        # turn off marker visualization
        markers = ["gripper0_grip_site", "gripper0_grip_site_cylinder", "gripper_goal", "grasp_marker"]
        for marker in markers:
            self.env.sim.model.site_rgba[self.robot.robosim.env.sim.model.site_name2id(marker)][3] = 0
        
        self.env.step(np.zeros(self.env.action_dim))
        im = self.env._get_observations()
        img = Image.fromarray(im["robot0_eye_in_hand_image"][::-1])
        depth = im["robot0_eye_in_hand_depth"][::-1]

        # turn on marker visualization
        for marker in markers:
            self.env.sim.model.site_rgba[self.env.sim.model.site_name2id(marker)][3] = 0.25

        return img, depth
    
    async def get_grasp_image_and_depth_image(self):
        img, depth = await self.get_grasp_image_and_depth()
        squeezed_depth = np.squeeze(depth)
        normalized_depth = (squeezed_depth - np.min(squeezed_depth)) / (np.max(squeezed_depth) - np.min(squeezed_depth)) * 255
        depth_uint8 = normalized_depth.astype(np.uint8)
        depth_image = Image.fromarray(depth_uint8)
        return img, depth_image, depth
    
    async def get_grasp(self, obj_name, method=GraspMethod.GRASP_DET_SEG):
        if method == GraspMethod.GRASP_DET_SEG:
            log.debug("Getting grasp from grasp_det_seg...")
            return await self.get_grasp_grasp_det_seg(obj_name)
        elif method == GraspMethod.GR_CONVNET:
            log.debug("Getting grasp from grasp convnet...")
            return await self.get_grasp_gr_convnet(obj_name)
        else:
            raise ValueError(f"Invalid grasp method: {method}")

    async def get_grasp_grasp_det_seg(self, obj_name):
        # return await self.get_grasps()
        log.debug("Getting grasp image and depth...")
        img, depth = await self.get_grasp_image_and_depth()
        log.debug("Getting grasp from image...")
        grasps = await self.get_grasps_from_image(img)
        if len(grasps) == 0:
            log.error("No grasps found.")
            return None

        candidate_objs = [obj["cls_name"].replace("_", " ") for obj in grasps]
        log.info(f"Getting closest object to '{obj_name}' from {candidate_objs}")
        closest_obj = await llm_utils.get_closest_text(obj_name, candidate_objs)
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
    
    async def get_grasp_gr_convnet(self, obj_name):
        log.debug("Getting grasp image and depth...")
        img, depth_image, depth = await self.get_grasp_image_and_depth_image()
        log.debug("Getting grasp from image...")
        grasps = await _get_grasps_from_rgb_and_depth(img, depth_image)
        grasp = grasps[0]
        log.debug(f"r_bbox: {grasp['r_bbox']}")
        g_obj = Grasp(
            cls=None,
            cls_name=None,
            score=None,
            bbox=None,
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

