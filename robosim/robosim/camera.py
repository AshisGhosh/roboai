
import numpy as np
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world

import logging
log = logging.getLogger("robosim robot camera")
log.setLevel(logging.DEBUG)

class Camera:
    def __init__(self, env, name, camera_height=480, camera_width=640):
        self.env = env
        self.name = name
        self.camera_height = camera_height
        self.camera_width = camera_width
        log.debug(f"Getting intrinsic matrix for {name}")
        self.intrinsic_matrix = get_camera_intrinsic_matrix(env.sim, name, camera_height, camera_width)
        log.debug(f"Getting extrinsic matrix for {name}")
        self.extrinsic_matrix = get_camera_extrinsic_matrix(env.sim, name)
        log.debug(f"Getting transform matrix for {name}")
        self.transform_matrix = get_camera_transform_matrix(env.sim, name, camera_height, camera_width)
        log.debug(f"Getting camera to world transform for {name}")
        self.camera_to_world_transform = np.linalg.inv(self.transform_matrix)
        log.debug(f"Camera initialized for {name}")

    def get_world_coords_from_pixels(self, pixels, depth):
        # cv2.imshow("Depth", depth)
        # cv2.waitKey(0)
        log.debug(f"Getting world coordinates from pixels {pixels} and depth {depth.shape}")
        real_depth_map = get_real_depth_map(self.env.sim, depth)
        log.debug(f"Real depth map: {real_depth_map.shape}")
        log.debug(f"pixels leading shape: depth map leading shape -- {pixels.shape[:-1]} -- {real_depth_map.shape[:-3]}")
        return transform_from_pixels_to_world(pixels, real_depth_map, self.camera_to_world_transform)
    
    def pixel_to_world(self, pixel):
        depth = self.env._get_observations()["robot0_eye_in_hand_depth"][::-1]
        return self.get_world_coords_from_pixels(np.array(pixel), depth)
    