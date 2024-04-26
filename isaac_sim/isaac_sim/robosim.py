import sys
import time

import carb
import omni
import numpy as np
from omni.isaac.kit import SimulationApp

from roboai.robot import RobotActor
from roboai.tasks import TaskManager
from roboai.planner import Planner

global World, Robot, Franka, nucleus, stage, prims, rotations, viewports, Gf, UsdGeom, ArticulationMotionPolicy, RMPFlowController

WORLD_STAGE_PATH = "/World"
FRANKA_STAGE_PATH = WORLD_STAGE_PATH + "/Franka"
FRANKA_USD_PATH = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
CAMERA_PRIM_PATH = f"{FRANKA_STAGE_PATH}/panda_hand/geometry/realsense/realsense_camera"
BACKGROUND_STAGE_PATH = WORLD_STAGE_PATH + "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"
GRAPH_PATH = "/ActionGraph"
REALSENSE_VIEWPORT_NAME = "realsense_viewport"

class SimManager:
    def __init__(self):
        self.sim = None
        self.assets_root_path = None
        self.world = None
        self.cameras = {}
        self.robot_actor = None
        self.task_manager = None
    
    def start_sim(self, headless=True):
        CONFIG = {
            "renderer": "RayTracedLighting", 
            "headless": headless,
            "window_width":   2560,
            "window_height":  1440
        }

        start_time = time.time()
        self.sim = SimulationApp(CONFIG)
        carb.log_warn(f"Time taken to load simulation: {time.time() - start_time} seconds")
        if headless:
            self.sim.set_setting("/app/window/drawMouse", True)
            self.sim.set_setting("/app/livestream/proto", "ws")
            self.sim.set_setting("/app/livestream/websocket/framerate_limit", 120)
            self.sim.set_setting("/ngx/enabled", False)

            from omni.isaac.core.utils.extensions import enable_extension
            enable_extension("omni.kit.livestream.native")

        self._do_imports()
        
        if self.assets_root_path is None:
            self.assets_root_path = self._get_assets_root_path()
        
        start_time = time.time()
        self.world = World(stage_units_in_meters=1.0)
        carb.log_warn(f"Time taken to create world: {time.time() - start_time} seconds")

        self._load_stage()
        self._load_robot()
        self._load_objects()
        self._create_markers()
        self._init_cameras()

        franka = self.world.scene.get_object("franka")
        controller = RMPFlowController(name="target_follower_controller", robot_articulation=franka)
        articulation_controller = franka.get_articulation_controller()

        self.robot_actor = RobotActor(world=self.world, robot=franka, controller=controller, articulator=articulation_controller)
        self.task_manager = TaskManager(sim_manager=self, robot_actor=self.robot_actor)
        self.planner = Planner(sim_manager=self, robot_actor=self.robot_actor)

    def _do_imports(self):
        global World, Robot, Franka, nucleus, stage, prims, rotations, viewports, Gf, UsdGeom, ArticulationMotionPolicy, RMPFlowController
        
        from omni.isaac.core import World
        from omni.isaac.core.robots import Robot
        from omni.isaac.franka import Franka
        from omni.isaac.core.utils import (
            nucleus,
            stage,
            prims,
            rotations,
            viewports
        )
        from pxr import Gf, UsdGeom  # noqa E402
        from omni.isaac.motion_generation.articulation_motion_policy import ArticulationMotionPolicy
        from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController

    def _get_assets_root_path(self):
        start_time = time.time()
        assets_root_path = nucleus.get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            sim.close()
            sys.exit()
        carb.log_warn(f"Time taken to get assets root path: {time.time() - start_time} seconds")
        return assets_root_path
    
    def _load_stage(self):
        start_time = time.time()
        stage.add_reference_to_stage(
            self.assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH
        )
        carb.log_warn(f"Time taken to add reference to stage: {time.time() - start_time} seconds")

    def _load_robot(self):
        start_time = time.time()
        stage.add_reference_to_stage(
            self.assets_root_path + FRANKA_USD_PATH, FRANKA_STAGE_PATH
        )
        self.world.scene.add(
                Franka(
                    prim_path=FRANKA_STAGE_PATH, 
                    name="franka",
                    position=np.array([0, -0.64, 0]),
                    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90))
                )
            )
        carb.log_warn(f"Time taken to create Franka: {time.time() - start_time} seconds")
        self.sim.update()
    
    def _load_objects(self):
        start_time = time.time()
        prims.create_prim(
            WORLD_STAGE_PATH + "/cracker_box",
            "Xform",
            position=np.array([-0.2, -0.25, 0.15]),
            orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
            usd_path=self.assets_root_path
            + "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        )
        prims.create_prim(
            WORLD_STAGE_PATH + "/sugar_box",
            "Xform",
            position=np.array([-0.07, -0.25, 0.1]),
            orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)),
            usd_path=self.assets_root_path
            + "/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        )
        prims.create_prim(
            WORLD_STAGE_PATH + "/soup_can",
            "Xform",
            position=np.array([0.1, -0.25, 0.10]),
            orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
            usd_path=self.assets_root_path
            + "/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        )
        prims.create_prim(
            WORLD_STAGE_PATH + "/mustard_bottle",
            "Xform",
            position=np.array([0.0, 0.15, 0.12]),
            orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
            usd_path=self.assets_root_path
            + "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        )
        carb.log_warn(f"Time taken to create prims: {time.time() - start_time} seconds")
        self.sim.update()
    
    def _create_markers(self):
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.materials import OmniGlass
        marker_cube = self.world.scene.add(
            VisualCuboid(
                prim_path=WORLD_STAGE_PATH + "/marker_cube",
                name="marker_cube",
                scale=np.array([0.025, 0.2, 0.05]),
                position=np.array([0.0, 0.0, 0.0]),
                color=np.array([1.0, 0.0, 0.0]),
            )
        )
        material = OmniGlass(
            prim_path="/World/material/glass",  # path to the material prim to create
            ior=1.25,
            depth=0.001,
            thin_walled=True,
            color=np.array([1.5, 0.0, 0.0])
        )
        marker_cube.apply_visual_material(material)
        self.sim.update()
    
    def _init_cameras(self):
        viewports.create_viewport_for_camera(REALSENSE_VIEWPORT_NAME, CAMERA_PRIM_PATH)

        realsense_prim = camera_prim = UsdGeom.Camera(
            stage.get_current_stage().GetPrimAtPath(CAMERA_PRIM_PATH)
        )
        realsense_prim.GetHorizontalApertureAttr().Set(20.955)
        realsense_prim.GetVerticalApertureAttr().Set(15.7)
        realsense_prim.GetFocalLengthAttr().Set(18.8)
        realsense_prim.GetFocusDistanceAttr().Set(400)

        viewport = omni.ui.Workspace.get_window("Viewport")
        rs_viewport = omni.ui.Workspace.get_window(REALSENSE_VIEWPORT_NAME)
        rs_viewport.dock_in(viewport, omni.ui.DockPosition.RIGHT, ratio=0.3)
        carb.log_warn(f"{REALSENSE_VIEWPORT_NAME} docked in {viewport.title}: {rs_viewport.docked}")

        from omni.isaac.sensor import Camera
        self.cameras["realsense"] = Camera(
            prim_path=CAMERA_PRIM_PATH,
            name="realsense",
            resolution=(640, 480),
            )

        camera_rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), -90) * Gf.Rotation(Gf.Vec3d(1, 0, 0), 45)
        self.cameras["agentview"] = Camera(
            prim_path="/World/camera",
            name="agentview",
            resolution=(1024, 768),
            position=(0, 2.75, 2.67),
            orientation=rotations.gf_rotation_to_np_array(camera_rot)
            )        
        # self.cameras["agentview"].set_world_pose(
        #     position=(0, 2.75, 2.67),
        #     orientation=(0.00061, 0.0032, 0.38051, 0.92477)
        # )
        
        self.world.reset()
        for cam in self.cameras.values():
            cam.initialize()
    
    def close_sim(self):
        self.sim.close()       

    def run_sim(self):
        while self.sim.is_running():
            self.world.step(render=True)
            if self.world.is_playing():
                if self.world.current_time_step_index == 0:
                    self.world.reset()
                    controller.reset()
                self.task_manager.do_tasks()
            self.sim.update()
    
    def get_image(self, camera_name="realsense", rgba=False, visualize=False):
        self.world.step(render=True)
        camera = self.cameras[camera_name]
        try:
            if rgba: 
                img =  camera.get_rgba()
            else:
                img = camera.get_rgb()
            if visualize:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Color", img)
                cv2.waitKey(0)
            return img
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

if __name__ == "__main__":
    sim_manager = SimManager()
    sim_manager.start_sim(headless=False)
    sim_manager.task_manager.test_task()
    sim_manager.world.pause()
    sim_manager.run_sim()
    # sim_manager.get_image()
    sim_manager.close_sim()