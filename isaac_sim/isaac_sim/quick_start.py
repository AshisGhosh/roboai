import sys
import time

import carb
import omni
import numpy as np
from omni.isaac.kit import SimulationApp

WORLD_STAGE_PATH = "/World"
FRANKA_STAGE_PATH = WORLD_STAGE_PATH + "/Franka"
FRANKA_USD_PATH = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
CAMERA_PRIM_PATH = f"{FRANKA_STAGE_PATH}/panda_hand/geometry/realsense/realsense_camera"
BACKGROUND_STAGE_PATH = WORLD_STAGE_PATH + "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"
GRAPH_PATH = "/ActionGraph"
REALSENSE_VIEWPORT_NAME = "realsense_viewport"

CONFIG = {
    "renderer": "RayTracedLighting",
    "headless": False,
    "window_width": 2560,
    "window_height": 1440,
}

start_time = time.time()
sim = SimulationApp(CONFIG)
carb.log_warn(f"Time taken to load simulation: {time.time() - start_time} seconds")

from omni.isaac.core import World  # noqa E402
from omni.isaac.franka import Franka  # noqa E402
from omni.isaac.core.utils import (  # noqa E402
    nucleus,
    stage,
    prims,
    rotations,
    viewports,
)
from pxr import Gf, UsdGeom  # noqa E402
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController  # noqa E402
from omni.isaac.core.objects import VisualCuboid  # noqa E402
from omni.isaac.core.materials import OmniGlass  # noqa E402
from omni.isaac.sensor import Camera  # noqa E402

from roboai.robot import RobotActor  # noqa E402
from roboai.tasks import TaskManager  # noqa E402


# INITIALIZE WORLD
start_time = time.time()
world = World(stage_units_in_meters=1.0)
carb.log_warn(f"Time taken to create world: {time.time() - start_time} seconds")

# GET ASSET PATH
start_time = time.time()
assets_root_path = nucleus.get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    sim.close()
    sys.exit()
carb.log_warn(f"Time taken to get assets root path: {time.time() - start_time} seconds")

# LOAD STAGE/BACKGROUND
start_time = time.time()
stage.add_reference_to_stage(
    assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH
)
carb.log_warn(
    f"Time taken to add reference to stage: {time.time() - start_time} seconds"
)

# LOAD FRANKA
start_time = time.time()
stage.add_reference_to_stage(assets_root_path + FRANKA_USD_PATH, FRANKA_STAGE_PATH)
world.scene.add(Franka(prim_path=FRANKA_STAGE_PATH, name="franka"))
carb.log_warn(f"Time taken to create Franka: {time.time() - start_time} seconds")
sim.update()

# CREATE OBJECT PRIMS
start_time = time.time()
prims.create_prim(
    WORLD_STAGE_PATH + "/cracker_box",
    "Xform",
    position=np.array([-0.2, -0.25, 0.15]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
)
prims.create_prim(
    WORLD_STAGE_PATH + "/sugar_box",
    "Xform",
    position=np.array([-0.07, -0.25, 0.1]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
)
prims.create_prim(
    WORLD_STAGE_PATH + "/soup_can",
    "Xform",
    position=np.array([0.1, -0.25, 0.10]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
)
prims.create_prim(
    WORLD_STAGE_PATH + "/mustard_bottle",
    "Xform",
    position=np.array([0.0, 0.15, 0.12]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
)

carb.log_warn(f"Time taken to create prims: {time.time() - start_time} seconds")

sim.update()

# CREATE MARKER OBJECTS

marker_cube = world.scene.add(
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
    color=np.array([1.5, 0.0, 0.0]),
)
marker_cube.apply_visual_material(material)
# prims.set_prim_attribute_value(WORLD_STAGE_PATH + "/marker_cube", "primvars:displayOpacity", [0.5])
sim.update()

# CREATE REALSENSE VIEWPORT & CAMERA

viewports.create_viewport_for_camera(REALSENSE_VIEWPORT_NAME, CAMERA_PRIM_PATH)

# Fix camera settings since the defaults in the realsense model are inaccurate
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
carb.log_warn(
    f"{REALSENSE_VIEWPORT_NAME} docked in {viewport.title}: {rs_viewport.docked}"
)

camera = Camera(prim_path=CAMERA_PRIM_PATH)

world.reset()
camera.initialize()

franka = world.scene.get_object("franka")
controller = RMPFlowController(
    name="target_follower_controller", robot_articulation=franka
)
articulation_controller = franka.get_articulation_controller()

robot_actor = RobotActor(
    world=world,
    robot=franka,
    controller=controller,
    articulator=articulation_controller,
)

task_manager = TaskManager(robot_actor=robot_actor)
task_manager.test_task()

world.pause()

while sim.is_running():
    world.step(render=True)
    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()
            controller.reset()
        task_manager.do_tasks()

sim.close()
