import os
import sys
import time

import carb
import omni
import numpy as np
from omni.isaac.kit import SimulationApp

from roboai.enums import CameraMode
from roboai.robot import RobotActor
from roboai.tasks import TaskManager
from roboai.planner import Planner

global \
    World, \
    Robot, \
    Franka, \
    extensions, \
    nucleus, \
    stage, \
    prims, \
    rotations, \
    viewports, \
    Gf, \
    UsdGeom, \
    ArticulationMotionPolicy, \
    RMPFlowController, \
    og


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
        self.controller = None
        self.robot_actor = None
        self.task_manager = None

    def start_sim(self, headless=True):
        CONFIG = {
            "renderer": "RayTracedLighting",
            "headless": headless,
            "window_width": 2560,
            "window_height": 1440,
        }

        start_time = time.time()
        self.sim = SimulationApp(CONFIG)
        carb.log_warn(
            f"Time taken to load simulation: {time.time() - start_time} seconds"
        )
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
        # self._create_markers()
        self._init_cameras()
        self._enable_ros2_bridge_ext()
        self._load_omnigraph()

        franka = self.world.scene.get_object("franka")
        self.controller = RMPFlowController(
            name="target_follower_controller", robot_articulation=franka
        )
        articulation_controller = franka.get_articulation_controller()

        self.robot_actor = RobotActor(
            world=self.world,
            robot=franka,
            controller=self.controller,
            articulator=articulation_controller,
        )
        self.task_manager = TaskManager(sim_manager=self, robot_actor=self.robot_actor)
        self.planner = Planner(sim_manager=self, robot_actor=self.robot_actor)

    def _do_imports(self):
        global \
            World, \
            Robot, \
            Franka, \
            extensions, \
            nucleus, \
            stage, \
            prims, \
            rotations, \
            viewports, \
            Gf, \
            UsdGeom, \
            ArticulationMotionPolicy, \
            RMPFlowController, \
            og

        from omni.isaac.core import World
        from omni.isaac.core.robots import Robot
        from omni.isaac.franka import Franka
        from omni.isaac.core.utils import (
            extensions,
            nucleus,
            stage,
            prims,
            rotations,
            viewports,
        )
        from pxr import Gf, UsdGeom  # noqa E402
        from omni.isaac.motion_generation.articulation_motion_policy import (
            ArticulationMotionPolicy,
        )
        from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
        import omni.graph.core as og

    def _get_assets_root_path(self):
        start_time = time.time()
        assets_root_path = nucleus.get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.sim.close()
            sys.exit()
        carb.log_warn(
            f"Time taken to get assets root path: {time.time() - start_time} seconds"
        )
        return assets_root_path

    def _load_stage(self):
        start_time = time.time()
        stage.add_reference_to_stage(
            self.assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH
        )
        carb.log_warn(
            f"Time taken to add reference to stage: {time.time() - start_time} seconds"
        )

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
                orientation=rotations.gf_rotation_to_np_array(
                    Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)
                ),
            )
        )
        carb.log_warn(
            f"Time taken to create Franka: {time.time() - start_time} seconds"
        )
        self.sim.update()

    def __rand_position_in_bounds(self, bounds, height):
        x = np.random.uniform(bounds[0][0], bounds[1][0])
        y = np.random.uniform(bounds[0][1], bounds[1][1])
        z = height
        return np.array([x, y, z])

    def _load_objects(self):
        start_time = time.time()
        bounds = [[-0.5, -0.35], [0.5, 0.35]]
        prims.create_prim(
            WORLD_STAGE_PATH + "/cracker_box",
            "Xform",
            # position=np.array([-0.2, -0.25, 0.15]),
            position=self.__rand_position_in_bounds(bounds, 0.15),
            orientation=rotations.gf_rotation_to_np_array(
                Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
            ),
            usd_path=self.assets_root_path
            + "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            semantic_label="cracker_box",
        )
        prims.create_prim(
            WORLD_STAGE_PATH + "/sugar_box",
            "Xform",
            # position=np.array([-0.07, -0.25, 0.1]),
            position=self.__rand_position_in_bounds(bounds, 0.1),
            orientation=rotations.gf_rotation_to_np_array(
                Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)
            ),
            usd_path=self.assets_root_path
            + "/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            semantic_label="sugar_box",
        )
        prims.create_prim(
            WORLD_STAGE_PATH + "/soup_can",
            "Xform",
            # position=np.array([0.1, -0.25, 0.10]),
            position=self.__rand_position_in_bounds(bounds, 0.10),
            orientation=rotations.gf_rotation_to_np_array(
                Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
            ),
            usd_path=self.assets_root_path
            + "/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            semantic_label="soup_can",
        )
        prims.create_prim(
            WORLD_STAGE_PATH + "/mustard_bottle",
            "Xform",
            # position=np.array([0.0, 0.15, 0.12]),
            position=self.__rand_position_in_bounds(bounds, 0.12),
            orientation=rotations.gf_rotation_to_np_array(
                Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
            ),
            usd_path=self.assets_root_path
            + "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            semantic_label="mustard_bottle",
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
            color=np.array([1.5, 0.0, 0.0]),
        )
        marker_cube.apply_visual_material(material)
        self.sim.update()

    def _init_cameras(self):
        viewports.create_viewport_for_camera(REALSENSE_VIEWPORT_NAME, CAMERA_PRIM_PATH)

        realsense_prim = UsdGeom.Camera(
            stage.get_current_stage().GetPrimAtPath(CAMERA_PRIM_PATH)
        )
        realsense_prim.GetHorizontalApertureAttr().Set(20.955)
        realsense_prim.GetVerticalApertureAttr().Set(15.7)
        # realsense_prim.GetFocalLengthAttr().Set(18.8)
        realsense_prim.GetFocalLengthAttr().Set(12.7)
        realsense_prim.GetFocusDistanceAttr().Set(400)

        viewport = omni.ui.Workspace.get_window("Viewport")
        rs_viewport = omni.ui.Workspace.get_window(REALSENSE_VIEWPORT_NAME)
        rs_viewport.dock_in(viewport, omni.ui.DockPosition.RIGHT, ratio=0.3)
        carb.log_warn(
            f"{REALSENSE_VIEWPORT_NAME} docked in {viewport.title}: {rs_viewport.docked}"
        )

        from omni.isaac.sensor import Camera

        render_product_path = rs_viewport.viewport_api.get_render_product_path()

        self.cameras["realsense"] = Camera(
            prim_path=CAMERA_PRIM_PATH,
            name="realsense",
            resolution=(640, 480),
            render_product_path=render_product_path,
        )
        self.cameras["realsense"].add_distance_to_camera_to_frame()
        self.cameras["realsense"].add_distance_to_image_plane_to_frame()

        camera_rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), -90) * Gf.Rotation(
            Gf.Vec3d(1, 0, 0), 45
        )
        self.cameras["agentview"] = Camera(
            prim_path="/World/agentview_camera",
            name="agentview",
            resolution=(1024, 768),
            position=(0, 2.75, 2.67),
            orientation=rotations.gf_rotation_to_np_array(camera_rot),
        )

        # viewports.create_viewport_for_camera("agentview", "/World/agentview_camera")
        # agentview_prim = UsdGeom.Camera(
        #     stage.get_current_stage().GetPrimAtPath("/World/agentview_camera")
        # )
        # agentview_viewport = omni.ui.Workspace.get_window("agentview")
        # render_product_path = agentview_viewport.viewport_api.get_render_product_path()
        # self.cameras["agentview"].initialize(render_product_path)

        self.world.reset()
        for cam in self.cameras.values():
            cam.initialize()

    def _enable_ros2_bridge_ext(self):
        extensions.enable_extension("omni.isaac.ros2_bridge")

    def _load_omnigraph(self):
        carb.log_warn("Loading Omnigraph")

        try:
            ros_domain_id = int(os.environ["ROS_DOMAIN_ID"])
            print("Using ROS_DOMAIN_ID: ", ros_domain_id)
        except ValueError:
            print("Invalid ROS_DOMAIN_ID integer value. Setting value to 0")
            ros_domain_id = 0
        except KeyError:
            print("ROS_DOMAIN_ID environment variable is not set. Setting value to 0")
            ros_domain_id = 0

        try:
            og.Controller.edit(
                {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                        (
                            "ReadSimTime",
                            "omni.isaac.core_nodes.IsaacReadSimulationTime",
                        ),
                        ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                        (
                            "PublishJointState",
                            "omni.isaac.ros2_bridge.ROS2PublishJointState",
                        ),
                        (
                            "SubscribeJointState",
                            "omni.isaac.ros2_bridge.ROS2SubscribeJointState",
                        ),
                        (
                            "ArticulationController",
                            "omni.isaac.core_nodes.IsaacArticulationController",
                        ),
                        ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                        ("OnTick", "omni.graph.action.OnTick"),
                        # Realsense Camera Helper
                        ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                        (
                            "getRenderProduct",
                            "omni.isaac.core_nodes.IsaacGetViewportRenderProduct",
                        ),
                        (
                            "setCamera",
                            "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct",
                        ),
                        ("cameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                        ("cameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                        (
                            "cameraHelperDepth",
                            "omni.isaac.ros2_bridge.ROS2CameraHelper",
                        ),
                        (
                            "cameraHelperInstanceSegmentation",
                            "omni.isaac.ros2_bridge.ROS2CameraHelper",
                        ),
                        # Agent View Camera Helper
                        (
                            "createAgentView",
                            "omni.isaac.core_nodes.IsaacCreateViewport",
                        ),
                        (
                            "getRenderProductAgentView",
                            "omni.isaac.core_nodes.IsaacGetViewportRenderProduct",
                        ),
                        (
                            "setCameraAgentView",
                            "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct",
                        ),
                        (
                            "cameraHelperAgentViewRgb",
                            "omni.isaac.ros2_bridge.ROS2CameraHelper",
                        ),
                        (
                            "cameraHelperAgentViewInfo",
                            "omni.isaac.ros2_bridge.ROS2CameraHelper",
                        ),
                        (
                            "cameraHelperAgentViewDepth",
                            "omni.isaac.ros2_bridge.ROS2CameraHelper",
                        ),
                        (
                            "cameraHelperAgentViewInstanceSegmentation",
                            "omni.isaac.ros2_bridge.ROS2CameraHelper",
                        ),
                    ],
                    og.Controller.Keys.CONNECT: [
                        (
                            "OnImpulseEvent.outputs:execOut",
                            "PublishJointState.inputs:execIn",
                        ),
                        (
                            "OnImpulseEvent.outputs:execOut",
                            "SubscribeJointState.inputs:execIn",
                        ),
                        (
                            "OnImpulseEvent.outputs:execOut",
                            "PublishClock.inputs:execIn",
                        ),
                        (
                            "OnImpulseEvent.outputs:execOut",
                            "ArticulationController.inputs:execIn",
                        ),
                        ("Context.outputs:context", "PublishJointState.inputs:context"),
                        (
                            "Context.outputs:context",
                            "SubscribeJointState.inputs:context",
                        ),
                        ("Context.outputs:context", "PublishClock.inputs:context"),
                        (
                            "ReadSimTime.outputs:simulationTime",
                            "PublishJointState.inputs:timeStamp",
                        ),
                        (
                            "ReadSimTime.outputs:simulationTime",
                            "PublishClock.inputs:timeStamp",
                        ),
                        (
                            "SubscribeJointState.outputs:jointNames",
                            "ArticulationController.inputs:jointNames",
                        ),
                        (
                            "SubscribeJointState.outputs:positionCommand",
                            "ArticulationController.inputs:positionCommand",
                        ),
                        (
                            "SubscribeJointState.outputs:velocityCommand",
                            "ArticulationController.inputs:velocityCommand",
                        ),
                        (
                            "SubscribeJointState.outputs:effortCommand",
                            "ArticulationController.inputs:effortCommand",
                        ),
                        # Realsense Camera Key Connects
                        ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                        (
                            "createViewport.outputs:execOut",
                            "getRenderProduct.inputs:execIn",
                        ),
                        (
                            "createViewport.outputs:viewport",
                            "getRenderProduct.inputs:viewport",
                        ),
                        ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                        (
                            "getRenderProduct.outputs:renderProductPath",
                            "setCamera.inputs:renderProductPath",
                        ),
                        ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                        ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                        (
                            "setCamera.outputs:execOut",
                            "cameraHelperDepth.inputs:execIn",
                        ),
                        (
                            "setCamera.outputs:execOut",
                            "cameraHelperInstanceSegmentation.inputs:execIn",
                        ),
                        ("Context.outputs:context", "cameraHelperRgb.inputs:context"),
                        ("Context.outputs:context", "cameraHelperInfo.inputs:context"),
                        ("Context.outputs:context", "cameraHelperDepth.inputs:context"),
                        (
                            "Context.outputs:context",
                            "cameraHelperInstanceSegmentation.inputs:context",
                        ),
                        (
                            "getRenderProduct.outputs:renderProductPath",
                            "cameraHelperRgb.inputs:renderProductPath",
                        ),
                        (
                            "getRenderProduct.outputs:renderProductPath",
                            "cameraHelperInfo.inputs:renderProductPath",
                        ),
                        (
                            "getRenderProduct.outputs:renderProductPath",
                            "cameraHelperDepth.inputs:renderProductPath",
                        ),
                        (
                            "getRenderProduct.outputs:renderProductPath",
                            "cameraHelperInstanceSegmentation.inputs:renderProductPath",
                        ),
                        # Agent View Camera Key Connects
                        ("OnTick.outputs:tick", "createAgentView.inputs:execIn"),
                        (
                            "createAgentView.outputs:execOut",
                            "getRenderProductAgentView.inputs:execIn",
                        ),
                        (
                            "createAgentView.outputs:viewport",
                            "getRenderProductAgentView.inputs:viewport",
                        ),
                        (
                            "getRenderProductAgentView.outputs:execOut",
                            "setCameraAgentView.inputs:execIn",
                        ),
                        (
                            "getRenderProductAgentView.outputs:renderProductPath",
                            "setCameraAgentView.inputs:renderProductPath",
                        ),
                        (
                            "setCameraAgentView.outputs:execOut",
                            "cameraHelperAgentViewRgb.inputs:execIn",
                        ),
                        (
                            "setCameraAgentView.outputs:execOut",
                            "cameraHelperAgentViewInfo.inputs:execIn",
                        ),
                        (
                            "setCameraAgentView.outputs:execOut",
                            "cameraHelperAgentViewDepth.inputs:execIn",
                        ),
                        (
                            "setCameraAgentView.outputs:execOut",
                            "cameraHelperAgentViewInstanceSegmentation.inputs:execIn",
                        ),
                        (
                            "Context.outputs:context",
                            "cameraHelperAgentViewRgb.inputs:context",
                        ),
                        (
                            "Context.outputs:context",
                            "cameraHelperAgentViewInfo.inputs:context",
                        ),
                        (
                            "Context.outputs:context",
                            "cameraHelperAgentViewDepth.inputs:context",
                        ),
                        (
                            "Context.outputs:context",
                            "cameraHelperAgentViewInstanceSegmentation.inputs:context",
                        ),
                        (
                            "getRenderProductAgentView.outputs:renderProductPath",
                            "cameraHelperAgentViewRgb.inputs:renderProductPath",
                        ),
                        (
                            "getRenderProductAgentView.outputs:renderProductPath",
                            "cameraHelperAgentViewInfo.inputs:renderProductPath",
                        ),
                        (
                            "getRenderProductAgentView.outputs:renderProductPath",
                            "cameraHelperAgentViewDepth.inputs:renderProductPath",
                        ),
                        (
                            "getRenderProductAgentView.outputs:renderProductPath",
                            "cameraHelperAgentViewInstanceSegmentation.inputs:renderProductPath",
                        ),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("Context.inputs:domain_id", ros_domain_id),
                        # Setting the /Franka target prim to Articulation Controller node
                        ("ArticulationController.inputs:usePath", True),
                        ("ArticulationController.inputs:robotPath", FRANKA_STAGE_PATH),
                        ("PublishJointState.inputs:topicName", "isaac_joint_states"),
                        (
                            "SubscribeJointState.inputs:topicName",
                            "isaac_joint_commands",
                        ),
                        # Realsense Camera Key Values
                        ("createViewport.inputs:name", REALSENSE_VIEWPORT_NAME),
                        ("createViewport.inputs:viewportId", 1),
                        ("cameraHelperRgb.inputs:frameId", "sim_camera"),
                        ("cameraHelperRgb.inputs:topicName", "rgb"),
                        ("cameraHelperRgb.inputs:type", "rgb"),
                        ("cameraHelperInfo.inputs:frameId", "sim_camera"),
                        ("cameraHelperInfo.inputs:topicName", "camera_info"),
                        ("cameraHelperInfo.inputs:type", "camera_info"),
                        ("cameraHelperDepth.inputs:frameId", "sim_camera"),
                        ("cameraHelperDepth.inputs:topicName", "depth"),
                        ("cameraHelperDepth.inputs:type", "depth"),
                        (
                            "cameraHelperInstanceSegmentation.inputs:frameId",
                            "sim_camera",
                        ),
                        (
                            "cameraHelperInstanceSegmentation.inputs:topicName",
                            "instance_segmentation",
                        ),
                        (
                            "cameraHelperInstanceSegmentation.inputs:type",
                            "instance_segmentation",
                        ),
                        # Agent View Camera Key Values
                        ("createAgentView.inputs:name", "agentview"),
                        ("createAgentView.inputs:viewportId", 2),
                        (
                            "setCameraAgentView.inputs:cameraPrim",
                            "/World/agentview_camera",
                        ),
                        ("cameraHelperAgentViewRgb.inputs:frameId", "agentview"),
                        ("cameraHelperAgentViewRgb.inputs:topicName", "agentview/rgb"),
                        ("cameraHelperAgentViewRgb.inputs:type", "rgb"),
                        ("cameraHelperAgentViewInfo.inputs:frameId", "agentview"),
                        (
                            "cameraHelperAgentViewInfo.inputs:topicName",
                            "agentview/camera_info",
                        ),
                        ("cameraHelperAgentViewInfo.inputs:type", "camera_info"),
                        ("cameraHelperAgentViewDepth.inputs:frameId", "agentview"),
                        (
                            "cameraHelperAgentViewDepth.inputs:topicName",
                            "agentview/depth",
                        ),
                        ("cameraHelperAgentViewDepth.inputs:type", "depth"),
                        (
                            "cameraHelperAgentViewInstanceSegmentation.inputs:frameId",
                            "agentview",
                        ),
                        (
                            "cameraHelperAgentViewInstanceSegmentation.inputs:topicName",
                            "agentview/instance_segmentation",
                        ),
                        (
                            "cameraHelperAgentViewInstanceSegmentation.inputs:type",
                            "instance_segmentation",
                        ),
                        (
                            "cameraHelperAgentViewInstanceSegmentation.inputs:enableSemanticLabels",
                            True,
                        ),
                        (
                            "cameraHelperAgentViewInstanceSegmentation.inputs:semanticLabelsTopicName",
                            "agentview/semantic_labels",
                        ),
                    ],
                },
            )
            carb.log_warn("Omnigraph loaded successfully")
        except Exception as e:
            carb.log_error(e)
            carb.log_warn("Failed to load Omnigraph")

        from omni.isaac.core_nodes.scripts.utils import set_target_prims

        set_target_prims(
            primPath="/ActionGraph/PublishJointState",
            targetPrimPaths=[FRANKA_STAGE_PATH],
        )

        prims.set_targets(
            prim=stage.get_current_stage().GetPrimAtPath(GRAPH_PATH + "/setCamera"),
            attribute="inputs:cameraPrim",
            target_prim_paths=[CAMERA_PRIM_PATH],
        )

    def close_sim(self):
        self.sim.close()

    def run_sim(self):
        while self.sim.is_running():
            self.world.step(render=True)
            if self.world.is_playing():
                if self.world.current_time_step_index == 0:
                    self.world.reset()
                    self.controller.reset()
                # Tick the Publish/Subscribe JointState, Publish TF and Publish Clock nodes each frame
                og.Controller.set(
                    og.Controller.attribute(
                        "/ActionGraph/OnImpulseEvent.state:enableImpulse"
                    ),
                    True,
                )
                # self.task_manager.do_tasks()
            self.sim.update()

    def get_image(self, camera_name="realsense", mode=CameraMode.RGB, visualize=False):
        self.world.step(render=True)
        camera = self.cameras[camera_name]
        try:
            if mode == CameraMode.RGB:
                img = camera.get_rgb()
            if mode == CameraMode.RGBA:
                img = camera.get_rgba()
            if mode == CameraMode.DEPTH:
                img = camera.get_depth()
            if visualize:
                import cv2

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Color", img)
                cv2.waitKey(0)
            return img
        except Exception as e:
            print(e)


if __name__ == "__main__":
    sim_manager = SimManager()
    sim_manager.start_sim(headless=False)
    sim_manager.task_manager.test_task()
    sim_manager.world.pause()
    sim_manager.run_sim()
    # sim_manager.get_image()
    sim_manager.close_sim()
