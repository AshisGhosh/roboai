import os
import yaml
import numpy as np
import asyncio
import multiprocessing

import omnigibson as og
from omnigibson.macros import gm  # noqa F401
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)  # noqa F401
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)
from omnigibson.robots import Tiago
from omnigibson.utils.asset_utils import get_og_scene_path
from .visualize_scene_graph import visualize_scene_graph, visualize_ascii_scene_graph


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
from starlette.responses import StreamingResponse


# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True


class ActionHandler:
    def __init__(self, env, controller, scene, task_queue):
        self.env = env
        self.controller = controller
        self.scene = scene
        self.actions = task_queue
        self._last_camera_action = None

    async def add_action(self, action: str):
        """
        Add an action to the list of actions to be executed
        """
        self.actions.put(action)

    def execute_controller(self, ctrl_gen):
        robot = self.env.robots[0]
        for action in ctrl_gen:
            # target_obj_pose = self.controller._tracking_object.get_position_orientation()
            # print(f"current: {robot.get_joint_positions()[robot.camera_control_idx]}, desired: {self.controller._get_head_goal_q(target_obj_pose)}")
            # print(f"{action} - camera action: {action[robot.controller_action_idx['camera']]}")
            state, reward, done, info = self.env.step(action)
            self._last_camera_action = action[robot.controller_action_idx["camera"]]

    def execute_action(self, action):
        """
        Execute the action at the top of the list
        """
        # robot = self.env.robots[0]
        action, args = action[0], action[1:]

        if action == "pick":
            print(f"Attempting: 'pick' with args: {args}")
            obj_name = args[0]
            grasp_obj = self.scene.object_registry("name", obj_name)
            print(f"navigating to object {grasp_obj.name}")
            self.controller._tracking_object = grasp_obj
            self.execute_controller(
                self.controller.apply_ref(
                    SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO,
                    grasp_obj,
                    attempts=10,
                )
            )
            print(f"grasping object {grasp_obj.name}")
            self.execute_controller(
                self.controller.apply_ref(
                    SymbolicSemanticActionPrimitiveSet.GRASP, grasp_obj
                )
            )
        elif action == "place":
            print(f"Attempting: 'place' with args: {args}")
            obj_name = args[0]
            table = self.scene.object_registry("name", obj_name)
            print(f"navigating to object {table.name}")
            self.controller._tracking_object = table
            self.execute_controller(
                self.controller.apply_ref(
                    SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO, table, attempts=10
                )
            )
            print(f"placing object on top of {table.name}")
            self.execute_controller(
                self.controller.apply_ref(
                    SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, table
                )
            )
        elif action == "navigate_to":
            print(f"Attempting: 'navigate_to' with args: {args}")
            obj_name = args[0]
            obj = self.scene.object_registry("name", obj_name)
            self.controller._tracking_object = obj
            print(f"navigating to object {obj.name}")
            self.execute_controller(
                self.controller.apply_ref(
                    SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO, obj, attempts=10
                )
            )

        elif action == "pick_test":
            print("Executing pick")
            grasp_obj = self.scene.object_registry("name", "cologne")
            print(f"navigating to object {grasp_obj.name}")
            self.controller._tracking_object = grasp_obj
            self.execute_controller(
                self.controller.apply_ref(
                    SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO,
                    grasp_obj,
                    attempts=10,
                )
            )
            print(f"grasping object {grasp_obj.name}")
            # self.execute_controller(self.controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj))
            self.execute_controller(
                self.controller.apply_ref(
                    SymbolicSemanticActionPrimitiveSet.GRASP, grasp_obj
                )
            )
            print("Finished executing pick")

        elif action == "place_test":
            print("Executing place")
            table = self.scene.object_registry("name", "table")
            # print(f"navigating to object {table.name}")
            self.controller._tracking_object = table
            # self.execute_controller(self.controller.apply_ref(SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO, table, attempts=10))
            print(f"placing object on top of {table.name}")
            # self.execute_controller(self.controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, table))
            self.execute_controller(
                self.controller.apply_ref(
                    SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, table
                )
            )
            print("Finished executing place")

        elif action == "navigate_to_coffee_table":
            # print("Executing navigate_to_coffee_table")
            coffee_table = self.scene.object_registry("name", "coffee_table_fqluyq_0")
            self.controller._tracking_object = coffee_table
            print(f"navigating to object {coffee_table.name}")
            self.execute_controller(
                self.controller.apply_ref(
                    SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO,
                    coffee_table,
                    attempts=10,
                )
            )
            print("Finished executing navigate_to_coffee_table")
        
        elif action == "viz":
            print("Visualizing scene graph")
            graph = self.env.get_scene_graph()
            print (graph)
            visualize_ascii_scene_graph(self.scene, graph)
            # visualize_scene_graph(self.scene, graph)
            print("Finished visualizing scene graph")

    def check_for_action(self):
        """
        Check if there is an action to be executed
        """
        if not self.actions.empty():
            action = self.actions.get()
            self.execute_action(action)
            return True
        action = np.zeros(self.env.robots[0].action_dim)
        if self._last_camera_action is not None:
            action[self.env.robots[0].controller_action_idx["camera"]] = (
                self._last_camera_action
            )

        # print(f"ACTION - {action}")
        state, reward, done, info = self.env.step(action)
        # print(f"info: {info}")
        return False


class SimWrapper:
    def __init__(self, task_queue, image_queue, scene_graph_queue):
        self.task_queue = task_queue
        self.image_queue = image_queue
        self.scene_graph_queue = scene_graph_queue
        asyncio.run(self.run())

    async def run(self):
        """
        Demonstrates how to use the action primitives to pick and place an object in an empty scene.

        It loads Rs_int with a Fetch robot, and the robot picks and places a bottle of cologne.
        """
        # Load the config
        # config_filename = os.path.join(og.example_config_path, "fetch_primitives.yaml")
        config_filename = os.path.join(
            "/omnigibson-src/roboai", "tiago_primitives.yaml"
        )
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # robot0_cfg = dict()
        # robot0_cfg["type"] = "Tiago"
        # robot0_cfg["obs_modalities"] = ["rgb", "depth", "seg_instance", "normal", "scan", "occupancy_grid"]
        # robot0_cfg["action_type"] = "continuous"
        # robot0_cfg["action_normalize"] = True

        # config["robots"] = [robot0_cfg]

        # Update it to create a custom environment and run some actions
        # config["scene"]["scene_model"] = "Rs_int"
        config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

        # # SHOW TRAVERSABLE AREA
        # import matplotlib.pyplot as plt
        # import cv2
        # scene_model = "Rs_int"
        # trav_map_size = 200
        # trav_map_erosion = 2

        # trav_map = Image.open(os.path.join(get_og_scene_path(scene_model), "layout", "floor_trav_0.png"))
        # trav_map = np.array(trav_map.resize((trav_map_size, trav_map_size)))
        # trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))

        # plt.figure(figsize=(12, 12))
        # plt.imshow(trav_map)
        # plt.title(f"Traversable area of {scene_model} scene")

        # plt.show()

        config["scene"]["not_load_object_categories"] = ["ceilings"]
        
        config["objects"] = [
            {
                "type": "DatasetObject",
                "name": "cologne",
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [-0.3, -0.8, 0.5],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "DatasetObject",
                "name": "apple",
                "category": "apple",
                "model": "agveuv",
                "position": [-0.3, -1.1, 0.5],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "DatasetObject",
                "name": "cleaner",
                "category": "bottle_of_cleaner",
                "model": "svzbeq",
                "position": [-0.5, -0.8, 0.6],
                "orientation": [0, 1, 0, 0],
            },
            {
                "type": "DatasetObject",
                "name": "tomato_can",
                "category": "can_of_tomatoes",
                "model": "ckdouu",
                "position": [-0.6, -1.1, 0.5],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "DatasetObject",
                "name": "table",
                "category": "breakfast_table",
                "model": "rjgmmy",
                "scale": [0.3, 0.3, 0.3],
                "position": [-0.7, 0.5, 0.2],
                "orientation": [0, 0, 0, 1],
            },
        ]

        # Load the environment
        env = og.Environment(configs=config)
        scene = env.scene
        robot = env.robots[0]
        print(type(robot))
        print(robot.default_arm)
        delattr(Tiago, "simplified_mesh_usd_path")
        # del robot.simplified_mesh_usd_path
        # print(robot.simplified_mesh_usd_path)

        # Allow user to move camera more easily
        og.sim.enable_viewer_camera_teleoperation()
        cam = og.sim.viewer_camera
        # camera pose: array([0.92048866, -5.66129052,  5.39363818]), array([0.44288347, 0.04140454, 0.08336682, 0.89173419])
        cam.set_position_orientation(
            position=np.array([0.92048866, -5.66129052, 5.39363818]),
            orientation=np.array([0.44288347, 0.04140454, 0.08336682, 0.89173419]),
        )

        # controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
        controller = SymbolicSemanticActionPrimitives(env)

        action_handler = ActionHandler(
            env, controller, scene, task_queue=self.task_queue
        )
        # visualize_scene_graph(scene, env.get_scene_graph())
        # graph = env.get_scene_graph()
        # print (graph)

        if False:
            print("\n\n####### TASK DATA #######\n")
            task_str, _, _ = env.task.show_instruction()
            print(task_str)
            task_obs, _ = env.task._get_obs(env)
            agent_pos = task_obs["agent.n.01_1_pos"]
            print(task_obs)
            print(env.task.object_scope)
            for k,v in env.task.object_scope.items():
                dist = np.linalg.norm(np.array(task_obs[f"{k}_pos"]) - np.array(agent_pos))
                print(f"{k}: {v.name} {v.category} {v.exists} {dist:.3f}")
            print("\n#########################\n\n")


       
        while True:
            await asyncio.sleep(0.1)
            img = robot.get_obs()[0]["robot0:eyes:Camera:0"]["rgb"]
            if self.image_queue.full():
                self.image_queue.get()
            self.image_queue.put(img)
            action_handler.check_for_action()
            # task_str, _, _ = env.task.show_instruction()
            # print(task_str)

            # if self.scene_graph_queue.full():
            #     self.scene_graph_queue.get()
            # graph = env.get_scene_graph()
            # self.scene_graph_queue.put(graph)
            # current = robot.get_joint_positions(normalized=False)
            # print(f"current: {current}")
            # arm_left = robot._controllers["arm_left"]
            # print(f"arm_left: {arm_left.control}")


app = FastAPI()

# List of allowed origins (you can use '*' to allow all origins)
origins = [
    "http://localhost:3000",  # Allow your Next.js app
    # Add any other origins as needed
]

# Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

task_queue = multiprocessing.Queue()
image_queue = multiprocessing.Queue(1)
scene_graph_queue = multiprocessing.Queue(1)
sim = multiprocessing.Process(target=SimWrapper, args=(task_queue, image_queue, scene_graph_queue))


@app.post("/add_action")
async def add_action(action: str):
    action = action.split(",")
    action = (action[0], *action[1:])
    print(f"Adding action: {action}")
    task_queue.put(action)
    return {"action": action}


@app.get("/get_image")
async def get_image():
    image = image_queue.get()
    # img_array = np.frombuffer(image.data, np.uint8).reshape(
    #     image.height, image.width, 3
    # )
    img_array = image
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# @app.get("/get_scene_graph")
# async def get_scene_graph():
#     graph = scene_graph_queue.get()
#     return {"graph": graph}

if __name__ == "__main__":
    sim.start()
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
