#!/usr/bin/python -u

import io

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel

from robosim.robosim import RoboSim

import logging

logging.basicConfig(level=logging.DEBUG)

# Create FastAPI instance
app = FastAPI()

robosim = RoboSim()

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


class Task(BaseModel):
    name: str
    type: str
    args: list | str


# Example route
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI! This is the robosim server."}


@app.on_event("startup")
async def startup_event():
    return robosim.setup()


@app.post("/get_feedback")
async def get_feedback():
    return await robosim.get_feedback("grasp-selection-feedback", "cereal")


@app.post("/pick")
async def pick(object_name: str):
    return await robosim.pick(object_name)


@app.post("/test")
async def test():
    robosim.clear_tasks()

    await add_task(Task(name="go to pick", type="go_to_pick_center", args=""))
    await add_task(Task(name="get grasp", type="get_grasp", args="cereal"))
    await add_task(Task(name="go to pre grasp", type="go_to_pre_grasp", args=""))
    await add_task(Task(name="open gripper", type="open_gripper", args=""))
    await add_task(Task(name="go to grasp pos", type="go_to_grasp_position", args=""))
    await add_task(Task(name="close gripper", type="close_gripper", args=""))
    await add_task(Task(name="go to pre grasp", type="go_to_pre_grasp", args=""))
    await add_task(Task(name="go to drop", type="go_to_drop", args=""))
    await add_task(Task(name="open gripper", type="open_gripper", args=""))

    await robosim.start_execution()
    return {"msg": "Test task added and execution started."}


@app.post("/start")
async def start():
    return await robosim.start_async()


@app.post("/move_pose")
async def move_pose(pose: list[float]):
    await add_task(Task(name="move pose", type="go_to_pose", args=pose))
    await robosim.start_execution()
    return {"msg": "Pose move task added and execution started."}


@app.post("/move_orientation")
async def move_orientation(orientation: list[float]):
    await add_task(
        Task(name="move orientation", type="go_to_orientation", args=orientation)
    )
    await robosim.start_execution()
    return {"msg": "Orientation move task added and execution started."}


@app.post("/move_position")
async def move_position(position: list[float]):
    await add_task(Task(name="move position", type="go_to_position", args=position))
    await robosim.start_execution()
    return {"msg": "Position move task added and execution started."}


@app.get("/move_gripper_goal_to_gripper")
async def move_gripper_goal_to_gripper():
    return robosim.move_gripper_goal_to_gripper()


@app.get("/get_gripper_orientation")
async def get_gripper_orientation():
    return str(robosim.robot.get_gripper_orientation_as_euler())


@app.get("/get_gripper_orientation_in_world")
async def get_gripper_orientation_in_world():
    return str(robosim.robot.get_gripper_orientation_in_world_as_euler())


@app.post("/pixel_to_marker")
async def pixel_to_marker(pixel: list[int]):
    return robosim.pixel_to_marker(pixel)


@app.post("/add_marker")
async def add_marker(position: list[float]):
    return robosim.add_marker(position)


@app.post("/move_marker")
async def move_marker(
    name: str, position: list[float] | None, orientation: list[float] | None
):
    return robosim.move_marker(name=name, position=position, orientation=orientation)


@app.get("/get_grasp_image")
async def get_grasp_image():
    logging.info("Getting grasp image...")
    img = await robosim.robot.grasp.get_grasp_image()
    logging.debug("Image received.")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    logging.debug("Image saved. Ready to stream.")
    return StreamingResponse(buf, media_type="image/png")


@app.get("/get_grasp_image_and_depth")
async def get_grasp_image_and_depth():
    logging.info("Getting grasp image and depth...")
    img, depth = await robosim.robot.grasp.get_grasp_image_and_depth()
    logging.debug("Image and depth received.")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    logging.debug("Image saved. Ready to stream.")
    return StreamingResponse(buf, media_type="image/png")


@app.get("/get_grasp_depth_image")
async def get_grasp_image_and_depth_image():
    logging.info("Getting grasp image and depth...")
    _img, depth = await robosim.robot.grasp.get_grasp_image_and_depth_image()
    logging.debug("Image and depth received.")
    buf_depth = io.BytesIO()
    depth.save(buf_depth, format="PNG")
    buf_depth.seek(0)
    return StreamingResponse(buf_depth, media_type="image/png")


@app.get("/get_image")
async def get_image():
    logging.info("Getting image...")
    img = await robosim.get_image()
    logging.debug("Image received.")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    logging.debug("Image saved. Ready to stream.")
    return StreamingResponse(buf, media_type="image/png")


@app.get("/get_image_with_markers")
async def get_image_with_markers():
    logging.info("Getting image with markers...")
    img = await robosim.get_image_with_markers()
    logging.debug("Image received.")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    logging.debug("Image saved. Ready to stream.")
    return StreamingResponse(buf, media_type="image/png")


@app.post("/pause")
async def pause():
    return await robosim.pause_execution()


@app.post("/resume")
async def resume():
    return await robosim.resume_execution()


@app.post("/close")
async def close():
    return await robosim.close_renderer()


@app.post("/execute_tasks")
async def execute_tasks():
    return await robosim.execute_async()


@app.post("/add_task")
async def add_task(task: Task):
    logging.info(f"Adding task: {task.name} of type {task.type} with args {task.args}")
    try:
        robosim.add_task(task.name, task.type, task.args)
        return {"msg": "Task added"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get_tasks")
async def get_tasks():
    tasks = [
        Task(name=t.name, type=t.function.__name__, args=t.args)
        for t in robosim.get_tasks()
    ]
    return tasks


@app.get("/get_objects")
async def get_objects():
    return robosim.get_object_names()
