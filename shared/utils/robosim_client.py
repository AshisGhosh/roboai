import asyncio
from shared.utils.http_client import get_request, post_request, get_image_request
import io
from PIL import Image

SERVER_NAME = "http://localhost:8000"

async def _get_objects_on_table() -> list[str]:
    return await get_request(f"{SERVER_NAME}/get_objects")

def get_objects_on_table():
    return asyncio.run(_get_objects_on_table())

async def _get_image() -> Image:
    img_data = await get_image_request(f"{SERVER_NAME}/get_image")
    return Image.open(io.BytesIO(img_data))

def get_image():
    return asyncio.run(_get_image())

async def _get_grasp_image() -> Image:
    img_data = await get_image_request(f"{SERVER_NAME}/get_grasp_image")
    return Image.open(io.BytesIO(img_data))

def get_grasp_image():
    return asyncio.run(_get_grasp_image())

async def _open_gripper():
    return await post_request(f"{SERVER_NAME}/add_task", data={"name": "open gripper", "type": "open_gripper", "args": ""})

def open_gripper():
    return asyncio.run(_open_gripper())

async def _close_gripper():
    return await post_request(f"{SERVER_NAME}/add_task", data={"name": "close gripper", "type": "close_gripper", "args": ""})

def close_gripper():
    return asyncio.run(_close_gripper())

async def _go_to_pick_center():
    return await post_request(f"{SERVER_NAME}/add_task", data={"name": "go to pick center", "type": "go_to_pick_center", "args": ""})

async def _get_grasp(object_name: str):
    return await post_request(f"{SERVER_NAME}/add_task", data={"name": f"get grasp {object_name}", "type": "get_grasp", "args": object_name})

def get_grasp(object_name: str):
    asyncio.run(_go_to_pick_center())
    return asyncio.run(_get_grasp(object_name))

async def _go_to_pre_grasp():
    return await post_request(f"{SERVER_NAME}/add_task", data={"name": "go to pre grasp", "type": "go_to_pre_grasp", "args": ""})

def go_to_pre_grasp():
    return asyncio.run(_go_to_pre_grasp())

async def _go_to_grasp_position():
    return await post_request(f"{SERVER_NAME}/add_task", data={"name": "go to grasp pos", "type": "go_to_grasp_position", "args": ""})

def go_to_grasp_position():
    return asyncio.run(_go_to_grasp_position())

async def _go_to_drop():
    return await post_request(f"{SERVER_NAME}/add_task", data={"name": "go to drop", "type": "go_to_drop", "args": ""})

def go_to_drop():
    return asyncio.run(_go_to_drop())

def pick(object_name: str):
    get_grasp(object_name)
    go_to_pre_grasp()
    open_gripper()
    go_to_grasp_position()
    close_gripper()
    go_to_pre_grasp()

def place(object_name: str):
    go_to_drop()
    open_gripper()