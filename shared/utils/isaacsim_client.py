import asyncio
from shared.utils.http_client import get_request, post_request, get_image_request
import io
from PIL import Image

SERVER_NAME = "http://localhost:8080"



async def _get_image() -> Image:
    img_data = await get_image_request(f"{SERVER_NAME}/get_image")
    return Image.open(io.BytesIO(img_data))

def get_image():
    return asyncio.run(_get_image())


def _add_task(task: str):
    return post_request(
        f"{SERVER_NAME}/add_task",
        data=task,
    )

def add_task(task: str):
    return asyncio.run(_add_task(task))

def pick(object_name: str):
    print(f"picking {object_name}")
    task = {"task": "pick"}
    add_task(task)

def place():
    print("placing object")
    task = {"task": "place"}
    print(f"Dummy task: {task}")

