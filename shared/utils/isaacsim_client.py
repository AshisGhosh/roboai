import asyncio
from shared.utils.http_client import post_request, get_image_request
import io
from PIL import Image
from functools import wraps


SERVER_NAME = "http://localhost:8080"


async def _get_image() -> Image:
    img_data = await get_image_request(f"{SERVER_NAME}/get_image")
    return Image.open(io.BytesIO(img_data))


def get_image():
    return asyncio.run(_get_image())


async def _add_task(task: str):
    return await post_request(
        f"{SERVER_NAME}/add_task",
        params=task,
    )


def add_task(task: str):
    return asyncio.run(_add_task(task))


def add_test_mode(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if globals().get("test_mode", False):
            print("TEST MODE ENABLED")
            return True
        else:
            return func(*args, **kwargs)

    return wrapper


def pick(object_name: str):
    print(f"picking {object_name}")
    task = {"task": "pick"}
    add_task(task)


def place(object_name: str):
    print(f"placing {object_name}")
    print("placing object")
    task = {"task": "place"}
    print(f"Dummy task: {task}")