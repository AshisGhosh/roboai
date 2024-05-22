import asyncio
from shared.utils.http_client import post_request, get_image_request
import io
from PIL import Image

SERVER_NAME = "http://localhost:8000"


async def _get_image() -> Image:
    img_data = await get_image_request(f"{SERVER_NAME}/get_image")
    return Image.open(io.BytesIO(img_data))

def get_image():
    return asyncio.run(_get_image())

async def _add_action(action: str):
    return await post_request(
        f"{SERVER_NAME}/add_action",
        params=action,
    )

def add_action(action: str):
    return asyncio.run(_add_action(action))

def pick(object_name: str):
    print(f"picking {object_name}")
    action = {"action": f"pick,{object_name}"}
    add_action(action)


def place(location: str):
    print(f"placing object in {location}")
    print("placing object")
    action = {"action": f"place,{location}"}
    add_action(action)

def navigate_to(location: str):
    print(f"navigating to {location}")
    action = {"action": f"navigate_to,{location}"}
    add_action(action)