import asyncio
from shared.utils.http_client import post_request, get_image_request, get_request
import io
from PIL import Image

from shared.utils.llm_utils import get_closest_text_sync as get_closest_text

SERVER_NAME = "http://localhost:8000"

OMNIGIBSON_TIMEOUT = 30


async def _get_image() -> Image:
    img_data = await get_image_request(
        f"{SERVER_NAME}/get_image", timeout=OMNIGIBSON_TIMEOUT
    )
    return Image.open(io.BytesIO(img_data))


def get_image():
    response = asyncio.run(_get_image())
    if isinstance(response, dict):
        return response.get("success", True)
    return response


async def _get_visible_objects() -> dict:
    return await get_request(
        f"{SERVER_NAME}/get_visible_objects", timeout=OMNIGIBSON_TIMEOUT
    )


def get_visible_objects():
    response = asyncio.run(_get_visible_objects())
    if "success" in response.keys():
        return response.get("success", True)
    return response["objects"]


async def _get_obj_in_hand() -> dict:
    return await get_request(
        f"{SERVER_NAME}/get_obj_in_hand", timeout=OMNIGIBSON_TIMEOUT
    )


def get_obj_in_hand():
    try:
        response = asyncio.run(_get_obj_in_hand())
        return response["obj_in_hand"]
    except Exception as e:
        print(f"Error getting object in hand: {e}")
        return False


async def _wait_until_ready() -> dict:
    await asyncio.sleep(1)
    return await get_request(
        f"{SERVER_NAME}/wait_until_ready", timeout=OMNIGIBSON_TIMEOUT
    )


def wait_until_ready():
    try:
        response = asyncio.run(_wait_until_ready())
        return response["is_ready"]
    except Exception as e:
        print(f"Error waiting until ready: {e}")
        return False


async def _add_action(action: str):
    return await post_request(
        f"{SERVER_NAME}/add_action", params=action, timeout=OMNIGIBSON_TIMEOUT
    )


def add_action(action: str):
    response = asyncio.run(_add_action(action))
    if isinstance(response, dict):
        return response.get("success", True)
    return response


def pick(object_name: str):
    print(f"Attempting to pick {object_name}. Referencing against visible objects.")
    objects = get_visible_objects()
    object_name = get_closest_text(object_name, objects, threshold=0.2)
    print(f"picking object {object_name}")
    action = {"action": f"pick,{object_name}"}
    return add_action(action)


def place(location: str):
    print(f"placing object in {location}")
    print("placing object")
    action = {"action": f"place,{location}"}
    return add_action(action)


def navigate_to(object_name: str, location: str = None):
    print(f"navigating to {object_name}, {location}")
    if location:
        action = {"action": f"navigate_to,{object_name},{location}"}
    else:
        action = {"action": f"navigate_to_object,{object_name}"}

    return add_action(action)
