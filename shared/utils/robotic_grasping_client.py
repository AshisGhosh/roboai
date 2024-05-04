import io

import asyncio
from shared.utils.http_client import get_request, post_request

from typing import Any, Dict
from PIL import Image

import logging

log = logging.getLogger("robotic_grasping_client")
log.setLevel(logging.DEBUG)

SERVER_NAME = "http://localhost:8003"


async def _get_grasps_from_rgb_and_depth(
    rgb_image: Image, depth_image: Image
) -> Dict[str, Any]:
    log.debug("Getting grasp from GR-ConvNet")
    timeout = 30.0

    image_byte_array = io.BytesIO()
    rgb_image.save(image_byte_array, format="JPEG")
    image_byte_array = image_byte_array.getvalue()
    log.debug("RGB image byte array saved")

    depth_image_byte_array = io.BytesIO()
    depth_image.save(depth_image_byte_array, format="JPEG")
    depth_image_byte_array = depth_image_byte_array.getvalue()
    log.debug("Depth image byte array saved")

    files = {
        "rgb_image": ("rgb_image.jpg", image_byte_array, "image/jpeg"),
        "depth_image": ("depth_image.jpg", depth_image_byte_array, "image/jpeg"),
    }
    response = await post_request(
        f"{SERVER_NAME}/get_grasps", files=files, timeout=timeout
    )
    return response


def get_grasps_from_rgb_and_depth(
    rgb_image: Image, depth_image: Image
) -> Dict[str, Any]:
    return asyncio.run(_get_grasps_from_rgb_and_depth(rgb_image, depth_image))


async def _check_server() -> str:
    response = await get_request(f"{SERVER_NAME}/")
    return response
