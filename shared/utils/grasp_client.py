import io

import asyncio
from shared.utils.http_client import get_request, post_request

from typing import Any, Dict
from PIL import Image

SERVER_NAME = "http://localhost:8005"

async def _get_grasp_from_image(image: Image) -> Dict[str, Any]:
    timeout = 30.0

    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="JPEG")
    image_byte_array = image_byte_array.getvalue()

    files = {'file': ('image.jpg', image_byte_array, 'image/jpeg')}
    response = await post_request(f"{SERVER_NAME}/detect", files=files, timeout=timeout)
    return response

def get_grasp_from_image(image: Image) -> Dict[str, Any]:
    return asyncio.run(_get_grasp_from_image(image))

async def _check_server() -> str:
    response = await get_request(f"{SERVER_NAME}/")
    return response