import io
import asyncio
from PIL import Image
from typing import Any, Dict
from shared.utils.http_client import post_request, get_request, get_image_request

import logging
log = logging.getLogger("model_server_client")
log.setLevel(logging.INFO)

SERVER_NAME = "http://localhost:8002"

async def _answer_question_from_image(image: Image, question: str) -> Dict[str, Any]:
    timeout = 120.0

    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="JPEG")
    image_byte_array = image_byte_array.getvalue()

    files = {'file': ('image.jpg', image_byte_array, 'image/jpeg')}
    response = await post_request(f"{SERVER_NAME}/answer_question", files=files, params={"question": question}, timeout=timeout)
    return response

def answer_question_from_image(image: Image, question: str) -> Dict[str, Any]:
    return asyncio.run(_answer_question_from_image(image, question))


async def _embed(text: str) -> Dict[str, Any]:
    log.debug(f"async _embed call: Embedding text: {text}")
    return await post_request(f"{SERVER_NAME}/embed", params={"text": text})

def embed(text: str) -> Dict[str, Any]:
    log.debug(f"Embedding text: {text}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(_embed(text))
    loop.close()
    return result
