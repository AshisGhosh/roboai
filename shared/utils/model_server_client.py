import io
import asyncio
from PIL import Image
from typing import Any, Dict
from shared.utils.http_client import post_request, get_request, get_image_request


SERVER_NAME = "http://localhost:8002"

async def _answer_question_from_image(image: Image, question: str) -> Dict[str, Any]:
    timeout = 120.0

    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="JPEG")
    image_byte_array = image_byte_array.getvalue()

    files = {'file': ('image.jpg', image_byte_array, 'image/jpeg')}
    response = await post_request(f"{SERVER_NAME}/answer_question", files=files, data={"question": question}, timeout=timeout)
    return response

def answer_question_from_image(image: Image, question: str) -> Dict[str, Any]:
    return asyncio.run(_answer_question_from_image(image, question))