from PIL import Image
from typing import Dict, Any
import io
import base64
import gradio_client
from gradio_client import Client


def gradio_answer_question_from_image(image: Image, question: str) -> Dict[str, Any]:
    client = Client("vikhyatk/moondream2")
    # result = client.predict(
    #         "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# filepath  in 'Upload an Image' Image component
    #         "Hello!!",	# str  in 'Input' Textbox component
    #         api_name="/answer_question"
    # )
    # image_byte_array = io.BytesIO()
    # image.save(image_byte_array, format="JPEG")
    # image_byte_array = image_byte_array.getvalue()
    # image = base64.b64encode(image_byte_array).decode("utf-8")
    image.save("/app/shared/data/tmp.png")
    
    result = client.predict(
        gradio_client.file("/app/shared/data/tmp.png"),
        question,
        api_name="/answer_question"
    )
    print(result)
    return {"result": result}