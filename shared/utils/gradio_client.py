from PIL import Image
from typing import Dict, Any
import gradio_client
from gradio_client import Client
import time


def moondream_answer_question_from_image(image: Image, question: str) -> Dict[str, Any]:
    client = Client("vikhyatk/moondream2")
    # client = Client("Kartik2503/ImageToText")

    image.save("/app/shared/data/tmp.png")
    
    start_time = time.time()
    result = client.predict(
        gradio_client.file("/app/shared/data/tmp.png"),
        question,
        api_name="/answer_question"
    )
    print(f"[Gradio] Time taken: {time.time() - start_time}")
    return {"result": result}

def qwen_vl_max_answer_question_from_image(image: Image, question: str) -> Dict[str, Any]:
    client = Client("https://qwen-qwen-vl-max.hf.space/--replicas/fi9fr/")

    image.save("/app/shared/data/tmp.png")
    start_time = time.time()
    # result = client.predict(
	# 	fn_index=3
    # )

    # json_str = "/tmp/gradio/tmp0af5pyui.json"
    # result = client.predict(
	# 	json_str,
	# 	img_path,	# str (filepath on your computer (or URL) of file) in '📁 Upload (上传文件)' Uploadbutton component
	# 	fn_index=5
    # )

    result = client.predict(
        # json_str,
        # "Hi",
        fn_index=2
    )

    
    print(f"[Gradio] Time taken: {time.time() - start_time}")
    return {"result": result}