from PIL import Image
from typing import Dict, Any
import replicate
import time


def moondream_answer_question_from_image(image: Image, question: str) -> Dict[str, Any]:
    image.save("/app/shared/data/tmp.png")
    image_handler = open("/app/shared/data/tmp.png", "rb")

    input = {"image": image_handler, "prompt": question}

    start_time = time.time()
    output = replicate.run(
        "lucataco/moondream2:392a53ac3f36d630d2d07ce0e78142acaccc338d6caeeb8ca552fe5baca2781e",
        input=input,
    )
    output = "".join(output)
    print(f"[Replicate] Time taken: {time.time() - start_time}")
    return {"result": output}
