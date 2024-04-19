# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

from PIL import Image
import time

import logging
log = logging.getLogger("model-server")
log.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
log.addHandler(handler)

class HuggingFaceIdefics:
    def __init__(self):
        model_load_start = time.time()
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        self.model =  AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b").to("cpu")
        log.info(f"Model loaded in {time.time() - model_load_start} seconds.")
    
    def answer_question_from_image(self, image, question):
        image1 = load_image("/app/shared/data/test2.png")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What do we see in this image?"},
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image1], return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)


        return generated_texts
    


if __name__ == "__main__":
    model = HuggingFaceIdefics()
    img_path = "/app/shared/data/test2.png"
    image = Image.open(img_path)
    question = "Describe this image."
    print(model.answer_question_from_image(image, question))
