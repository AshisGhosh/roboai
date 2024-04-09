from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time

import logging
log = logging.getLogger("model-server")
log.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
log.addHandler(handler)

class HuggingFaceMoonDream2:
    def __init__(self):
        self.model_id = "vikhyatk/moondream2"
        self.revision = "2024-04-02"
        model_load_start = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, revision=self.revision
        )
        log.info(f"Model loaded in {time.time() - model_load_start} seconds.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
    
    def encode_image(self, image):
        start_encode = time.time()
        encoded_image = self.model.encode_image(image)
        log.info(f"Image encoded in {time.time() - start_encode} seconds.")
        return encoded_image
    
    def answer_question(self, enc_image, question):
        start_model = time.time()
        answer = self.model.answer_question(enc_image, question, self.tokenizer)
        log.info(f"Answered question in {time.time() - start_model} seconds.")
        return answer
    
    def answer_question_from_image(self, image, question):
        enc_image = self.encode_image(image)
        return self.answer_question(enc_image, question)
    


if __name__ == "__main__":
    model = HuggingFaceMoonDream2()
    img_path = "/app/shared/data/test2.png"
    image = Image.open(img_path)
    enc_image = model.encode_image(image)
    question = "Describe this image."
    print(model.answer_question(enc_image, question))
