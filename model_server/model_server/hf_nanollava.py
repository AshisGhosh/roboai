from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
import torch

import logging

log = logging.getLogger("model-server")
log.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
log.addHandler(handler)


class HuggingFaceNanoLLaVA:
    def __init__(self):
        torch.set_default_device("cpu")

        model_load_start = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            "qnguyen3/nanoLLaVA",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        log.info(f"Model loaded in {time.time() - model_load_start} seconds.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "qnguyen3/nanoLLaVA", trust_remote_code=True
        )

    def process_image(self, image):
        start_process = time.time()
        image_tensor = self.model.process_images([image], model.config).to(
            dtype=model.dtype
        )
        log.info(f"Image processed in {time.time() - start_process} seconds.")
        return image_tensor

    def answer_question(self, image_tensor, prompt):
        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_chunks = [
            self.tokenizer(chunk).input_ids for chunk in text.split("<image>")
        ]
        input_ids = torch.tensor(
            text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long
        ).unsqueeze(0)
        start_model = time.time()
        output_ids = model.generate(
            input_ids, images=image_tensor, max_new_tokens=2048, use_cache=True
        )[0]
        log.info(f"Answered question in {time.time() - start_model} seconds.")
        output = self.tokenizer.decode(
            output_ids[input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        return output


if __name__ == "__main__":
    model = HuggingFaceNanoLLaVA()
    img_path = "/app/shared/data/test2.png"
    image = Image.open(img_path)
    image_tensor = model.encode_image(image)
    question = "Describe this image."
    print(model.answer_question(image_tensor, question))
