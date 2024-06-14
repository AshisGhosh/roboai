import argparse
import json
import re
import os
import datetime as dt
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process an image file or a directory of images."
    )
    parser.add_argument(
        "path", type=str, help="Path to an image file or a directory containing images."
    )
    return parser.parse_args()


def get_timestamp():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def initialize_model(model_id):
    processor = AutoProcessor.from_pretrained(
        model_id,
        size={"longest_edge": 672, "shortest_edge": 672},
        do_image_splitting=False,
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )
    return processor, model


def process_images(image_paths, processor, model, prompt, device):
    results = {}
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            prompt_full = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = processor(text=prompt_full, images=[image], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated_ids = model.generate(**inputs, max_new_tokens=500)
            generated_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            results[image_path.name] = {"response": generated_texts[0]}
            print(f"{image_path.name}: {generated_texts[0]}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results[image_path.name] = {"error": str(e)}
    return results


def adjust_file_permissions(output_file):
    """
    Fixes file permission when running script as root in docker
    """
    if os.getuid() == 0:  # Running as root
        host_uid, host_gid = 1000, 1000  # Default IDs, adjust as necessary
        os.chown(output_file, host_uid, host_gid)
        print(f"File ownership changed for {output_file}")


def main():
    args = parse_arguments()
    device = "cuda:0"
    model_id = "HuggingFaceM4/idefics2-8b-chatty"
    prompt = "Name the objects on the table from left to right."
    processor, model = initialize_model(model_id)

    start_time = get_timestamp()
    path = Path(args.path)
    results = {}
    if path.exists():
        if path.is_dir():
            image_paths = sorted(list(path.glob("rgb_????.png")))
            results = process_images(image_paths, processor, model, prompt, device)
        elif path.is_file() and re.match(r"rgb_\d{4}\.png", path.name):
            results = process_images([path], processor, model, prompt, device)
        else:
            print("The file or directory does not match the expected pattern.")
    else:
        print("The provided path does not exist.")

    end_time = get_timestamp()
    output_data = {
        "model_id": model_id,
        "prompt": prompt,
        "start_time": start_time,
        "end_time": end_time,
        "responses": results,
    }
    output_file = (
        f"idefics2_responses_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Output written to {output_file}")
    # Change file ownership to host user if script is running as root in docker
    adjust_file_permissions(output_file)


if __name__ == "__main__":
    main()
