import argparse
import json
import re
import os
import datetime as dt
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process an image file or a directory of images.")
    parser.add_argument('path', type=str, help='Path to an image file or a directory containing images.')
    return parser.parse_args()

def get_timestamp():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def initialize_model(device, dtype, model_id):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=device,
        revision="bfloat16",
        quantization_config=quantization_config
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return processor, model

def process_images(image_paths, processor, model):
    results = {}
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = "Name the objects on the table from left to right."
            model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            input_len = model_inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]
                decoded = processor.decode(generation, skip_special_tokens=True)
                print(decoded)
            results[image_path.name] = {"response": decoded}
            print(f"{image_path.name}: {decoded}")
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
    dtype = torch.bfloat16
    model_id = "../models/paligemma-3b-mix-448"
    processor, model = initialize_model(device, dtype, model_id)

    start_time = get_timestamp()
    path = Path(args.path)
    results = {}
    if path.exists():
        if path.is_dir():
            image_paths = sorted(list(path.glob('rgb_????.png')))
            results = process_images(image_paths, processor, model)
        elif path.is_file() and re.match(r'rgb_\d{4}\.png', path.name):
            results = process_images([path], processor, model)
        else:
            print("The file or directory does not match the expected pattern.")
    else:
        print("The provided path does not exist.")

    end_time = get_timestamp()
    output_data = {"model_id": model_id, "start_time": start_time, "end_time": end_time, "responses": results}
    output_file = f"paligemma_responses_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Output written to {output_file}")
    # Change file ownership to host user if script is running as root in docker
    adjust_file_permissions(output_file)

if __name__ == "__main__":
    main()