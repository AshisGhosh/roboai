import argparse
import json
import re
import os
import datetime as dt
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process an image file or a directory of images.")
    parser.add_argument('path', type=str, help='Path to an image file or a directory containing images.')
    return parser.parse_args()

def get_timestamp():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def initialize_model(device, model_id, revision):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

def process_images(image_paths, model, tokenizer):
    results = {}
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            enc_image = model.encode_image(image)
            answer = model.answer_question(enc_image, "Name the objects on the table from left to right.", tokenizer)
            results[image_path.name] = {"response": answer}
            print(f"{image_path.name}: {answer}")
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
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-08"
    model, tokenizer = initialize_model(device, model_id, revision)

    start_time = get_timestamp()
    path = Path(args.path)
    results = {}
    if path.exists():
        if path.is_dir():
            image_paths = sorted(list(path.glob('rgb_????.png')))
            results = process_images(image_paths, model, tokenizer)
        elif path.is_file() and re.match(r'rgb_\d{4}\.png', path.name):
            results = process_print([path], model, tokenizer)
        else:
            print("The file or directory does not match the expected pattern.")
    else:
        print("The provided path does not exist.")

    end_time = get_timestamp()
    output_data = {"model_id": model_id, "start_time": start_time, "end_time": end_time, "responses": results}
    output_file = f"moondream2_responses_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Output written to {output_file}")
    # Change file ownership to host user if script is running as root in docker
    adjust_file_permissions(output_file)


if __name__ == "__main__":
    main()
