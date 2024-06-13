import argparse
import json
import re
import os
import datetime as dt
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process an image file or a directory of images.")
    parser.add_argument('path', type=str, help='Path to an image file or a directory containing images.')
    return parser.parse_args()

def get_timestamp():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def initialize_model(model_id):
    model = AutoModel.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer

def process_images(image_paths, tokenizer, model, prompt):
    results = {}
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            msgs = [{'role': 'user', 'content': prompt}]

            response = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True, # if sampling=False, beam_search will be used by default
                temperature=0.7,
                # system_prompt='' # pass system_prompt if needed
            )
            results[image_path.name] = {"response": response}
            print(f"{image_path.name}: {response}")
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
    model_id = "openbmb/MiniCPM-Llama3-V-2_5-int4"
    prompt = "Name the objects on the table from left to right."
    model, tokenizer = initialize_model(model_id)
    model.eval()
    
    start_time = get_timestamp()
    path = Path(args.path)
    results = {}
    if path.exists():
        if path.is_dir():
            image_paths = sorted(list(path.glob('rgb_????.png')))
            results = process_images(image_paths, tokenizer, model, prompt)
        elif path.is_file() and re.match(r'rgb_\d{4}\.png', path.name):
            results = process_images([path], tokenizer, model, prompt)
        else:
            print("The file or directory does not match the expected pattern.")
    else:
        print("The provided path does not exist.")

    end_time = get_timestamp()
    output_data = {"model_id": model_id, "prompt": prompt, "start_time": start_time, "end_time": end_time, "responses": results}
    output_file = f"minicpm-llama3-2-5-int4_responses_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Output written to {output_file}")
    # Change file ownership to host user if script is running as root in docker
    adjust_file_permissions(output_file)


if __name__ == "__main__":
    main()