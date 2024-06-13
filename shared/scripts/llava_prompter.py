import argparse
import json
import re
import io
import os
import base64
from pathlib import Path
from PIL import Image
import datetime as dt
import ollama


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process an image file or a directory of images.")
    parser.add_argument('path', type=str, help='Path to an image file or a directory containing images.')
    return parser.parse_args()

def get_timestamp():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def preload_model(model_id):
    try:
        preload_response = ollama.generate(model=model_id, keep_alive=-1)
        print("Model preloaded successfully.\n", preload_response)
    except ollama.ResponseError as e:
        print(f"ollama Response Error during preload: {e.error}")
    except Exception as err:
        print(f"Unexpected error during model preload: {err}")

def unload_model(model_id):
    try:
        unload_response = ollama.generate(model=model_id, keep_alive=0)
        print("Model unloaded successfully.\n", unload_response)
    except ollama.ResponseError as e:
        print(f"ollama Response Error during unload: {e.error}")
    except Exception as err:
        print(f"Unexpected error during model unload: {err}")

def encode_image_to_base64(resized_image):
    buffered = io.BytesIO()
    resized_image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_string

def get_caption_with_image(image_path, model_id, prompt):
    try:
        with Image.open(image_path) as img:
            resized_image = img.resize((672, 672))  # max rez for llava
            resized_image_base64 = encode_image_to_base64(resized_image)
        response = ollama.generate(model=model_id, prompt=prompt, format="json", stream=False, images=[resized_image_base64])
        print(f"{image_path.name}: {response}")
        return response
    except ollama.ResponseError as e:
        print(f"ollama Response Error: {e.error}")
        return {"error": "ollama Response Error", "status_code": e.status_code, "details": e.error}
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
        return {"error": "Unexpected error", "details": str(err)}

def process_images(image_paths, model_id, prompt):
    results = {}
    for image_path in image_paths:
        answer = get_caption_with_image(image_path, model_id, prompt)
        results[image_path.name] = {"response": answer}
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
    model_id = "llava:latest"
    prompt = """Name the objects on the table from left to right. Respond using JSON. 
    response format: {"objects_left_to_right:" ["<first object from left>", "<next object from left>", "<last object from left>"]}"""

    start_time = get_timestamp()
    path = Path(args.path)
    if path.is_dir():
        # Process all images that match the specific pattern
        image_paths = sorted(list(path.glob('rgb_????.png')))
        preload_model(model_id)
        results = process_images(image_paths, model_id, prompt)
        unload_model(model_id)
    elif path.is_file():
        # Check if the file matches the pattern and process
        if re.match(r'rgb_\d{4}\.png', path.name):
            preload_model(model_id)
            results = process_images([path], model_id, prompt)
            unload_model(model_id)
        else:
            print("The file does not match the expected 'rgb_XXXX.png' pattern.")
    else:
        print("The provided path does not exist or is not a file/directory.")

    end_time = get_timestamp()
    output_data = {"model_id": model_id, "prompt": prompt, "start_time": start_time, "end_time": end_time, "responses": results}
    output_file = f"llava_responses_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Output written to {output_file}")
    # Change file ownership to host user if script is running as root in docker
    adjust_file_permissions(output_file)


if __name__ == "__main__":
    args = parse_arguments()
    main()