import argparse
import json
import re
import base64
from pathlib import Path
from PIL import Image
import ollama

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Process an image file or a directory of images.")
parser.add_argument('path', type=str, help='Path to an image file or a directory containing images.')
args = parser.parse_args()

model_name = "llava:latest"
prompt = """Name the objects on the table from left to right as a list. Respond using JSON. 
ex: 
{"objects_left_to_right:" ["first object from left", "next object from left", "last object from left"]}"""

def preload_model(model_name):
    try:
        preload_response = ollama.generate(model=model_name, keep_alive=-1)
        print("Model preloaded successfully.\n", preload_response)
    except ollama.ResponseError as e:
        print(f"ollama Response Error during preload: {e.error}")
    except Exception as err:
        print(f"Unexpected error during model preload: {err}")

def unload_model(model_name):
    try:
        unload_response = ollama.generate(model=model_name, keep_alive=0)
        print("Model unloaded successfully.\n", unload_response)
    except ollama.ResponseError as e:
        print(f"ollama Response Error during unload: {e.error}")
    except Exception as err:
        print(f"Unexpected error during model unload: {err}")

def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_caption_with_image(image_path, model_name, prompt):
    try:
        image_base64 = encode_image_to_base64(image_path)
        response = ollama.generate(model=model_name, prompt=prompt, format="json", stream=False, images=[image_base64])
        return response
    except ollama.ResponseError as e:
        print(f"ollama Response Error: {e.error}")
        return {"error": "ollama Response Error", "status_code": e.status_code, "details": e.error}
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
        return {"error": "Unexpected error", "details": str(err)}

def process_images(image_paths, model_name, prompt):
    results = {}
    for image_path in image_paths:
        answer = get_caption_with_image(image_path, model_name, prompt)
        results[image_path.name] = {"response": answer}
    return results

# Check if the path is a directory or a file
path = Path(args.path)
if path.is_dir():
    # Process all images that match the specific pattern
    image_paths = sorted(list(path.glob('rgb_????.png')))
    preload_model(model_name)
    results = process_images(image_paths, model_name, prompt)
    unload_model(model_name)
elif path.is_file():
    # Check if the file matches the pattern and process
    if re.match(r'rgb_\d{4}\.png', path.name):
        preload_model(model_name)
        results = process_images([path], model_name, prompt)
        unload_model(model_name)
    else:
        print("The file does not match the expected 'rgb_XXXX.png' pattern.")
else:
    print("The provided path does not exist or is not a file/directory.")

print(json.dumps(results, indent=4))
