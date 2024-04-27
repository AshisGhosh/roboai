# to do: replace httpx with ollama wrapper


import datetime as dt
import os
import json
import sys
import base64
import argparse
import logging
import httpx
from dotenv import load_dotenv
# from litellm import completion
# import ollama
# import pandas as pd


# Configure logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("roboai")
load_dotenv("shared/.env")

def get_model_info(model_name):
    api_url = "http://localhost:11434/api/show"
    data = {"name": model_name}
    try:
        with httpx.Client() as client:
            response = client.post(api_url, json=data)
            response.raise_for_status()
            model_data = response.json()
            # Construct a dictionary with the relevant fields including model name
            model_info = {
                "name": model_name,
                "modelfile": model_data.get("modelfile"),
                "parameters": model_data.get("parameters"),
                "template": model_data.get("template"),
                "details": {
                    "format": model_data.get("details", {}).get("format"),
                    "family": model_data.get("details", {}).get("family"),
                    "families": model_data.get("details", {}).get("families"),
                    "parameter_size": model_data.get("details", {}).get("parameter_size"),
                    "quantization_level": model_data.get("details", {}).get("quantization_level")
                }
            }
            return model_info
    except httpx.RequestError as e:
        logger.error(f"An error occurred while requesting {e.request.url!r}.")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error response {e.response.status_code} while requesting {e.request.url!r}.")
    return {}

def preload_model(model_name):
    api_url = "http://localhost:11434/api/generate"
    data = {"model": model_name, "keep_alive": -1}
    try:
        with httpx.Client() as client:
            response = client.post(api_url, json=data)
            response.raise_for_status()
            print("Model preloaded successfully.")
    except httpx.RequestError as e:
        logger.error(f"Request error during model preloading: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during model preloading: {e.response.status_code}")

def unload_model(model_name):
    api_url = "http://localhost:11434/api/generate"
    data = {"model": model_name, "keep_alive": 0}
    try:
        with httpx.Client() as client:
            response = client.post(api_url, json=data)
            response.raise_for_status()
            print("Model unloaded successfully.")
    except httpx.RequestError as e:
        logger.error(f"Request error during model unloading: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during model unloading: {e.response.status_code}")

def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_caption_with_image(image_path, model_name, prompt):
    """using generate endpoint instead of chat"""
    api_url = "http://localhost:11434/api/generate"
    image_base64 = encode_image_to_base64(image_path)
    data = {"model": model_name, "prompt": prompt, "stream": False, "images": [image_base64]}
    
    try:
        with httpx.Client() as client:
            response = client.post(api_url, json=data)
            response.raise_for_status()
            json_response = response.json()
            if 'response' not in json_response:
                logger.error(f"Missing 'response' key in response for image {image_path}. Response: {json_response}")
                return {"error": "Missing 'response' key", "response": json_response}
            return json_response
    except httpx.RequestError as e:
        logger.error(f"Request error for {image_path}: {str(e)}")
        return {"error": "Request error", "details": str(e)}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error for {image_path}: Status {e.response.status_code}")
        return {"error": "HTTP error", "status": e.response.status_code, "details": str(e)}

# def get_caption_with_image(image_path):
#     """LiteLLM version, may need update"""
#     image_base64 = encode_image_to_base64(image_path)
#     prompt = "This image is a robot's 3rd-person view of a wooden table with objects on it. What are these objects? Imagine the table divided into quadrants with respect to your view: top/bottom, left/right. Which quadrants are each object in?"
#     try:
#         response = completion(
#             model="ollama/llava",
#             api_base="http://localhost:11434",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": prompt
#                         },
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": image_base64
#                             }
#                         }
#                     ]
#                 }
#             ]
#         )
#         # Extract only the JSON content or relevant parts of the response
#         return response.json()  # Assuming 'completion' returns a response object like 'httpx.Response'
#     except Exception as e:
#         logger.error(f"Failed to get caption for image {image_path}: {e}")
#         return {}  # Return an empty dict in case of failure

def get_dir_path(latest_symlink_path, dir_name=None):
    print(f"Latest symlink path: {latest_symlink_path}")
    print(f"Directory name: {dir_name}")
    
    if dir_name:
        dir_path = os.path.join("/app/shared/data/image_exports", dir_name)
    else:
        try:
            resolved_symlink = os.path.join(os.path.dirname(latest_symlink_path), os.readlink(latest_symlink_path))
            print(f"Resolved symlink '{latest_symlink_path}' to directory '{resolved_symlink}'")
            dir_path = resolved_symlink
        except OSError as e:
            print(f"Failed to read symlink {latest_symlink_path}: {e}")
            sys.exit(1)
            
    print(f"Final directory path: {dir_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory exists: {os.path.isdir(dir_path)}")
    
    if not os.path.isdir(dir_path):
        print(f"Directory '{dir_path}' not found.")
        sys.exit(1)
    
    return dir_path

def process_dir(dir_path, model_name, prompt):
    """preloads model, makes image/prompt calls to mode, then closes model"""
    sim_metadata_path = os.path.join(dir_path, f"{os.path.basename(dir_path)}.json")
    with open(sim_metadata_path, 'r') as file:
        sim_metadata = json.load(file)

    preload_model(model_name)

    results = {}
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".png"):
            image_path = os.path.join(dir_path, filename)
            # This function call blocks until the response is received
            api_response = get_caption_with_image(image_path, model_name, prompt)
            if 'response' in api_response:  # Handling direct ollama API response for 'generate' endpoint
                results[filename] = api_response
                print(f"Processed {filename}: {api_response.get('response', 'No content')}")
            elif 'message' in api_response:  # Handling litellm & ollama 'chat' endpoint
                message_content = api_response['message'].get('content', 'No content') if isinstance(api_response['message'], dict) else 'No content'
                results[filename] = api_response
                print(f"Processed {filename}: {message_content}")
            else:
                error_message = f"Error processing {filename}: {api_response.get('error', 'Unknown error')}"
                print(error_message)
                results[filename] = {"error": "Failed to process image", "details": api_response}

    unload_model(model_name)
    return results, sim_metadata

def compare_results(results):
    # Placeholder for comparison logic
    print("Comparison results (placeholder):", results)

def generate_json_output(results, sim_metadata, model_info, prompt, output_file='output.json'):
    analysis_start_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    output_data = {
        "sim_info": sim_metadata,
        "model_info": model_info,
        "analysis_start_time": analysis_start_time,
        "prompt:": prompt,
        "results": results
    }

    # Debugging: Log each item's type in results to check for non-serializable objects
    for key, value in results.items():
        logger.debug(f"Key: {key}, Type of value: {type(value)}")

    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)
    logger.info(f"JSON output generated at {output_file}")

def load_output_json(o_json_file_path):
    try:
        with open(o_json_file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {o_json_file_path} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {o_json_file_path} could not be decoded.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def html_from_output_json():
    pass

def main():
    parser = argparse.ArgumentParser(description="Process image dir and compare LLM captions.")
    parser.add_argument("dir_name", nargs="?", help="Name of the dir containing the dataset. If not specified, the script will use the 'latest' symlink.")
    args = parser.parse_args()

    latest_symlink_path = "/app/shared/data/image_exports/latest"
    dir_path = get_dir_path(latest_symlink_path, args.dir_name)

    analysis_dir = os.path.join(dir_path, "analysis")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
        os.chmod(analysis_dir, 0o755)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_session_dir = os.path.join(analysis_dir, timestamp)
    os.makedirs(analysis_session_dir)
    os.chmod(analysis_session_dir, 0o755)

    model_name = "llava"
    prompt = "This image is a robot's 3rd-person view of a wooden table with objects on it. What are these objects? Imagine the table divided into quadrants with respect to your view: top/bottom, left/right. Which quadrants are each object in?"    

    model_info = get_model_info(model_name)
    if model_info:
        print(f"Model Info: {model_info}")
        results, sim_metadata = process_dir(dir_path, model_name, prompt)
        compare_results(results)

        output_json_path = os.path.join(analysis_session_dir, 'output.json')
        generate_json_output(results, sim_metadata, model_info, prompt, output_file=output_json_path)
    else:
        print("Failed to fetch model information. Aborting image processing.")

    # Set file permissions right after creation
    os.chmod(output_json_path, 0o644)
    if os.getuid() == 0:  # Optionally change ownership to match the host user, if the script runs as root
        host_uid = 1000  # Replace with actual host user ID
        host_gid = 1000  # Replace with actual host group ID
        os.chown(analysis_dir, host_uid, host_gid)
        os.chown(analysis_session_dir, host_uid, host_gid)
        os.chown(output_json_path, host_uid, host_gid)

if __name__ == "__main__":
    main()
