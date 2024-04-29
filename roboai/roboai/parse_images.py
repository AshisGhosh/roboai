import datetime as dt
import os
import json
import sys
import base64
import argparse
import logging
import ollama
import pandas as pd

from dotenv import load_dotenv
from html import escape

# Configure logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("roboai")
load_dotenv("shared/.env")

def get_model_info(model_name):
    model_data = ollama.show(model_name)
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

def preload_model(model_name):
    try:
        preload_response = ollama.generate(model=model_name, keep_alive=-1)
        print("Model preloaded successfully.\n", preload_response)
    except ollama.ResponseError as e:
        logger.error(f"ollama Response Error during preload: {e.error}")
    except Exception as err:
        logger.error(f"Unexpected error during model preload: {err}")

def unload_model(model_name):
    try:
        unload_response = ollama.generate(model=model_name, keep_alive=0)
        print("Model unloaded successfully.\n", unload_response)
    except ollama.ResponseError as e:
        logger.error(f"ollama Response Error during unload: {e.error}")
    except Exception as err:
        logger.error(f"Unexpected error during model unload: {err}")

def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_caption_with_image(image_path, model_name, prompt):
    try:
        image_base64 = encode_image_to_base64(image_path)
        response = ollama.generate(model=model_name, prompt=prompt, stream=False, images=[image_base64])
        return response
    except ollama.ResponseError as e:
        logger.error(f"ollama Response Error: {e.error}")
        return {"error": "ollama Response Error", "status_code": e.status_code, "details": e.error}
    except Exception as err:
        logger.error(f"An unexpected error occurred: {err}")  # Log unexpected errors
        return {"error": "Unexpected error", "details": str(err)}

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

def html_from_output_json(json_file_path, html_output_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        results = data['results']
        sim_info = data['sim_info']['runs']
        rows = []
        for filename, content in results.items():
            response = escape(content.get('response', '')).replace("\n", "<br>")
            details_json = json.dumps(content, indent=4).replace("\n", "<br>")
            details = f"<details><summary>View Details</summary><pre>{details_json}</pre></details>"

            sim_details = next((run for run in sim_info if run['image_file'] == filename), None)
            sim_details_json = json.dumps(sim_details, indent=4).replace("\n", "<br>") if sim_details else "No sim details available"
            sim_details_formatted = f"<details><summary>View Sim Details</summary><pre><code>{sim_details_json}</code></pre></details>"

            rows.append({
                'Run': filename.split('.')[0],
                'Image Filename': filename,
                'Image': f'<img src="../../{filename}" alt="{filename}" class="expandable">',
                'Prompt': escape(data['prompt:']).replace("\n", "<br>"),
                'Response Message': response,
                'Full Response': details,
                'Sim Details': sim_details_formatted
            })
        df = pd.DataFrame(rows)
        
        html_style_script = '''
        <style>
            body { background-color: #263238; color: #ECEFF1; font-family: monospace; font-size: 1.1em; }
            pre { white-space: pre-wrap; font-size: smaller; }
            details summary { cursor: pointer; }
            img.expandable {width: 100px; height: auto; cursor: pointer; transition: transform 0.25s ease; }
            img.expandable:hover { transform: scale(1.05); }
            table { width: 100%; border-collapse: collapse; }
            th, td { border: 1px solid #37474F; padding: 12px; text-align: left; font-size: 0.9em; overflow: visible; position: relative; }
        </style>
        <script>
            document.addEventListener('DOMContentLoaded', function () {
                document.querySelectorAll('img.expandable').forEach(img => {
                    img.onclick = function () {
                        const cell = img.closest('td'); // Get the parent cell of the image
                        if (img.style.width === '100px' || img.style.width === '') { // Check current state
                            img.style.position = 'absolute';
                            img.style.width = 'auto'; // Allow the image to grow naturally
                            img.style.maxWidth = '100%'; // Ensure it does not exceed the screen width
                            img.style.zIndex = '1000'; // Ensure it overlays other content
                            cell.style.position = 'static'; // Ensure cell does not limit the image size
                        } else {
                            img.style.position = '';
                            img.style.width = '100px'; // Reset to thumbnail size
                            img.style.maxWidth = ''; // Remove max width constraint
                            img.style.zIndex = '';
                        }
                    };
                });
            });
        </script>
        '''
        
        html_table = df.to_html(escape=False, index=False)
        html_content = f"{html_style_script}{html_table}"

        with open(html_output_path, 'w') as f:
            f.write(html_content)

        print(f"HTML output generated at {html_output_path}")
        
    except Exception as e:
        print(f"An error occurred while generating HTML: {e}")

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
    prompt = "You are a robot with a camera input. This image is the feed from your camera pointed at a wooden table with objects on it. What are these objects? Visualize the table divided into quadrants relative to the perspective of the image: top-left, top-right, bottom-left, bottom-right. What objects are located in which quadrants?"    

    model_info = get_model_info(model_name)
    if model_info:
        logger.info(f"Model Info: {model_info}")
        results, sim_metadata = process_dir(dir_path, model_name, prompt)
        compare_results(results)

        output_json_path = os.path.join(analysis_session_dir, 'output.json')
        output_html_path = os.path.join(analysis_session_dir, 'output.html')
        generate_json_output(results, sim_metadata, model_info, prompt, output_file=output_json_path)
        html_from_output_json(output_json_path, output_html_path)

        # Set file permissions right after creation
        os.chmod(output_json_path, 0o644)
        os.chmod(output_html_path, 0o644)
        if os.getuid() == 0:  # Optionally change ownership to match the host user, if the script runs as root
            host_uid = 1000  # Replace with actual host user ID
            host_gid = 1000  # Replace with actual host group ID
            os.chown(analysis_dir, host_uid, host_gid)
            os.chown(analysis_session_dir, host_uid, host_gid)
            os.chown(output_json_path, host_uid, host_gid)
            os.chown(output_html_path, host_uid, host_gid)
    else:
        logger.error("Failed to fetch model information. Aborting image processing.")


if __name__ == "__main__":
    main()
