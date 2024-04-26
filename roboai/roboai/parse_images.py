import datetime as dt
import os
import json
import base64
import argparse
import logging
from PIL import Image
from dotenv import load_dotenv
from litellm import completion

# Configure logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("roboai")
load_dotenv("shared/.env")

def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_caption_with_image(image_path):
    image_base64 = encode_image_to_base64(image_path)
    prompt_text = "This image is a robot's 3rd-person view of a wooden table with objects on it. What are these objects? Imagine the table divided into quadrants with respect to your view: top/bottom, left/right. Which quadrants are each object in?"
    response = completion(
        model="ollama/llava",
        api_base="http://localhost:11434",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        }
                    }
                ]
            }
        ]
    )
    caption = response["choices"][0]["message"]["content"]
    return caption, prompt_text

def process_folder(folder_path):
    sim_metadata_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}.json")
    with open(sim_metadata_path, 'r') as file:
        sim_metadata = json.load(file)
    
    sim_metadata['folder_path'] = folder_path 

    results = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            try:
                caption, prompt = get_caption_with_image(image_path)
                results[filename] = {
                    'caption': caption,
                    'prompt': prompt  
                }
                print(f"Processed {filename}: {caption}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    return results, sim_metadata

def compare_results(results):
    # Placeholder for comparison logic
    print("Comparison results (placeholder):", results)

def set_permissions(path):
    os.chmod(path, 0o644)

def generate_json_output(results, sim_metadata, output_file='output.json'):
    analysis_start_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_data = {
        'sim_info': sim_metadata,
        'analysis_start_time': f'{analysis_start_time} UTC',
        'results': results
    }
    
    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)
    print(f"JSON output generated at {output_file}")

def generate_html_output(results, sim_metadata, output_file='output.html'):
    current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = [
        '<html>',
        '<head><title>Roboai Simulation Image Analysis</title></head>',
        '<body>',
        '<h1>Roboai Simulation Image Analysis</h1>',
        f'<p>Generated at {current_time} UTC</p>',
        '<table border="1">',
        '<tr><th>Run</th><th>Image</th><th>Details</th><th>Prompt</th><th>LLM Output</th></tr>'
    ]

    for run_info in sim_metadata['runs']:
        run_number = run_info['run']
        image_file = run_info['image_file']
        html_image_path = f"../../{image_file}"  # Correct, assuming HTML is two levels deeper

        # Add the main image row
        html_content.append('<tr>')
        html_content.append(f'<td>{run_number}</td>')
        html_content.append(f'<td><img src="{html_image_path}" alt="{html_image_path}"></td>')
        html_content.append(f'<td><pre>{json.dumps(run_info, indent=4)}</pre></td>')
        html_content.append(f'<td>{results[image_file]["prompt"]}</td>')
        html_content.append(f'<td>{results[image_file]["caption"]}</td>')
        html_content.append('</tr>')

        # Check for and add a flipped image row
        flipped_image_file = image_file.replace('.png', '_flipped.png')
        flipped_image_path = f"../../{flipped_image_file}"  # Relative path must match the file structure
        full_flipped_image_path = os.path.join(sim_metadata['folder_path'], flipped_image_file)  # Full path for existence check
        if os.path.exists(full_flipped_image_path):
            html_content.append('<tr>')
            html_content.append(f'<td>{run_number} (Flipped)</td>')
            html_content.append(f'<td><img src="{flipped_image_path}" alt="{flipped_image_path}"></td>')
            html_content.append('<td>-</td>')
            html_content.append(f'<td>{results[flipped_image_file]["prompt"]}</td>')
            html_content.append(f'<td>{results[flipped_image_file]["caption"]}</td>')
            html_content.append('</tr>')

    html_content.append('</table>')
    html_content.append('</body></html>')

    with open(output_file, 'w') as file:
        file.write('\n'.join(html_content))
    print(f"HTML output generated at {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process image folder and compare LLM captions.")
    parser.add_argument("folder_name", nargs="?", help="Name of the folder containing the dataset. If not specified, the script will use the 'latest' symlink.")
    args = parser.parse_args()

    latest_symlink_path = "/app/shared/data/image_exports/latest"
    folder_name = args.folder_name if args.folder_name else os.path.basename(os.readlink(latest_symlink_path))
    folder_path = os.path.join("/app/shared/data/image_exports", folder_name)
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_name}' not found.")
        sys.exit(1)

    analysis_folder = os.path.join(folder_path, "analysis")
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)
        os.chmod(analysis_folder, 0o755)  # Set directory permissions to 755 immediately after creation

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    specific_analysis_folder = os.path.join(analysis_folder, timestamp)
    os.makedirs(specific_analysis_folder)
    os.chmod(specific_analysis_folder, 0o755)  # Set directory permissions to 755

    results, sim_metadata = process_folder(folder_path)
    compare_results(results)

    # Generate outputs in the new specific analysis subfolder
    output_html_path = os.path.join(specific_analysis_folder, 'output.html')
    output_json_path = os.path.join(specific_analysis_folder, 'output.json')
    generate_html_output(results, sim_metadata, output_file=output_html_path)
    generate_json_output(results, sim_metadata, output_file=output_json_path)
    
    # Set file permissions right after creation
    os.chmod(output_html_path, 0o644)
    os.chmod(output_json_path, 0o644)

    # Optionally change ownership to match the host user, if the script runs as root
    if os.getuid() == 0:  # Check if the current script is running as root
        host_uid = 1000  # Replace with actual host user ID
        host_gid = 1000  # Replace with actual host group ID
        os.chown(analysis_folder, host_uid, host_gid)
        os.chown(specific_analysis_folder, host_uid, host_gid)
        os.chown(output_html_path, host_uid, host_gid)
        os.chown(output_json_path, host_uid, host_gid)


if __name__ == "__main__":
    main()
