import os
import json
import base64
import argparse
import logging
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
    response = completion(
        model="openrouter/huggingfaceh4/zephyr-7b-beta:free",
        messages=[
            {
                "role": "system",
                "content": "You are a robot looking at a tabletop. Your camera detects various objects on the table and the image file you receive in base64 is the feed. Compared to the milk carton, rank the objects in order from closest to farthest."
            },
            {
                "role": "user",
                "content": {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + image_base64,  # Correct format for embedding base64 images
                        "detail": "auto"
                    }
                }
            }
        ]
    )
    return response["choices"][0]["message"]["content"]

def process_folder(folder_path):
    metadata_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}.json")
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            try:
                caption = get_caption_with_image(image_path)
                results[filename] = caption
                print(f"Processed {filename}: {caption}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return results, metadata

def compare_results(results, metadata):
    # Placeholder for comparison logic
    print("Comparison results (placeholder):", results)

def main():
    parser = argparse.ArgumentParser(description="Process image folder and compare LLM captions.")
    parser.add_argument("folder_name", help="Name of the folder containing the dataset")
    args = parser.parse_args()

    folder_path = os.path.join("/app/shared/data/image_exports", args.folder_name)
    if not os.path.isdir(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return

    results, metadata = process_folder(folder_path)
    compare_results(results, metadata)

if __name__ == "__main__":
    main()
