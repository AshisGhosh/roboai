import argparse
import json
import re
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Process an image file or a directory of images.")
parser.add_argument('path', type=str, help='Path to an image file or a directory containing images.')
args = parser.parse_args()

device = "cuda:0"
model_id = "vikhyatk/moondream2"
revision = "2024-05-08"

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16, attn_implementation="flash_attention_2"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Define a function to process images sequentially
def process_images(image_paths):
    results = {}
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            enc_image = model.encode_image(image)
            answer = model.answer_question(enc_image, "Name the objects on the table from left to right.", tokenizer)
            results[image_path.name] = {"response": answer}
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results[image_path.name] = {"error": str(e)}
    return results

# Check if the path is a directory or a file
path = Path(args.path)
if path.is_dir():
    # Process all images that match the specific pattern
    image_paths = sorted(list(path.glob('rgb_????.png')))
    results = process_images(image_paths)
elif path.is_file():
    # Check if the file matches the pattern and process
    if re.match(r'rgb_\d{4}\.png', path.name):
        results = process_images([path])
    else:
        print("The file does not match the expected 'rgb_XXXX.png' pattern.")
else:
    print("The provided path does not exist or is not a file/directory.")


print(json.dumps(results, indent=4))
output_file = f"moondream2_responses_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


# Write the compiled data to a JSON file
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4, default=default_converter)

# Change file ownership to host user if script is running as root in docker
if os.getuid() == 0:
        host_uid = 1000  # Replace with actual host user ID
        host_gid = 1000  # Replace with actual host group ID
        os.chown(output_file, host_uid, host_gid)
        
print(f"Output written to {output_file}")