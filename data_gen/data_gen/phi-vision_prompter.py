import argparse
import json
import re
import os
import datetime as dt
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process an image file or a directory of images."
    )
    parser.add_argument(
        "path", type=str, help="Path to an image file or a directory containing images."
    )
    return parser.parse_args()


def get_timestamp():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def initialize_model(model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def process_images(image_paths, processor, device, model, prompt):
    results = {}
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            prompt_phi_format = (
                f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
            )
            inputs = processor(prompt_phi_format, image, return_tensors="pt").to(device)
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=1000,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            response = processor.batch_decode(
                generate_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
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
    model_id = "microsoft/Phi-3-vision-128k-instruct"
    prompt = "Name the objects on the table from left to right."
    model, processor = initialize_model(model_id)

    start_time = get_timestamp()
    path = Path(args.path)
    results = {}
    if path.exists():
        if path.is_dir():
            image_paths = sorted(list(path.glob("rgb_????.png")))
            results = process_images(image_paths, processor, device, model, prompt)
        elif path.is_file() and re.match(r"rgb_\d{4}\.png", path.name):
            results = process_images([path], processor, device, model, prompt)
        else:
            print("The file or directory does not match the expected pattern.")
    else:
        print("The provided path does not exist.")

    end_time = get_timestamp()
    output_data = {
        "model_id": model_id,
        "prompt": prompt,
        "start_time": start_time,
        "end_time": end_time,
        "responses": results,
    }
    output_file = (
        f"phi-3-vision-128k-instruct_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Output written to {output_file}")
    # Change file ownership to host user if script is running as root in docker
    adjust_file_permissions(output_file)


if __name__ == "__main__":
    main()
