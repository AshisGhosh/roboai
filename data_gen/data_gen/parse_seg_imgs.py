import cv2
import json
import numpy as np
import os
import tarfile
import argparse
import datetime as dt
from collections import Counter

def find_leftmost_pixel(mask, color):
    matched_pixels = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
    if not np.any(matched_pixels):
        print(f"No pixels found for color: {color}")  # Debug: No pixels found
        return float('inf')
    x_indices = np.where(matched_pixels)[1]
    return np.min(x_indices)  # Return the smallest x-index (leftmost pixel)

def punctuator(word):
    """
    Cleans up object name strings
    
    (Basicwriter from Nvidia Omniverse Replicator can't take punctuation in the semantic fields.)
    """
    replacement_dict = {
        "cheezit": "Cheez-It",
        "domino": "Domino",
        "campbells": "Campbell's",
        "frenchs": "French's"
    }
    words = word.split()
    cleaned_words = [replacement_dict.get(w.lower(), w) for w in words]
    return ' '.join(cleaned_words)

def process_image(base_name, image_file, archive):
    print(f"Processing image: {base_name}")  # Debug: Processing image
    if isinstance(archive, tarfile.TarFile):
        json_path = next((m for m in archive.getmembers() if m.name.endswith(f'semantic_segmentation_labels_{base_name}.json')), None)
        json_file = archive.extractfile(json_path)
    else:
        json_path = os.path.join(archive, f'semantic_segmentation_labels_{base_name}.json')
        json_file = open(json_path, 'r')

    if json_file is None:
        print(f"Failed to load JSON file for {base_name}")  # Debug: JSON file not found
        return {}

    mask = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"Failed to load image for {base_name}")  # Debug: Image not found
        return {}

    # Convert image from BGRA to RGBA
    mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA)

    colors = json.load(json_file)
    json_file.close()

    print(f"Colors from JSON for {base_name}: {colors}")  # Debug: Print colors from JSON

    flat_mask = mask.reshape(-1, mask.shape[-1])
    unique_colors = Counter(map(tuple, flat_mask))

    print(f"Unique colors in the image (converted to RGBA) for {base_name}: {unique_colors}")  # Debug: Print unique colors

    objects_left_to_right = []

    for color_str, data in colors.items():
        rgba = tuple(map(int, color_str.strip('()').split(',')))
        if data['class'] == 'UNLABELLED':
            continue  # Skip 'UNLABELLED' entries
        print(f"Checking color {rgba} for class {data['class']}")
        if rgba in unique_colors:
            leftmost_x = find_leftmost_pixel(mask, rgba)
            if leftmost_x is not None:
                objects_left_to_right.append((leftmost_x, data['class']))
        else:
            print(f"Color {rgba} not found in unique colors")  # Debug: Color not found

    objects_left_to_right.sort()
    ordered_objects = [obj[1] for obj in objects_left_to_right]
    corrected_objects = [punctuator(obj) for obj in ordered_objects]

    print(f"Corrected objects for rgb_{base_name}.png: {corrected_objects}")  # Debug: Found objects

    return {f"rgb_{base_name}.png": {"objects_left_to_right": corrected_objects}}

def process_images_from_folder(image_folder):
    result = {}
    for root, dirs, files in os.walk(image_folder):
        for file_name in sorted(files):
            if file_name.startswith('semantic_segmentation') and file_name.endswith('.png'):
                base_name = file_name.split('_')[-1].split('.')[0]
                with open(os.path.join(root, file_name), 'rb') as f:
                    result.update(process_image(base_name, f, root))
    return result

def process_images_from_tar(tar_path):
    result = {}
    with tarfile.open(tar_path, 'r:*') as tar:
        for member in sorted(tar.getmembers(), key=lambda x: x.name):
            if member.name.startswith('_output/semantic_segmentation') and member.name.endswith('.png'):
                base_name = member.name.split('_')[-1].split('.')[0]
                file_name = tar.extractfile(member)
                result.update(process_image(base_name, file_name, tar))
    return result

def main(args):
    if os.path.isdir(args.path):
        ordered_objects = process_images_from_folder(args.path)
    elif tarfile.is_tarfile(args.path):
        ordered_objects = process_images_from_tar(args.path)
    else:
        raise ValueError("The path must be a directory or a tarball.")
    
    output_file = f"gt_responses_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(ordered_objects, f, indent=4)    
    print(json.dumps(ordered_objects, indent=4))
    print(f"Output written to {output_file}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a folder or tarball of synthetic images for object ordering.')
    parser.add_argument('path', type=str, help='The path to the dataset directory or tarball.')
    args = parser.parse_args()
    main(args)
