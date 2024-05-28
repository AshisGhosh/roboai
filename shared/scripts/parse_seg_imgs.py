import cv2
import json
import numpy as np
import os
import tarfile
import argparse
from collections import Counter

def find_leftmost_pixel(mask, color):
    matched_pixels = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
    if not np.any(matched_pixels):
        print(f"No pixels found for color: {color}")  # Debug: No pixels found
        return float('inf')
    x_indices = np.where(matched_pixels)[1]
    return np.min(x_indices)  # Return the smallest x-index (leftmost pixel)

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

    print(f"Found objects for {base_name}.png: {ordered_objects}")  # Debug: Found objects

    return {f"{base_name}.png": {"objects_left_to_right": ordered_objects}}

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
    print(json.dumps(ordered_objects, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a folder or tarball of synthetic images for object ordering.')
    parser.add_argument('path', type=str, help='The path to the dataset directory or tarball.')
    args = parser.parse_args()
    main(args)

# import cv2
# import numpy as np
# from collections import Counter

# # Load the image
# image_path = '/home/user/omni.replicator_out/_output/semantic_segmentation_0003.png'  # Update with the correct path
# mask = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# # Check if the image is loaded correctly
# if mask is None:
#     print(f"Failed to load image from {image_path}")
# else:
#     print(f"Image loaded successfully with shape {mask.shape} and dtype {mask.dtype}")

# # Flatten the image array and convert it to a list of tuples (each representing an RGBA color)
# flat_mask = mask.reshape(-1, mask.shape[-1])
# unique_colors = Counter(map(tuple, flat_mask))

# # Convert the found colors from BGRA to RGBA
# unique_colors_rgba = {tuple(np.array(color)[[2, 1, 0, 3]]): count for color, count in unique_colors.items()}

# # Print all unique colors in the image in RGBA format
# print("Unique colors in the image (converted to RGBA):")
# for color, count in unique_colors_rgba.items():
#     print(f"Color: {color}, Count: {count}")

# # Define colors as seen in JSON (assuming RGBA format)
# colors_to_check = {
#     "mustard bottle": (140, 255, 25, 255),
#     "cracker box": (140, 25, 255, 255),
#     "soup can": (255, 197, 25, 255),
#     "sugar box": (25, 255, 82, 255)
# }

# # Check each color from JSON
# for item, color in colors_to_check.items():
#     if color in unique_colors_rgba:
#         print(f"Color for {item} found in image.")
#     else:
#         print(f"Color for {item} not found in image.")


# import cv2
# import json
# import numpy as np
# import os
# import tarfile
# import argparse


# def process_images_from_folder(image_folder):
#     result = {}
#     for root, dirs, files in os.walk(image_folder):
#         for file_name in sorted(files):
#             if file_name.startswith('semantic_segmentation') and file_name.endswith('.png'):
#                 base_name = file_name.split('_')[-1].split('.')[0]
#                 result.update(process_image(base_name, os.path.join(root, file_name), root))
#     return result

# def process_images_from_tar(tar_path):
#     result = {}
#     with tarfile.open(tar_path, 'r:*') as tar:
#         for member in sorted(tar.getmembers(), key=lambda x: x.name):
#             print(f"Processing file: {member.name}")  # Debug: print file names being processed
#             if member.name.endswith('.png') and 'semantic_segmentation' in member.name:
#                 base_name = member.name.split('_')[-1].split('.')[0]
#                 file_name = tar.extractfile(member)
#                 result.update(process_image(base_name, file_name, tar))
#     return result

# def process_image(base_name, image_file, archive):
#     if isinstance(archive, tarfile.TarFile):
#         json_path = next((m for m in archive.getmembers() if m.name.endswith(f'semantic_segmentation_labels_{base_name}.json')), None)
#         json_file = archive.extractfile(json_path)
#     else:
#         json_path = os.path.join(archive, f'semantic_segmentation_labels_{base_name}.json')
#         json_file = open(json_path, 'r')

#     mask = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
#     colors = json.load(json_file)
#     json_file.close()

#     object_positions = {}
#     for color_str, info in colors.items():
#         rgba = tuple(map(int, color_str.strip('()').split(',')))  # Convert RGBA string to tuple of integers
#         if info['class'] not in ['BACKGROUND', 'UNLABELLED']:
#             print(f"Searching for {info['class']} with RGBA color {rgba}")  # Debug: Color searching
#             leftmost_x = find_leftmost_pixel(mask, rgba)
#             if leftmost_x != float('inf'):
#                 object_positions[leftmost_x] = info['class']

#     ordered_objects = [object_positions[x] for x in sorted(object_positions)]
#     print(f"Found objects for {base_name}.png: {ordered_objects}")  # Debug: Objects found
#     return {f'{base_name}.png': {'objects_left_to_right': ordered_objects}}

# def find_leftmost_pixel(mask, color):
#     # Create a boolean array where all channels match the target color
#     matched_pixels = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
#     if not np.any(matched_pixels):
#         print(f"No pixels found for color: {color}")  # Debug: No pixels found
#         return float('inf')
#     # Find the x-coordinates of the matched pixels
#     x_indices = np.where(matched_pixels)[1]
#     return np.min(x_indices)  # Return the smallest x-index (leftmost pixel)

# def main(args):
#     if os.path.isdir(args.path):
#         ordered_objects = process_images_from_folder(args.path)
#     elif tarfile.is_tarfile(args.path):
#         ordered_objects = process_images_from_tar(args.path)
#     else:
#         raise ValueError("The path must be a directory or a tarball.")
#     print(json.dumps(ordered_objects, indent=4))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Process a folder or tarball of synthetic images for object ordering.')
#     parser.add_argument('path', type=str, help='The path to the dataset directory or tarball.')
#     args = parser.parse_args()
#     main(args)
