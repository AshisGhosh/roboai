from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, Dataset as HuggingFaceDataset, Features, Image, Value, DatasetDict
import json
import os
from PIL import Image as PILImage


PKG = "model_training/data/"

class ChessDataset(Dataset):
    def __init__(self, split='train'):
        self.data = load_dataset(
            "Trelis/chess_pieces",
            # revision="refs/convert/parquet",
        )[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": sample["image"], # Should be a PIL image
            "qa": [
                {
                    "question": "What do you see?",
                    "answer": sample["caption"],
                }
            ]
        }

class YCBIsaacDataset(Dataset):
    def __init__(self, split='train'):
        self.data = load_from_disk(
            PKG + "ycb_isaac",
        )[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "image": item["image"], 
            "qa": [
                {
                    "question": "Name the objects on the table from left to right",
                    "answer": item["caption"],
                }
            ]
        }
    
    
def create_ycb_isaac_dataset(json_file, image_folder, output_folder=PKG + "ycb_isaac", test_split_size=100):
    with open(json_file, "r") as f:
        captions = json.load(f)
    
    data = {
        "image": [],
        "caption": []
    }
    for image_file, caption in captions.items():
        image_path = os.path.join(image_folder, image_file)
        if os.path.exists(image_path):
            concatenated_caption = ", ".join(caption["objects_left_to_right"])
            # Convert png to jpg
            if image_file.endswith(".png"):
                im = PILImage.open(image_path)
                image_path = image_path.replace(".png", ".jpg")
                im.convert("RGB").save(image_path)
            data["image"].append(image_path)
            data["caption"].append(concatenated_caption)

    features = Features({
        "image": Image(),  # Image feature
        "caption": Value("string"),  # Value feature
    })
    dataset = HuggingFaceDataset.from_dict(data, features=features)
    
    # Split the dataset
    test_indices = list(range(len(dataset) - test_split_size, len(dataset)))
    train_indices = list(range(len(dataset) - test_split_size))
    
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)
    
    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # Save the DatasetDict
    dataset_dict.save_to_disk(output_folder)
    
    print("DatasetDict created and saved locally.")

def load_datasets(name:str) -> dict:
    if name == "chess":
        datasets = {
            "train": ChessDataset("train"),
            "test": ChessDataset("test"),
        }
    elif name == "ycb_isaac":
        datasets = {
            "train": YCBIsaacDataset("train"),
            "test": YCBIsaacDataset("test"),
        }
    else:
        raise ValueError(f"Dataset {name} not found.")
    return datasets

def main():
    create_ycb_isaac_dataset(PKG + "ycb_isaac_raw/gt_responses_20240606_083247.json", PKG + "ycb_isaac_raw")

if __name__ == "__main__":
    main()