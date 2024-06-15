
from .data import load_datasets



from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = "models--vikhyatk--moondream2/snapshots/9ba2958f5a886de83fa18a235d651295a05b4d13"

import torch
DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16 # CPU doesn't support float16. Also, switch to bfloat16 for Ampere architectures.


def get_model(model_id="vikhyatk/moondream2", revision ="2024-04-02", use_4bit=False, use_lora=False, lora_path=False, use_flash_attn=False) -> AutoModelForCausalLM:
    quantization_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE
        )
        quantization_config = bnb_config

    if use_lora:
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(lora_path)
        print(config)

    flash_attn = None
    if use_flash_attn:
        flash_attn = "flash_attention_2" if DEVICE == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        attn_implementation=flash_attn,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
        quantization_config=quantization_config
    )

    if use_lora:
        model = PeftModel.from_pretrained(model, lora_path)


    return model, tokenizer



def compare_models():
    model1, tokenizer = get_model()
    model1.eval()
    model1.to("cpu")

    model2, _ = get_model(use_4bit=False, use_lora=True)
    model2.eval()
    model2.to("cpu")

    def compare_model_weights(model1, model2):
        model1_dict = model1.state_dict()
        model2_dict = model2.state_dict()

        for name, param in model1_dict.items():
            if name in model2_dict:
                if not torch.equal(param.data, model2_dict[name].data):
                    print(f"Difference found in parameter: {name}")
                    return False
            else:
                print(f"Parameter {name} not found in the fine-tuned model.")
                return False

        for name in model2_dict.keys():
            if name not in model1_dict:
                print(f"Parameter {name} not found in the base model.")
                return False

        print("No differences found in the model parameters.")
        return True
    
    def compare_model_weights_by_values(model1, model2):
        model1_weights = list(model1.parameters())
        model2_weights = list(model2.parameters())

        if len(model1_weights) != len(model2_weights):
            print(f"The number of parameters is different: {len(model1_weights)} vs {len(model2_weights)}")
            return False

        for i, (param1, param2) in enumerate(zip(model1_weights, model2_weights)):
            if not torch.equal(param1.data, param2.data):
                print(f"Difference found in parameter at index {i}")
                return False

        print("No differences found in the model parameters.")
        return True
    
    def compare_common_weights(model1, model2):
        model1_dict = model1.state_dict()
        model2_dict = model2.state_dict()

        common_keys = set(model1_dict.keys()).intersection(set(model2_dict.keys()))

        if not common_keys:
            print("No common parameters found between the models.")
            return False

        for key in common_keys:
            if not torch.equal(model1_dict[key], model2_dict[key]):
                print(f"Difference found in parameter: {key}")
                return False

        print("No differences found in the common parameters.")
        return True
    
    weights_are_same = compare_model_weights(model1, model2)
    # weights_are_same = compare_model_weights_by_values(model1, model2)
    # weights_are_same = compare_common_weights(model1, model2)


    if not weights_are_same:
        print("The models have different weights, indicating training has modified the weights.")
    else:
        print("The models have identical weights, indicating training may not have modified the weights.")

    def check_lora_weights(model):
        lora_params = {name: param for name, param in model.named_parameters() if "lora" in name}
        if not lora_params:
            print("No LoRA parameters found.")
            return False
        
        for name, param in lora_params.items():
            if torch.sum(param.data != 0) > 0:
                print(f"LoRA parameter {name} has non-zero values.")
            else:
                print(f"LoRA parameter {name} is all zeros.")
        return True
    
    check_lora_weights(model2)


def display_image(image):
    import cv2
    import numpy as np
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_dataset():
    from .data import load_datasets
    datasets = load_datasets("ycb_isaac")
    sample = datasets['train'][0]
    for qa in sample['qa']:
        print('Question:', qa['question'])
        print('Ground Truth:', qa['answer'])
    display_image(sample["image"])

    sample = datasets['test'][1]
    for qa in sample['qa']:
        print('Question:', qa['question'])
        print('Ground Truth:', qa['answer'])
    display_image(sample["image"])


def eval_sample(dataset='ycb_isaac', split="test", sample_index=0):
    datasets = load_datasets(dataset)
    sample = datasets[split][sample_index]

    model, tokenizer = get_model(use_lora=True)
    # model, tokenizer = get_model()
    model.eval()

    for qa in sample['qa']:
        print('Question:', qa['question'])
        print('Ground Truth:', qa['answer'])
        print('model:', model.answer_question(
            model.encode_image(sample['image']),
            qa['question'],
            tokenizer=tokenizer,
        ))
    display_image(sample["image"])


import hydra
from omegaconf import DictConfig
@hydra.main(config_path="conf", config_name="eval_config")
def moondream_eval(cfg: DictConfig):
    import datetime
    import json
    import re
    import pathlib
    from shared.scripts.moondream_prompter import process_images
    model_id = cfg.model.id
    revision = cfg.model.revision

    if cfg.finetune.is_finetune:
        use_lora = cfg.finetune.peft.use_lora 
        lora_path = "/app/"+cfg.finetune.path

    use_flash_attn = cfg.model.use_flash_attn
    model, tokenizer = get_model(model_id, revision, use_lora=use_lora, lora_path=lora_path, use_flash_attn=use_flash_attn)

    dataset = cfg.data.dataset
    prompt = cfg.data.prompt
    subset = cfg.data.subset

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = pathlib.Path("/app/data/" + dataset)
    results = {}
    if path.exists():
        if path.is_dir():
            image_paths = sorted(list(path.glob("rgb_????.png")))
            if subset:
                subset_idx = int(len(image_paths) * subset)
                image_paths = image_paths[:subset_idx]
            results = process_images(image_paths, model, prompt, tokenizer)
        elif path.is_file() and re.match(r"rgb_\d{4}\.png", path.name):
            results = process_images([path], model, prompt, tokenizer)
        else:
            print("The file or directory does not match the expected pattern.")
    else:
        print("The provided path does not exist.")

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_data = {
        "model_id": model_id,
        "prompt": prompt,
        "start_time": start_time,
        "end_time": end_time,
        "responses": results,
    }
    output_name = cfg.model.slug + "_responses"
    if cfg.finetune.is_finetune:
        output_name = "finetune_" + output_name

    output_file = (
        f"{output_name}_{cfg.timestamp}.json"
    )
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Output written to {output_file}")

def main():
    # compare_models()
    # test_dataset()
    # eval_sample(dataset='ycb_isaac', split="test", sample_index=10)
    moondream_eval()


if __name__ == "__main__":
    main()
    
