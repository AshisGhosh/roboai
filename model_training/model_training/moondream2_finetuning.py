import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import DataLoader
from bitsandbytes.optim import Adam8bit
import math
from einops import rearrange
from tqdm import tqdm

from .data import load_datasets

DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16 # CPU doesn't support float16. Also, switch to bfloat16 for Ampere architectures.


ANSWER_EOS = "<|endoftext|>"

IMG_TOKENS = 729    # Number of tokens used to represent each image.

def get_model(model_id, revision=None, use_4bit=False) -> AutoModelForCausalLM:
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

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
        quantization_config=quantization_config
    )

    return model, tokenizer


def get_dataloaders(datasets:dict, model:AutoModelForCausalLM, tokenizer, batch_size, train_split=None) -> dict:
    def collate_fn(batch):
        images = [sample['image'] for sample in batch]
        images = torch.stack(model.vision_encoder.preprocess(images))
        images = rearrange(images,
                        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                        p1=14, p2=14)

        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)

            for qa in sample['qa']:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}",
                    add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)

            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images.to(dtype=DTYPE),
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )

    if train_split is not None:
        from torch.utils.data import random_split
        train_split_idx = int(len(datasets["train"]) * train_split)
        train_subset, _ = random_split(datasets["train"], [train_split_idx, len(datasets["train"]) - train_split_idx])
        datasets["train"] = train_subset

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=3,
            collate_fn=collate_fn,
        ),
    }
    return dataloaders


def setup_lora(model, lora_alpha=32, lora_rank=64, lora_dropout=0.1, set_other_trainable=True):
    # if use_4bit:
    #     from peft import prepare_model_for_kbit_training
    #     model.gradient_checkpointing_enable()
    #     model = prepare_model_for_kbit_training(model)



    ## Apply LoRA (if use_lora is True in the config)
    from peft import LoraConfig
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            'proj','fc1','fc2',
            'Wqkv','out_proj'
        ],
        lora_dropout=lora_dropout,  # Example value, adjust as needed
        bias="none",  # Example setting, adjust as needed
        task_type="CAUSAL_LM",
        # modules_to_save=['lm_head','embd'], #won't work with the trainer unless using a hf trainer, not custom.
    )

    from peft import get_peft_model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if set_other_trainable:
        trainable_params_names = ['lm_head','embd']
        # trainable_params_names = None

        # Set modules to be trainable
        for n, p in model.named_parameters():
            if any(k in n for k in trainable_params_names):
                p.requires_grad_(True)
            # else:
            #     p.requires_grad_(False)  # Optional: Set the rest to be not trainable

        # Make a dictionary of trainable parameters
        trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}

        # Convert trainable_params to state_dict format
        trainable_params_state_dict = {n: p.data for n, p in trainable_params.items()}
    
    return model, trainable_params_state_dict




def compute_loss(model, batch):
    images, tokens, labels, attn_mask = batch

    images = images.to(DEVICE)
    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)

    with torch.no_grad():
        img_embs = model.vision_encoder.encoder(images)
        img_embs = model.vision_encoder.projection(img_embs)

    tok_embs = model.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = model.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss


def lr_schedule(learning_rate, step, max_steps, schedule_type="cosine"):
    if schedule_type == "cosine":
        x = step / max_steps
        if x < 0.1:
            return 0.1 * learning_rate + 0.9 * learning_rate * x / 0.1
        else:
            return 0.1 * learning_rate + 0.9 * learning_rate * (1 + math.cos(math.pi * (x - 0.1))) / 2
    elif schedule_type == "constant":
        x = step / max_steps
        if x < 0.1:
            return 0.1 * learning_rate + 0.9 * learning_rate * x / 0.1
        else:
            return learning_rate
    else:
        raise NotImplementedError



def get_optimizer(model, learning_rate, lora_params=None, param_selection="lora"):
    if param_selection == "lora":
        return Adam8bit(
            [
                {"params": lora_params},
            ],
            lr=learning_rate * 0.1,
            betas=(0.9, 0.95),
            eps=1e-6
        )
    elif param_selection == "all":
        return Adam8bit(
            [
                {"params": model.text_model.parameters()},
            ],
            lr=learning_rate * 0.1,
            betas=(0.9, 0.95),
            eps=1e-6
        )
    else:
        raise NotImplementedError

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    model_id = cfg.model.id
    revision = cfg.model.revision
    dataset = cfg.data.dataset
    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
    learning_rate = cfg.training.learning_rate
    use_4bit = cfg.peft.use_4bit
    use_lora = cfg.peft.use_lora
    train_split = cfg.data.train_split


    print(f"Training model {model_id} on dataset {dataset} for {epochs} epochs.")

    print(f"Loading datasets {dataset}...")
    datasets = load_datasets(dataset)

    print(f"Loading model {model_id} {revision}...")
    model, tokenizer = get_model(model_id, revision=revision, use_4bit=use_4bit)
    dataloaders = get_dataloaders(datasets, model, tokenizer, batch_size, train_split=train_split)

    if use_lora:
        print("Setting up LoRA...")
        lora_alpha = cfg.peft.lora_alpha
        lora_rank = cfg.peft.lora_rank
        lora_dropout = cfg.peft.lora_dropout
        model, trainable_params_state_dict = setup_lora(model, lora_alpha=lora_alpha, lora_rank=lora_rank, lora_dropout=lora_dropout)
        LR_scaling = lora_alpha / (lora_rank**0.5)
        print("Using an LR scaling for LoRA adapters of: ", LR_scaling)

    ## For fine-tuning LoRA params
    lora_params = []
    for name, module in model.named_modules():
        if "lora" in name:
            lora_params.extend([p for p in module.parameters() if p.requires_grad])

    print("Training model...")
    optimizer = get_optimizer(model, learning_rate, lora_params, param_selection="lora")

    total_steps = epochs * len(dataloaders["train"]) // gradient_accumulation_steps

    # Eval steps
    eval_freq = 0.25 # means run every such fraction of total steps.
    eval_steps=total_steps*eval_freq

    model.text_model.train()
    model.text_model.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False},) #this fixes the no grad issues...


    if cfg.use_wandb:
        import wandb
        wandb.init(
            project="model-ft",
            config={
                "EPOCHS": epochs,
                "BATCH_SIZE": batch_size,
                "GRAD_ACCUM_STEPS": gradient_accumulation_steps,
                "LR": learning_rate,
            }
        )

    i = 0
    for epoch in range(epochs):
        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{epochs}"):
            i += 1

            loss = compute_loss(model, batch)
            loss.backward()

            if i % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            lr = lr_schedule(learning_rate, i / gradient_accumulation_steps, total_steps)
            for param_group in optimizer.param_groups:
                if param_group['params'] == lora_params:
                    param_group['lr'] = lr * LR_scaling  # Apply scaling only to lora_params
                else:
                    param_group['lr'] = lr  # Apply base lr to all other params

            if i % eval_steps == 0 and cfg.use_wandb:
                # Calculate validation loss
                val_loss = 0
                for val_batch in tqdm(dataloaders["test"], desc="Validation"):
                    with torch.no_grad():
                        val_loss += compute_loss(model, val_batch).item()
                val_loss /= len(dataloaders["test"])

            if cfg.use_wandb:
                wandb.log({
                    "loss/train": loss.item(),
                    "lr": optimizer.param_groups[0]['lr']
                } | ({"loss/val": val_loss} if i % eval_steps == 0 else {}))

    # Save the final model
    save_directory = "final_model"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)


    if cfg.use_wandb:
        wandb.finish()

    print("Training complete.")


def main():
    train()

if __name__ == "__main__":
    main()