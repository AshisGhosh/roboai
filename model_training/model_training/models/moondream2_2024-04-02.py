import torch
from transformers import AutoModelForCausalLM
from einops import rearrange
from torch.utils.data import DataLoader


DEVICE = "cuda"
DTYPE = (
    torch.float32 if DEVICE == "cpu" else torch.bfloat16
)  # CPU doesn't support float16. Also, switch to bfloat16 for Ampere architectures.

ANSWER_EOS = "<|endoftext|>"
IMG_TOKENS = 729  # Number of tokens used to represent each image.


def get_dataloaders(
    datasets: dict, model: AutoModelForCausalLM, tokenizer, batch_size, train_split=None
) -> dict:
    def collate_fn(batch):
        images = [sample["image"] for sample in batch]
        images = torch.stack(model.vision_encoder.preprocess(images))
        images = rearrange(
            images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14
        )

        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)

            for qa in sample["qa"]:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False,
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}", add_special_tokens=False
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
            torch.stack(
                [torch.tensor(label, dtype=torch.long) for label in labels_acc]
            ),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )

    if train_split is not None:
        from torch.utils.data import random_split

        train_split_idx = int(len(datasets["train"]) * train_split)
        train_subset, _ = random_split(
            datasets["train"],
            [train_split_idx, len(datasets["train"]) - train_split_idx],
        )
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
    inputs_embeds = torch.cat(
        (tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1
    )

    outputs = model.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss
