import os
import json
import random
import torch
import numpy as np
import torch.distributed as dist

from datasets import Dataset
from transformers import TrainingArguments, Trainer
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)
from peft import LoraConfig, get_peft_model, TaskType

def init_distributed():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, torch.device("cuda", local_rank)

local_rank, device = init_distributed()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

MODEL_NAME = "mistralai/Ministral-3-8B-Instruct-2512"

TRAIN_PATH = "train.json"

OUTPUT_DIR = "mistral_gen"
DS_CONFIG = "deepspeed_config.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)

    for x in data:
        x["answer"] = x["answer"].strip().lower()
        assert x["answer"] in ["yes", "no", "undecided"]
        x["answer"] = x["answer"].capitalize()

    return Dataset.from_list(data)

train_dataset = load_dataset(TRAIN_PATH)

tokenizer = MistralCommonBackend.from_pretrained(MODEL_NAME)

base_model = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

model = get_peft_model(
    base_model,
    LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
)

model.to(device)

def tokenize_fn(example):

    messages = [
        {"role": "user", "content": example["question"]}
    ]

    tokenized = tokenizer.apply_chat_template(
        messages,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
        add_generation_prompt=True
    )

    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]

    labels = torch.full_like(input_ids, -100)

    answer_ids = tokenizer(example["answer"], add_special_tokens=False)["input_ids"]

 
    for i in range(len(input_ids) - len(answer_ids), -1, -1):
        if input_ids[i:i + len(answer_ids)].tolist() == answer_ids:
            labels[i:i + len(answer_ids)] = input_ids[i:i + len(answer_ids)]
            break

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

train_dataset = train_dataset.map(
    tokenize_fn,
    remove_columns=train_dataset.column_names,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=3e-5,
    fp16=True,
    warmup_steps=100,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    deepspeed=DS_CONFIG,
    local_rank=local_rank,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

if local_rank == 0:
    save_path = os.path.join(OUTPUT_DIR, "lora")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[INFO] LoRA saved to {save_path}")
