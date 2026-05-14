import os
import json
import random
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

def init_distributed():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device

local_rank, device = init_distributed()
print(f"[INFO] Using device {device}, local_rank={local_rank}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class GPUUsageCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if local_rank == 0:
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserv = torch.cuda.memory_reserved() / 1024**2
            print(
                f"[Step {state.global_step}] "
                f"Allocated {alloc:.1f}MB | Reserved {reserv:.1f}MB"
            )

LABEL_MAP = {"no": 0, "yes": 1}

def load_json_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["answer"] = df["answer"].str.lower()
    df["labels"] = df["answer"].map(LABEL_MAP).astype(int)
    return Dataset.from_pandas(df[["question", "labels"]])

TRAIN_PATH = "train.json"
train_dataset = load_json_dataset(TRAIN_PATH)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Qwen 通常没有 pad_token，手动设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.float16,
)

base_model.config.pad_token_id = tokenizer.pad_token_id

model = get_peft_model(
    base_model,
    LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
)

model.to(device)

def tokenize_function(examples):
    return tokenizer(
        examples["question"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["question"]
)

training_args = TrainingArguments(
    label_names=["labels"],
    output_dir="qwen_ontology",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=3e-5,
    warmup_steps=100,
    weight_decay=0.01,
    num_train_epochs=5,
    logging_dir="./logs",
    report_to="none",
    fp16=True,
    eval_strategy="no",
    deepspeed="deepspeed_config.json",
    local_rank=local_rank,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    callbacks=[GPUUsageCallback()],
)

trainer.train()

if local_rank == 0:
    save_path = "qwen_ontology_lora"
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"[INFO] LoRA adapter saved to {save_path}")

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
    print("[INFO] Distributed process group destroyed")