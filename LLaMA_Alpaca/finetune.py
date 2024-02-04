import os
import sys
from typing import List
import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

BASE_MODEL = "decapoda-research/llama-7b-hf"
FILE_PATH = "data\icml_test.csv"
OUT_PATH = "\model"
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
NUM_EPOCHS = 4
LEARNING_RATE = 3e-4
DROPOUT = 0.05
MAX_LEN = 1024
V_SIZE = 2000

LORA_WEIGHTS = "tloen/alpaca-lora-7b"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

# # model/data params
# base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
# data_path: str = "data\icml_test.csv",
# output_dir: str = "\model",
# # training hyperparams
# batch_size: int = 128,
# micro_batch_size: int = 4,
# num_epochs: int = 3,
# learning_rate: float = 3e-4,
# cutoff_len: int = 256,
# val_set_size: int = 2000,
# # lora hyperparams
# lora_r: int = 8,
# lora_alpha: int = 16,
# lora_dropout: float = 0.05,
# lora_target_modules: List[str] = [
#     "q_proj",
#     "v_proj",
# ],
# # llm hyperparams
# train_on_inputs: bool = True,  # if False, masks out inputs in loss
# group_by_length: bool = False,  # faster, but produces an odd training loss curve,
# resume_from_checkpoint: str = None,  # either training checkpoint or final adapter

# print(
#     f"Training Alpaca-LoRA model with params:\n"
#     f"base_model: {base_model}\n"
#     f"data_path: {data_path}\n"
#     f"output_dir: {output_dir}\n"
#     f"batch_size: {batch_size}\n"
#     f"micro_batch_size: {micro_batch_size}\n"
#     f"num_epochs: {num_epochs}\n"
#     f"learning_rate: {learning_rate}\n"
#     f"cutoff_len: {cutoff_len}\n"
#     f"val_set_size: {val_set_size}\n"
#     f"lora_r: {lora_r}\n"
#     f"lora_alpha: {lora_alpha}\n"
#     f"lora_dropout: {lora_dropout}\n"
#     f"lora_target_modules: {lora_target_modules}\n"
#     f"train_on_inputs: {train_on_inputs}\n"
#     f"group_by_length: {group_by_length}\n"
#     f"resume_from_checkpoint: {resume_from_checkpoint}\n"
# )
    
gradient_accumulation_steps = BATCH_SIZE // MICRO_BATCH_SIZE

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    device_map=device_map,
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = 0  

data = load_dataset("json", data_files = FILE_PATH)

if resume_from_checkpoint:
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        resume_from_checkpoint = False  # So the trainer won't try loading its state
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        model = set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")

model.print_trainable_parameters()  # Be more transparent about the % of trainable params.


train_val = data["train"].train_test_split(
    test_size=V_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size = MICRO_BATCH_SIZE,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 100,
        num_train_epochs = NUM_EPOCHS,
        learning_rate = LEARNING_RATE,
        fp16=True,
        logging_steps = 10,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps = 200,
        save_steps = 200,
        output_dir = OUT_PATH,
        save_total_limit=3,
        load_best_model_at_end = True,
        ddp_find_unused_parameters = False if ddp else None,
        group_by_length = group_by_length,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

trainer.train(resume_from_checkpoint=resume_from_checkpoint)

model.save_pretrained(OUT_PATH)

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation = True,
        max_length = MAX_LEN,
        padding = False,
        return_tensors = None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < MAX_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt



