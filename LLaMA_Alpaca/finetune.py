import os
import json
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

BASE_MODEL = "decapoda-research/llama-7b-hf"
FILE_PATH = "data\snippet.csv"
OUT_PATH = "\model"
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
NUM_EPOCHS = 4
LEARNING_RATE = 3e-4
DROPOUT = 0.05
MAX_LEN = 1024
V_SIZE = 2000

# llm hyperparams
LORA_WEIGHTS = "tloen/alpaca-lora-7b"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
TRAIN_STEPS = 300


def load_data(path):
    # Load and preprocess the CSV data
    df = pd.read_csv(os.path.abspath(path))
    df['labels'] = df['labels'].replace(-1, 0)  # Replace -1 with 0 for labels
    
    # Create the dataset
    dataset = [
        {
            "instruction": "Detect the sentiment of the movie review.",
            "input": row["documents"],
            "output": ("Negative" if row["labels"] == 0 else "Positive")
        } for index, row in df.iterrows()
    ]

    return dataset

# Process and save the data
dataset_data = load_data(FILE_PATH)
with open("snippet.json", "w") as f:
   json.dump(dataset_data, f)
data = load_dataset("json", data_files="snippet.json")

# Initialize device
device = torch.device("cuda")# if torch.cuda.is_initialized() else "cpu")
#device = torch.device("cpu")
print(f"Device name: {torch.cuda.get_device_name(device.index)}")
print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

# Initialize model
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Initialize tokenizer
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = (0)  # UNK
tokenizer.padding_side = "left"

# Generates prompts by combining the instruction & input & output values for datapoints.
def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
 
# Tokenize and add End_Of_Sentence token
def tokenize(prompt, add_eos_token=True):
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
    return tokenized_full_prompt

# Prepare the training data
train_val = data["train"].train_test_split(
    test_size=200, shuffle=True, seed=42
)
train_data = (
    train_val["train"].map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].map(generate_and_tokenize_prompt)
)

# Set the model
model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r = LORA_R,
    lora_alpha = LORA_ALPHA,
    target_modules = LORA_TARGET_MODULES,
    lora_dropout = LORA_DROPOUT,
    bias = "none",
    task_type = "CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# Set training arguments
training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size = MICRO_BATCH_SIZE,
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
    warmup_steps = 100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps = 10,
    optim="adamw_torch",
    evaluation_strategy = "steps",
    save_strategy = "steps",
    eval_steps = 50,
    save_steps = 50,
    output_dir = OUT_PATH,
    save_total_limit = 3,
    load_best_model_at_end = True,
    report_to = "tensorboard"
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

# Training
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))
 
model = torch.compile(model)
trainer.train()

model.save_pretrained(OUT_PATH)