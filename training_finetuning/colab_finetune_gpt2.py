import os
os.environ["WANDB_DISABLED"] = "true"


!pip install --quiet transformers datasets accelerate

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# === Config ===
dataset_list = [
    "mlabonne/guanaco-llama2-1k",
    "facebook/empathetic_dialogues",
    "lemonilia/LimaRP", # erp idk
    "PygmalionAI/PIPPA"
    ]

max_length = 256
num_train_epochs = 2
gradient_accumulation_steps = 8
learning_rate = 5e-5
output_dir = "/content/gpt2-124M-finetuned"

# === Formatting ===
def format_example(example):
    # Try standard format (instruction datasets)
    instruction = example.get("instruction") or example.get("prompt") or ""
    input_text = example.get("input") or ""
    response = example.get("output") or example.get("completion") or example.get("response") or example.get("text") or ""

    if instruction or input_text:
        return {"text": f"User: {instruction}\nInput: {input_text}\nAssistant: {response}".strip()}
    else:
        return {"text": response.strip()}

def load_and_prepare_datasets(dataset_names):
    datasets = []
    for ds_name in dataset_names:
        try:
            print(f"Loading {ds_name}...")
            ds = load_dataset(ds_name, split="train", download_mode="force_redownload")
            ds = ds.map(format_example, remove_columns=ds.column_names)
            datasets.append(ds)
        except Exception as e:
            print(f"❌ Failed to load {ds_name}: {e}")
    return concatenate_datasets(datasets)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

# === Train ===
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

train_dataset = load_and_prepare_datasets(dataset_list)
tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=250,
    save_total_limit=1,
    logging_steps=50,
    learning_rate=learning_rate,
    weight_decay=0.01,
    fp16=True if device == "cuda" else False,
    report_to=None,
)

trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(output_dir)
print(f"✅ Model saved to {output_dir}")
