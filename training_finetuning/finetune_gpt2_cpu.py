import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# 1. Define your datasets here (HF dataset ids)
instruction_datasets = [
    "yahma/alpaca-cleaned",  # Instruction tuning
    "stanfordnlp/dialogue-dataset"  # Example roleplay dataset (replace as needed)
]

# 2. Prompt template for all data samples
def format_example(example):
    # For instruction-following data: keys may differ per dataset
    # Adjust keys accordingly or map fields from the dataset
    instruction = example.get("instruction") or example.get("prompt") or ""
    input_text = example.get("input") or ""
    response = example.get("output") or example.get("completion") or example.get("response") or ""

    if input_text.strip():
        prompt = f"User: {instruction}\nInput: {input_text}\nAssistant:"
    else:
        prompt = f"User: {instruction}\nAssistant:"

    return {"text": f"{prompt} {response.strip()}"}

def load_and_prepare_datasets(dataset_names):
    datasets = []
    for ds_name in dataset_names:
        print(f"Loading dataset {ds_name}...")
        ds = load_dataset(ds_name, split="train")
        ds = ds.map(format_example)
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    print(f"Combined dataset size: {len(combined)}")
    return combined

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add the EOS token as PAD token to avoid warnings if missing
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Load and prepare datasets
    train_dataset = load_and_prepare_datasets(instruction_datasets)

    # Tokenize
    tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])

    # Data collator for language modeling (causal LM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args for CPU training, small batch size due to memory limits
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned-instruction-roleplay",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="no",
        report_to=None,
        fp16=False,
        optim="adamw_torch",
        weight_decay=0.01,
        learning_rate=5e-5,
        warmup_steps=100,
        gradient_accumulation_steps=4,  # To simulate batch size 4
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model("./gpt2-finetuned-final")

if __name__ == "__main__":
    main()
