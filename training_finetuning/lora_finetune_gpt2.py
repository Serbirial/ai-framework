import os
import argparse
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

# === Configurable datasets for instruction tuning ===
INSTRUCTION_DATASETS = [
    "yahma/alpaca-cleaned",
    "tatsu-lab/alpaca_gpt4",
]

def format_example(example):
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
        ds = ds.map(format_example, remove_columns=ds.column_names)
        datasets.append(ds)
    combined = concatenate_datasets(datasets)
    print(f"Combined dataset size: {len(combined)}")
    return combined

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tune GPT-2 124M")
    parser.add_argument("--colab", action="store_true", help="Use Colab settings (fp16, gradient_accumulation_steps=8)")
    parser.add_argument("--output_dir", type=str, default="./gpt2-lora-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],  # GPT2 attention linear layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    train_dataset = load_and_prepare_datasets(INSTRUCTION_DATASETS)
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer, args.max_length), batched=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args with conditional adjustments for Colab
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8 if args.colab else 1,
        evaluation_strategy="no",
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-4,
        weight_decay=0.01,
        fp16=True if (args.colab and device == "cuda") else False,
        optim="adamw_torch",
        report_to=None,
    )

    trainer = Trainer(
        model=model.to(device),
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
