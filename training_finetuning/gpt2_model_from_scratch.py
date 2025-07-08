import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, AdamW, get_linear_schedule_with_warmup

# === CONFIG ===
MODEL_DIR = "./gpt2_from_scratch"
DATA_PATH = "./data.txt"  # Your plain text file (one or multiple paragraphs)
SEQ_LEN = 1024
BATCH_SIZE = 1             # Increase if you have more RAM/CPU
EPOCHS = 3
LR = 5e-4                  # Start with 5e-4, you can tune later
WARMUP_STEPS = 500
SAVE_EVERY = 500           # Save checkpoint every N batches
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# === LOAD TOKENIZER (GPT2 pretrained) ===
print("Loading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.model_max_length = SEQ_LEN

# === CREATE MODEL CONFIG & MODEL ===
print("Creating GPT-2 config and model from scratch...")
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=SEQ_LEN,
    n_ctx=SEQ_LEN,
    n_embd=768,
    n_layer=12,
    n_head=12,
)
model = GPT2LMHeadModel(config).to(DEVICE)

# === DATASET CLASS ===
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize entire corpus into input IDs
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, (len(self.tokens) - self.seq_len))

    def __getitem__(self, idx):
        # Input: seq_len tokens; Target: next tokens shifted by 1
        input_ids = self.tokens[idx : idx + self.seq_len]
        target_ids = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(input_ids), torch.tensor(target_ids)

# === DATA LOADER ===
print("Loading dataset...")
dataset = TextDataset(DATA_PATH, tokenizer, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === OPTIMIZER & SCHEDULER ===
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

# === TRAIN LOOP ===
print(f"Training on device: {DEVICE}")
model.train()

global_step = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)

        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        global_step += 1

        if global_step % 10 == 0:
            avg_loss = epoch_loss / (batch_idx + 1)
            print(f"Step {global_step} - loss: {avg_loss:.4f}")

        if global_step % SAVE_EVERY == 0:
            save_path = os.path.join(MODEL_DIR, f"checkpoint-step-{global_step}")
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving checkpoint to {save_path}...")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

print("Training complete!")
# Save final model
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Final model saved to {MODEL_DIR}")

"""
NOTES:
Raspberry Pi Zero 2 W (2GB)
Batch size: 1
Learning rate: 2e-4
Epochs: 1-3
Notes: Very slow training; best for tiny toy datasets only; heavy swap use likely.

ThinkPad T550 (i7-5600U, 16GB RAM)
Batch size: 2-4 (start with 2)
Learning rate: 5e-4 to 1e-3
Epochs: 3-5
Notes: CPU-only, slow training; enable MKL/OpenMP for speed.

Dedicated Server (24 cores, 120GB RAM)
Batch size: 16-64
Learning rate: 5e-4 to 1e-3
Epochs: 5+
Notes: use threading controls; save checkpoints frequently.



"""