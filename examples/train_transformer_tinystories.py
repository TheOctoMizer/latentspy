import os
import torch
import gc
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
import latentspy as ls

# 0. Setup Environment and Device
# LatentSpy works across all PyTorch devices (CUDA/MPS/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Dataset Preparation (Pre-Tokenized)
# In real training, pre-tokenizing with Map is much faster than streaming.
print("Loading and Tokenizing dataset...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("roneneldan/TinyStories", split="train")
# We only take a tiny subset for this example
train_ds = dataset.select(range(100))
val_ds = dataset.select(range(100, 110))

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

# Map tokenization once upfront
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# Move to PyTorch format for the DataLoader
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Native DataLoader for maximal performance
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# 2. Model Initialization
print("Initializing tiny GPT-2 model...")
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=128,
    n_embd=128,
    n_layer=2,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config).to(device)

# 3. LatentSpy Setup (The Core Utility)
# This attaches hooks to the model to monitor its internal representation geometry.
monitor = ls.watch(
    model,
    layers='auto',
    metrics=[
        ls.Metric.ACTIVATION_NORM,
        ls.Metric.PATCHINESS,      # Best for detecting representation collapse
        ls.Metric.RECONSTRUCTION,  # Advanced clustering quality
        ls.Metric.EEE              # Eigenvalue Enrichment (Learning pace index)
    ],
    experiment_name="example_train_run",
    log_type="db",                 # Log to internal SQLite for dashboard viewing
    dashboard=True,                # Automatically start the LatentSpy dashboard UI
    val_interval=5                 # Compute validation statistics every 5 steps
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# 4. Minimal Training Loop
print("\n--- Starting Training Example ---")
print("Access dashboard at: http://localhost:8000")
model.train()

# We only run for a few steps to demonstrate the utility
TOTAL_STEPS = 10
global_step = 0

for epoch in range(1):
    for step, batch in enumerate(train_loader):
        if global_step >= TOTAL_STEPS: break
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # IMPORTANT: Tell LatentSpy we are starting a new training step
        monitor.step()

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Handle Geometric Validation Rounds (Computing Patchiness/Clusters)
        # We manually trigger this when desired, often at set intervals.
        if monitor.should_run_validation():
            print(f" ► [Step {global_step+1}] Running Geometric Validation Round...")
            # We pull a few batches for a statistically significant sample
            val_batches = [next(iter(val_loader)) for _ in range(2)]
            val_batches = [{k: v.to(device) for k, v in b.items()} for b in val_batches]
            
            # This triggers the heavy math in a background thread to keep training fast
            monitor.run_validation_pp(val_batches)

        # Log training metrics (norms, rank, etc.)
        monitor.log()
        
        global_step += 1
        print(f"Step {global_step}/{TOTAL_STEPS} | Loss: {loss.item():.4f}")

# 5. Final Stability Assessment
print("\n--- Training Finished ---")
print("Calculating final metrics report...")
results = monitor.log()

if results:
    # Let's peek at the final 'Patchiness' of the first attention layer
    layer_name = list(results.keys())[0]
    metrics = results[layer_name]
    if 'patchiness' in metrics:
        p = metrics['patchiness']
        print(f"Final Patchiness for {layer_name}: {p:.4f}")
        if p > 10.0:
            print("ALERT: High patchiness detected. Latent space may be collapsing!")
        else:
            print("STATUS: Latent space appears healthy and well-distributed.")

# 6. Smooth Cleanup
monitor.remove()
print("\nLatentSpy handles removed. Process complete.")

# Manual GC for clean script termination
del model; del optimizer; del monitor
if torch.cuda.is_available(): torch.cuda.empty_cache()
gc.collect()
