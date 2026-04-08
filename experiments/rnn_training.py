import os
import sys
import gc

# Environment configuration for stability on macOS
os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "True",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1"
})

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
import latentspy as ls

class TinyLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # batch_first=True makes tensors [batch, seq, feature]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        # input: [batch, seq]
        x = self.embedding(x) # [batch, seq, embedding_dim]
        lstm_out, _ = self.lstm(x) # [batch, seq, hidden_dim]
        logits = self.fc(lstm_out) # [batch, seq, vocab_size]
        return logits

def run_experiment():
    EXPERIMENT_NAME: str = "lstm_healthy_baseline"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # HEALTHY HYPERPARAMETERS
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    MAX_SEQ_LEN = 128
    VAL_INTERVAL = 750
    SAMPLE_INTERVAL = 200
    WARMUP_STEPS = 500
    MAX_GRAD_NORM = 1.0

    print(f"\n[Run Config] {EXPERIMENT_NAME}")
    
    # 1. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    model = TinyLSTM(vocab_size).to(DEVICE)
    print(f"  Model: LSTM (2 layers, 256 hidden)")

    # 2. Dataset
    print("\nLoading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    train_dataset = dataset.select(range(100000))
    val_dataset = dataset.select(range(100000, 102000))
    
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)
    
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"], desc="Train")
    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text"], desc="Val")
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=0)
    
    def get_val_batch(loader):
        iterator = iter(loader)
        while True:
            try: yield next(iterator)
            except StopIteration: iterator = iter(loader); yield next(iterator)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=0)
    val_iterator = get_val_batch(val_loader)

    # 3. LatentSpy
    # We explicitly watch the 'lstm' and 'fc' layers
    monitor = ls.watch(
        model,
        layers=["embedding", "lstm", "fc"],
        metrics=[
            ls.Metric.ACTIVATION_NORM,
            ls.Metric.SPARSITY,
            ls.Metric.KURTOSIS,
            ls.Metric.PATCHINESS,
            ls.Metric.RECONSTRUCTION,
        ],
        sample_interval=SAMPLE_INTERVAL,
        val_interval=VAL_INTERVAL,
        experiment_name=EXPERIMENT_NAME,
        log_type="db",
        alert_interval=10,
        dashboard=True,
        alert_warmup_steps=WARMUP_STEPS
    )

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    print(f"\n--- MISSION START: {EXPERIMENT_NAME} ---")
    try:
        model.train()
        global_step = 0
        for epoch in range(1, NUM_EPOCHS + 1):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                targets = input_ids.clone() # Predict next token
                
                global_step += 1
                monitor.step()
                
                optimizer.zero_grad()
                logits = model(input_ids)
                
                # Reshape for loss: [batch*seq, vocab]
                loss = criterion(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, 1:].reshape(-1))
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()

                if monitor.should_run_validation():
                    val_batches = []
                    for _ in range(8):
                        vb = next(val_iterator)["input_ids"].to(DEVICE)
                        val_batches.append(vb)
                    
                    monitor.run_validation_pp(val_batches)
                    
                    model.eval()
                    with torch.no_grad():
                        v_input = val_batches[0]
                        v_logits = model(v_input)
                        v_loss = criterion(v_logits[:, :-1, :].reshape(-1, vocab_size), v_input[:, 1:].reshape(-1))
                        monitor.log_scalar("val_loss", v_loss.item())
                        print(f"   [Val] Step: {global_step:5d} | Val Loss: {v_loss.item():.4f}")
                    model.train()
                    
                    del val_batches
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                monitor.log()
                monitor.log_scalar("loss", loss.item())
                monitor.log_scalar("lr", scheduler.get_last_lr()[0])
                
                if global_step % 100 == 0 or global_step == 1:
                    print(f"      Step: {global_step:5d} | Train Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    except KeyboardInterrupt:
        print("\n[!] Halted.")
    finally:
        monitor.remove()
        gc.collect()

if __name__ == "__main__":
    run_experiment()
