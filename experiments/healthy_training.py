import os
import sys
import gc

# Set the environment before any other imports to prevent FAISS MacOS crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
import latentspy as ls

def run_experiment():
    EXPERIMENT_NAME: str = "healthy_training_baseline"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # HEALTHY BASELINE HYPERPARAMETERS
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    VAL_INTERVAL = 750  # Give the val worker enough time between rounds to avoid queue-full skips
    SAMPLE_INTERVAL = 200
    WARMUP_STEPS = 500
    MAX_GRAD_NORM = 1.0

    print(f"=== {EXPERIMENT_NAME} ===")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Validation Interval: {VAL_INTERVAL}")

    # 1. Model & Tokenizer
    print("\nInitializing model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=128,
        n_embd=256,
        n_layer=4,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config).to(DEVICE)

    # 2. Dataset Preparation (Pre-Tokenized)
    print("\nLoading dataset...")
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        train_dataset = dataset.select(range(100000))
        val_dataset = dataset.select(range(100000, 102000))
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)
        
        print("Tokenizing datasets... (this makes training much faster)")
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing Train")
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing Val")
        
        # Set to PyTorch format so DataLoader returns raw tensors immediately
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        # Native Dataloaders for maximal acceleration
        num_workers = 2 if DEVICE.type == "cuda" else 0
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
        # Validation loader is an iterator we can loop repeatedly from
        def get_val_batch(loader):
            iterator = iter(loader)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    yield next(iterator)
        
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
        val_iterator = get_val_batch(val_loader)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # 3. LatentSpy Setup
    monitor = ls.watch(
        model,
        layers='auto',
        metrics=[
            ls.Metric.ACTIVATION_NORM,
            ls.Metric.SPARSITY,
            ls.Metric.KURTOSIS,
            ls.Metric.COSINE_SIMILARITY,
            ls.Metric.PATCHINESS,
            ls.Metric.RECONSTRUCTION,
            ls.Metric.EEE,
        ],
        sample_interval=SAMPLE_INTERVAL,
        val_interval=VAL_INTERVAL,
        experiment_name=EXPERIMENT_NAME,
        log_type="db",
        alert_interval=10,
        dashboard=True,
        metric_kwargs={"patchiness": {"k": 16}},
        val_metric_kwargs={"patchiness": {"k": 256}},
        alert_warmup_steps=500,
        deep_metric_interval=5
    )

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    loss_history = []
    global_step = 0

    print("\n--- Starting Training ---")
    try:
        model.train()
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\n[Epoch {epoch}/{NUM_EPOCHS}]")
            
            for step_in_epoch, batch in enumerate(train_loader, 1):
                input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
                attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

                global_step += 1
                monitor.step()
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                
                # Gradient clipping to stabilize learning pace
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                optimizer.step()
                scheduler.step()

                if monitor.should_run_validation():
                    # 1. Evaluate LatentSpy Metrics (Geometry)
                    val_batches = []
                    # Pull enough batches to safely clear the 10,000 token subsample cutoff
                    for _ in range(8):  
                        val_batch = next(val_iterator)
                        val_batch = {k: v.to(DEVICE, non_blocking=True) for k, v in val_batch.items()}
                        val_batches.append(val_batch)
                    
                    monitor.run_validation_pp(val_batches)
                    
                    # 2. Evaluate Standard Base Model Loss (Ground Truth Performance)
                    model.eval()
                    total_val_loss = 0.0
                    with torch.no_grad():
                        for v_batch in val_batches:
                            v_outputs = model(v_batch['input_ids'], attention_mask=v_batch['attention_mask'], labels=v_batch['input_ids'])
                            total_val_loss += v_outputs.loss.item()
                    
                    avg_val_loss = total_val_loss / len(val_batches)
                    monitor.log_scalar("val_loss", avg_val_loss)
                    print(f"   ► [Val Step {global_step}] Standard Val Loss: {avg_val_loss:.4f}")
                    model.train()

                monitor.log()
                monitor.log_scalar("loss", loss.item())
                monitor.log_scalar("lr", scheduler.get_last_lr()[0])
                
                loss_history.append(loss.item())
                
                if global_step % 100 == 0 or global_step == 1:
                    print(f"Step {global_step:4d} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    except KeyboardInterrupt:
        print("\n\n[!] Training interupted by user (Ctrl+C). Initiating graceful shutdown...")
    
    finally:
        print(f"\n--- Training Completed / Halted ---")
        if loss_history:
            print(f"Final logged train loss: {loss_history[-1]:.4f}")
            print(f"Loss improvement (first to last): {loss_history[0] - loss_history[-1]:.4f}")

        if 'train_loader' in locals():
            del train_loader
        gc.collect()

        print(f"\n--- Storage Analysis ---")
        print(f"Experiment data stored in: {monitor.storage.run_database}")
        
        export_path = monitor.storage.export_experiment_data(EXPERIMENT_NAME)
        print(f"Full export path: {export_path}")

        print(f"\nSaving model to models/healthy_baseline...")
        os.makedirs("models", exist_ok=True)
        model.save_pretrained("models/healthy_baseline")
        print("Model saved successfully.")

        monitor.remove()
        del model, optimizer, monitor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        print(f"\nFatal error in experiment: {e}")
    finally:
        gc.collect()
