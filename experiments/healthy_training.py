import os
import sys
import gc

# Set the environment before any other imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
import latentspy as ls

def get_batches(ds, tokenizer, batch_size=4):
    """Create batches from dataset"""
    batch = []
    try:
        for item in ds:
            batch.append(item["text"])
            if len(batch) == batch_size:
                yield tokenizer(batch, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
                batch = []
    except Exception as e:
        print(f"Error during dataset iteration: {e}")
        return

def run_experiment():
    EXPERIMENT_NAME: str = "healthy_training_baseline"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # HEALTHY BASELINE HYPERPARAMETERS
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 8
    NUM_EPOCHS = 3
    VAL_INTERVAL = 750  # Give the val worker enough time between rounds to avoid queue-full skips
    SAMPLE_INTERVAL = 200

    print(f"=== {EXPERIMENT_NAME} ===")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Validation Interval: {VAL_INTERVAL}")

    print("\nLoading dataset...")
    try:
        dataset = load_dataset("roneneldan/TinyStories", streaming=False, split="train")
        train_dataset = dataset.select(range(100000))
        val_dataset = dataset.select(range(100000, 102000))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print("Initializing model...")
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

    monitor = ls.watch(
        model,
        layers='auto',
        metrics=[
            # TIER 1 — Fast (every sample_interval steps): O(N), monotonic, actionable thresholds
            ls.Metric.ACTIVATION_NORM,
            ls.Metric.SPARSITY,
            ls.Metric.KURTOSIS,
            ls.Metric.COSINE_SIMILARITY,
            # TIER 3 — Val-only:
            #   PP:     r=0.902 with GLUE — crown jewel, needs 10k+ tokens
            #   RS/RE:  secondary quantization metrics, same KMeans pass as PP
            #   EEE:    r=-0.557 but NON-MONOTONIC — logged as trend, no alerts
            # NOTE: effective_rank dropped (redundant with EEE, both SVD-based)
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
        deep_metric_interval=5  # Deep metrics every 5 sampled=every 1000 training steps
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    train_iterator = get_batches(train_dataset, tokenizer, batch_size=BATCH_SIZE)
    val_iterator = get_batches(val_dataset, tokenizer, batch_size=BATCH_SIZE)

    model.train()
    loss_history = []
    global_step = 0

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
            
            # Shuffle the dataset and create a new iterator for each epoch
            shuffled_dataset = train_dataset.shuffle(seed=42 + epoch)
            train_iterator = get_batches(shuffled_dataset, tokenizer, batch_size=BATCH_SIZE)
            
            for step_in_epoch, batch in enumerate(train_iterator, 1):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)

                global_step += 1
                monitor.step()
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                if monitor.should_run_validation():
                    # 16 batches × 8 × 128 = 16,384 tokens — already above the 10k subsample
                    # cap in log_val(), so quality is identical to 50 batches but 68% fewer
                    # model forward passes, keeping the val worker fast enough to not skip rounds.
                    val_batches = []
                    for _ in range(16):
                        try:
                            val_batch = next(val_iterator)
                        except StopIteration:
                            val_iterator = get_batches(val_dataset, tokenizer, batch_size=BATCH_SIZE)
                            val_batch = next(val_iterator)
                        
                        val_batch = {k: v.to(DEVICE) for k, v in val_batch.items()}
                        val_batches.append(val_batch)
                    
                    monitor.run_validation_pp(val_batches)

                monitor.log()
                monitor.log_scalar("loss", loss.item())
                current_lr = optimizer.param_groups[0]['lr']
                monitor.log_scalar("lr", current_lr)
                
                loss_history.append(loss.item())
                
                if global_step % 100 == 0 or global_step == 1:
                    print(f"Epoch {epoch} | Step {global_step:4d} | Loss: {loss.item():.4f}")

    finally:
        print(f"\n--- Training Completed ---")
        if loss_history:
            print(f"Final loss: {loss_history[-1]:.4f}")
            print(f"Loss improvement: {loss_history[0] - loss_history[-1]:.4f}")


        del train_iterator
        del val_iterator
        if 'dataset' in locals():
            del dataset
        gc.collect()

        print(f"\n--- Storage Analysis ---")
        print(f"Experiment data stored in: {monitor.storage.run_database}")
        
        export_path = monitor.storage.export_experiment_data(EXPERIMENT_NAME)
        print(f"Full export path: {export_path}")

        # Save the model
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
    finally:
        gc.collect()
