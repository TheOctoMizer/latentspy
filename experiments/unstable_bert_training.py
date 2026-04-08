import os
import sys
import gc
import random

# Environment configuration for stability on macOS
os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "True",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1"
})

import torch
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
import latentspy as ls


def run_experiment():
    EXPERIMENT_NAME: str = "unstable_bert_training"
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {DEVICE}")

    # ------------------------------------------------------------------
    # UNSTABLE HYPERPARAMETERS (The point of failure demo)
    # ------------------------------------------------------------------
    LEARNING_RATE   = 0.05       # Extremely high for BERT pre-training
    BATCH_SIZE      = 16
    NUM_EPOCHS      = 3
    MLM_PROBABILITY = 0.15
    MAX_SEQ_LEN     = 128
    WARMUP_STEPS    = 0          # No warmup
    MAX_GRAD_NORM   = 100.0      # No clipping

    # LatentSpy observation cadence
    VAL_INTERVAL    = 750
    SAMPLE_INTERVAL = 200

    # Config summary
    print(f"\n[Run Config] {EXPERIMENT_NAME} (UNSTABLE)")
    print(f"  Params : LR={LEARNING_RATE}, BS={BATCH_SIZE}, Epochs={NUM_EPOCHS}")
    print(f"  Monitor: Val_Int={VAL_INTERVAL}, Sample_Int={SAMPLE_INTERVAL}")

    # 1. Model & Tokenizer
    print("\nInitializing BERT model...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=MAX_SEQ_LEN,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config).to(DEVICE)

    # 2. Dataset Preparation
    print("\nLoading dataset...")
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        train_raw = dataset.select(range(100_000))
        val_raw   = dataset.select(range(100_000, 102_000))

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LEN,
                return_token_type_ids=True,
            )

        print("Tokenizing datasets...")
        train_dataset = train_raw.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Train")
        val_dataset   = val_raw.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Val")

        BERT_COLS = ["input_ids", "attention_mask", "token_type_ids"]
        train_dataset.set_format(type="torch", columns=BERT_COLS)
        val_dataset.set_format(type="torch", columns=BERT_COLS)

        mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=MLM_PROBABILITY,
            return_tensors="pt",
        )

        # num_workers=0 and pin_memory=False for system stability
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=mlm_collator, pin_memory=False, num_workers=0)
        
        def get_val_batch(loader):
            iterator = iter(loader)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    yield next(iterator)

        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=mlm_collator, pin_memory=False, num_workers=0)
        val_iterator = get_val_batch(val_loader)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # 3. LatentSpy Monitor
    monitor = ls.watch(
        model,
        layers="auto",
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
        deep_metric_interval=5,
    )

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # 5. Training Loop
    loss_history = []
    global_step  = 0

    print(f"\n--- MISSION START: {EXPERIMENT_NAME} (UNSTABLE) ---")
    try:
        model.train()
        for epoch in range(1, NUM_EPOCHS + 1):
            for step_in_epoch, batch in enumerate(train_loader, 1):
                input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
                attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE, non_blocking=True)
                labels         = batch["labels"].to(DEVICE, non_blocking=True)

                global_step += 1
                monitor.step()

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()

                if monitor.should_run_validation():
                    val_batches = []
                    for _ in range(8):
                        vb = next(val_iterator)
                        vb = {k: v.to(DEVICE, non_blocking=True) for k, v in vb.items()}
                        val_batches.append(vb)

                    monitor.run_validation_pp(val_batches)

                    model.eval()
                    total_val_loss = 0.0
                    with torch.no_grad():
                        for vb in val_batches:
                            v_out = model(input_ids=vb["input_ids"], attention_mask=vb["attention_mask"], token_type_ids=vb.get("token_type_ids"), labels=vb["labels"])
                            total_val_loss += v_out.loss.item()

                    avg_val_loss = total_val_loss / len(val_batches)
                    monitor.log_scalar("val_loss", avg_val_loss)
                    print(f"   [Val] Step: {global_step:5d} | Val Loss: {avg_val_loss:.4f}")
                    model.train()
                    
                    # Cleanup val tensors
                    del val_batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                monitor.log()
                monitor.log_scalar("loss", loss.item())
                monitor.log_scalar("lr", scheduler.get_last_lr()[0])
                loss_history.append(loss.item())

                if global_step % 100 == 0 or global_step == 1:
                    print(f"      Step: {global_step:5d} | Train Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted.")

    finally:
        print(f"\n--- Training Completed / Halted ---")
        if 'train_loader' in locals():
            del train_loader
        monitor.remove()
        gc.collect()

if __name__ == "__main__":
    run_experiment()
