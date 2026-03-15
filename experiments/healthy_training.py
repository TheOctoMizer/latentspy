"""
Experiment A: Healthy Training (The Control)

Train model using standard, proven hyperparameters to establish baseline PP behavior.
"""

import torch
import gc
import os
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
import latentspy as ls

EXPERIMENT_NAME = "healthy_training"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_STEPS = 50
VAL_INTERVAL = 5
SAMPLE_INTERVAL = 2

print(f"=== {EXPERIMENT_NAME} ===")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Training Steps: {NUM_STEPS}")
print(f"Validation Interval: {VAL_INTERVAL}")

print("\nLoading dataset...")
try:
    dataset = load_dataset("roneneldan/TinyStories", streaming=True, split="train")
    train_dataset = dataset.take(2000)
    val_dataset = dataset.skip(2000).take(400)
except Exception as e:
    print(f"Error loading dataset: {e}")
    os._exit(0)

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
        ls.Metric.PATCHINESS,
        ls.Metric.ACTIVATION_NORM,
        ls.Metric.EFFECTIVE_RANK,
        ls.Metric.COSINE_SIMILARITY
    ],
    sample_interval=SAMPLE_INTERVAL,
    val_interval=VAL_INTERVAL,
    experiment_name=EXPERIMENT_NAME
)


optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

def get_batches(ds, batch_size=4):
    """Create batches from dataset"""
    batch = []
    for item in ds:
        batch.append(item["text"])
        if len(batch) == batch_size:
            yield tokenizer(batch, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            batch = []

train_batches = list(get_batches(train_dataset, batch_size=BATCH_SIZE))
val_batches = list(get_batches(val_dataset, batch_size=BATCH_SIZE))

print(f"\nTraining batches: {len(train_batches)}")
print(f"Validation batches: {len(val_batches)}")

print(f"\n--- Starting Healthy Training ---")
model.train()

pp_history = []
loss_history = []

for step in range(1, NUM_STEPS + 1):
    if step >= len(train_batches):
        break
        
    batch = train_batches[step % len(train_batches)]
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    monitor.step()
    
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()

    val_pp_results = {}
    if monitor.should_run_validation():
        print(f"\n--- Validation PP at Step {step} ---")
        val_batch = val_batches[step % len(val_batches)]
        val_batch = {k: v.to(DEVICE) for k, v in val_batch.items()}
        
        val_pp_results = monitor.run_validation_pp(val_batch)
        
        if val_pp_results:
            step_pp_values = []
            for layer_name, metrics in val_pp_results.items():
                if 'patchiness' in metrics:
                    pp_value = metrics['patchiness']
                    step_pp_values.append(pp_value)
                    print(f"  {layer_name}: PP = {pp_value:.4f}")
            
            if step_pp_values:
                avg_pp = sum(step_pp_values) / len(step_pp_values)
                pp_history.append((step, avg_pp))
                print(f"  Average PP: {avg_pp:.4f}")

    results = monitor.log()
    loss_history.append(loss.item())
    
    print(f"Step {step}/{NUM_STEPS} | Loss: {loss.item():.4f}")
    
    if step % 5 == 0 and results:
        training_layers = [name for name in results.keys() if not name.startswith('val_')]
        if training_layers:
            layer_name = training_layers[0]
            m = results[layer_name]
            print(f"  Training {layer_name}:")
            print(f"    Norm: {m.get('activation_norm', 0):.2f}")
            print(f"    Rank: {m.get('effective_rank', 0):.2f}")
            print(f"    Patchiness: {m.get('patchiness', 0):.4f}")

print(f"\n--- Training Completed ---")
print(f"Final loss: {loss_history[-1]:.4f}")
print(f"Loss improvement: {loss_history[0] - loss_history[-1]:.4f}")

print(f"\n--- PP Analysis ---")
if pp_history:
    print(f"PP measurements: {len(pp_history)}")
    first_pp = pp_history[0][1]
    last_pp = pp_history[-1][1]
    pp_change = last_pp - first_pp
    
    print(f"Initial PP: {first_pp:.4f}")
    print(f"Final PP: {last_pp:.4f}")
    print(f"PP change: {pp_change:+.4f}")
    
    if pp_change < -0.01:
        print("PP decreased - Healthy representation learning!")
    elif abs(pp_change) <= 0.01:
        print("PP stable - Model converged")
    else:
        print("PP increased - Potential training issues")
    
    print(f"\nPP progression:")
    for step, pp_val in pp_history:
        print(f"  Step {step:2d}: PP = {pp_val:.4f}")
else:
    print("No PP measurements recorded")

print(f"\n--- Final Validation Report ---")
if val_batches:
    final_val_batch = val_batches[0]
    final_val_batch = {k: v.to(DEVICE) for k, v in final_val_batch.items()}
    
    monitor.start_val()
    with torch.no_grad():
        model(final_val_batch['input_ids'], attention_mask=final_val_batch['attention_mask'])
    
    final_results = monitor.log_val()
    
    if final_results:
        print("Final layer-wise PP:")
        for layer_name, metrics in sorted(final_results.items()):
            if 'patchiness' in metrics:
                pp_val = metrics['patchiness']
                rank_val = metrics.get('effective_rank', 0)
                norm_val = metrics.get('activation_norm', 0)
                
                print(f"  {layer_name}:")
                print(f"    PP: {pp_val:.4f}")
                print(f"    Rank: {rank_val:.2f}")
                print(f"    Norm: {norm_val:.2f}")
                
                if pp_val < 0.1:
                    assessment = "Very uniform"
                elif pp_val < 0.5:
                    assessment = "Healthy distribution"
                elif pp_val < 1.0:
                    assessment = "Moderately clustered"
                else:
                    assessment = "Highly clustered/collapsed"
                print(f"    Assessment: {assessment}")

print(f"\n--- Storage Analysis ---")
print(f"Experiment data stored in: {monitor.storage.run_database}")

pp_progression = monitor.storage.get_pp_progression(EXPERIMENT_NAME)
if pp_progression:
    print(f"\nStored PP progression ({len(pp_progression)} measurements):")
    for step, avg_pp, is_val in pp_progression:
        val_marker = " (val)" if is_val else " (train)"
        print(f"  Step {step:2d}: PP = {avg_pp:.4f}{val_marker}")

export_path = monitor.storage.export_experiment_data(EXPERIMENT_NAME)
print(f"\nExperiment data exported to: {export_path}")

all_experiments = monitor.storage.list_experiments()
print(f"\nAll experiments in database ({len(all_experiments)}):")
for exp in all_experiments[:5]:  # Show first 5
    print(f"  - {exp['name']} (created: {exp['created_at']})")

monitor.remove()
del model, optimizer, monitor

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

gc.collect()
print(f"\n=== {EXPERIMENT_NAME} Complete ===")
