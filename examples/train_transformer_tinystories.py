import torch
import gc
import os
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
import latentspy as ls

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading dataset...")
try:
    dataset = load_dataset("roneneldan/TinyStories", streaming=True, split="train")
    dataset = dataset.take(1000) 
    # Create separate validation dataset
    val_dataset = dataset.skip(800).take(200)  # Use last 200 for validation
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
    n_embd=128,
    n_layer=2,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config).to(device)

monitor = ls.watch(
    model,
    layers='auto',
    metrics=[
        ls.Metric.ACTIVATION_NORM,
        ls.Metric.EFFECTIVE_RANK,
        ls.Metric.COSINE_SIMILARITY,
        ls.Metric.PATCHINESS
    ],
    val_interval=3  # Compute validation PP every 3 steps
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

def get_batches(ds, batch_size=4):
    batch = []
    for item in ds:
        batch.append(item["text"])
        if len(batch) == batch_size:
            yield tokenizer(batch, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            batch = []

print("\n--- Starting Training ---")
model.train()
num_steps = 10
batch_size = 4

batches = get_batches(dataset, batch_size=batch_size)
val_batches = list(get_batches(val_dataset, batch_size=batch_size))  # Convert to list for repeated access

for step in range(1, num_steps + 1):
    try:
        batch = next(batches)
    except StopIteration:
        break
        
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    monitor.step()

    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()

    # Run validation PP if it's time
    if monitor.should_run_validation():
        print(f"\n--- Running Validation PP at Step {step} ---")
        val_batch = val_batches[step % len(val_batches)]
        val_batch = {k: v.to(device) for k, v in val_batch.items()}
        
        val_results = monitor.run_validation_pp(val_batch)
        
        if val_results:
            print("Validation PP Results:")
            for layer_name, metrics in val_results.items():
                if 'patchiness' in metrics:
                    print(f"  {layer_name}: PP = {metrics['patchiness']:.4f}")

    results = monitor.log()

    print(f"Step {step}/{num_steps} | Loss: {loss.item():.4f}")
    
    if step % 2 == 0:
        print(f"  [LatentSpy] Training Stability Metrics:")
        if results:
            # Show stats for a specific layer (e.g., first attention layer)
            layer_name = list(results.keys())[0]
            if not layer_name.startswith('val_'):  # Only show training metrics
                m = results[layer_name]
                print(f"    Layer: {layer_name}")
                print(f"      Norm: {m.get('activation_norm', 0):.2f}")
                print(f"      Rank: {m.get('effective_rank', 0):.2f}")
                print(f"      CosSim: {m.get('cosine_similarity', 0):.2f}")
                print(f"      Patchiness: {m.get('patchiness', 0):.2f} (lower is usually more stable)")
            
            # Show validation metrics if available
            val_layers = [name for name in results.keys() if name.startswith('val_')]
            if val_layers:
                print(f"    Validation PP (consistent distribution):")
                for val_layer in val_layers[:2]:  # Show first 2 validation layers
                    val_m = results[val_layer]
                    if 'patchiness' in val_m:
                        print(f"      {val_layer}: {val_m['patchiness']:.4f}")

print("\n--- Training Finished ---")

# --- Stability Report ---
print("\n" + "="*40)
print("FINAL TRAINING STABILITY REPORT")
print("="*40)

# We can perform a deeper analysis on the last step's activations
results = monitor.log()
if results:
    for layer_name, metrics in results.items():
        patchiness_val = metrics.get('patchiness', 0)
        rank_val = metrics.get('effective_rank', 0)
        
        print(f"\nLayer: {layer_name}")
        print(f"  Final Patchiness: {patchiness_val:.4f}")
        print(f"  Final Effective Rank: {rank_val:.4f}")
        
        if patchiness_val > 1.5:
             print("  [Assessment] High patchiness detected. Potential representation collapse.")
        elif patchiness_val < 0.1:
             print("  [Assessment] Very low patchiness. Latent space is highly uniform.")
        else:
             print("  [Assessment] Normal patchiness. Representation is well-distributed.")

print("-" * 40)

# Cleanup
monitor.remove()
print("LatentSpy monitor removed.")

del model
del optimizer
del monitor

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

gc.collect()
os._exit(0)
