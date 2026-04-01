import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import latentspy as ls

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f'Using device: {device}')

# 1. Define a minimal architecture to watch
class SimpleTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(32, 32)
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, x):
        return self.mlp(self.attn(x))

model = SimpleTransformerModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 2. Attach LatentSpy to the model
monitor = ls.watch(
    model, 
    layers='auto',
    metrics=[
        ls.Metric.ACTIVATION_NORM, 
        ls.Metric.EFFECTIVE_RANK, 
        ls.Metric.PATCHINESS,
        ls.Metric.RECONSTRUCTION,
        ls.Metric.EEE
    ],
    experiment_name="simple_transformer_example",
    log_type="db",
    val_interval=10
)

# 3. Model Execution & Training Loop
print("\n--- Starting Training Example ---")
model.train()
TOTAL_STEPS = 50

for step in range(1, TOTAL_STEPS + 1):
    input_data = torch.randn(8, 32, 32, device=device) # [batch, seq, dim]
    target = torch.randn(8, 32, 32, device=device)
    
    monitor.step()
    optimizer.zero_grad()
    
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    monitor.log_scalar('train_loss', loss.item())
    
    # 4. Handle Geometric Validation Rounds
    if monitor.should_run_validation():
        print(f" ► [Step {step}] Running Geometric Validation Round...")
        val_batches = [torch.randn(8, 32, 32, device=device) for _ in range(2)]
        monitor.run_validation_pp(val_batches)
        
    monitor.log()
    
    if step % 10 == 0:
        print(f"Step {step:03d}/{TOTAL_STEPS} | Loss: {loss.item():.4f}")

# 5. Extracting Results
print("\n--- Diagnostic Results (Quick Glance) ---")
results = monitor.log()

if results:
    for layer, metrics in results.items():
        if layer == '__scalars__': continue
        print(f"\nLayer: {layer}")
        for k in ['activation_norm', 'patchiness', 'effective_rank']:
            if k in metrics:
                print(f"  {k:<16}: {metrics[k]:.4f}")

# 6. Cleanup
monitor.remove()
print("\nLatentSpy handles detached successfully.")
