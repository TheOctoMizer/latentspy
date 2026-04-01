import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import latentspy as ls

# LatentSpy works across all PyTorch devices (CUDA/MPS/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f'Using device: {device}')

# 1. Define a minimal architecture to watch
class SimpleCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20, 10)
    
    def forward(self, x):
        # We can watch any part of the forward flow
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = SimpleCNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 2. Attach LatentSpy to the model
monitor = ls.watch(
    model, 
    layers=["conv1", "conv2", "fc1"],
    metrics=[
        ls.Metric.ACTIVATION_NORM, 
        ls.Metric.EFFECTIVE_RANK, 
        ls.Metric.PATCHINESS,
        ls.Metric.RECONSTRUCTION
    ],
    experiment_name="simple_cnn_example",
    log_type="db",
    val_interval=10 # Example validation interval
)

# 3. Model Execution & Training Loop
print("\n--- Starting Training Example ---")
model.train()
TOTAL_STEPS = 50

for step in range(1, TOTAL_STEPS + 1):
    input_data = torch.randn(8, 1, 9, 9, device=device) # [batch, channel, height, width]
    target = torch.randint(0, 10, (8,), device=device)
    
    # Let LatentSpy know a new training step has started
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
        val_batches = [torch.randn(8, 1, 9, 9, device=device) for _ in range(2)]
        monitor.run_validation_pp(val_batches)
        
    monitor.log() # Log standard metrics (norms, sparsity)
    
    if step % 10 == 0:
        print(f"Step {step:03d}/{TOTAL_STEPS} | Loss: {loss.item():.4f}")

# 5. Extracting Results
print("\n--- Diagnostic Results (Quick Glance) ---")
results = monitor.log()

if results:
    for layer, metrics in results.items():
        if layer == '__scalars__': continue
        print(f"\nLayer: {layer}")
        # Only show the most interesting highlights for this example
        for k in ['activation_norm', 'patchiness', 'effective_rank']:
            if k in metrics:
                print(f"  {k:<16}: {metrics[k]:.4f}")

# 6. Cleanup
monitor.remove()
print("\nLatentSpy handles detached successfully.")
