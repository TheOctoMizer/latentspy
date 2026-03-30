import torch
import torch.nn as nn
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
input_data = torch.randn(1, 1, 9, 9).to(device) # [batch, channel, height, width]

# 2. Attach LatentSpy to the model
# You can specify exact layers by name or use 'auto' to capture everything
monitor = ls.watch(
    model, 
    layers=["conv1", "conv2", "fc1"],
    metrics=[
        ls.Metric.ACTIVATION_NORM, 
        ls.Metric.EFFECTIVE_RANK, 
        ls.Metric.PATCHINESS,
        ls.Metric.RECONSTRUCTION
    ]
)

# 3. Model Execution
# monitor.step() increments the internal step counter for time-series logging
monitor.step()

print("\n--- Executing Forward Pass ---")
output = model(input_data)

# 4. Extracting Results
# monitor.log() computes the metrics for the captured activations
results = monitor.log()

print("\n--- Diagnostic Results (Quick Glance) ---")
if results:
    for layer, metrics in results.items():
        print(f"\nLayer: {layer}")
        # Only show the most interesting highlights for this example
        for k in ['activation_norm', 'patchiness', 'effective_rank']:
            if k in metrics:
                print(f"  {k:<16}: {metrics[k]:.4f}")

# 5. Cleanup
# Always remove the handles to prevent memory buildup in long-running environments
monitor.remove()
print("\nLatentSpy handles detached successfully.")