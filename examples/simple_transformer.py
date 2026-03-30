import torch
import torch.nn as nn
import latentspy as ls

# LatentSpy works across all PyTorch devices (CUDA/MPS/CPU)
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
        # We can watch any part of the forward flow
        return self.mlp(self.attn(x))

model = SimpleTransformerModel().to(device)
input_data = torch.randn(1, 32, 32).to(device) # [batch, seq, dim]

# 2. Attach LatentSpy to the model
# 'layers=auto' will find all Linear, Conv, and Attention layers automatically
monitor = ls.watch(
    model, 
    layers='auto',
    metrics=[
        ls.Metric.ACTIVATION_NORM, 
        ls.Metric.EFFECTIVE_RANK, 
        ls.Metric.PATCHINESS,
        ls.Metric.RECONSTRUCTION,
        ls.Metric.EEE
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
