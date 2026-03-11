import torch
import torch.nn as nn
import latentspy as ls

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f'Using device: {device}')

class SimpleTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(10, 10)
        self.mlp = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
    
    def forward(self, x):
        return self.mlp(self.attn(x))

model = SimpleTransformerModel()
input_data = torch.randn(16, 10)

monitor = ls.watch(
    model, 
    layers='auto',
    metrics=[
        ls.Metric.ACTIVATION_NORM, 
        ls.Metric.EFFECTIVE_RANK, 
        ls.Metric.COSINE_SIMILARITY, 
        ls.Metric.PATCHINESS
    ]
)

monitor.step()

print("--- Watching Model ---")
print(f"Tracking layers: {len(monitor.handles)}")
output = model(input_data)

results = monitor.log()

print("\n--- Results ---")
for layer, metrics in results.items():
    print(f"\nLayer: {layer}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

monitor.remove()
