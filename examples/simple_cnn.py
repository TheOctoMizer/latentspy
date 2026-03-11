import torch
import torch.nn as nn
import latentspy as ls

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f'Using device: {device}')

class SimpleCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = SimpleCNNModel()
input_data = torch.randn(16, 1, 9, 9)

monitor = ls.watch(
    model, 
    layers=["conv1", "conv2", "fc1"],
    metrics=[
        ls.Metric.ACTIVATION_NORM, 
        ls.Metric.EFFECTIVE_RANK, 
        ls.Metric.COSINE_SIMILARITY, 
        ls.Metric.PATCHINESS
    ]
)

monitor.step()

print("Running forward pass...")
output = model(input_data)

results = monitor.log()

print("\n--- Captured Metrics ---")
for layer, metrics in results.items():
    print(f"\nLayer: {layer}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

monitor.remove()