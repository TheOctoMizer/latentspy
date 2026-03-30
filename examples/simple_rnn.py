import torch
import torch.nn as nn
import latentspy as ls

# LatentSpy works across all PyTorch devices (CUDA/MPS/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f'Using device: {device}')

# 1. Define a minimal architecture to watch
class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size=50, embedding_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        # We can watch any part of the forward flow
        x = self.embedding(x)
        output, _ = self.rnn(x) # RNN layer emits both output and hidden
        x = self.fc(output)
        return x

vocab_size = 50
model = SimpleRNNModel(vocab_size=vocab_size).to(device)
input_data = torch.randint(0, vocab_size, (1, 10)).to(device) # [batch, seq]

# 2. Attach LatentSpy to the model
# Using 'auto' finds the common layers, or specify them for exact control
monitor = ls.watch(
    model, 
    layers=["embedding", "rnn", "fc"],
    metrics=[
        ls.Metric.ACTIVATION_NORM, 
        ls.Metric.SPARSITY, 
        ls.Metric.KURTOSIS,
        ls.Metric.PATCHINESS
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
        # Only show a few metrics for this example
        for k in ['activation_norm', 'patchiness', 'kurtosis']:
            if k in metrics:
                print(f"  {k:<16}: {metrics[k]:.4f}")

# 5. Cleanup
# Always remove the handles to prevent memory buildup in long-running environments
monitor.remove()
print("\nLatentSpy handles detached successfully.")