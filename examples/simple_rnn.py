import torch
import torch.nn as nn
import latentspy as ls

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f'Using device: {device}')

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size=50, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        x = self.fc(output)
        return x

vocab_size = 50
model = SimpleRNNModel(vocab_size=vocab_size)
input_data = torch.randint(0, vocab_size, (16, 20))

monitor = ls.watch(
    model, 
    layers=["embedding", "rnn", "fc"],
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
if results:
    for layer, metrics in results.items():
        print(f"\nLayer: {layer}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
else:
    print("No metrics captured. Ensure layers are correctly specified.")

monitor.remove()