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
class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size=50, embedding_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x) # RNN layer emits both output and hidden
        x = self.fc(output)
        return x

vocab_size = 50
model = SimpleRNNModel(vocab_size=vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 2. Attach LatentSpy to the model
monitor = ls.watch(
    model, 
    layers=["embedding", "rnn", "fc"],
    metrics=[
        ls.Metric.ACTIVATION_NORM, 
        ls.Metric.SPARSITY, 
        ls.Metric.KURTOSIS,
        ls.Metric.PATCHINESS
    ],
    experiment_name="simple_rnn_example",
    log_type="db",
    val_interval=10
)

# 3. Model Execution & Training Loop
print("\n--- Starting Training Example ---")
model.train()
TOTAL_STEPS = 50

for step in range(1, TOTAL_STEPS + 1):
    input_data = torch.randint(0, vocab_size, (8, 10), device=device) # [batch, seq]
    target = torch.randint(0, vocab_size, (8, 10), device=device)
    
    monitor.step()
    optimizer.zero_grad()
    
    output = model(input_data)
    # Reshape for cross entropy: [batch*seq, vocab_size] vs [batch*seq]
    loss = criterion(output.view(-1, vocab_size), target.view(-1))
    loss.backward()
    optimizer.step()
    
    monitor.log_scalar('train_loss', loss.item())
    
    # 4. Handle Geometric Validation Rounds
    if monitor.should_run_validation():
        print(f" ► [Step {step}] Running Geometric Validation Round...")
        val_batches = [torch.randint(0, vocab_size, (8, 10), device=device) for _ in range(2)]
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
        for k in ['activation_norm', 'patchiness', 'kurtosis']:
            if k in metrics:
                print(f"  {k:<16}: {metrics[k]:.4f}")

# 6. Cleanup
monitor.remove()
print("\nLatentSpy handles detached successfully.")