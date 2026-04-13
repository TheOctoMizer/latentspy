# LatentSpy 🕵️‍♂️

  "LatentSpy is heavily inspired by and built upon the research presented in 'Exploring the Impact of a Transformer's Latent Space Geometry on Downstream Task Performance' by Anna Marbut et al. (2024). We highly recommend reading the original paper for the theoretical foundation of these metrics."
### Catch Model Failures Before the Loss Does.

LatentSpy is a real-time diagnostic and monitoring utility for PyTorch models. It hooks into your model's internal activations to detect "representation collapse," "activation explosions," and other structural pathologies that often occur long before your training loss starts to diverge.

> [!IMPORTANT]
> **The Honest Take:** LatentSpy is a researcher's tool built for deep observability. It is NOT a "plug-and-play" performance booster. It adds a small amount of overhead (mostly async) and requires you to actually look at the metrics to understand what's happening inside your "black box."

## Why LatentSpy?
Most monitoring tools (WandB, TensorBoard) focus on **outcomes** (loss, accuracy). LatentSpy focuses on **internal health**. It helps you identify:
- **Representation Death:** When >95% of your neurons stop firing.
- **Dimensional Collapse:** When your model maps all inputs to a tiny sub-space.
- **Activation Explosions:** Early signs of numerical instability.
- **Structural Fragmentation:** Using reconstruction error to see if your clusters are meaningful.

## Features
- **Real-Time Dashboard:** A Bokeh/Lite-inspired "Mission Control" (FastAPI + WebSockets) for live metric tracking.
- **Asynchronous Execution:** Metric computation happens in background threads to minimize training stalls.
- **Multi-Device Support:** Optimized for CUDA (NVIDIA), MPS (Apple Silicon), and CPU. 
- **Tiered Metrics:**
  - **Tier 1 (Fast):** Scalar-based (Norms, Sparsity, Kurtosis) computed Every Step.
  - **Tier 2 (Deep):** Vector-based (Cosine Similarity) computed periodically.
  - **Tier 3 (Validation Only):** Heavy geometric metrics (Patchiness, EEE, Reconstruction) computed on validation data.
- **3D Latent Projections:** Interactive visualization of how your hidden states are distributed.

## Installation

### Standard (CPU/MPS)
```bash
pip install latentspy
```

### GPU Support (CUDA)
*Requires Python < 3.11 for specific dependency compatibility.*
```bash
pip install latentspy[cuda]
```

## Quick Start

```python
from latentspy import LatentMonitor
import torch

model = MyTransformer()
# Setup monitor to track Attention and MLP layers automatically
monitor = LatentMonitor(model, layers="auto", dashboard=True)

for epoch in range(10):
    for batch in dataloader:
        monitor.step() # Tick the internal step counter
        
        output = model(batch)
        loss = criterion(output, target)
        
        # Log custom metrics alongside latents
        monitor.log_scalar("loss", loss.item())
        
        # Capture current activations
        monitor.log()
        
        loss.backward()
        optimizer.step()
```

## Caveats
- **Memory Pressure:** Monitoring large models requires capturing high-dimensional tensors. While LatentSpy uses asynchronous processing and subsampling, you should be mindful of CPU RAM if monitoring many layers at once.
- **XLA/TPU Support:** Currently **NOT supported**. The internal hooking mechanism disrupts XLA graph compilation and will crash your training.
- **SQLite Locking:** All runs are stored in a local `.latentspy/` directory using SQLite. If you are running massive distributed jobs with many writers, you might encounter database contention.

## The Dashboard
Start the dashboard standalone or from within your script:
```python
import latentspy
latentspy.serve(port=8000)
```
Then visit `http://localhost:8000` to see **Mission Control**.

## Core Metrics Explained
- **EEE (Eigenvalue Early Enrichment):** Measures how quickly the model "learns" the dominant directions of data. High EEE early on is a sign of healthy representation learning.
- **Patchiness (PP):** A measure of how "clumpy" the latent space is. If PP spikes, your model might be collapsing into a few modes.
- **Reconstruction Skew:** Measures the imbalance in how well different parts of the latent space are captured by a linear probe or cluster.

---
*Built for the AAI-590 Capstone Project/Research. For bugs, features, or architectural debates, please open an issue.*