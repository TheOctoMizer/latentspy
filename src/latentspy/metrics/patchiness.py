import torch
import numpy as np

def patchiness(activations: torch.Tensor, k: int = 256) -> float:
    """
    Compute the Patchiness Proportion (PP) of activations.

    PP = Var(cell_density) / Mean(cell_density)  [Fano Factor — paper formula]

    Uses random centroid sampling + PyTorch's cdist for 1-NN assignment.
    This is mathematically equivalent to FAISS flat L2 search but is stable
    across all platforms (FAISS conflicts with PyTorch OpenMP on macOS).

    Args:
        activations (torch.Tensor): Output activations of shape (Batch, ...).
        k (int): Number of bins. Paper default is 256.

    Returns:
        float: The PP score (Fano Factor of cluster densities).
    """
    X = activations.flatten(1).detach().float()
    batch_size = X.size(0)

    k = min(k, batch_size // 2)
    if k < 2:
        return 0.0

    if X.std().item() < 1e-8:
        return 1.0

    centroid_indices = torch.randperm(batch_size)[:k]
    centroids = X[centroid_indices]
    dists = torch.cdist(X, centroids)
    bin_indices = torch.argmin(dists, dim=1)
    counts = torch.bincount(bin_indices, minlength=k).float()
    densities = counts / batch_size
    mean_density = densities.mean()
    var_density = densities.var()

    if mean_density < 1e-10:
        return 0.0

    return (var_density / mean_density).item()
