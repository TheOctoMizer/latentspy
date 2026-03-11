import torch
import numpy as np
import faiss

def patchiness(activations: torch.Tensor, k: int = 256) -> float:
    """
    Compute the Patchiness Proportion (PP) of activations.

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

    try:
        return _patchiness_faiss(X, k)
    except Exception:
        # Fallback to PyTorch implementation
        return _patchiness_pytorch(X, k)


def _patchiness_faiss(X: torch.Tensor, k: int) -> float:
    """FAISS-based clustering implementation."""
    X_np = X.cpu().numpy().astype('float32')
    
    if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
        # CUDA Implementation
        try:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(X_np.shape[1])
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(X_np)
            
            kmeans = faiss.Kmeans(X_np.shape[1], k, niter=20, verbose=False)
            kmeans.train(X_np)
            centroids = kmeans.centroids
            
            D, I = gpu_index.search(centroids, 1)
            
        except Exception:
            index = faiss.IndexFlatL2(X_np.shape[1])
            index.add(X_np)

            kmeans = faiss.Kmeans(X_np.shape[1], k, niter=20, verbose=False)
            kmeans.train(X_np)
            centroids = kmeans.centroids

            D, I = index.search(X_np, 1)
            bin_indices = I.flatten()
    else:
        # CPU-only implementation
        index = faiss.IndexFlatL2(X_np.shape[1])
        index.add(X_np)

        kmeans = faiss.Kmeans(X_np.shape[1], k, niter=20, verbose=False)
        kmeans.train(X_np)
        centroids = kmeans.centroids

        centroid_index = faiss.IndexFlatL2(X_np.shape[1])
        centroid_index.add(centroids)

        D, I = centroid_index.search(X_np, 1)
        bin_indices = I.flatten()

    counts = np.bincount(bin_indices, minlength=k).astype('float32')
    densities = counts / X_np.shape[0]
    mean_density = densities.mean()
    var_density = densities.var()
    
    if mean_density < 1e-10:
        return 0.0
    
    return (var_density / mean_density).item()


def _patchiness_pytorch(X: torch.Tensor, k: int) -> float:
    """PyTorch fallback implementation."""
    batch_size = X.size(0)

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
