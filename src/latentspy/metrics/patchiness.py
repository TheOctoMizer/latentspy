import torch
import numpy as np
import faiss
from .activation_utils import prepare_activations_for_faiss, validate_activations_format

def patchiness(activations: torch.Tensor, k: int = 256) -> float:
    """
    Compute the Patchiness Proportion (PP) of activations.

    Args:
        activations (torch.Tensor): Output activations of shape (Batch, Seq, Hidden) or 
                                  (Batch, ..., Hidden).
        k (int): Number of bins. Paper default is 256.

    Returns:
        float: The PP score (Fano Factor of cluster densities).
    """
    try:
        activations_np, hidden_dim = prepare_activations_for_faiss(activations)
        validate_activations_format(activations_np)
    except Exception as e:
        X = activations.flatten(1).detach().float()
        batch_size = X.size(0)
        k = min(k, batch_size // 2)
        if k < 2:
            return 0.0
        if X.std().item() < 1e-8:
            return 1.0
        return _patchiness_pytorch(X, k)
    
    total_tokens = activations_np.shape[0]
    k = min(k, total_tokens // 2)
    if k < 2:
        return 0.0

    if np.std(activations_np) < 1e-8:
        return 1.0

    try:
        return _patchiness_faiss(activations_np, k)
    except Exception:
        X = activations.flatten(1).detach().float()
        return _patchiness_pytorch(X, k)


def cluster_activations(activations_np: np.ndarray, k: int, index: faiss.Index) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform k-means clustering on activation vectors using FAISS.

    Args:
        activations_np: numpy array of activations of shape (num_vectors, hidden_dim).
        k: number of clusters to create
        index: FAISS index (either CPU or GPU) containing the activation vectors

    Returns:
        tuple containing:
        - centroids: numpy array of cluster centers with shape (k, hidden_dim)
        - bin_indices: numpy array of cluster assignments for each input vector with shape(num_vectors, )
    """
    hidden_dim = activations_np.shape[1]

    kmeans = faiss.Kmeans(hidden_dim, k, niter=20, verbose=False)
    kmeans.train(activations_np)
    centroids = kmeans.centroids

    centroid_index = faiss.IndexFlatL2(hidden_dim)
    centroid_index.add(centroids)

    D, I = index.search(activations_np, 1)
    bin_indices = I.flatten()

    return centroids, bin_indices

def _patchiness_faiss(activations_np: np.ndarray, k: int) -> float:
    """FAISS-based clustering implementation."""
    hidden_dim = activations_np.shape[1]
    
    if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
        # CUDA Implementation
        try:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(hidden_dim)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(activations_np)
            centroids, bin_indices = cluster_activations(activations_np, k, gpu_index)

        except Exception:
            index = faiss.IndexFlatL2(hidden_dim)
            index.add(activations_np)

            kmeans = faiss.Kmeans(hidden_dim, k, niter=20, verbose=False)
            kmeans.train(activations_np)
            centroids = kmeans.centroids

            D, I = index.search(activations_np, 1)
            bin_indices = I.flatten()
    else:
        # CPU-only implementation
        index = faiss.IndexFlatL2(hidden_dim)
        index.add(activations_np)
        centroids, bin_indices = cluster_activations(activations_np, k, index)
    
    counts = np.bincount(bin_indices, minlength=k).astype('float32')
    densities = counts / activations_np.shape[0]
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
