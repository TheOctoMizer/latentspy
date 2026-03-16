import torch
import numpy as np
import faiss
from .activation_utils import prepare_activations_for_faiss, validate_activations_format
from .clustering import quantize_latent_space, get_cluster_statistics

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
    if hasattr(activations, 'shape'):
        total_points = activations.flatten(0, -2).shape[0] if activations.dim() > 2 else activations.shape[0]
        max_k = total_points // 39
        k = min(k, max_k, total_points // 2)
        k = max(k, 2)
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
    """FAISS-based clustering implementation using the dedicated clustering function."""
    cluster_labels, centroids, clustering_info = quantize_latent_space(activations_np, k)
    cluster_stats = get_cluster_statistics(cluster_labels, k)
    densities = cluster_stats['cluster_densities']
    mean_density = densities.mean()
    var_density = densities.var()
    
    if mean_density < 1e-10:
        return 0.0

    m = mean_density
    V = var_density
    
    # Fano Factor (V/m) normalized for densities
    # For counts c = d * N, V_c = V * N^2, m_c = m * N.
    # F = V_c / m_c = (V * N^2) / (m * N) = (V/m) * N.
    # However, to keep it independent of N (batch size), we use V/m^2 which is 0 for uniform.
    if m < 1e-10:
        return 0.0
        
    # Relative Variance (squared coefficient of variation)
    # 0 for uniform, k-1 for totally collapsed.
    return float(V / (m**2 + 1e-10))


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
    
    m = mean_density
    V = var_density
    
    if m < 1e-10:
        return 0.0
    
    # Relative Variance
    return float(V / (m**2 + 1e-10))
