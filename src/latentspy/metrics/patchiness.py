import torch
import numpy as np
import faiss
from .activation_utils import prepare_activations_for_faiss, validate_activations_format
from .clustering import quantize_latent_space, get_cluster_statistics


def _center_activations(activations_np: np.ndarray) -> np.ndarray:
    """Mean-center activations so patchiness reflects geometry, not layer-scale.
    
    Without centering, layers with very different activation magnitudes will show
    artificially different patchiness values purely due to L2 norm differences,
    not latent-space structure. Mean-centering removes the DC component while
    preserving relative geometry within the space.
    """
    return activations_np - activations_np.mean(axis=0, keepdims=True)


def patchiness(activations: torch.Tensor, k: int = 256) -> float:
    """
    Compute Lloyd's Patchiness Index (PP) of activations.

    PP measures how non-uniformly tokens are distributed across k quantized bins
    in the latent space. Formally (Lloyd 1967, Eq. 3 in Marbut et al. 2024):

        PP = (m* / m)  where  m* = m + V/m - 1
           = 1 + V/m² - 1/m

    where m and V are the mean and variance of RAW COUNTS (not normalized densities)
    per cluster bin. For a perfectly uniform distribution, PP approaches 1.0.
    For a representation-collapsed space (all tokens in one bin), PP approaches k.

    This metric is validated in the paper with Pearson r=0.902 against GLUE on
    BERT-small. It requires >= 10k tokens to be statistically meaningful; use only
    in validation rounds, not per-training-step.

    Args:
        activations (torch.Tensor): Activations of shape (Batch, Seq, Hidden) or
                                    any shape where the last dim is Hidden.
        k (int): Number of quantization bins. Paper default is 256.

    Returns:
        float: Lloyd's PP score. ~1.0 for uniform; higher = more patchy (clustered).
    """
    try:
        activations_np, hidden_dim = prepare_activations_for_faiss(activations)
        validate_activations_format(activations_np)
    except Exception:
        # Fallback: reshape manually
        activations_np = activations.flatten(0, -2).detach().cpu().float().numpy().astype('float32')
        if activations_np.ndim != 2 or activations_np.shape[0] < 4:
            return 1.0
    
    total_tokens = activations_np.shape[0]
    k = min(k, total_tokens // 2)
    if k < 2:
        return 1.0

    if np.std(activations_np) < 1e-8:
        # All representations identical → fully collapsed → PP = k (maximum patchiness)
        return float(k)

    # Mean-center so geometry is not confounded by layer-level magnitude differences
    activations_np = _center_activations(activations_np)

    try:
        return _patchiness_faiss(activations_np, k)
    except Exception:
        X = torch.from_numpy(activations_np)
        return _patchiness_pytorch(X, k)


def cluster_activations(activations_np: np.ndarray, k: int, index: faiss.Index) -> tuple:
    """
    Perform k-means clustering on activation vectors using FAISS.

    Args:
        activations_np: numpy array of activations of shape (num_vectors, hidden_dim).
        k: number of clusters to create
        index: FAISS index (either CPU or GPU) containing the activation vectors

    Returns:
        tuple containing:
        - centroids: numpy array of cluster centers with shape (k, hidden_dim)
        - bin_indices: numpy array of cluster assignments for each input vector
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
    """FAISS k-means implementation of Lloyd's Patchiness Index.
    
    Uses raw bin counts (not normalized proportions) to match the ecological
    definition of Lloyd (1967) and Equation 3 in Marbut et al. (2024).
    """
    cluster_labels, centroids, clustering_info = quantize_latent_space(activations_np, k)
    cluster_stats = get_cluster_statistics(cluster_labels, k)

    # Use raw integer counts (not normalized densities) — this is what Lloyd's
    # formula requires. For N tokens in k equal bins: m = N/k, V ≈ 0 → PP ≈ 1.0.
    counts = cluster_stats['cluster_counts'].astype(np.float64)
    m = counts.mean()  # mean tokens per bin
    V = counts.var()   # variance of token counts
    
    if m < 1e-10:
        return 1.0

    # Lloyd's Patchiness Index: PP = 1 + V/m² - 1/m
    # - PP = 1.0 for a Poisson/uniform process (V ≈ m)
    # - PP → k  for complete collapse (all tokens in one bin)
    PP = 1.0 + V / (m ** 2 + 1e-10) - 1.0 / (m + 1e-10)
    return float(max(PP, 0.0))


def _patchiness_pytorch(X: torch.Tensor, k: int) -> float:
    """PyTorch fallback implementation of Lloyd's Patchiness Index."""
    n = X.shape[0]
    k = min(k, n // 2)
    if k < 2:
        return 1.0
    
    centroid_indices = torch.randperm(n)[:k]
    centroids = X[centroid_indices].clone()
    
    for _ in range(20):  # match FAISS niter=20
        dists = torch.cdist(X.float(), centroids.float())
        bin_indices = torch.argmin(dists, dim=1)
        new_centroids = torch.zeros_like(centroids)
        counts_vec = torch.zeros(k, device=X.device)
        new_centroids.index_add_(0, bin_indices, X.float())
        counts_vec.index_add_(0, bin_indices, torch.ones(n, device=X.device))
        mask = counts_vec > 0
        new_centroids[mask] = new_centroids[mask] / counts_vec[mask].unsqueeze(1)
        centroids = new_centroids
    
    counts = torch.bincount(bin_indices, minlength=k).float()
    m = counts.mean().item()
    V = counts.var().item()
    
    if m < 1e-10:
        return 1.0
    
    PP = 1.0 + V / (m ** 2 + 1e-10) - 1.0 / (m + 1e-10)
    return float(max(PP, 0.0))
