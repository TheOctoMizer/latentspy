import torch
import numpy as np
import faiss
from typing import Tuple, Dict, Any
from .density import calculate_cell_densities, analyze_density_distribution


def quantize_latent_space(activations_np: np.ndarray, k: int = 256, niter: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Quantize the latent space using FAISS K-Means clustering.
    
    This function breaks the high-dimensional latent space into distinct "subspaces" or bins.
    
    Args:
        activations_np (np.ndarray): Prepared activations array of shape (total_tokens, hidden_dim)
        k (int): Number of cluster centroids. Paper default is 256.
        niter (int): Number of K-Means iterations. Default is 5.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - cluster_labels: Array of integer labels (0 to k-1) for each token
            - centroids: Array of cluster centroids of shape (k, hidden_dim)
            - clustering_info: Dictionary with clustering statistics
    """
    if not isinstance(activations_np, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(activations_np)}")
    
    if activations_np.dtype != np.float32:
        raise ValueError(f"Expected float32 dtype, got {activations_np.dtype}")
    
    if activations_np.ndim != 2:
        raise ValueError(f"Expected 2D array, got {activations_np.ndim}D")
    
    total_tokens, hidden_dim = activations_np.shape
    
    k = min(k, total_tokens // 2)
    if k < 2:
        raise ValueError(f"Not enough data points for clustering. Need at least 4 points, got {total_tokens}")
    
    kmeans = faiss.Kmeans(
        d=hidden_dim,
        k=k,
        niter=niter,
        verbose=False,
        gpu=torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
    )
    
    kmeans.train(activations_np)
    
    centroids = kmeans.centroids
    
    index = faiss.IndexFlatL2(hidden_dim)
    index.add(centroids)
    
    distances, labels = index.search(activations_np, 1)
    cluster_labels = labels.flatten()
    
    clustering_info = {
        'k': k,
        'total_tokens': total_tokens,
        'hidden_dim': hidden_dim,
        'centroids_shape': centroids.shape,
        'labels_shape': cluster_labels.shape,
        'inertia': float(kmeans.obj[0]) if hasattr(kmeans, 'obj') and len(kmeans.obj) > 0 else None,
        'niter': niter,
        'gpu_used': torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'),
        'unique_labels': len(np.unique(cluster_labels)),
        'empty_clusters': k - len(np.unique(cluster_labels))
    }
    
    return cluster_labels, centroids, clustering_info


def get_cluster_statistics(cluster_labels: np.ndarray, k: int = 256) -> Dict[str, Any]:
    """
    Calculate statistics about the clustering results.
    
    Args:
        cluster_labels (np.ndarray): Array of cluster labels for each token
        k (int): Total number of clusters expected
    
    Returns:
        Dict[str, Any]: Statistics about cluster distribution and densities
    """
    cell_densities, density_info = calculate_cell_densities(cluster_labels, k)

    density_analysis = analyze_density_distribution(cell_densities)

    cluster_densities = cell_densities.astype('float32') / cell_densities.sum()
    
    stats = {
        'cluster_counts': cell_densities,
        'cluster_densities': cluster_densities,
        'density_info': density_info,
        'density_analysis': density_analysis
    }
    
    return stats
