import numpy as np
from typing import Dict, Any, Tuple


def calculate_cell_densities(cluster_labels: np.ndarray, k: int = 256) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calculate cell densities from cluster assignments.
    
    This function counts the number of points assigned to each cluster centroid,
    representing how crowded each bin is in the latent space.
    
    Args:
        cluster_labels (np.ndarray): Array of cluster assignments (0 to k-1) for each token
        k (int): Total number of clusters/bins. Paper default is 256.
    
    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - cell_densities: Array of raw point counts for each cluster
            - density_info: Dictionary with density statistics
    """
    if not isinstance(cluster_labels, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(cluster_labels)}")
    
    if cluster_labels.ndim != 1:
        raise ValueError(f"Expected 1D array, got {cluster_labels.ndim}D")
    
    if cluster_labels.min() < 0 or cluster_labels.max() >= k:
        raise ValueError(f"Cluster labels must be in range [0, {k-1}], got range [{cluster_labels.min()}, {cluster_labels.max()}]")
    
    cell_densities = np.bincount(cluster_labels, minlength=k)
    
    total_points = len(cluster_labels)
    non_empty_clusters = np.count_nonzero(cell_densities)
    empty_clusters = k - non_empty_clusters
    
    density_info = {
        'k': k,
        'total_points': total_points,
        'cell_densities': cell_densities,
        'non_empty_clusters': non_empty_clusters,
        'empty_clusters': empty_clusters,
        'mean_density': float(cell_densities.mean()),
        'std_density': float(cell_densities.std()),
        'min_density': int(cell_densities.min()),
        'max_density': int(cell_densities.max()),
        'most_crowded_bin': int(cell_densities.argmax()),
        'least_crowded_bin': int(cell_densities.argmin()) if non_empty_clusters > 0 else None,
        'density_range': int(cell_densities.max() - cell_densities.min()),
        'coefficient_of_variation': float(cell_densities.std() / (cell_densities.mean() + 1e-10))
    }
    
    return cell_densities, density_info


def analyze_density_distribution(cell_densities: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the distribution of cell densities.
    
    This function provides additional insights into how uniform the latent space is.
    A perfectly uniform space would have all bins with similar densities.
    
    Args:
        cell_densities (np.ndarray): Array of point counts for each cluster
    
    Returns:
        Dict[str, Any]: Analysis of density distribution
    """
    total_points = cell_densities.sum()
    density_proportions = cell_densities.astype('float32') / total_points
    
    entropy = -np.sum(density_proportions * np.log(density_proportions + 1e-10))
    max_entropy = np.log(len(cell_densities))
    entropy_ratio = entropy / max_entropy
    
    sorted_densities = np.sort(cell_densities)
    n = len(cell_densities)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_densities)) / (n * np.sum(sorted_densities)) - (n + 1) / n
    
    avg_density = cell_densities.mean()
    dominant_clusters = np.where(cell_densities > 2 * avg_density)[0]
    dominant_density_fraction = cell_densities[dominant_clusters].sum() / total_points if len(dominant_clusters) > 0 else 0
    
    analysis = {
        'entropy': float(entropy),
        'max_entropy': float(max_entropy),
        'entropy_ratio': float(entropy_ratio),
        'gini_coefficient': float(gini),
        'dominant_clusters': dominant_clusters.tolist(),
        'num_dominant_clusters': len(dominant_clusters),
        'dominant_density_fraction': float(dominant_density_fraction),
        'density_proportions': density_proportions,
        'is_uniform': entropy_ratio > 0.9,
        'is_collapsed': float(gini) > 0.7
    }
    
    return analysis


def get_density_summary(cell_densities: np.ndarray, k: int = 256) -> str:
    """
    Get a human-readable summary of density distribution.
    
    Args:
        cell_densities (np.ndarray): Array of point counts for each cluster
        k (int): Total number of clusters
    
    Returns:
        str: Human-readable summary
    """
    if cell_densities.sum() == 0:
        return "Cell Density Summary: No points found"
    
    total_points = cell_densities.sum()
    non_empty_clusters = np.count_nonzero(cell_densities)
    
    density_proportions = cell_densities.astype('float32') / total_points
    
    entropy = -np.sum(density_proportions * np.log(density_proportions + 1e-10))
    max_entropy = np.log(len(cell_densities))
    entropy_ratio = entropy / max_entropy
    
    sorted_densities = np.sort(cell_densities)
    n = len(cell_densities)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_densities)) / (n * np.sum(sorted_densities)) - (n + 1) / n
    
    summary = f"""Cell Density Summary (k={k}):
- Total points: {total_points}
- Non-empty clusters: {non_empty_clusters}/{k} ({non_empty_clusters/k*100:.1f}%)
- Density range: {cell_densities.min()} to {cell_densities.max()} (range: {cell_densities.max() - cell_densities.min()})
- Mean density: {cell_densities.mean():.2f} ± {cell_densities.std():.2f}
- Entropy ratio: {entropy_ratio:.3f} (1.0 = perfectly uniform)
- Gini coefficient: {gini:.3f} (0.0 = perfectly equal)
- Assessment: {"Uniform distribution" if entropy_ratio > 0.9 else "Collapsed space" if gini > 0.7 else "Moderately clustered"}
"""
    
    return summary
