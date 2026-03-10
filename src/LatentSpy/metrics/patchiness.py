import torch

def patchiness(activations: torch.Tensor, k: int = 5) -> float:
    """
    Computes the patchiness of activations using K-Means clustering.
    
    A higher score indicates that activations are clumped into unequal, 
    isolated clusters (potentially indicating memorization).
    """
    X = activations.flatten(1)
    batch_size = X.size(0)
    
    if batch_size <= k:
        return 0.0

    indices = torch.randperm(batch_size)[:k]
    centroids = X[indices]
    
    for _ in range(5):
        dist = torch.cdist(X, centroids)
        assignments = torch.argmin(dist, dim=1)
        
        new_centroids = []
        for i in range(k):
            mask = (assignments == i)
            if mask.any():
                new_centroids.append(X[mask].mean(0))
            else:
                new_centroids.append(centroids[i])
        centroids = torch.stack(new_centroids)

    counts = torch.bincount(assignments, minlength=k).float()
    
    densities = counts / batch_size
    
    mean_density = densities.mean()
    std_density = densities.std()
    
    score = std_density / (mean_density + 1e-8)
    
    return score.item()
