import torch

def reconstruction_metrics(activations: torch.Tensor, k: int = 256):
    """
    Computes Reconstruction Error (RE) and Reconstruction Skew (RS).
    Uses a fast Torch-based K-means implementation for stability across platforms.
    """
    X = activations.flatten(0, -2).detach().float()
    
    n_samples, hidden_dim = X.shape
    k = min(k, n_samples // 2)
    if k < 1:
        return {"reconstruction_error": 0.0, "reconstruction_skew": 0.0}

    indices = torch.randperm(n_samples, device=X.device)[:k]
    centroids = X[indices].clone()
    
    for _ in range(8):
        x_norm = (X**2).sum(dim=1, keepdim=True)
        c_norm = (centroids**2).sum(dim=1).unsqueeze(0)
        dist = x_norm + c_norm - 2 * torch.mm(X, centroids.t())
        bin_indices = torch.argmin(dist, dim=1)
        
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, 1, device=X.device)
        new_centroids.index_add_(0, bin_indices, X)
        counts.index_add_(0, bin_indices, torch.ones(n_samples, 1, device=X.device))
        centroids = new_centroids / (counts + 1e-6)

    X_reconstructed = centroids[bin_indices]
    
    error_magnitudes = torch.norm(X - X_reconstructed, dim=1)
    
    avg_sq_norm = (torch.norm(X, dim=1)**2).mean() + 1e-10
    RE = (error_magnitudes ** 2).mean() / avg_sq_norm
    
    mean_err = error_magnitudes.mean()
    std_err = error_magnitudes.std()
    
    if std_err < 1e-8:
        RS = 0.0
    else:
        n = len(error_magnitudes)
        skew_num = ((error_magnitudes - mean_err) ** 3).sum() / n
        skew_den = std_err ** 3
        RS = (skew_num / (skew_den + 1e-10))
        
    return {"reconstruction_error": float(RE.item()), "reconstruction_skew": float(RS.item())}