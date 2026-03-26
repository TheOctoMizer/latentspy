import torch

def eigenvalue_early_enrichment(activations: torch.Tensor) -> float:
    """
    Measures how quickly the cumulative variance is explained by top principal components.

    Args:
        activations (torch.Tensor): Tensor of shape (n_samples, n_features)

    Returns:
        float: The EEE score, where higher values indicate earlier enrichment.
    """

    X = activations.flatten(0, -2).float()
    X_centered = X - X.mean(dim=0)

    try:
        _, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
    except RuntimeError:
        # Fallback for MPS
        U, S, Vh = torch.linalg.svd(X_centered.cpu(), full_matrices=False)
        S = S.to(X.device)

    eigenvalues = S ** 2

    explained_variance = eigenvalues / (eigenvalues.sum() + 1e-10)
    cumulative_variance = torch.cumsum(explained_variance, dim=0)

    d = len(cumulative_variance)
    reference = torch.linspace(1/d, 1.0, steps=d, device=X.device)

    diff = cumulative_variance - reference
    auc_diff = diff.sum().item() / d

    return float(auc_diff)