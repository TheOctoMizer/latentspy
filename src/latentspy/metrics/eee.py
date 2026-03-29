import torch


def eigenvalue_early_enrichment(activations: torch.Tensor) -> float:
    """
    Eigenvalue Early Enrichment (EEE) measures how front-loaded the explained
    variance is across principal components.

    A high EEE means nearly all variance is concentrated in a few dimensions
    (anisotropic / "cone" geometry). A low EEE means variance is spread evenly.
    Per Marbut et al. (2024) the relationship with GLUE is non-monotonic, so
    EEE is logged as a trend metric only — no alert threshold is set.
    Returns:
        float: EEE score (area between cumulative variance curve and uniform reference).
               Positive = more front-loaded (anisotropic).
    """
    X = activations.flatten(0, -2).float().cpu()
    X_centered = X - X.mean(dim=0)

    _, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = S ** 2

    explained_variance = eigenvalues / (eigenvalues.sum() + 1e-10)
    cumulative_variance = torch.cumsum(explained_variance, dim=0)

    d = len(cumulative_variance)
    reference = torch.linspace(1 / d, 1.0, steps=d)  # always on CPU

    diff = cumulative_variance - reference
    auc_diff = diff.sum().item() / d

    return float(auc_diff)