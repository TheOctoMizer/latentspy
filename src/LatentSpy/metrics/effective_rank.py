import torch

def effective_rank(activations: torch.Tensor) -> float:
    """
    Compute the effective rank of the activation matrix.
    
    Args:
        activations (torch.Tensor): Output activations of shape (Batch, ...).
        
    Returns:
        float: The effective rank (scalar).
    """
    X = activations.flatten(1)

    try:
        s = torch.linalg.svdvals(X)
    except RuntimeError:
        _, s, _ = torch.svd(X)
    s_sum = s.sum()
    if s_sum == 0:
        return 0.0
    p = s / s_sum
    eps = 1e-10
    entropy = -torch.sum(p * torch.log(p + eps))
    erank = torch.exp(entropy)
    
    return erank.item()
