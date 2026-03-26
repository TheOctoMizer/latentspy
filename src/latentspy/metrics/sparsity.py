import torch

def sparsity(activations: torch.Tensor, threshold: float = 1e-5) -> float:
    """
    Compute the sparsity ratio of activations.
    
    Args:
        activations (torch.Tensor): Output activations.
        threshold (float): Threshold below which a value is considered zero.
        
    Returns:
        float: The sparsity ratio (0.0 to 1.0).
    """
    total_elements = activations.numel()
    if total_elements == 0:
        return 0.0
    zero_elements = torch.sum(torch.abs(activations) < threshold).item()
    return float(zero_elements / total_elements)
