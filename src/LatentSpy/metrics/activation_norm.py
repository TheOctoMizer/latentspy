import torch

def activation_norm(activations: torch.Tensor) -> float:
    """
    Compute the mean activation norm of the given activations.
    
    Args:
        activations (torch.Tensor): The activations to compute the norm of.
    
    Returns:
        float: The mean activation norm.
    """
    flat = activations.flatten(1)
    norms = torch.norm(flat, p=2, dim=-1)
    return norms.mean().item()