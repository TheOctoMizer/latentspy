import torch

def activation_norm(activations: torch.Tensor) -> float:
    norms = torch.norm(activations, p=2, dim=-1)
    norms_mean = norms.mean().item()
    return norms_mean