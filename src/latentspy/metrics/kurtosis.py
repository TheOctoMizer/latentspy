import torch

def kurtosis(activations: torch.Tensor) -> float:
    """
    Compute the excess kurtosis of activations.
    
    Excess Kurtosis = (E[(X-mu)^4] / sigma^4) - 3.
    0 for normal distribution, > 0 for heavy-tailed (leptokurtic).
    
    Args:
        activations (torch.Tensor): Output activations.
        
    Returns:
        float: The excess kurtosis.
    """
    X = activations.float()
    mu = X.mean()
    sigma = X.std()
    
    if sigma < 1e-8:
        return 0.0
        
    m4 = torch.mean((X - mu)**4)
    kurt = (m4 / (sigma**4)) - 3
    return float(kurt.item())
