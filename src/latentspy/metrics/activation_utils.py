import torch
import numpy as np
from typing import Union, Tuple


def prepare_activations_for_faiss(activations: torch.Tensor) -> Tuple[np.ndarray, int]:
    """
    Prepare activations tensor for FAISS clustering.
    
    This function formats the activations for FAISS clustering
    by converting PyTorch tensors to NumPy arrays.
    
    Args:
        activations (torch.Tensor): Raw activations from hooks with shape 
                                   (batch_size, seq_len, hidden_dim) or 
                                   (batch_size, *, hidden_dim)
    
    Returns:
        Tuple[np.ndarray, int]: 
            - Prepared activations array of shape (total_tokens, hidden_dim)
            - Original hidden dimension for reference
    """
    if not isinstance(activations, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(activations)}")

    hidden_dim = activations.shape[-1]
    
    if activations.dim() > 2:
        flattened = activations.flatten(0, -2)
    else:
        flattened = activations
    
    detached = flattened.detach().cpu()
    
    activations_np = detached.numpy().astype('float32')
    
    return activations_np, hidden_dim

def validate_activations_format(activations_np: np.ndarray) -> bool:
    """
    Validate that activations are properly formatted for FAISS.
    
    Args:
        activations_np (np.ndarray): Activations array to validate
    
    Returns:
        bool: True if format is correct for FAISS
    
    Raises:
        ValueError: If format is incorrect
    """
    if not isinstance(activations_np, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(activations_np)}")
    
    if activations_np.dtype != np.float32:
        raise ValueError(f"Expected float32 dtype, got {activations_np.dtype}")
    
    if activations_np.ndim != 2:
        raise ValueError(f"Expected 2D array, got {activations_np.ndim}D")
    
    if activations_np.shape[0] == 0:
        raise ValueError("Empty activations array (no tokens)")
    
    if activations_np.shape[1] == 0:
        raise ValueError("Empty feature dimension")
    
    return True


def get_activation_stats(activations_np: np.ndarray) -> dict:
    """
    Get statistics about the prepared activations.
    
    Args:
        activations_np (np.ndarray): Prepared activations array
    
    Returns:
        dict: Statistics including shape, mean, std, min, max
    """
    return {
        'shape': activations_np.shape,
        'total_tokens': activations_np.shape[0],
        'hidden_dimension': activations_np.shape[1],
        'mean': float(activations_np.mean()),
        'std': float(activations_np.std()),
        'min': float(activations_np.min()),
        'max': float(activations_np.max()),
        'has_nan': bool(np.isnan(activations_np).any()),
        'has_inf': bool(np.isinf(activations_np).any()),
    }
