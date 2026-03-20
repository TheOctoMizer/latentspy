import torch
import numpy as np
from sklearn.decomposition import PCA
from .activation_utils import prepare_activations_for_faiss

def project_to_3d(activations: torch.Tensor, max_points: int = 1000) -> np.ndarray:
    """
    Project high-dimensional activations to 3D space using PCA.
    
    Args:
        activations (torch.Tensor): Activations tensor.
        max_points (int): Maximum number of points to project (to keep it light).
        
    Returns:
        np.ndarray: Projected points of shape (N, 3).
    """
    # 1. Prepare activations (flatten to 2D)
    act_np, _ = prepare_activations_for_faiss(activations)
    
    # 2. Sample points if there are too many
    if act_np.shape[0] > max_points:
        indices = np.random.choice(act_np.shape[0], max_points, replace=False)
        act_np = act_np[indices]
    
    # 3. Handle very small dimensions
    if act_np.shape[1] < 3:
        # Pad with zeros if less than 3 dims
        padded = np.zeros((act_np.shape[0], 3), dtype=np.float32)
        padded[:, :act_np.shape[1]] = act_np
        return padded
    
    # 4. Perform PCA
    pca = PCA(n_components=3)
    projected = pca.fit_transform(act_np)
    
    return projected.astype(np.float32)
