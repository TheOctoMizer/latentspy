import torch

def cosine_similarity(activations: torch.Tensor) -> float:
    X = activations.flatten(1)
    batch_size = X.size(0)
    if batch_size <= 1:
        return 0.0
    X_norm = torch.nn.functional.normalize(X, p=2, dim=1, eps=1e-8)
    sim_matrix = torch.mm(X_norm, X_norm.t())
    total_sim = sim_matrix.sum() - batch_size
    num_pairs = batch_size * (batch_size - 1)
    return (total_sim / num_pairs).item()
