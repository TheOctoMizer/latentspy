import torch
from src.latentspy.metrics import (
    activation_norm, 
    effective_rank, 
    cosine_similarity, 
    patchiness
)

def test_patchiness_low():
    # Points spread out in different directions
    # Hand-picked to ensure they fall into different clusters
    data = torch.zeros(10, 10)
    for i in range(5):
        data[i, i] = 10.0
        data[i+5, i] = -10.0
    
    score = patchiness(data, k=2)
    # With K=2, the two clusters should be roughly equal size (5 each)
    # std should be low
    assert score < 1.0

def test_patchiness_high():
    # 9 points identical, 1 point far away
    # This is 'patchy' because one cluster will have 9 members, other has 1
    data = torch.zeros(10, 10)
    data[9, 0] = 100.0 # The 'far' point
    
    score = patchiness(data, k=2)
    assert score > 1.0

def test_cosine_similarity_perfect():
    # All vectors are identical -> Similarity 1.0
    data = torch.ones(4, 10) 
    sim = cosine_similarity(data)
    assert abs(sim - 1.0) < 1e-5

def test_cosine_similarity_orthogonal():
    # Vectors are orthogonal (basis vectors) -> Similarity 0.0
    data = torch.zeros(2, 2)
    data[0, 0] = 1.0
    data[1, 1] = 1.0
    sim = cosine_similarity(data)
    assert abs(sim - 0.0) < 1e-5

def test_activation_norm():
    # Setup test data
    # Matrix of 1s, shape [2, 10]
    # L2 norm of each row (length 10) is sqrt(10)
    data = torch.ones(2, 10)
    norm = activation_norm(data)
    expected = torch.sqrt(torch.tensor(10.0)).item()
    assert abs(norm - expected) < 1e-5

def test_effective_rank_max():
    # An identity matrix should have full rank. 
    # For a square matrix of size N, the effective rank should be N.
    N = 10
    data = torch.eye(N)
    # Note:svd of Identity is all 1s.
    # p = [1/N, 1/N, ..., 1/N]
    # Entropy = - sum(1/N * log(1/N)) = log(N)
    # e^Entropy = N
    rank = effective_rank(data)
    assert abs(rank - N) < 1e-4

def test_effective_rank_min():
    # A matrix of all 1s has rank 1.
    data = torch.ones(10, 10)
    # svd will have one large value and rest zeros
    rank = effective_rank(data)
    assert abs(rank - 1.0) < 1e-4
