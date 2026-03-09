import torch
import pytest
from src.latentspy.metrics import activation_norm, effective_rank

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
