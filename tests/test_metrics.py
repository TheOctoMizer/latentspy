import torch
from src.latentspy.metrics import (
    activation_norm, 
    effective_rank, 
    cosine_similarity, 
    patchiness,
    eigenvalue_early_enrichment,
    sparsity,
    kurtosis,
    reconstruction_metrics
)

def test_patchiness_uniform():
    # 512 random high-dim points -> roughly uniform bins -> low PP
    torch.manual_seed(0)
    data = torch.randn(512, 64)
    score = patchiness(data)  # default k=256
    assert score < 1.0, f"Expected uniform data to have low patchiness, got {score:.4f}"

def test_patchiness_clustered():
    # 512 identical points -> FAISS puts them all in 1 cluster -> high PP
    data = torch.zeros(512, 64)
    score = patchiness(data)
    assert score > 0.1, f"Expected clustered data to have high patchiness, got {score:.4f}"

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

def test_eee_uniform():
    # Uniform variance distribution -> Low EEE
    torch.manual_seed(0)
    data = torch.randn(1000, 100)
    score = eigenvalue_early_enrichment(data)
    # Theoretically close to 0
    assert score < 0.1, f"Expected low EEE for uniform data, got {score:.4f}"

def test_eee_collapsed():
    # All variance in first component -> High EEE
    data = torch.zeros(100, 10)
    data[:, 0] = torch.randn(100) # Only 1st dimension has variance
    score = eigenvalue_early_enrichment(data)
    # Theoretically close to 0.45 for d=10
    # (10-1)/(2*10) = 9/20 = 0.45
    assert score > 0.4, f"Expected high EEE for collapsed data, got {score:.4f}"

def test_sparsity():
    # Mostly zeros -> High sparsity
    data = torch.zeros(100, 10)
    data[0, 0] = 1.0 # Only 1 element non-zero
    score = sparsity(data)
    assert score == 0.999

def test_kurtosis_normal():
    # Normal distribution -> Kurtosis 0
    torch.manual_seed(0)
    data = torch.randn(10000)
    score = kurtosis(data)
    assert abs(score) < 0.2

def test_kurtosis_outliers():
    # Large outliers -> High kurtosis
    torch.manual_seed(0)
    data = torch.randn(1000)
    data[0] = 100.0 # One huge outlier
    score = kurtosis(data)
    assert score > 100

def test_reconstruction():
    # Create simple clusters
    torch.manual_seed(0)
    c1 = torch.randn(50, 64) + 10
    c2 = torch.randn(50, 64) - 10
    data = torch.cat([c1, c2], dim=0) # 100 x 64
    
    res = reconstruction_metrics(data, k=2)
    assert "reconstruction_error" in res
    assert "reconstruction_skew" in res
    assert res["reconstruction_error"] < 0.1 # Should be low for clear clusters
    assert isinstance(res["reconstruction_error"], float)
