import torch
import numpy as np
from .activation_utils import prepare_activations_for_faiss
from .clustering import quantize_latent_space, get_cluster_statistics


def reconstruction_metrics(activations: torch.Tensor, k: int = 256) -> dict:
    """
    Computes Reconstruction Error (RE) and Reconstruction Skew (RS).

    Uses the shared FAISS k-means backend (quantize_latent_space) to ensure
    consistency with the patchiness metric — both are computed from the same
    clustering pass during validation. Activations are mean-centered before
    clustering so metrics reflect geometry, not layer-scale magnitude.

    RE  (Equation 15 in Marbut et al. 2024): normalised mean squared reconstruction
         error. Lower = more clustered / easier to compress.
    RS  (Equation 5): skewness of the normalised per-point error magnitude.
         Near 0 for uniform clusters; negative (left-skew) for highly clustered data
         where most points sit very close to a centroid.
    """
    try:
        activations_np, _ = prepare_activations_for_faiss(activations)
    except Exception:
        activations_np = activations.flatten(0, -2).detach().cpu().float().numpy().astype('float32')
    
    n_samples = activations_np.shape[0]
    k = min(k, n_samples // 2)
    if k < 1:
        return {"reconstruction_error": 0.0, "reconstruction_skew": 0.0}

    # Mean-center for geometry-only signal (matches patchiness pre-processing)
    activations_np = activations_np - activations_np.mean(axis=0, keepdims=True)

    try:
        cluster_labels, centroids, _ = quantize_latent_space(activations_np, k)
    except Exception as e:
        # Graceful degradation if FAISS fails (e.g. dimension mismatch)
        print(f"LatentSpy: reconstruction FAISS fallback: {e}")
        return {"reconstruction_error": 0.0, "reconstruction_skew": 0.0}

    # Reconstruct each point from its assigned centroid
    reconstructed = centroids[cluster_labels]          # (N, H)
    diff = activations_np - reconstructed              # (N, H)
    error_magnitudes = np.linalg.norm(diff, axis=1)   # (N,)

    # Reconstruction Error: normalised by the mean squared norm of original points
    avg_sq_norm = (np.linalg.norm(activations_np, axis=1) ** 2).mean() + 1e-10
    RE = float((error_magnitudes ** 2).mean() / avg_sq_norm)

    # Reconstruction Skew
    mean_err = error_magnitudes.mean()
    std_err = error_magnitudes.std()
    if std_err < 1e-8:
        RS = 0.0
    else:
        n = len(error_magnitudes)
        skew_num = (((error_magnitudes - mean_err) ** 3).sum() / n)
        skew_den = std_err ** 3
        # Fisher-Pearson standardised moment skewness
        RS = float((n * n / ((n - 1) * (n - 2) + 1e-10)) * skew_num / (skew_den + 1e-10))

    return {"reconstruction_error": RE, "reconstruction_skew": RS}