from .activation_norm import activation_norm
from .effective_rank import effective_rank
from .cosine_similarity import cosine_similarity
from .patchiness import patchiness
from .eee import eigenvalue_early_enrichment
from .sparsity import sparsity
from .kurtosis import kurtosis
from .reconstruction import reconstruction_metrics

class Metric:
    """Namespace for available metrics to provide better IDE hinting."""
    ACTIVATION_NORM = "activation_norm"
    EFFECTIVE_RANK = "effective_rank"
    COSINE_SIMILARITY = "cosine_similarity"
    PATCHINESS = "patchiness"
    EEE = "eigenvalue_early_enrichment"
    SPARSITY = "sparsity"
    KURTOSIS = "kurtosis"
    RECONSTRUCTION = "reconstruction_metrics"
    
    @classmethod
    def all_metrics(cls):
        return [
            cls.ACTIVATION_NORM,
            cls.EFFECTIVE_RANK,
            cls.COSINE_SIMILARITY,
            cls.PATCHINESS,
            cls.EEE,
            cls.SPARSITY,
            cls.KURTOSIS,
            cls.RECONSTRUCTION
        ]
