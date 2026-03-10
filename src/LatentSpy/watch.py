from .monitor import LatentMonitor
from typing import List, Optional, Union
import torch.nn as nn

def watch(
    model: nn.Module, 
    layers: Union[str, List[str]] = "auto", 
    metrics: Optional[List[str]] = None,
    sample_interval: int = 1,
    distributed: bool = False
) -> LatentMonitor:
    """
    Start watching a model's latent activations.
    
    This is the primary entry point for LatentSpy. It initializes a LatentMonitor,
    attaches the necessary forward hooks, and returns the monitor instance.
    
    Args:
        model (nn.Module): The PyTorch model to monitor.
        layers (Union[str, List[str]]): Either "auto" to detect standard layers (MLP, Attention)
            or a list of specific layer names to track.
        metrics (Optional[List[str]]): Metrics to compute. Use `ls.Metric` constants 
            (e.g., `[ls.Metric.ACTIVATION_NORM]`). Defaults to `[ls.Metric.ACTIVATION_NORM]`.
        sample_interval (int): Only capture activations every N steps.
        distributed (bool): If True, synchronize and average metrics across all GPUs.
            
    Returns:
        LatentMonitor: An active monitor instance. Use monitor.log() to record metrics.
    """
    monitor = LatentMonitor(
        model, 
        layers=layers, 
        metrics=metrics, 
        sample_interval=sample_interval, 
        distributed=distributed
    )
    monitor.attach()
    return monitor
