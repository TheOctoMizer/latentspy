from .monitor import LatentMonitor
from typing import List, Optional, Union
import torch.nn as nn

def watch(
    model: nn.Module, 
    layers: Union[str, List[str]] = "auto", 
    metrics: Optional[List[str]] = None,
    sample_interval: int = 1,
    distributed: bool = False,
    val_interval: int = None,
    experiment_name: str = None
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
        val_interval (int): Compute validation-based PP every N steps. If None, disabled.
        experiment_name (str): Name for experiment tracking in storage. If None, auto-generated.
            
    Returns:
        LatentMonitor: An active monitor instance. Use monitor.log() to record metrics.
    """
    monitor = LatentMonitor(
        model, 
        layers=layers, 
        metrics=metrics, 
        sample_interval=sample_interval, 
        distributed=distributed,
        val_interval=val_interval,
        experiment_name=experiment_name
    )
    monitor.attach()
    return monitor
