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
    experiment_name: str = None,
    log_type: str = "db"
) -> LatentMonitor:
    """
    Start watching a model's latent activations.
    
    Args:
        model (nn.Module): The PyTorch model to monitor.
        layers (Union[str, List[str]]): Either "auto" to detect standard layers (MLP, Attention)
            or a list of specific layer names to track.
        metrics (Optional[List[str]]): Metrics to compute. Use `ls.Metric` constants.
        sample_interval (int): Only capture activations every N steps.
        distributed (bool): If True, synchronize and average metrics across all GPUs.
        val_interval (int): Compute validation-based PP every N steps. If None, disabled.
        experiment_name (str): Name for experiment tracking.
        log_type (str): Storage format ("db", "json", "csv", "none"). Defaults to "db".
            
    Returns:
        LatentMonitor: An active monitor instance.
    """
    monitor = LatentMonitor(
        model, 
        layers=layers, 
        metrics=metrics, 
        sample_interval=sample_interval, 
        distributed=distributed,
        val_interval=val_interval,
        experiment_name=experiment_name,
        log_type=log_type
    )
    monitor.attach()
    return monitor
