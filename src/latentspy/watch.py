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
    log_type: str = "db",
    alert_interval: int = 50,
    dashboard: bool = False,
    dashboard_port: int = 8000
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
        alert_interval (int): Minimum steps between identical console warnings. Defaults to 50.
        dashboard (bool): If True, start the real-time web dashboard in the background.
        dashboard_port (int): Port to run the dashboard on. Defaults to 8000.
            
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
        log_type=log_type,
        alert_interval=alert_interval,
        dashboard=dashboard,
        dashboard_port=dashboard_port
    )
    monitor.attach()
    return monitor
