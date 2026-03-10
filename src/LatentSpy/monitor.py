import torch
import torch.nn as nn
from . import metrics
from .hooks import register_hooks
from .storage import store

class LatentMonitor:
    def __init__(self, model: nn.Module, layers="auto", metrics=None, sample_interval: int = 1, distributed: bool = False):
        self.model = model
        self.layers = layers
        self.metrics = metrics or ["activation_norm"]
        self.sample_interval = sample_interval
        self.distributed = distributed
        self.global_step = 0
        self.activations = {"__enabled__": False}
        self.handles = []
        self._last_results = {}

    def __repr__(self):
        status = "ENABLED" if self.activations["__enabled__"] else "IDLE"
        return f"LatentMonitor(layers={self.layers}, interval={self.sample_interval}, status={status}, step={self.global_step})"

    def step(self):
        """
        Increment the internal step counter and toggle recording state based on interval.
        Call this at the START of your training loop.
        """
        should_record = (self.global_step % self.sample_interval == 0)
        self.activations["__enabled__"] = should_record
        
        self.global_step += 1

    def enable(self):
        """Enable activation gathering."""
        self.activations["__enabled__"] = True

    def disable(self):
        """Disable activation gathering."""
        self.activations["__enabled__"] = False

    def attach(self):
        target_layers = [name for name, _ in self.model.named_modules() if self._should_track(name)]
        self.handles = register_hooks(self.model, target_layers, self.activations)

    def _should_track(self, name):
        if self.layers == "auto":
            return "attn" in name.lower() or "mlp" in name.lower()
        return name in self.layers

    def all_available_layers(self):
        all_layers = [name for name, _ in self.model.named_modules()]
        return all_layers

    def compute(self):
        results = {}
        if len(self.activations) <= 1:
            return results

        for name, act in self.activations.items():
            if name == "__enabled__": continue
            
            results[name] = {}
            for metric_name in self.metrics:
                metric_fn = getattr(metrics, metric_name, None)
                if metric_fn:
                    val = metric_fn(act)

                    if self.distributed and torch.distributed.is_initialized():
                        tensor_val = torch.tensor(val, device=act.device)
                        torch.distributed.all_reduce(tensor_val, op=torch.distributed.ReduceOp.SUM)
                        val = (tensor_val / torch.distributed.get_world_size()).item()
                    
                    results[name][metric_name] = val
                else:
                    print(f"Warning: Metric '{metric_name}' not found in latentspy.metrics")
        return results

    def log(self):
        if self.activations.get("__enabled__", False):
            results = self.compute()
            if results:
                store.update(results, step=self.global_step)
                self._last_results = results
                self.clear() 
        
        return self._last_results

    def clear(self):
        enabled = self.activations.get("__enabled__", True)
        self.activations.clear()
        self.activations["__enabled__"] = enabled

    def remove(self):
        for h in self.handles:
            h.remove()
        