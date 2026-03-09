import torch
import torch.nn as nn
from . import metrics
from .hooks import register_hooks
from .storage import store

class LatentMonitor:
    def __init__(self, model: nn.Module, layers="auto", metrics=None):
        self.model = model
        self.layers = layers
        self.metrics = metrics or ["activation_norm"]
        self.activations = {}
        self.handles = []

    def __repr__(self):
        return f"LatentMonitor({self.model}, {self.layers}, {self.metrics}, {self.activations}, {self.handles})"

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
        for name, act in self.activations.items():
            results[name] = {}
            for metric_name in self.metrics:
                metric_fn = getattr(metrics, metric_name, None)
                if metric_fn:
                    results[name][metric_name] = metric_fn(act)
                else:
                    print(f"Warning: Metric '{metric_name}' not found in latentspy.metrics")
        return results

    def log(self):
        results = self.compute()
        store.update(results)
        return results

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        