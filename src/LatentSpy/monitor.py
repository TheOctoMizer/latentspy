import torch
import torch.nn as nn
from .metrics.activation_norm import activation_norm
from .hooks import register_hooks
from .storage import store

class LatentMonitor:
    def __init__(self, model: nn.Module, layers="auto", metrics=None):
        self.model = model
        self.layers = layers
        self.metrics = metrics or ["activation_norm"]
        self.activations = {}
        self.handles = []
    
    def attach(self):
        target_layers = [name for name, _ in self.model.named_modules() if self._should_track(name)]
        self.handles = register_hooks(self.model, target_layers, self.activations)

    def _should_track(self, name):
        if self.layers == "auto":
            return "attn" in name.lower() or "mlp" in name.lower()
        return name in self.layers

    def compute(self):
        results = {}
        for name, act in self.activations.items():
            results[name] = {}
            if "activation_norm" in self.metrics:
                results[name]["activation_norm"] = activation_norm(act)
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
        