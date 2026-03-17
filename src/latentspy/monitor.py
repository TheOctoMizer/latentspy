import torch
import torch.nn as nn
from . import metrics
from .hooks import register_hooks
from .storage import store

class LatentMonitor:
    def __init__(self, model: nn.Module, layers="auto", metrics=None, sample_interval: int = 1, distributed: bool = False, val_interval: int = None, experiment_name: str = None, log_type: str = "db", alert_interval: int = 50):
        self.model = model
        self.layers = layers
        self.metrics = metrics or ["activation_norm"]
        self.sample_interval = sample_interval
        self.val_interval = val_interval
        self.distributed = distributed
        self.experiment_name = experiment_name
        self.log_type = log_type
        self.alert_interval = alert_interval
        self.global_step = 0
        self.activations = {"__enabled__": False}
        self.val_activations = {"__enabled__": False}
        self.in_val_mode = False
        self.handles = []
        self._last_results = {}
        self._last_val_results = {}
        
        # State for health warnings to avoid spam
        self._health_states = {} # layer -> {metric: state}
        self._warned_metrics = set()
        
        # ANSI Colors
        self.CLR_RED = "\033[91m"
        self.CLR_YEL = "\033[93m"
        self.CLR_END = "\033[0m"
        self.CLR_BOLD = "\033[1m"
        
        # Initialize enhanced storage
        from .storage import MetricStorage
        self.storage = MetricStorage(experiment_name, log_type=log_type) if experiment_name else store

    def __repr__(self):
        status = "ENABLED" if self.activations["__enabled__"] else "IDLE"
        return f"LatentMonitor(layers={self.layers}, interval={self.sample_interval}, status={status}, step={self.global_step}, log_type={self.log_type})"

    def step(self):
        """
        Increment the internal step counter and toggle recording state based on interval.
        Call this at the START of your training loop.
        """
        should_record = (self.global_step % self.sample_interval == 0)
        self.activations["__enabled__"] = should_record
        
        self.global_step += 1

    def should_run_validation(self):
        """Check if validation-based PP should be computed at this step."""
        if self.val_interval is None:
            return False
        return self.global_step % self.val_interval == 0

    def run_validation_pp(self, val_batch):
        """
        Run validation-based PP computation on a provided validation batch.
        """
        if not self.should_run_validation():
            return {}

        self.start_val()
        
        with torch.no_grad():
            if isinstance(val_batch, dict):
                input_ids = val_batch.get("input_ids")
                attention_mask = val_batch.get("attention_mask")
                if input_ids is not None:
                    self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                self.model(val_batch)
        
        val_results = self.log_val()
        self._last_val_results = val_results
        return val_results

    def enable(self):
        """Enable activation gathering."""
        self.activations["__enabled__"] = True

    def disable(self):
        """Disable activation gathering."""
        self.activations["__enabled__"] = False

    def attach(self):
        target_layers = [name for name, _ in self.model.named_modules() if self._should_track(name)]
        self.handles = register_hooks(
            self.model, target_layers, self.activations, val_dict=self.val_activations
        )

    def _should_track(self, name):
        if self.layers == "auto":
            return "attn" in name.lower() or "mlp" in name.lower()
        return name in self.layers

    def _check_health(self, layer_name, metrics_dict, act_tensor=None):
        """Check metrics against health thresholds and issue colored warnings."""
        warnings = []
        
        # 1. Effective Rank Collapse
        if "effective_rank" in metrics_dict:
            rank = metrics_dict["effective_rank"]
            if rank < 1.1:
                warnings.append((f"RANK COLLAPSE DETECTED: {layer_name} rank is {rank:.2f}. Represents total capacity loss.", "CRITICAL"))
            elif rank < 2.0:
                warnings.append((f"Low rank detected in {layer_name} (rank={rank:.2f}).", "WARNING"))

        # 2. Activation Explosion (Norm)
        if "activation_norm" in metrics_dict:
            norm = metrics_dict["activation_norm"]
            if norm > 1e6:
                warnings.append((f"ACTIVATION EXPLOSION: {layer_name} norm is {norm:.2e}. Training is likely diverging.", "CRITICAL"))
            elif norm > 1e3:
                warnings.append((f"High activation norm in {layer_name} (norm={norm:.2e}).", "WARNING"))

        # 3. Patchiness (Representation Collapse)
        if "patchiness" in metrics_dict:
            pp = metrics_dict["patchiness"]
            # New formula (Relative Variance): 
            # 0.0 = Perfectly Uniform (Healthy)
            # >> 1.0 = Highly Clustered (Anomaly)
            # For small batches, 1.0-5.0 might be normal variance.
            if pp > 20.0:
                warnings.append((f"REPRESENTATION COLLAPSE: {layer_name} patchiness is {pp:.2f}. Features are dying or highly redundant.", "CRITICAL"))
            elif pp > 10.0:
                warnings.append((f"High representation clustering in {layer_name} (pp={pp:.2f}).", "WARNING"))

        for msg, level in warnings:
            # 1. Persist alert to database
            self.storage.log_alert(self.global_step, layer_name, level, msg)
            
            # 2. Console warning (Rate limited)
            warn_key = f"{layer_name}_{msg}"
            if warn_key not in self._health_states or (self.global_step - self._health_states[warn_key] >= self.alert_interval):
                self._health_states[warn_key] = self.global_step
                color = self.CLR_RED if level == "CRITICAL" else self.CLR_YEL
                prefix = f"[{level}]"
                print(f"{color}{self.CLR_BOLD}{prefix} {msg}{self.CLR_END}")

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
                    if metric_name not in self._warned_metrics:
                        print(f"Warning: Metric '{metric_name}' not found in latentspy.metrics")
                        self._warned_metrics.add(metric_name)
            
            # Internal health check
            self._check_health(name, results[name], act)
            
        return results

    def log(self):
        if self.activations.get("__enabled__", False):
            results = self.compute()
            if results:
                self.storage.update(results, step=self.global_step, is_validation=False)
                self._last_results = results
                self.clear() 
        
        # Return combined results
        combined_results = self._last_results.copy()
        for layer_name, metrics in self._last_val_results.items():
            combined_results[f"val_{layer_name}"] = metrics
                
        return combined_results

    def clear(self):
        enabled = self.activations.get("__enabled__", True)
        self.activations.clear()
        self.activations["__enabled__"] = enabled


    def start_val(self):
        """Enter validation mode."""
        self.in_val_mode = True
        self.val_activations.clear()
        self.val_activations["__enabled__"] = True
        self.activations["__enabled__"] = False

    def end_val(self):
        """Exit validation mode."""
        self.in_val_mode = False
        self.val_activations["__enabled__"] = False

    def log_val(self):
        """Compute metrics over the full accumulated validation buffer."""
        results = {}
        for name, act in self.val_activations.items():
            if name == "__enabled__":
                continue
            results[name] = {}
            for metric_name in self.metrics:
                metric_fn = getattr(metrics, metric_name, None)
                if metric_fn:
                    results[name][metric_name] = metric_fn(act)
                else:
                    if metric_name not in self._warned_metrics:
                        print(f"Warning: Metric '{metric_name}' not found in latentspy.metrics")
                        self._warned_metrics.add(metric_name)

        if results:
            self.storage.update(results, step=self.global_step, is_validation=True)
            # Run health checks on val results too
            for name, metrics_dict in results.items():
                self._check_health(f"val_{name}", metrics_dict)

        # Clear the buffer
        self.val_activations.clear()
        self.val_activations["__enabled__"] = False
        return results

    def remove(self):
        for h in self.handles:
            h.remove()
        