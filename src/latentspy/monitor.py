import torch
import torch.nn as nn
import multiprocessing
from . import metrics
from .hooks import register_hooks
from .storage import get_storage
from .metrics.projection import project_to_3d

class LatentMonitor:
    def __init__(
        self, 
        model: nn.Module, 
        layers="auto", 
        metrics=None, 
        sample_interval: int = 1, 
        distributed: bool = False, 
        val_interval: int = None, 
        experiment_name: str = None, 
        log_type: str = "db", 
        alert_interval: int = 50, 
        dashboard: bool = False, 
        dashboard_port: int = 8000,
        metric_kwargs=None,
        val_metric_kwargs=None,
        alert_warmup_steps: int = 0
    ):
        self.model = model
        self.layers = layers
        self.metrics = metrics or ["activation_norm"]
        self.sample_interval = sample_interval
        self.val_interval = val_interval
        self.distributed = distributed
        self.experiment_name = experiment_name
        self.log_type = log_type
        self.alert_interval = alert_interval
        self.dashboard = dashboard
        self.dashboard_port = dashboard_port
        self.metric_kwargs = metric_kwargs or {}
        self.val_metric_kwargs = val_metric_kwargs or {}
        self.alert_warmup_steps = alert_warmup_steps
        self.server_process = None
        
        if self.dashboard:
            self._start_dashboard()
        
        self.global_step = 0
        self.activations = {"__enabled__": False}
        self.val_activations = {"__enabled__": False}
        self.in_val_mode = False
        self.handles = []
        self._last_results = {}
        self._last_val_results = {}
        
        self._health_states = {}
        self._warned_metrics = set()
        
        self.CLR_RED = "\033[91m"
        self.CLR_YEL = "\033[93m"
        self.CLR_END = "\033[0m"
        self.CLR_BOLD = "\033[1m"
        
        self.storage = get_storage(experiment_name, log_type=log_type)

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

    def run_validation_pp(self, val_batches):
        """
        Run validation-based PP computation on a provided validation batch or list of batches.
        """
        if not self.should_run_validation():
            return {}

        self.start_val()
        
        with torch.no_grad():
            if isinstance(val_batches, dict):
                val_batches = [val_batches]
                
            for val_batch in val_batches:
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
        if self.global_step < self.alert_warmup_steps:
            return
            
        warnings = []
        
        if "effective_rank" in metrics_dict:
            rank = metrics_dict["effective_rank"]
            if rank < 1.1:
                warnings.append((f"RANK COLLAPSE DETECTED: {layer_name} rank is {rank:.2f}. Represents total capacity loss.", "CRITICAL"))
            elif rank < 2.0:
                warnings.append((f"Low rank detected in {layer_name} (rank={rank:.2f}).", "WARNING"))

        if "activation_norm" in metrics_dict:
            norm = metrics_dict["activation_norm"]
            if norm > 1e6:
                warnings.append((f"ACTIVATION EXPLOSION: {layer_name} norm is {norm:.2e}. Training is likely diverging.", "CRITICAL"))
            elif norm > 1e3:
                warnings.append((f"High activation norm in {layer_name} (norm={norm:.2e}).", "WARNING"))

        if "patchiness" in metrics_dict:
            pp = metrics_dict["patchiness"]
            # 0.0 = Perfectly Uniform (Healthy)
            # >> 1.0 = Highly Clustered (Anomaly)
            if pp > 20.0:
                warnings.append((f"REPRESENTATION COLLAPSE: {layer_name} patchiness is {pp:.2f}. Features are dying or highly redundant.", "CRITICAL"))
            elif pp > 10.0:
                warnings.append((f"High representation clustering in {layer_name} (pp={pp:.2f}).", "WARNING"))

        if "eigenvalue_early_enrichment" in metrics_dict:
            eee = metrics_dict["eigenvalue_early_enrichment"]
            if eee > 0.45:
                warnings.append((f"SPECTRAL COLLAPSE: {layer_name} EEE is {eee:.3f}. Variance is concentrated in very few dimensions.", "CRITICAL"))
            elif eee > 0.35:
                warnings.append((f"High spectral enrichment in {layer_name} (EEE={eee:.3f}).", "WARNING"))

        if "sparsity" in metrics_dict:
            sp = metrics_dict["sparsity"]
            if sp > 0.95:
                warnings.append((f"REPRESENTATION DEATH: {layer_name} sparsity is {sp:.2f}. >95% of units are inactive.", "CRITICAL"))
            elif sp > 0.8:
                warnings.append((f"High sparsity in {layer_name} (Sparsity={sp:.2f}). Possible over-pruning.", "WARNING"))

        if "kurtosis" in metrics_dict:
            kurt = metrics_dict["kurtosis"]
            if kurt > 1000:
                warnings.append((f"EXTREME OUTLIERS: {layer_name} kurtosis is {kurt:.1f}. Numerical instability likely.", "CRITICAL"))
            elif kurt > 100:
                warnings.append((f"High kurtosis in {layer_name} (Kurtosis={kurt:.1f}). Strong outlier features detected.", "WARNING"))

        if "reconstruction_error" in metrics_dict:
            re = metrics_dict["reconstruction_error"]
            if re > 0.4:
                warnings.append((f"POOR RECONSTRUCTION: {layer_name} RE is {re:.3f}. Latent structure is highly fragmented.", "CRITICAL"))
            elif re > 0.2:
                warnings.append((f"High reconstruction error in {layer_name} (RE={re:.3f}). Clusters are poorly defined.", "WARNING"))

        if "reconstruction_skew" in metrics_dict:
            rs = metrics_dict["reconstruction_skew"]
            if rs > 5.0:
                warnings.append((f"EXTREME ERROR SKEW: {layer_name} RS is {rs:.1f}. Some features are significantly under-represented.", "CRITICAL"))
            elif rs > 2.0:
                warnings.append((f"High reconstruction skew in {layer_name} (RS={rs:.1f}). Non-uniform cluster quality.", "WARNING"))

        for msg, level in warnings:
            self.storage.log_alert(self.global_step, layer_name, level, msg)
            
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
                    kwargs = self.metric_kwargs.get(metric_name, {})
                    val = metric_fn(act, **kwargs)

                    if self.distributed and torch.distributed.is_initialized():
                        tensor_val = torch.tensor(val, device=act.device)
                        torch.distributed.all_reduce(tensor_val, op=torch.distributed.ReduceOp.SUM)
                        val = (tensor_val / torch.distributed.get_world_size()).item()
                    
                    if isinstance(val, dict):
                        results[name].update(val)
                    else:
                        results[name][metric_name] = val
                else:
                    if metric_name not in self._warned_metrics:
                        print(f"Warning: Metric '{metric_name}' not found in latentspy.metrics")
                        self._warned_metrics.add(metric_name)
            
            self._check_health(name, results[name], act)
            
        return results

    def log(self):
        if self.activations.get("__enabled__", False):
            results = self.compute()
            if results:
                self.storage.update(results, step=self.global_step, is_validation=False)
                
                for name, act in self.activations.items():
                    if name == "__enabled__": continue
                    if "patchiness" in self.metrics:
                        try:
                            projected = project_to_3d(act, max_points=500)
                            self.storage.log_projections(self.global_step, name, projected)
                        except Exception as e:
                            print(f"Error computing 3D projection for {name}: {e}")

                self._last_results = results
                self.clear() 
        
        combined_results = self._last_results.copy()
        for layer_name, metrics in self._last_val_results.items():
            combined_results[f"val_{layer_name}"] = metrics
                
        return combined_results

    def clear(self):
        enabled = self.activations.get("__enabled__", True)
        self.activations.clear()
        self.activations["__enabled__"] = enabled

    def log_scalar(self, name: str, value: float):
        """
        Log a scalar metric (e.g. loss, learning rate) directly to storage.
        
        Args:
            name (str): Name of the metric.
            value (float): Value to log.
        """
        results = {"__scalars__": {name: value}}
        self.storage.update(results, step=self.global_step, is_validation=False)


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
        for name, act_list in self.val_activations.items():
            if name == "__enabled__":
                continue
            act = torch.cat(act_list, dim=0)
            results[name] = {}
            for metric_name in self.metrics:
                metric_fn = getattr(metrics, metric_name, None)
                if metric_fn:
                    kwargs = self.val_metric_kwargs.get(metric_name, self.metric_kwargs.get(metric_name, {}))
                    val = metric_fn(act, **kwargs)
                    if isinstance(val, dict):
                        results[name].update(val)
                    else:
                        results[name][metric_name] = val
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

    def _start_dashboard(self):
        """Start the dashboard server in a background process."""
        from .server import start_server
        self.server_process = multiprocessing.Process(
            target=start_server,
            kwargs={"host": "0.0.0.0", "port": self.dashboard_port},
            daemon=True
        )
        self.server_process.start()
        print(f"LatentSpy Dashboard started at http://localhost:{self.dashboard_port}")

    def remove(self):
        if self.server_process and self.server_process.is_alive():
            self.server_process.terminate()
            self.server_process.join(timeout=1)
            if self.server_process.is_alive():
                self.server_process.kill()
        
        for h in self.handles:
            h.remove()
        