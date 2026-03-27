import torch
import torch.nn as nn
import multiprocessing
import threading
import queue
import time
from typing import List, Union, Optional, Dict, Any
from . import metrics
from .hooks import register_hooks
from .storage import get_storage
from .metrics.projection import project_to_3d

# Metrics that run every sampled step — lightweight, O(N), best for early-warning detection
_FAST_METRICS = frozenset([
    "activation_norm",
    "sparsity",
    "kurtosis",
    "cosine_similarity",
])

# Metrics that run only on deep steps — involve SVD/KMeans, expensive but essential for structural diagnostics
_DEEP_METRICS = frozenset([
    "effective_rank",
    "eigenvalue_early_enrichment",
    "patchiness",
    "reconstruction",
])

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
        log_type: Union[str, List[str]] = "db", 
        alert_interval: int = 50, 
        dashboard: bool = False, 
        dashboard_port: int = 8000,
        metric_kwargs=None,
        val_metric_kwargs=None,
        alert_warmup_steps: int = 0,
        deep_metric_interval: int = 10
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
        self.deep_metric_interval = deep_metric_interval  # Every N *sampled* steps, run deep metrics
        self._sampled_step = 0  # Tracks how many samples have been taken
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
        
        # Performance: Two separate background threads
        # 1. Fast worker: handles scalars and training-step metrics (must stay real-time)
        self._queue = queue.Queue(maxsize=2000)
        # 2. Val worker: handles slow validation (k=256 KMeans etc.), completely isolated
        self._val_queue = queue.Queue(maxsize=50)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="LatentSpy-Fast")
        self._val_worker_thread = threading.Thread(target=self._val_worker, daemon=True, name="LatentSpy-Val")
        self._worker_thread.start()
        self._val_worker_thread.start()

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

    def _check_health(self, layer_name, metrics_dict, step=None):
        """Check metrics against health thresholds and issue colored warnings."""
        step = step or self.global_step
        if step < self.alert_warmup_steps:
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
            if eee > 0.60:
                warnings.append((f"SPECTRAL COLLAPSE: {layer_name} EEE is {eee:.3f}. Variance is concentrated in very few dimensions.", "CRITICAL"))
            elif eee > 0.45:
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
            if re > 0.6:
                warnings.append((f"POOR RECONSTRUCTION: {layer_name} RE is {re:.3f}. Latent structure is highly fragmented.", "CRITICAL"))
            elif re > 0.4:
                warnings.append((f"High reconstruction error in {layer_name} (RE={re:.3f}). Clusters are poorly defined.", "WARNING"))

        if "reconstruction_skew" in metrics_dict:
            rs = metrics_dict["reconstruction_skew"]
            if rs > 8.0:
                warnings.append((f"EXTREME ERROR SKEW: {layer_name} RS is {rs:.1f}. Some features are significantly under-represented.", "CRITICAL"))
            elif rs > 4.0:
                warnings.append((f"High reconstruction skew in {layer_name} (RS={rs:.1f}). Non-uniform cluster quality.", "WARNING"))

        for msg, level in warnings:
            self.storage.log_alert(step, layer_name, level, msg)
            
            warn_key = f"{layer_name}_{msg}"
            if warn_key not in self._health_states or (step - self._health_states[warn_key] >= self.alert_interval):
                self._health_states[warn_key] = step
                color = self.CLR_RED if level == "CRITICAL" else self.CLR_YEL
                prefix = f"[{level}]"
                print(f"{color}{self.CLR_BOLD}{prefix} {msg}{self.CLR_END}")

    def log(self):
        """
        Push a snapshot of current activations to the background worker for processing.
        This is now non-blocking to training.
        """
        if not self.activations.get("__enabled__", False):
            return {}

        # Capture snapshots immediately while activations are fresh
        # We must clone to CPU here to release GPU memory and avoid synchronization later
        snapshots = {}
        for name, act in self.activations.items():
            if name != "__enabled__" and isinstance(act, torch.Tensor):
                snapshots[name] = act.detach().cpu().clone()
        
        if not snapshots:
            return {}

        self._sampled_step += 1
        is_deep = (self._sampled_step % self.deep_metric_interval == 0)

        try:
            self._queue.put_nowait({
                "type": "log",
                "step": self.global_step,
                "snapshots": snapshots,
                "is_deep": is_deep
            })
        except queue.Full:
            # Skip if worker is too far behind
            pass

        self.clear()
        return {} # Results are now computed asynchronously

    def _worker(self):
        """Fast worker: handles scalars and training-step metrics. Must stay real-time."""
        uncommitted_tasks = 0
        MAX_UNCOMMITTED = 20
        
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=1.0)
                if task["type"] == "stop":
                    break

                # Use commit=False for all updates in the worker loop
                if task["type"] == "log":
                    self._process_log(task["step"], task["snapshots"], task.get("is_deep", True), commit=False)
                elif task["type"] == "scalar":
                    self._process_scalar(task["name"], task["value"], task["step"], commit=False)

                uncommitted_tasks += 1
                self._queue.task_done()

                if uncommitted_tasks >= MAX_UNCOMMITTED or self._queue.empty():
                    self.storage.flush()
                    uncommitted_tasks = 0

            except queue.Empty:
                if uncommitted_tasks > 0:
                    self.storage.flush()
                    uncommitted_tasks = 0
                continue
            except Exception as e:
                print(f"LatentSpy Worker Error: {e}")

    def _val_worker(self):
        """Dedicated validation worker — runs heavy KMeans/SVD without blocking the fast worker."""
        while not self._stop_event.is_set():
            try:
                task = self._val_queue.get(timeout=1.0)
                if task["type"] == "stop":
                    break
                if task["type"] == "val":
                    self._process_val(task["step"], task["snapshots"], commit=True)
                self._val_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"LatentSpy Val-Worker Error: {e}")


    def _process_scalar(self, name, value, step, commit=True):
        """Process a scalar metric in the background thread."""
        results = {"__scalars__": {name: value}}
        
        if name.lower() == "loss":
            import math
            try:
                results["__scalars__"]["perplexity"] = math.exp(value)
            except OverflowError:
                results["__scalars__"]["perplexity"] = float('inf')
                
        self.storage.update(results, step=step, is_validation=False, commit=commit)

    def _process_log(self, step, snapshots, is_deep: bool = True, commit: bool = True):
        """Internal method to compute metrics for a training step.
        
        Args:
            step: Training step count.
            snapshots: Dict of layer_name -> activation tensor (on CPU).
            is_deep: If True, compute ALL metrics (fast + deep). If False, only compute fast metrics.
        """
        results = {}
        for name, act in snapshots.items():
            layer_results = {}
            for metric in self.metrics:
                # Two-tier metric system: skip deep metrics on non-deep steps
                if not is_deep and metric in _DEEP_METRICS:
                    continue

                metric_fn = getattr(metrics, metric)
                if not metric_fn: continue
                
                kwargs = self.metric_kwargs.get(metric, {})
                try:
                    val = metric_fn(act, **kwargs)
                    if isinstance(val, dict):
                        layer_results.update(val)
                    else:
                        layer_results[metric] = val
                except Exception as e:
                    print(f"Error computing {metric} for {name}: {e}")
            
            if layer_results:
                results[name] = layer_results
                self._check_health(name, layer_results, step=step)

            # 3D Projection: only on deep steps (expensive PCA)
            if is_deep and "patchiness" in self.metrics:
                try:
                    proj = project_to_3d(act)
                    self.storage.log_projections(step, name, proj, cluster_ids=None, commit=commit)
                except Exception as e:
                     print(f"Error computing projection for {name}: {e}")

        self._last_results = results
        self.storage.update(results, step=step, is_validation=False, commit=commit)

    def clear(self):
        enabled = self.activations.get("__enabled__", True)
        self.activations.clear()
        self.activations["__enabled__"] = enabled

    def log_scalar(self, name: str, value: float):
        """
        Push a scalar metric to the background worker for asynchronous logging.
        Throttled to every 10 steps for non-critical scalars.
        """
        # Always log 'loss' but throttle other scalars like 'lr' to reduce task pressure
        if name.lower() != "loss" and self.global_step % 10 != 0:
            return

        try:
            self._queue.put_nowait({
                "type": "scalar",
                "name": name,
                "value": value,
                "step": self.global_step
            })
        except queue.Full:
            pass


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
        """
        Push validation snapshots (list of tensors) to the background worker.
        Does NOT concatenate or clone on the main thread to avoid GPU stalls.
        """
        if not self.val_activations.get("__enabled__", False):
            return {}

        # Just move the list pointers. Shallow copy is fast.
        snapshots = {}
        for name, act_list in self.val_activations.items():
            if name != "__enabled__" and isinstance(act_list, list) and len(act_list) > 0:
                snapshots[name] = act_list 

        if snapshots:
            try:
                # Route to the DEDICATED val worker to avoid blocking the fast metric worker
                self._val_queue.put_nowait({
                    "type": "val",
                    "step": self.global_step,
                    "snapshots": snapshots
                })
            except queue.Full:
                pass

        # Clear local buffer but keep enabled
        self.val_activations.clear()
        self.val_activations["__enabled__"] = True
        return {}

    def _process_val(self, step, snapshots, commit=True):
        """Internal method to compute metrics for a validation round asynchronously."""
        results = {}
        for name, act_list in snapshots.items():
            try:
                # 1. Concatenate on GPU (async) then move to CPU (sync point in worker only)
                with torch.no_grad():
                    # Move to CPU first if total tokens are massive to avoid OOM on tiny GPUs
                    # Or cat and then move. Cat on device is usually faster.
                    full_act = torch.cat(act_list, dim=0).detach()
                    
                    # 2. Sub-sample tokens if validation set is massive (e.g. > 10,000 tokens)
                    # This keeps KMeans and SVD fast in the background.
                    max_val_tokens = 10000 
                    n_tokens = full_act.size(0)
                    if n_tokens > max_val_tokens:
                        indices = torch.randperm(n_tokens)[:max_val_tokens]
                        full_act = full_act[indices]
                    
                    act = full_act.cpu().clone()
                    del full_act
                    
                layer_results = {}
                for metric in self.metrics:
                    metric_fn = getattr(metrics, metric)
                    if not metric_fn: continue
                    
                    kwargs = self.val_metric_kwargs.get(metric, self.metric_kwargs.get(metric, {}))
                    try:
                        val = metric_fn(act, **kwargs)
                        if isinstance(val, dict):
                            layer_results.update(val)
                        else:
                            layer_results[metric] = val
                    except Exception as e:
                        print(f"Error computing {metric} (val) for {name}: {e}")
                
                if layer_results:
                    results[name] = layer_results
                    self._check_health(name, layer_results, step=step)
            except Exception as e:
                print(f"Error processing validation snapshot for {name}: {e}")
        
        self._last_val_results = results
        self.storage.update(results, step=step, is_validation=True, commit=commit)

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
        """Stop dashboard server and background threads, and remove hooks."""
        self._stop_event.set()
        # Signal both workers to stop
        for q in [self._queue, self._val_queue]:
            if hasattr(self, '_queue') or hasattr(self, '_val_queue'):
                try:
                    q.put({"type": "stop"}, block=False)
                except:
                    pass
        
        if hasattr(self, 'storage'):
            self.storage.close()

        if self.server_process and self.server_process.is_alive():
            self.server_process.terminate()
            self.server_process.join(timeout=1)
            if self.server_process.is_alive():
                self.server_process.kill()
        
        for h in self.handles:
            h.remove()