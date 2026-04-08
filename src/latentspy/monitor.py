import torch
import torch.nn as nn
import multiprocessing
import threading
import queue
import time
import gc
from typing import List, Union, Optional, Dict, Any
from . import metrics
from .hooks import register_hooks
from .storage import get_storage
from .metrics.projection import project_to_3d

_FAST_METRICS = frozenset([
    "activation_norm",
    "sparsity",
    "kurtosis",
    "cosine_similarity",
])

_DEEP_METRICS = frozenset()

_VAL_ONLY_METRICS = frozenset([
    "patchiness",
    "reconstruction_metrics",
    "eigenvalue_early_enrichment",
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
        
        try:
            device_type = next(self.model.parameters()).device.type
            if device_type == "xla":
                raise RuntimeError("LatentSpy does not support TPU/XLA devices. The hooking mechanism heavily disrupts XLA graph compilation. Please use CUDA, MPS, or CPU.")
        except StopIteration:
            pass
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
        # 2. Val worker: handles slow validation (k=256 KMeans etc.), completely isolated.
        # IMPORTANT: Keep this small! Each item holds concatenated activation tensors.
        # A large queue here causes massive memory blowup (e.g. 50 items × 200MB = 10GB).
        self._val_queue = queue.Queue(maxsize=3)
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

    def _check_health(self, layer_name, metrics_dict, step=None, is_validation=False):
        """Check metrics against health thresholds and issue colored warnings.
        
        Args:
            is_validation: Whether this is being called from validation context.
                           Some metrics (EEE) only produce meaningful alerts on
                           validation data due to non-monotonic training dynamics
                           (Marbut et al. 2024).
        """
        step = step or self.global_step
        if step < self.alert_warmup_steps:
            return
            
        warnings = []

        if "activation_norm" in metrics_dict:
            norm = metrics_dict["activation_norm"]
            if norm > 1e6:
                warnings.append((f"ACTIVATION EXPLOSION: {layer_name} norm={norm:.2e}. Training is likely diverging.", "CRITICAL", "norm_explosion"))
            elif norm > 1e3:
                warnings.append((f"High activation norm in {layer_name} (norm={norm:.2e}).", "WARNING", "norm_high"))

        if "sparsity" in metrics_dict:
            sp = metrics_dict["sparsity"]
            if sp > 0.95:
                warnings.append((f"REPRESENTATION DEATH: {layer_name} sparsity={sp:.2f}. >95% of units are inactive.", "CRITICAL", "sparsity_death"))
            elif sp > 0.8:
                warnings.append((f"High sparsity in {layer_name} (Sparsity={sp:.2f}). Possible over-pruning.", "WARNING", "sparsity_high"))

        if "kurtosis" in metrics_dict:
            kurt = metrics_dict["kurtosis"]
            if kurt > 1000:
                warnings.append((f"EXTREME OUTLIERS: {layer_name} kurtosis={kurt:.1f}. Numerical instability likely.", "CRITICAL", "kurtosis_extreme"))
            elif kurt > 100:
                warnings.append((f"High kurtosis in {layer_name} (Kurtosis={kurt:.1f}). Strong outlier features detected.", "WARNING", "kurtosis_high"))

        # Patchiness: only meaningful with sufficient token coverage (validation)
        # Lloyd's PP scale: ~1.0 for uniform, higher for clustered/collapsed.
        # From Marbut et al. (2024) Fig 3: healthy BERT-small lands in 1.02–1.06;
        # severely degraded models approach 1.00 (too uniform) or much higher (collapsed).
        # We alert on very HIGH PP (representation collapse) only; for the low end,
        # track the EEE trend on the dashboard instead of hard-thresholding.
        if "patchiness" in metrics_dict and is_validation:
            pp = metrics_dict["patchiness"]
            if pp > 50.0:
                warnings.append((f"REPRESENTATION COLLAPSE: {layer_name} PP={pp:.2f}. Most tokens collapsed to one region.", "CRITICAL", "patchiness_collapse"))
            elif pp > 10.0:
                warnings.append((f"High patchiness in {layer_name} (PP={pp:.2f}). Latent space strongly non-uniform.", "WARNING", "patchiness_high"))

        # EEE: logged as a trend metric during validation, but NOT alerted.
        # Marbut et al. (2024) show EEE has a non-monotonic relationship with
        # downstream performance — absolute thresholds produce false alarms on
        # healthy models. Watch the trend on the dashboard instead.
        # (No alert block for EEE by design.)

        # Reconstruction: only meaningful with sufficient token coverage (validation)
        if "reconstruction_error" in metrics_dict and is_validation:
            re = metrics_dict["reconstruction_error"]
            if re > 0.6:
                warnings.append((f"POOR RECONSTRUCTION: {layer_name} RE={re:.3f}. Latent structure is highly fragmented.", "CRITICAL", "reconstruction_error_high"))
            elif re > 0.4:
                warnings.append((f"High reconstruction error in {layer_name} (RE={re:.3f}). Clusters poorly defined.", "WARNING", "reconstruction_error_warn"))

        if "reconstruction_skew" in metrics_dict and is_validation:
            rs = metrics_dict["reconstruction_skew"]
            if rs > 8.0:
                warnings.append((f"EXTREME ERROR SKEW: {layer_name} RS={rs:.1f}. Some features significantly under-represented.", "CRITICAL", "recon_skew_high"))
            elif rs > 4.0:
                warnings.append((f"High reconstruction skew in {layer_name} (RS={rs:.1f}). Non-uniform cluster quality.", "WARNING", "recon_skew_warn"))

        for msg, level, pathology_key in warnings:
            self.storage.log_alert(step, layer_name, level, msg)
            
            # Dedup key: (layer, level, pathology_type) — NOT the message text,
            # which embeds the metric value and would create a new key every step.
            warn_key = f"{layer_name}_{level}_{pathology_key}"
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

        # Capture snapshots — activations are already on CPU (moved by the hook)
        # so .detach().cpu() is a no-op move + we just .clone() to own the data.
        snapshots = {}
        for name, act in self.activations.items():
            if name != "__enabled__" and isinstance(act, torch.Tensor):
                snapshots[name] = act.clone()
        
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
        """Compute training-step metrics (TIER 1 fast + TIER 2 deep only).
        
        TIER 3 (VAL_ONLY) metrics — patchiness, reconstruction, EEE — are always
        skipped here regardless of is_deep. They require validation-scale token
        coverage (10k+) to be statistically meaningful (Marbut et al. 2024).
        
        Args:
            step: Training step count.
            snapshots: Dict of layer_name -> activation tensor (on CPU).
            is_deep: If True, also compute TIER 2 metrics. If False, TIER 1 only.
        """
        results = {}
        for name, act in snapshots.items():
            layer_results = {}
            for metric in self.metrics:
                # Never run val-only metrics on training steps
                if metric in _VAL_ONLY_METRICS:
                    continue
                # Skip deep metrics on non-deep steps
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
                self._check_health(name, layer_results, step=step, is_validation=False)

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
        Push validation snapshots to the background worker.
        Eagerly concatenates and subsamples tensors HERE on the main thread so
        the raw accumulated list (potentially 50 batches per layer) is freed
        immediately, instead of living in the queue for minutes while the val
        worker is busy. This is the primary fix for the memory blowup.
        """
        if not self.val_activations.get("__enabled__", False):
            return {}

        MAX_VAL_TOKENS = 10000  # Cap tokens *before* queuing to bound memory per task

        snapshots = {}
        for name, act_list in self.val_activations.items():
            if name == "__enabled__" or not isinstance(act_list, list) or len(act_list) == 0:
                continue
            try:
                with torch.no_grad():
                    # Concatenate and immediately free the individual tensors
                    full_act = torch.cat(act_list, dim=0)  # already on CPU from hooks
                    # Subsample here so the queued payload is bounded
                    n_tokens = full_act.size(0)
                    if n_tokens > MAX_VAL_TOKENS:
                        indices = torch.randperm(n_tokens, device='cpu')[:MAX_VAL_TOKENS]
                        full_act = full_act[indices].clone()
                    else:
                        full_act = full_act.clone()
                snapshots[name] = full_act  # single compact tensor, not a list
            except Exception as e:
                print(f"LatentSpy: Error pre-processing val snapshot for {name}: {e}")
            finally:
                # Explicitly free the list elements to release memory NOW
                act_list.clear()

        if snapshots:
            try:
                # Route to the DEDICATED val worker to avoid blocking the fast metric worker
                self._val_queue.put_nowait({
                    "type": "val",
                    "step": self.global_step,
                    "snapshots": snapshots
                })
            except queue.Full:
                # Val worker is still busy — skip this validation round rather than OOM.
                # If this fires repeatedly, increase VAL_INTERVAL so the worker has more
                # time to finish before the next validation triggers.
                print(f"LatentSpy: Val queue full at step {self.global_step}, skipping validation round. "
                      f"Consider increasing val_interval (current: {self.val_interval}).")
                for t in snapshots.values():
                    del t
                gc.collect()

        # Clear local buffer and fully disable validation mode
        self.val_activations.clear()
        self.val_activations["__enabled__"] = False
        self.in_val_mode = False
        return {}

    def _process_val(self, step, snapshots, commit=True):
        """Internal method to compute metrics for a validation round asynchronously.
        
        snapshots is now a dict of layer_name -> single pre-concatenated CPU tensor
        (the concatenation and subsampling happens in log_val() before queuing).
        """
        results = {}
        for name, act in snapshots.items():
            try:
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
                    self._check_health(name, layer_results, step=step, is_validation=True)
            except Exception as e:
                print(f"Error processing validation snapshot for {name}: {e}")
            finally:
                # Explicitly free the tensor after processing each layer
                del act
        
        self._last_val_results = results
        self.storage.update(results, step=step, is_validation=True, commit=commit)
        # Force a GC cycle after processing all validation tensors.
        # Large numpy/FAISS allocations don't always release immediately on `del`;
        # this ensures memory is reclaimed before the next validation round arrives.
        gc.collect()
        # Flush the CUDA caching allocator's free-block pool so that memory
        # released by GC is actually returned to the OS / available to other
        # allocations, rather than sitting in PyTorch's internal cache.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        
        # Signal both workers to stop gracefully
        for q in [self._queue, self._val_queue]:
            try:
                q.put_nowait({"type": "stop"})
            except Exception:
                pass
        
        # Wait for workers to finish pending tasks before closing storage.
        # This prevents the "NoneType object has no attribute 'cursor'" error
        # where the worker tries to log a final result but the DB is already closed.
        if hasattr(self, '_worker_thread') and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if hasattr(self, '_val_worker_thread') and self._val_worker_thread.is_alive():
            # Val worker does heavier work, give it slightly more time
            self._val_worker_thread.join(timeout=5.0)

        if hasattr(self, 'storage'):
            self.storage.close()

        if self.server_process and self.server_process.is_alive():
            self.server_process.terminate()
            self.server_process.join(timeout=1)
            if self.server_process.is_alive():
                self.server_process.kill()
        
        for h in self.handles:
            h.remove()