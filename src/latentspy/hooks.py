import torch

def register_hooks(model, layer_names, activations_dict, val_dict=None):
    """
    Register forward hooks to capture activations for specified layers.
    
    Args:
        model (nn.Module): The model to register hooks to.
        layer_names (list): List of layer names to register hooks to.
        activations_dict (dict): Dictionary to store training activations.
        val_dict (dict or None): When provided and activated (via __enabled__), 
            hook outputs are concatenated into this dict (for validation-mode patchiness).
    
    Returns:
        list: List of hook handles.
    """
    handles = []
    
    def get_hook(name):
        def hook(module, input, output):
            actual_output = output[0] if isinstance(output, (tuple, list)) else output
            if val_dict is not None and val_dict.get("__enabled__", False):
                val_dict.setdefault(name, []).append(actual_output.detach().cpu())
            elif activations_dict.get("__enabled__", True):
                device_type = actual_output.device.type
                if device_type == "cuda":
                    # CUDA: keep on device; log() will do a non-blocking async copy
                    activations_dict[name] = actual_output.detach()
                else:
                    # MPS / CPU: move to CPU now to avoid deferred memory growth
                    activations_dict[name] = actual_output.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)
            
    return handles