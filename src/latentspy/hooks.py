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
            tensor = actual_output.detach()
            
            # Val mode: accumulate into val buffer (cat along batch dim)
            if val_dict is not None and val_dict.get("__enabled__", False):
                if name in val_dict:
                    val_dict[name] = torch.cat([val_dict[name], tensor], dim=0)
                else:
                    val_dict[name] = tensor
            # Training mode: standard single-pass capture
            elif activations_dict.get("__enabled__", True):
                activations_dict[name] = tensor

        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)
            
    return handles