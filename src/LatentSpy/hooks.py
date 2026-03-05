import torch

def register_hooks(model, layer_names, activations_dict):
    """Register forward hooks to capture activations for specified layers."""
    handles = []
    
    def get_hook(name):
        def hook(module, input, output):
            activations_dict[name] = output.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)
            
    return handles