class FeatureExtractor:
    """Overrides outputs.hidden_states with hidden_states from target_modules"""
    def __init__(self, model, target_modules=None):
        self.model = model
        self.target_modules = target_modules or []

    def __call__(self, *args, **kwargs):
        self._hidden_states = {}
        hooks = []

        # Register hooks
        for i, layer in enumerate(self.model.layers):
            for module_name in self.target_modules:
                if not hasattr(layer, module_name):
                    print(f"Warning: {module_name} not in layer, falling back to model.layers[{i}]")
                def func(module, input, output, name=module_name):
                    self._hidden_states.setdefault(name, []).append(output.detach().cpu())
                module = getattr(layer, module_name, layer)
                hooks.append(module.register_forward_hook(func))
        
        # Forward pass
        outputs = self.model(*args, **kwargs)
        outputs.hidden_states = self._hidden_states or [output.detach().cpu() for output in outputs.hidden_states]
        del self._hidden_states

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs
