import os
import pickle

# snippets from: https://github.com/dmbeaglehole/neural_controllers/blob/xrfm/neural_controllers.py
class NeuralController:
    def __init__(self, model, tokenizer, control_method='rfm', n_components=5, 
                 rfm_iters=10, batch_size=2):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.control_method = control_method

        print(f"n_components: {n_components}")

        hparams = {
            'control_method' : control_method,
            'rfm_iters' : rfm_iters,
            'forward_batch_size' : batch_size,
            'M_batch_size' : 2048,
            'n_components' : n_components
        }
        self.hyperparams = hparams
        
        self.hidden_layers = list(range(-1, -self.model.config.num_hidden_layers-1, -1))

        print('Hidden layers:', self.hidden_layers)
        print("\nController hyperparameters:")
        for n_, v_ in self.hyperparams.items():
            print(f"{n_:<20} : {v_}")
        print()

    def generate(self, prompt, layers_to_control=[], control_coef=0.4, **kwargs):
        if len(layers_to_control) == 0:
            control = False
        else:
            control = True     
            
        if control:               
            return self._controlled_generate(prompt, layers_to_control, control_coef, **kwargs)
        else:
            return generate_on_text(self.model, self.tokenizer, prompt, **kwargs)
        
    def _controlled_generate(self, prompt, layers_to_control, control_coef, **kwargs):
        ## define hooks
        hooks = hook_model(self.model, self.directions, layers_to_control, control_coef)

        ## do forward pass
        out = generate_on_text(self.model, self.tokenizer, prompt, **kwargs)

        ## clear hooks
        clear_hooks(hooks)
        return out

    
    def load(self, concept, model_name, path='./', composite=False):
        if composite:
            filename = os.path.join(path, f'{self.control_method}_composite_{concept}_{model_name}.pkl')
        else:
            filename = os.path.join(path, f'{self.control_method}_{concept}_{model_name}.pkl')
        with open(filename, 'rb') as f:
            self.directions = pickle.load(f)
            self.hidden_layers = self.directions.keys()
        
        detector_path = os.path.join(path, f'{self.control_method}_{concept}_{model_name}_detector.pkl')
        if os.path.exists(detector_path):
            print("Detector found")
            with open(detector_path, 'rb') as f:
                self.detector_coefs = pickle.load(f)

# snippets from: https://github.com/dmbeaglehole/neural_controllers/blob/xrfm/py
def generate_on_text(model, tokenizer, input_text, **kwargs):
        
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    # Generate output
    outputs = model.generate(
        **inputs,
        **kwargs,
    )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0])
    return generated_text
    
def hook_model(model, directions, layers_to_control, control_coef, component_idx=0):
    hooks = {}
    for layer_idx in layers_to_control:
        control_vec = directions[layer_idx][component_idx]
        if len(control_vec.shape)==1:
            control_vec = control_vec.reshape(1,1,-1)
               
               
        block = model.model.layers[layer_idx]

        def block_hook(module, input, output, control_vec=control_vec, control_coef=control_coef):
            """
            note that module, input are unused, but are
            required by torch.
            """ 
            
            new_output = output[0]

            new_output = new_output + control_coef*control_vec.to(dtype=new_output.dtype, device=new_output.device)
            
            if isinstance(output, tuple):
                new_output = (new_output,) + output[1:] 
            
            return new_output
        
        hook_handle = block.register_forward_hook(block_hook)
        hooks[layer_idx] = hook_handle
    
    return hooks

def clear_hooks(hooks) -> None:
    for hook_handle in hooks.values():
        hook_handle.remove()