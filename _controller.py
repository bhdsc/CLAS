import copy

import torch
import torch.nn.functional as F

import _base_controller
from transformers import AutoTokenizer, AutoModelForCausalLM

import _config as config
from _utils import format_prompt

def from_pretrained(model_id, controller_class=None, control_method='rfm', cache_dir=None, device_map="auto", torch_dtype="auto", **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, device_map=device_map, torch_dtype=torch_dtype)
    
    # ValueError: Asking to pad but the tokenizer does not have a padding token.
    # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
    if tokenizer.pad_token is None:
        pad_token = tokenizer.eos_token
        pad_token = '<|finetune_right_pad_id|>'
        assert pad_token in tokenizer.get_vocab()
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = "left"
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    if controller_class is None:
        controller_class = NeuralController
        
    if kwargs:
        return controller_class(
            model,
            tokenizer,
            **kwargs
        )
    return controller_class(
        model,
        tokenizer,
        rfm_iters=8,
        batch_size=2,
        n_components=10,
        control_method=control_method
    )

def generate(controller, prompt, generated_text="", add_special_tokens=False, use_chat_template=True, **kwargs):
    prompt = format_prompt(controller.tokenizer, prompt, generated_text, add_special_tokens, use_chat_template)
    output = controller.generate(prompt, **kwargs)
    return output

def generate_time(controller, prompt, **kwargs):
    import time, sys
    t0 = time.perf_counter()
    output = generate(controller, prompt, **kwargs)
    print(prompt, '\n---\n', output, file=sys.stderr)
    t1 = time.perf_counter()
    return output, t1 - t0
    
def add(hidden_state, control_vec, control_coef, rescale_out, *args, **kwargs):
    return hidden_state + control_coef @ control_vec + hidden_state * rescale_out

def add_proj(hidden_state, control_vec, control_coef, rescale_out, control_bias, *args, **kwargs):
    control_coef = hidden_state @ control_vec.mT * -control_coef + control_bias
    return hidden_state + control_coef @ control_vec + hidden_state * rescale_out

def add_dynamic(hidden_state, control_vec, control_coef, rescale_out, control_bias, *args, **kwargs):
    control_coef = hidden_state @ control_coef.mT + control_bias
    return hidden_state + control_coef @ control_vec + hidden_state * rescale_out

steer_funcs = [add, add_proj, add_dynamic]
steer_funcs = {func.__name__: func for func in steer_funcs}

class NeuralController(_base_controller.NeuralController):
    def __init__(self, *args, **kwargs):
        _base_controller.NeuralController.__init__(self, *args, **kwargs)
        self.directions = None
        self.params = None

    def load(self, concept, model_name, path):
        super().load(concept, model_name, path)
        self.directions = {k: v.to(device=self.model.device) for k, v in self.directions.items()}
        self.hidden_layers = self.directions.keys()
        assert len(self.hidden_layers) == self.model.config.num_hidden_layers
        self.update_params()
        self.name_or_path = f"{self.control_method}_{concept}_{model_name}"
        return self.directions
    
    def update_params(self, params={}):
        self.params = copy.deepcopy(params)

    def generate(self, prompt, layers_to_control=[], control_coef=0, steer_func="add", block_attr="layer", output_only=False, **kwargs):
        if layers_to_control and not self.params:
            name = self.__class__.__name__
            print(f"{name}.params={self.params}, continuing generation without full params. Use {name}.update_params(...) to generate with full params.")
        if isinstance(steer_func, str):
            steer_func = steer_funcs[steer_func]
        self.steer_func = steer_func
        self.block_attr = block_attr
        self.tokenizer.padding_side = "left"
        self.model.apply(remove_hooks)
        output = self._controlled_generate(prompt, layers_to_control, control_coef, **kwargs)
        if output_only:
            input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt = self.tokenizer.decode(input_ids, clean_up_tokenization_spaces=True)
            return output[len(prompt):]
        return output
        
    def _controlled_generate(self, prompt, layers_to_control, control_coef, **kwargs):        
        parent_hook_model = _base_controller.hook_model
        _base_controller.hook_model = self.hook_model
        out = super()._controlled_generate(prompt, layers_to_control, control_coef, **kwargs)
        _base_controller.hook_model = parent_hook_model
        return out

    def block_hook(self, module, input, output, control_vec, control_coef, rescale_out, control_bias, layer_idx, rep_token=-1):
        new_output = output
        if isinstance(output, tuple):
            new_output = output[0]
        control_vec = control_vec.to(device=new_output.device)
        control_coef = control_coef.to(device=new_output.device)
        rescale_out = rescale_out.to(device=new_output.device)
        control_bias = control_bias.to(device=new_output.device)
        assert new_output.ndim == 3

        # Steer activations/hidden_states
        if layer_idx in self._layers_to_control:
            new_output = self.steer_func(new_output.float(), control_vec, control_coef, rescale_out, control_bias).to(new_output.dtype)
        
        if isinstance(output, tuple):
            new_output = (new_output,) + output[1:] 
            
        return new_output

    def hook_model(self, model, directions, layers_to_control, control_coefs, component_idx=0):
        if directions is None:
            return {}
        
        self._layers_to_control = layers_to_control
        
        if not isinstance(control_coefs, dict):
            control_coefs = {layer_idx: float(control_coefs) for layer_idx in self.hidden_layers}
            
        hooks = {}
        for layer_idx in self.hidden_layers:
            control_vec = directions[layer_idx][component_idx]
            if len(control_vec.shape)==1:
                control_vec = control_vec.reshape(1,1,-1)
            
            control_coef = torch.as_tensor(control_coefs[layer_idx])
            if len(control_coef.shape)==1:
                control_coef = control_coef.reshape(1,1,-1)
            elif control_coef.ndim < 1:
                control_coef = control_coef.reshape(1,1,-1)
            
            #############################################################################################################
            # NOTE: Call update_params(...) to set
            rescale_out = self.params["rescale_out"][layer_idx] if "rescale_out" in self.params else torch.tensor(0.0)
            control_bias = self.params["control_bias"][layer_idx] if "control_bias" in self.params else torch.tensor(0.0)
            #############################################################################################################
            
            def block_hook(
                module, input, output, 
                control_vec=control_vec, control_coef=control_coef, rescale_out=rescale_out, control_bias=control_bias,
                layer_idx=layer_idx, rep_token=-1
            ):
                return self.block_hook(module, input, output, 
                                       control_vec, control_coef, rescale_out, control_bias,
                                       layer_idx, rep_token)
            
            block = model.model.layers[layer_idx]
            block = getattr(block, self.block_attr, block)
            hook_handle = block.register_forward_hook(block_hook)
            hooks[layer_idx] = hook_handle
        
        return hooks

class Activation(NeuralController):
    def __init__(self, *args, **kwargs):
        NeuralController.__init__(self, *args, **kwargs)
        
    def _controlled_generate(self, prompt, layers_to_control, control_coef, **kwargs):
        self._activations = {}
        self.activations = {}
        self._coefficients = {}
        self.coefficients = {}
        self._projections = {}
        self.projections = {}
        
        out = super()._controlled_generate(prompt, layers_to_control, control_coef, **kwargs)
        
        self._activations = {k: self._activations[k] for k in self.hidden_layers}
        self.activations = {k: self.activations[k] for k in self.hidden_layers}
        self._coefficients = {k: self._coefficients[k] for k in self.hidden_layers}
        self.coefficients = {k: self.coefficients[k] for k in self.hidden_layers}
        self._projections = {k: self._projections[k] for k in self.hidden_layers}
        self.projections = {k: self.projections[k] for k in self.hidden_layers}
        
        return out

    def block_hook(self, module, input, output, control_vec, control_coef, rescale_out, control_bias, layer_idx, rep_token=-1):
        new_output = output
        if isinstance(output, tuple):
            new_output = output[0]
        rep_output = new_output[:, rep_token, :].float()
        control_vec = control_vec.to(device=new_output.device)
        control_coef = control_coef.to(device=new_output.device)
        rescale_out = rescale_out.to(device=new_output.device)
        control_bias = control_bias.to(device=new_output.device)
        assert new_output.ndim == 3

        n = self.model.config.hidden_size
        cpu_dtype = lambda x: x.detach().cpu().to(new_output.dtype)
        
        # Save activation value of rep_token before steering
        if layer_idx not in self._activations:
            self._activations[layer_idx] = []
        self._activations[layer_idx].append(cpu_dtype(rep_output))

        # Save coefficient value of rep_token activations onto control_coef before steering
        if layer_idx not in self._coefficients:
            self._coefficients[layer_idx] = []
        self._coefficients[layer_idx].append(cpu_dtype(rep_output @ control_coef.mT + control_bias if control_coef.numel() >= n else control_coef))
    
        # Save projection value of rep_token activations onto control_vec before steering
        if layer_idx not in self._projections:
            self._projections[layer_idx] = []
        self._projections[layer_idx].append(cpu_dtype(rep_output @ control_vec.mT))
        
        # Steer activations/hidden_states
        if layer_idx in self._layers_to_control:
            new_output = self.steer_func(new_output.float(), control_vec, control_coef, rescale_out, control_bias).to(new_output.dtype)

        rep_output = new_output[:, rep_token, :].float()
    
        # Save activation value of rep_token after steering
        if layer_idx not in self.activations:
            self.activations[layer_idx] = []
        self.activations[layer_idx].append(cpu_dtype(rep_output))

        # Save coefficient value of rep_token activations onto control_coef after steering
        if layer_idx not in self.coefficients:
            self.coefficients[layer_idx] = []
        self.coefficients[layer_idx].append(cpu_dtype(rep_output @ control_coef.mT + control_bias if control_coef.numel() >= n else control_coef))
    
        # Save projection value of rep_token activations onto control_vec after steering
        if layer_idx not in self.projections:
            self.projections[layer_idx] = []
        self.projections[layer_idx].append(cpu_dtype(rep_output @ control_vec.mT))
        
        if isinstance(output, tuple):
            new_output = (new_output,) + output[1:] 
            
        return new_output

def remove_hooks(module, hook_name="block_hook"):
    def clear(hooks):
        f = lambda v: hook_name == getattr(v, "__name__", str(v))
        keys = [k for k, v in hooks.items() if f(v)]
        for k in keys:
            hooks.pop(k)
        return hooks
    
    module._forward_hooks = clear(module._forward_hooks)
    module._forward_pre_hooks = clear(module._forward_pre_hooks)
    module._backward_hooks = clear(module._backward_hooks)