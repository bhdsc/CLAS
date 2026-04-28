import os
import json
import pickle
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import _config as config
from _controller import add, add_proj, add_dynamic

#####################################################################################################################
# STEERING LAYERS

def init_linear(in_features, out_features, dtype, device, init_weight=None, unit_norm=False):
    dtype = torch.float32
    weight = nn.Linear(in_features, out_features, bias=False, dtype=dtype, device=device).weight
    with torch.no_grad():
        if init_weight is not None:
            if not torch.is_tensor(init_weight):
                return weight.fill_(float(init_weight))
            k = init_weight.shape[0]
            weight[:k].copy_(init_weight.to(dtype=dtype, device=device))
        if unit_norm:
            weight.div_(weight.norm(dim=-1, keepdim=True))
    return weight

class _Layer(nn.Module):
    def __init__(self, layer, control_vec, control_coef=0.0, rescale_out=0.0, control_bias=0.0, unit_norm=False, steer_func=add, r=1, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.layer = layer
        self.hidden_size = layer.hidden_size
        self.steer_func = steer_func
        self.control_coef = control_coef
        self.control_vec = init_linear(self.hidden_size, r, dtype, device, init_weight=control_vec, unit_norm=unit_norm)
        self.rescale_out = init_linear(self.hidden_size, 1, dtype, device, init_weight=rescale_out, unit_norm=False)
        self.bias = init_linear(r, 1, dtype, device, init_weight=control_bias, unit_norm=False)
        self.control_bias = self.bias
        self.attention_type = getattr(layer, 'attention_type', None)

    def extra_repr(self):
        attrs = ["control_coef", "control_vec", "rescale_out", "control_bias"]
        get = lambda attr: getattr(self, attr)
        params = [
            f'({attr}): nn.Parameter(shape={tuple(get(attr).shape)}, dtype={get(attr).dtype}, requires_grad={get(attr).requires_grad})'
            for attr in attrs
        ]
        return '\n'.join(params)
    
    def steer(self, out):
        return self.steer_func(out.float(), self.control_vec, self.control_coef, self.rescale_out, self.control_bias).to(out.dtype)

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        if isinstance(out, tuple):
            out, *attn = out
            out = self.steer(out)
            return (out, *attn)
        return self.steer(out)

    def get_param(self, module, name):
        if isinstance(module, nn.Module):
            return module.state_dict()
        if isinstance(module, nn.Parameter):
            return module
        if torch.is_tensor(module):
            return module

    def get_data(self, module, name):
        if isinstance(module, nn.Module):
            return module.weight.data
        if isinstance(module, nn.Parameter):
            return module.data
        if torch.is_tensor(module):
            return module
        
    def set_param(self, p, name):
        module = getattr(self, name)
        if isinstance(module, nn.Module):
            module.load_state_dict(p)
            return module.to(dtype=module.weight.dtype, device=module.weight.device)
        if isinstance(module, nn.Parameter):
            return setattr(self, name, nn.Parameter(p.to(dtype=module.dtype, device=module.device)))
        if torch.is_tensor(module):
            return setattr(self, name, p.to(dtype=module.dtype, device=module.device))

class StaticLayer(_Layer):
    def __init__(self, layer, control_vec, control_coef=0.0, rescale_out=0.0, control_bias=0.0, unit_norm=False, steer_func=add, r=1, dtype=torch.float32, device="cuda"):
        super().__init__(layer, control_vec, control_coef, rescale_out, control_bias, unit_norm, steer_func, r, dtype, device)
        if not isinstance(control_coef, nn.Parameter):
            self.control_coef = init_linear(r, 1, dtype, device, init_weight=control_coef, unit_norm=False)

class ProjLayer(_Layer):
    def __init__(self, layer, control_vec, control_coef=0.0, rescale_out=0.0, control_bias=0.0, unit_norm=False, steer_func=add_proj, r=1, dtype=torch.float32, device="cuda"):
        super().__init__(layer, control_vec, control_coef, rescale_out, control_bias, unit_norm, steer_func, r, dtype, device)
        if not isinstance(control_coef, nn.Parameter):
            self.control_coef = init_linear(r, 1, dtype, device, init_weight=control_coef, unit_norm=False)

class DynamicLayer(_Layer):
    def __init__(self, layer, control_vec, control_coef=0.0, rescale_out=0.0, control_bias=0.0, unit_norm=False, steer_func=add_dynamic, r=1, dtype=torch.float32, device="cuda"):
        super().__init__(layer, control_vec, control_coef, rescale_out, control_bias, unit_norm, steer_func, r, dtype, device)
        if not isinstance(control_coef, nn.Parameter):
            self.control_coef = init_linear(self.hidden_size, r, dtype, device, init_weight=control_coef, unit_norm=unit_norm)

#####################################################################################################################
# MODEL UTILS

def get_layer(layer, control_vec, control_coef=0.0, rescale_out=0.0, control_bias=0.0, unit_norm=False, steer_func=add, r=1, dtype=torch.float32, device="cuda"):
    if steer_func is None:
        return layer
    if not isinstance(steer_func, str):
        steer_func = steer_func.__name__
    if steer_func == "add":
        return StaticLayer(layer, control_vec, control_coef, rescale_out, control_bias, unit_norm, add, r, dtype, device)
    if steer_func == "add_proj":
        return ProjLayer(layer, control_vec, control_coef, rescale_out, control_bias, unit_norm, add_proj, r, dtype, device)
    if steer_func == "add_dynamic":
        return DynamicLayer(layer, control_vec, control_coef, rescale_out, control_bias, unit_norm, add_dynamic, r, dtype, device)
    raise ValueError(f"{steer_func} is not valid!")

def update_layer(model, hidden_layers, directions, coefficients=0.0, scales=0.0, biases=0.0, unit_norm=False, steer_func=add, r=1, target_modules=[]):
    if not isinstance(directions, dict):
        directions = {layer_idx: directions for layer_idx in hidden_layers}
    if not isinstance(coefficients, dict):
        coefficients = {layer_idx: coefficients for layer_idx in hidden_layers}
    if not isinstance(scales, dict):
        scales = {layer_idx: scales for layer_idx in hidden_layers}
    if not isinstance(biases, dict):
        biases = {layer_idx: biases for layer_idx in hidden_layers}
    if not isinstance(target_modules, str):
        target_modules = target_modules[0] if target_modules else ''
        
    for layer_idx in hidden_layers:
        if any(layer_idx not in d for d in [directions, coefficients, scales, biases]):
            continue
        layer = get_module(model.model.layers[layer_idx], target_modules)
        control_vec = directions[layer_idx]
        control_coef = coefficients[layer_idx]
        rescale_out = scales[layer_idx]
        control_bias = biases[layer_idx]
        if hasattr(layer, "layer"):
            layer = layer.layer
        layer = get_layer(layer, control_vec, control_coef, rescale_out, control_bias, unit_norm, steer_func, r, device=model.device)
        model.model.layers[layer_idx] = update_module(model.model.layers[layer_idx], target_modules, layer)
    
    if not isinstance(steer_func, str):
        steer_func = getattr(steer_func, "__name__", None)
    setattr(model.config, "steer_func", steer_func)
    setattr(model.config, "r", r)
    setattr(model.config, "target_modules", target_modules)

    freeze_model(model)
    unfreeze_control_coef(model)
    unfreeze_control_bias(model)

    return model

def get_best_model(controller, logging_dir, config=None, step=None, **kwargs):
    config = load_config(logging_dir) if config is None else config
    step = get_best_step(logging_dir) if step is None else step
    directions = load_control_vec(step, logging_dir)
    coefficients = load_control_coef(step, logging_dir)
    scales = load_rescale_out(step, logging_dir)
    biases = load_control_bias(step, logging_dir)

    model = controller.model
    hidden_layers = controller.hidden_layers
    return update_layer(model, hidden_layers, directions, coefficients, scales, biases, unit_norm=False, steer_func=config['steer_func'], r=config['r'], target_modules=config['target_modules'])

def get_module(module, name):
    return getattr(module, name, module)

def get_modules(modules, name):
    return [get_module(module, name) for module in modules]

def update_module(module, name, value):
    if hasattr(module, name):
        setattr(module, name, value)
        return module
    return value

def trainable_params(model):
    return ((name, param.shape, param.data) for name, param in model.named_parameters() if param.requires_grad)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_module(model, name):
    for layer in get_modules(model.model.layers, model.config.target_modules):
        if hasattr(layer, name):
            module = getattr(layer, name)
            if hasattr(module, "parameters"):
                for param in module.parameters():
                    param.requires_grad = True
            else:
                param = module
                param.requires_grad = True

def unfreeze_control_vec(model):
    unfreeze_module(model, name="control_vec")

def unfreeze_control_coef(model):
    unfreeze_module(model, name="control_coef")

def unfreeze_rescale_out(model):
    unfreeze_module(model, name="rescale_out")

def unfreeze_control_bias(model):
    unfreeze_module(model, name="control_bias")

def unfreeze_lm_head(model):
    model.lm_head.weight.requires_grad = True
    
def get_data(model, name):
    data = {}
    for i, layer in enumerate(get_modules(model.model.layers, model.config.target_modules)[::-1], 1):
        if hasattr(layer, name):
            module = getattr(layer, name)
            dat = layer.get_data(module, name).detach().cpu()
            if dat.numel() == 1:
                dat = dat.item()
            data[-i] = dat
    return data

def get_control_vec(model):
    return get_data(model, name="control_vec")

def get_control_coef(model):
    return get_data(model, name="control_coef")

def get_rescale_out(model):
    return get_data(model, name="rescale_out")

def get_control_bias(model):
    return get_data(model, name="control_bias")

def load_config(logging_dir="./"):
    path = f"{logging_dir}/config.json"
    path = os.path.normpath(path)
    with open(path, "r") as f:
        return json.load(f)

def load_checkpoint(filename, logging_dir="./"):
    path = f"{logging_dir}/{filename}"
    path = os.path.normpath(path)
    with open(path, "rb") as f:
        return  pickle.load(f)

def load_control_coef(step, logging_dir="./"):
    filename = f"checkpoint-{step}/control_coef.pkl"
    return load_checkpoint(filename, logging_dir)

def load_control_vec(step, logging_dir="./"):
    filename = f"checkpoint-{step}/control_vec.pkl"
    return load_checkpoint(filename, logging_dir)

def load_rescale_out(step, logging_dir="./"):
    filename = f"checkpoint-{step}/rescale_out.pkl"
    return load_checkpoint(filename, logging_dir)

def load_control_bias(step, logging_dir="./"):
    filename = f"checkpoint-{step}/control_bias.pkl"
    return load_checkpoint(filename, logging_dir)

def load_checkpoints(filename, logging_dir="./", load_fn=lambda *args: None):
    files = glob(f"{logging_dir}/**/{filename}", recursive=True)
    files = sorted(set(files))
    steps = [int(file.split('/')[-2].replace('checkpoint-', '')) for file in files]
    return {step: load_fn(step, logging_dir) for step in sorted(steps)}

def load_control_coefs(logging_dir):
    filename = "control_coef.pkl"
    return load_checkpoints(filename, logging_dir, load_control_coef)

def load_control_vecs(logging_dir):
    filename = "control_vec.pkl"
    return load_checkpoints(filename, logging_dir, load_control_vec)

def load_rescale_outs(logging_dir):
    filename = "rescale_out.pkl"
    return load_checkpoints(filename, logging_dir, load_rescale_out)

def load_control_biases(logging_dir):
    filename = "control_bias.pkl"
    return load_checkpoints(filename, logging_dir, load_control_bias)

def compute_norm(x, size=1024):
    if torch.is_tensor(x):
        x = x.norm(dim=-1) if x.numel() >= size else x
        x = x.squeeze().tolist()
    return x if isinstance(x, list) else [float(x)]
    
def compute_similarity(x, y):
    assert torch.is_tensor(x) and x.ndim
    assert torch.is_tensor(y) and y.ndim
    return F.cosine_similarity(x.squeeze(), y.squeeze(), dim=-1).mean().item()
    
def save_module(model, path, names=["control_coef", "control_vec", "rescale_out", "control_bias"]):
    modules = {}
    for i, layer in enumerate(get_modules(model.model.layers, model.config.target_modules)[::-1], 1):
        modules[-i] = {}
        for name in names:
            if hasattr(layer, name):
                module = getattr(layer, name)
                modules[-i][name] = layer.get_param(module, name)
    torch.save(modules, path)

def load_module(model, path, names=["control_coef", "control_vec", "rescale_out", "control_bias"]):
    modules = get_pretrained(path)
    for i, layer in enumerate(get_modules(model.model.layers, model.config.target_modules)[::-1], 1):
        for name in names:
            if hasattr(layer, name):
                p = modules[-i][name]
                layer.set_param(p, name)
    return model

def get_pretrained(save_directory, map_location=None, **kwargs):
    path = f"{save_directory}/adapter_model.pt"
    path = os.path.normpath(path)
    return torch.load(path, map_location=map_location)

# https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/modeling_utils.py#L3542
def save_pretrained(model, save_directory, **kwargs):
    os.makedirs(save_directory, exist_ok=True)
    path = f"{save_directory}/adapter_model.pt"
    save_module(model, path)

    modules = {}
    modules['control_coef'] = get_control_coef(model)
    modules['control_vec'] = get_control_vec(model)
    modules['rescale_out'] = get_rescale_out(model)
    modules['control_bias'] = get_control_bias(model)

    # NOTE: normalizes control_vec and scales control_coef
    for layer_idx in modules['control_coef']:
        control_coef = modules['control_coef'][layer_idx]
        control_vec = modules['control_vec'][layer_idx]
        control_vec_norm = control_vec.norm(dim=-1, keepdim=True)
        control_vec_norm = torch.where(control_vec_norm == 0, torch.ones_like(control_vec_norm), control_vec_norm)
        modules['control_vec'][layer_idx] = control_vec / control_vec_norm
        modules['control_coef'][layer_idx] = control_coef * control_vec_norm

    for name, module in modules.items():
        path = f"{save_directory}/{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(module, f)
        norm = {k: compute_norm(v) for k, v in module.items()} 
        text = "/".join(path.split("/")[:-2])
        plot_layers(norm, path=path.replace(".pkl", ".png"), title=text.replace(".pkl", ""))

    return modules

def get_hist(logging_dir, metric_name='eval_loss'):
    trainer_state = json.load(open(f"{logging_dir}/trainer_state.json", "r"))
    log_history = trainer_state['log_history']
    return [log for log in log_history if metric_name in log]

def get_best_hist(logging_dir, metric_name='eval_loss', func=min):
    logs = get_hist(logging_dir, metric_name)
    return func(logs, key=lambda x: x[metric_name])

def get_best_step(logging_dir, metric_name='eval_loss', func=min):
    step = get_best_hist(logging_dir, metric_name, func)['step']
    return int(step)

#####################################################################################################################
# DATA UTILS

from datasets import Dataset

def dataset(tokenizer, prompts, completions, max_length=None):
    data = {
        "prompt": prompts,
        "completion": completions,
    }
    fn_kwargs = {"tokenizer": tokenizer, "max_length": max_length}
    base_dataset = Dataset.from_dict(data)
    tokenized_dataset = base_dataset.map(tokenize_function, batched=True, fn_kwargs=fn_kwargs)
    formatted_dataset = tokenized_dataset.map(format_for_training, fn_kwargs=fn_kwargs)
    return formatted_dataset

def dataset_completion_only(tokenizer, prompts, completions, max_length=None):
    base_dataset = dataset(tokenizer, prompts, completions, max_length)
    formatted_dataset = base_dataset.map(ignore_prompt)
    return formatted_dataset
    
def tokenize_function(examples, **kwargs):
    tokenizer = kwargs["tokenizer"]
    max_length = kwargs["max_length"]
    prompt_tokens = tokenizer(
        examples["prompt"], 
        padding=False,
        truncation=True, 
        max_length=max_length,
        add_special_tokens=False,
    )
    completion_tokens = tokenizer(
        examples["completion"],
        padding=False,
        truncation=True, 
        max_length=max_length,
        add_special_tokens=False,
    )
    inputs = {}
    inputs["input_ids"] = [prompt_ids + completion_ids
    for prompt_ids, completion_ids in zip(
        prompt_tokens["input_ids"], 
        completion_tokens["input_ids"]
    )]
    inputs["attention_mask"] = [prompt_mask + completion_mask
    for prompt_mask, completion_mask in zip(
        prompt_tokens["attention_mask"], 
        completion_tokens["attention_mask"]
    )]
    inputs["prompt_length"] = [len(prompt_ids) for prompt_ids in prompt_tokens["input_ids"]]

    max_len = tokenizer.model_max_length if max_length is None else max_length
    assert all(len(input_ids) <= max_len for input_ids in prompt_tokens["input_ids"])
    assert all(len(input_ids) <= max_len for input_ids in completion_tokens["input_ids"])
    assert all(len(input_ids) <= max_len for input_ids in inputs["input_ids"])
    
    return inputs

def format_for_training(example, **kwargs):
    tokenizer = kwargs["tokenizer"]
    input_ids = example["input_ids"]
    labels = input_ids.copy()
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    return {"input_ids": input_ids, "labels": labels}

def ignore_prompt(example, **kwargs):
    input_ids = example["input_ids"]
    labels = example["labels"]
    prompt_len = example["prompt_length"]
    labels = [label if i >= prompt_len else -100 for i, label in enumerate(labels)]
    return {"input_ids": input_ids, "labels": labels}

#####################################################################################################################
# TRAINING UTILS

from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainerCallback

def data_collator(tokenizer, pad_to_multiple_of=8):
    kwargs = ['input_ids', 'attention_mask', 'labels']
    def func(features):
        tokenizer.padding_side = "right"
        max_length = max(len(feature['input_ids']) for feature in features)
        max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        for i, feature in enumerate(features):
            feature = {k: feature[k] for k in kwargs}
            pad_length = max_length - len(feature['labels'])
            feature['labels'] += [-100] * pad_length
            features[i] = feature
        return tokenizer.pad(features, padding='max_length', max_length=max_length, return_tensors="pt")
    return func

class DeterministicTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, step_size=1e-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.step_size = float(step_size)
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size,
            sampler_fn=SequentialSampler, # disables dataset shuffling during training.
            is_training=True,
        )
    
    # https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/trainer.py#L1343C16-L1343C30
    def create_optimizer(self):
        param_groups = super().create_optimizer().param_groups
        assert len(param_groups) == 2
        self.optimizer.param_groups[1]['lr'] = self.step_size
        return self.optimizer

# https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/trainer_callback.py#L574
# https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/trainer.py#L2538
class WeightNormLoggingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        model.save_pretrained = save_pretrained.__get__(model)
        save_directory = f"{args.output_dir}/checkpoint-{state.global_step}"
        model.save_pretrained(save_directory)

    def log(self, args, state, control, model=None, **kwargs):
        def to_tensor(xs):
            if all(isinstance(x, torch.Tensor) for x in xs):
                return torch.stack(xs)
            return torch.tensor(xs)

        control_coef = get_control_coef(model)
        control_vec = get_control_vec(model)
        rescale_out = get_rescale_out(model)
        control_bias = get_control_bias(model)

        state.log_history.append(
            {
                "coef_norm": compute_norm(to_tensor(list(control_coef.values()))), 
                "vec_norm": compute_norm(to_tensor(list(control_vec.values()))), 
                "out_norm": compute_norm(to_tensor(list(rescale_out.values()))), 
                "bias_norm": compute_norm(to_tensor(list(control_bias.values()))), 
                "n_layers": len(control_coef),
                "epoch": state.epoch, 
                "step": state.global_step,
            }
        )

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            self.log(args, state, control, model, **kwargs)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if control.should_log:
            self.log(args, state, control, model, **kwargs)

def fit(model, tokenizer, trainer_args, train_dataset, eval_dataset, step_size, trainer=None):
    is_quantized = getattr(model, "is_quantized", False)
    if is_quantized:
        model.is_quantized = False

    if trainer is None:
        trainer = DeterministicTrainer(
            model=model,
            tokenizer=tokenizer,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator(tokenizer),
            callbacks=[WeightNormLoggingCallback()],
            step_size=step_size,
        )
    output_dir = trainer.args.output_dir
    trainer.args.output_dir = trainer_args.logging_dir
    trainer.train()

    trainer.save_state()
    train_dataset.save_to_disk(f"{trainer.args.output_dir}/train_dataset")
    eval_dataset.save_to_disk(f"{trainer.args.output_dir}/eval_dataset")

    modules = {}
    modules['control_coef'] = load_control_coefs(trainer.args.output_dir)
    modules['control_vec'] = load_control_vecs(trainer.args.output_dir)
    modules['rescale_out'] = load_rescale_outs(trainer.args.output_dir)
    modules['control_bias'] = load_control_biases(trainer.args.output_dir)

    for name, module_list in modules.items():
        norms = {step: {k: compute_norm(v) for k, v in module.items()} for step, module in module_list.items()}
        plot_training_dynamics(
            norms, 
            path=f"{trainer.args.output_dir}/plot_{name}_training_dynamics.png", 
            title=f"{name}_norm".replace('_', ' ').title(),
        )
    
    plot_loss(
        trainer.state.log_history, 
        path=f"{trainer.args.output_dir}/plot_loss.png", 
        title=f"n_train={len(train_dataset)}, n_eval={len(eval_dataset)}, lr={trainer_args.learning_rate}",
    )
    json.dump(config.to_dict(), open(f"{trainer.args.output_dir}/config.json", "w"))
    print(f"Checkpoints saved at: {trainer.args.output_dir}")
    trainer.args.output_dir = output_dir

    if is_quantized:
        model.is_quantized = True

    return trainer

#####################################################################################################################
# PLOTTING UTILS

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

def plot_layers(dct, path=None, title="", xlabel="$Layer_i$", figsize=(10,6)):
    df = pd.DataFrame(dct).T
    ax = df.plot(kind='bar', figsize=figsize)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    plt.xticks(rotation=0)
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_training_dynamics(dct, path=None, title="Norm Training Dynamics", figsize=(12, 8)):
    df = pd.DataFrame(dct).T
    titles = df.columns.tolist()
    titles = [f"$Layer_{{{layer_idx}}}$" for layer_idx in titles]
    n = len(titles)
    l = np.ceil(np.sqrt(n)).astype(int)
    
    fig, axes = plt.subplots(l, l, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes[:len(df.columns)]):
        data = pd.DataFrame(df.iloc[:, i].tolist())
        for col in data.columns:
            ax.plot(df.index, data[col], '-o', label=col)
        ax.set_title(titles[i])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels), fontsize='small')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.xlabel("Steps")
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")    
    plt.show()
    plt.close()

def plot_loss(log_history, path=None, title=None, extra_metrics={}, figsize=(12, 4)):
    metrics = ['loss', 'eval_loss', 'grad_norm', 'coef_norm', 'vec_norm', 'out_norm', 'bias_norm']
    
    loss_history = {}
    for metric in metrics:
        loss_history[metric] = []
        loss_history[f"{metric}_step"] = []

    for log in log_history:
        for metric in metrics:
            if metric in log:
                loss_history[metric].append(log[metric])
                loss_history[f"{metric}_step"].append(log['step'])

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Training and evaluation losses
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    line1, = axes[0].plot(loss_history['loss_step'], loss_history['loss'], '-o', label='Training Loss', color=next(colors))
    line2, = axes[0].plot(loss_history['eval_loss_step'], loss_history['eval_loss'], '-o', label='Evaluation Loss', color=next(colors))
    lines = []
    for i, (name, metric) in enumerate(extra_metrics.items()):
        line, = axes[0].twinx().plot(metric['step'], metric['score'], '--o', label=name, color=next(colors))
        lines.append(line)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Evaluation Curves')
    
    lines = [line1, line2, *lines]
    labels = [line.get_label() for line in lines]
    axes[0].legend(lines, labels)
    axes[0].grid()

    # Gradient and weight norms
    f = lambda x: torch.tensor(x).flatten(start_dim=1).sum(dim=-1)
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    line1, = axes[1].plot(loss_history['grad_norm_step'], loss_history['grad_norm'], '-o', label='Gradient Norm', color=next(colors))
    line2, = axes[1].plot(loss_history['coef_norm_step'], f(loss_history['coef_norm']), '-o', label='Coefficient Norm', color=next(colors))
    line3, = axes[1].plot(loss_history['vec_norm_step'], f(loss_history['vec_norm']), '-o', label='Direction Norm', color=next(colors))
    line3, = axes[1].plot(loss_history['bias_norm_step'], f(loss_history['bias_norm']), '-o', label='Bias Norm', color=next(colors))
    line4, = axes[1].twinx().plot(loss_history['out_norm_step'], f(loss_history['out_norm']), '--o', label='Rescaled Norm', color=next(colors))
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Norm')
    axes[1].set_title('Gradient and Weight Norms')
    
    lines = [line1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    axes[1].legend(lines, labels)
    axes[1].grid()

    if title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
    
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")    
    plt.show()
    plt.close()