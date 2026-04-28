# https://github.com/huggingface/peft
# pip install peft

import os
import torch
from peft import get_peft_model, LoraConfig, PeftModel, TaskType

import _config as config
from _tune import data_collator, get_best_step

def update_layer(model, r, target_modules, layers_to_transform=None, lora_dropout=0.0):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=r*2,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)

def unfreeze_lm_head(model):
    model.base_model.lm_head.weight.requires_grad = True
    
def save_module(model, path):
    model.save_pretrained(path)

def load_module(model, path):
    return PeftModel.from_pretrained(model, path)

def get_best_model(controller, logging_dir, step=None, **kwargs):
    step = get_best_step(logging_dir) if step is None else step
    path = f"{logging_dir}/checkpoint-{step}"
    model = controller.model
    return load_module(model, path)

#####################################################################################################################
# TRAINING UTILS

from transformers import Trainer, TrainerCallback
from _tune import DeterministicTrainer

# https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/trainer_callback.py#L574
# https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/trainer.py#L2538
class WeightNormLoggingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        save_directory = f"{args.output_dir}/checkpoint-{state.global_step}"
        model.save_pretrained(save_directory)

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

def plot_loss(log_history, path=None, title=None, extra_metrics={}, figsize=(12, 4)):
    metrics = ['loss', 'eval_loss', 'grad_norm']
    
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
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    line1, = axes[1].plot(loss_history['grad_norm_step'], loss_history['grad_norm'], '-o', label='Gradient Norm', color=next(colors))
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Norm')
    axes[1].set_title('Gradient and Weight Norms')
    axes[1].legend()
    axes[1].grid()

    if title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
    
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")    
    plt.show()
    plt.close()