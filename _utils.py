import os
import random
import numpy as np
import torch
from transformers import set_seed as _set_seed

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _set_seed(seed)

def balanced_sample(X, y, n_samples=1000, seed=42):
    np.random.seed(seed)
    
    X = np.array(X) # N, D
    y = np.array(y) # N
    assert X.shape[0] == y.shape[0]

    n = y.shape[0]
    if n_samples < 1:
        n_samples = int(n * n_samples)

    indices = []
    ys = np.unique(y)
    n_classes = ys.shape[0]
    
    n_samples_per_class = n_samples // n_classes
    for i in np.random.permutation(ys):
        idx = np.where(y == i)[0]
        idx = np.random.choice(idx, size=n_samples_per_class)
        indices.append(idx)

    n_samples_per_class = n_samples % n_classes
    for i in np.random.permutation(ys)[:n_samples_per_class]:
        idx = np.where(y == i)[0]
        idx = np.random.choice(idx, size=1)
        indices.append(idx)

    idx = np.concatenate(indices)
    idx = np.random.permutation(idx)
    return X[idx].tolist(), y[idx].tolist()

def format_prompt(tokenizer, prompt, generated_text="", add_special_tokens=False, use_chat_template=True):
    if any(token in prompt for token in tokenizer.all_special_tokens):
        return prompt
    if use_chat_template and tokenizer.chat_template:
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_special_tokens=False, add_generation_prompt=True)
        return text + generated_text
    text = tokenizer.decode(tokenizer(prompt)['input_ids']) + (tokenizer.eos_token if add_special_tokens else '')
    if tokenizer.bos_token is not None:
        text = text.replace(tokenizer.bos_token, '')
    return text + generated_text

def format_eos(tokenizer):
    name = tokenizer.name_or_path.lower()
    eos_map = {
        'llama': '<|eot_id|>',
        'gemma': '<end_of_turn><eos>',
        'qwen': '<|im_end|>',
        'mixtral': '</s>',
    }
    for k, v in eos_map.items():
        if k in name:
            return v