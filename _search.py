import os
import time
import pandas as pd
from tqdm import tqdm

from _controller import generate
from _tune import save_pretrained, update_layer, load_control_coef, load_control_vec

class GridSearch:
    def __init__(self, controller, control_coefs, layers_to_control, judge, **kwargs):
        assert controller.directions is not None
        self.controller = controller
        self.control_coefs = sorted(set(control_coefs))
        self.layers_to_control = layers_to_control
        self.judge = judge()
        self.kwargs = kwargs
        self.search_fns = {
            "greedy": greedy_search,
            "linear": linear_search,
            "binary": binary_search,
            "ternary": ternary_search,
        }

    def fit(self, prompts, judge_inputs, judge_labels, search_fn="linear"):
        cache = {}
        arr = self.control_coefs
        func = lambda control_coef: self.score(prompts, judge_inputs, judge_labels, control_coef, cache)
        search_fn = self.search_fns[search_fn]
        hist = search_fn(arr, func)
        return hist, cache
    
    def score(self, prompts, judge_inputs, judge_labels, control_coef=0, cache=None):
        judge = self.judge
        kwargs = self.kwargs.copy()
        kwargs['layers_to_control'] = self.layers_to_control
        kwargs['control_coef'] = control_coef
        
        outputs = []
        times = []
        
        for prompt in tqdm(prompts):
            t0 = time.perf_counter()
            output = generate(self.controller, prompt, **kwargs)
            t1 = time.perf_counter()
            outputs.append(output)
            times.append(t1 - t0)

        df = {}
        df['base_input'] = judge_inputs
        df['model_output'] = outputs
        df['base_label'] = judge_labels
        df = pd.DataFrame(df)
        scores = judge(*judge.prepare(df))
        accs = judge.score_to_accuracy()

        if isinstance(cache, dict):
            cache[control_coef] = {}
            cache[control_coef]['prompt'] = prompts
            cache[control_coef]['output'] = outputs
            cache[control_coef]['score'] = scores
            cache[control_coef]['accuracy'] = accs
            cache[control_coef]['time'] = times

        return sum(accs) / len(accs)

    def save(self, hist, cache, save_directory, **kwargs):
        save_directory = os.path.join(save_directory, "checkpoint-0")
        model = self.controller.model
        hidden_layers = self.layers_to_control
        directions = self.controller.directions
        coefficients = max(hist, key=hist.get)
        model = update_layer(model, hidden_layers, directions, coefficients, steer_func="add")
        modules = save_pretrained(model, save_directory, **kwargs)
        model = update_layer(model, hidden_layers, directions, coefficients, steer_func=None)
        df = pd.DataFrame.from_dict(cache, orient='index')
        df.to_csv(save_directory + '/cache.csv', index_label='k')
        return modules
        
def get_best_model(controller, logging_dir, config=None, step=None, **kwargs):
    step = 0
    directions = load_control_vec(step, logging_dir)
    coefficients = load_control_coef(step, logging_dir)

    model = controller.model
    hidden_layers = list(coefficients.keys())
    return update_layer(model, hidden_layers, directions, coefficients)

#############################################################################################
# SEARCH UTILS

def greedy_search(arr, func):
    hist = {}
    best = 0
    for x in arr:
        hist[x] = func(x)
        if hist[x] < best: break
        best = hist[x]
    return hist

def linear_search(arr, func):
    hist = {x: func(x) for x in arr}
    return hist

def binary_search(arr, func):
    left = 0
    right = len(arr) - 1
    hist = {}
    fn = lambda d, k: d.get(k, func(k))

    while left < right:
        mid = left + (right - left) // 2
        val_mid = fn(hist, arr[mid])
        val_next = fn(hist, arr[mid+1])
        if val_mid < val_next:
            left = mid + 1
        else:
            right = mid
        hist[arr[mid]] = val_mid
        hist[arr[mid+1]] = val_next
    hist[arr[left]] = fn(hist, arr[left])
    return hist

def ternary_search(arr, func):
    left = 0
    right = len(arr) - 1
    hist = {}
    fn = lambda d, k: d.get(k, func(k))

    while right - left > 2:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        val_mid1 = fn(hist, arr[mid1])
        val_mid2 = fn(hist, arr[mid2])
        if val_mid1 < val_mid2:
            left = mid1 + 1
        else:
            right = mid2 - 1
        hist[arr[mid1]] = val_mid1
        hist[arr[mid2]] = val_mid2
    for i in range(left, right + 1):
        hist[arr[i]] = fn(hist, arr[i])
    return hist