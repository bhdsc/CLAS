import __import__; __import__.sys.path.insert(0, __import__.path)
print(__file__)

import os
import gc
import pandas as pd
from tqdm import tqdm 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import _config as config
from steer_personalities_mml_run import LLM, LLMType
from neural_controllers import NeuralController

def get_datasets(tokenizer):
    from _datasets import probe_datasets, load_probe_dataset
    from _utils import format_prompt

    dataset_ids = list(probe_datasets.keys())
    datasets = list(probe_datasets.values())
    from pprint import pprint; pprint(dataset_ids)
    
    datasets = {}
    for dataset_id in dataset_ids:
        dataset = load_probe_dataset(dataset_id)

        assert dataset.dataset['x'].is_unique
        assert dataset.dataset['y'].mean() == 0.5
        assert any(len(dataset.dataset) == dataset.n_total * (i + 1) for i in range(2))

        kwargs = {}
        kwargs['use_chat_template'] = dataset.use_chat_template

        def f(row):
            if row['y']:
                return format_prompt(tokenizer, row['x'])
            return format_prompt(tokenizer, row['x'], **kwargs)

        df = dataset.dataset
        df["x"] = df.apply(f, axis=1)
        dataset.dataset = df
        datasets[dataset_id] = dataset
    return datasets

def compute_save_directions(llm, dataset, concept, control_method='rfm', compute_directions=False):
    concept_types = [concept]
    for concept_type in concept_types:
        controller = NeuralController(
            llm,
            llm.tokenizer,
            rfm_iters=8,
            control_method=control_method,
            n_components=1,
        )
        if compute_directions:
            file_path = f"{__import__.path_directions}/{control_method}_{concept_type}_{llm.name}.pkl"
            if not os.path.isfile(file_path):
                controller.compute_directions(dataset['train']['x'].tolist(), dataset['train']['y'].tolist())
                controller.save(concept=f'{concept_type}', model_name=llm.name, path=__import__.path_directions)
            else: 
                return
        else:
            controller.load(concept=f'{concept_type}', model_name=llm.name, path=__import__.path_directions)
        controller.directions = {k: v / v.norm(keepdim=True) for k, v in controller.directions.items()}
            
        from _probe import evaluate
        valid_metrics, test_metrics, detector_coefs = evaluate(
            controller, 
            dataset['valid']['x'].tolist(),
            dataset['valid']['y'].tolist(),
            dataset['test']['x'].tolist(),
            dataset['test']['y'].tolist(),
            concept=concept_type,
            model_name=llm.name,
            path=__import__.path_directions,
            control_method=control_method,
        )
        print("valid_metrics"); print(pd.DataFrame(valid_metrics).T.describe())
        print("test_metrics"); print(pd.DataFrame(test_metrics).T.describe())

def main():
    torch.backends.cudnn.benchmark = True 
    torch.backends.cuda.matmul.allow_tf32 = True

    ####################################################
    # Environment variables
    model_id = config.model_id
    cache_dir = config.cache_dir
    device_map = "auto"
    torch_dtype = "auto" # NOTE: Change to torch.float32 for higher precision

    model_name = model_id.split('/')[-1]
    method = os.getenv("method")
    compute_directions = os.getenv("compute_directions", "false").lower() == "true"
    ####################################################

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, device_map=device_map, torch_dtype=torch_dtype)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

    processor = None
    llm_type = LLMType.TEXT
    llm = LLM(model, tokenizer, processor, model_name, llm_type)
    datasets = get_datasets(tokenizer)
    
    for dataset_id, dataset in tqdm(datasets.items()):
        concept = dataset_id
        df = {}
        df['train'] = dataset.get_train()
        df['valid'] = dataset.get_valid()
        df['test'] = dataset.get_test()
        df['train'] = pd.concat([df['train'], df['valid']])
        for k, v in df.items():
            print(concept, k, len(v))
        compute_save_directions(llm, df, concept, method, compute_directions)
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()