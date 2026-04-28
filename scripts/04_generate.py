import __import__
print(__file__)

import os
import pandas as pd
from tqdm import tqdm; tqdm.pandas()

import _config as config
from _controller import from_pretrained, generate_time
from _peft import get_best_model as get_best_peft_model
from _tune import get_best_model as get_best_tune_model
from _search import get_best_model as get_best_search_model
from _utils import format_eos, set_seed; set_seed(config.seed)

def get_datasets():
    from _datasets import steer_datasets, load_steer_dataset

    dataset_ids = list(steer_datasets.keys())
    datasets = list(steer_datasets.values())
    from pprint import pprint; pprint(dataset_ids)

    dfs = []
    for dataset_id, dataset_cls in zip(dataset_ids, datasets):
        path = __import__.path_datasets + '/' + dataset_cls.__name__ + '.csv'
        dataset = load_steer_dataset(dataset_id, path=path)

        assert dataset.dataset['x'].is_unique
        assert any(len(dataset.dataset) == dataset.n_total * (i + 1) for i in range(2))

        df = {}
        df['cls'] = dataset
        df['id'] = dataset_id 
        df['train'] = dataset.get_train()
        df['valid'] = dataset.get_valid()
        df['test'] = dataset.get_test()
        df['kwargs'] = dataset.generation_kwargs(config)
        dfs.append(df)

    return dfs

def get_best_model(controller, logging_dir, steer_func):
    if "search" in steer_func:
        return get_best_search_model(controller, logging_dir)
    if "lora" in steer_func:
        return get_best_peft_model(controller, logging_dir)
    return get_best_tune_model(controller, logging_dir)

def main():
    ####################################################
    # Environment variables
    model_id = config.model_id
    cache_dir = config.cache_dir
    torch_dtype = "auto" # NOTE: Change to torch.float32 for higher precision

    model_name = model_id.split('/')[-1]
    config.steer_func, steer_func = config.steer_func.split('-', 1)
    method = os.getenv("method")
    source_concept = os.getenv("concept")
    split = config.eval_split
    ####################################################

    controller = from_pretrained(model_id, control_method=method, cache_dir=cache_dir, torch_dtype=torch_dtype)
    tokenizer = controller.tokenizer
    model = controller.model
    datasets = get_datasets()

    for dataset in tqdm(datasets):
        concept = dataset['id']
        path = __import__.path_directions
        controller.load(concept=source_concept or concept, model_name=model_name, path=path)
        controller.directions = None

        if "prompt" not in steer_func:
            config.logging_dir = f'{__import__.path_coefficients}/{controller.name_or_path}/{steer_func}'
            n = 0 if source_concept is None else 1
            csv_path = f'{config.logging_dir}/{concept}_{n}_{split}.csv'
            if not config.overwrite and os.path.isfile(csv_path):
                print(f'{csv_path} exists, skipping...')
                continue
            model = get_best_model(controller, config.logging_dir, steer_func)
        
        else:
            config.logging_dir = f'{__import__.path_coefficients}/{controller.name_or_path}'
            n = int(steer_func[-1])
            csv_path = f'{config.logging_dir}/{concept}_{n}_{split}.csv'
            if not config.overwrite and os.path.isfile(csv_path):
                print(f'{csv_path} exists, skipping...')
                continue
            
        get_prompt = dataset['cls'].get_steer_prompt if n else dataset['cls'].get_base_prompt
        dataset[split]["x"] = get_prompt(dataset[split]['base_input'].tolist())

        controller.model = model
        print(config.logging_dir)
        print(model)

        df = dataset[split]
        kwargs = dataset['kwargs']
        kwargs['use_chat_template'] = True if n else dataset['cls'].use_chat_template
        df[["model_output", "time"]] = df["x"].progress_apply(lambda prompt: generate_time(controller, prompt, **kwargs)).apply(pd.Series)
        df["model_output"] = df["model_output"].apply(lambda output: output.replace(format_eos(tokenizer), ''))
        df["control_coef"] = config.control_coef
        df.to_csv(csv_path, index=False)
        
        # cleanup for lora
        if hasattr(model, "unload"):
            model = model.unload()
            controller.model = model
        
        ################################################################################################################################################

if __name__ == "__main__":
    main()