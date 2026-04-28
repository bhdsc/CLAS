import __import__
print(__file__)

import os
from tqdm import tqdm; tqdm.pandas()

import torch

import _config as config
from _controller import from_pretrained
from _search import GridSearch
from _judge import score_judges
from _utils import set_seed; set_seed(config.seed)


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
        df['id'] = dataset_id 
        df['train'] = dataset.get_train()
        df['valid'] = dataset.get_valid()
        df['test'] = dataset.get_test()
        df['kwargs'] = dataset.generation_kwargs(config)
        dfs.append(df)

    return dfs

def main():
    ####################################################
    # Environment variables
    model_id = config.model_id
    cache_dir = config.cache_dir
    torch_dtype = "auto" # NOTE: Change to torch.float32 for higher precision

    model_name = model_id.split('/')[-1]
    config.steer_func, steer_func = config.steer_func.split('-', 1)
    method = os.getenv("method")
    step_size = float(os.getenv("step_size", 1e-1))
    num_steps = config.num_train_epochs
    split = config.eval_split
    control_coefs = torch.arange(0, num_steps) * step_size
    control_coefs = [round(x, 4) for x in control_coefs.tolist()]
    ####################################################

    controller = from_pretrained(model_id, control_method=method, cache_dir=cache_dir, torch_dtype=torch_dtype)
    tokenizer = controller.tokenizer
    model = controller.model
    datasets = get_datasets()

    for dataset in tqdm(datasets):
        concept = dataset['id']
        path = __import__.path_directions
        controller.load(concept=concept, model_name=model_name, path=path)

        config.logging_dir = f'{__import__.path_coefficients}/{controller.name_or_path}/{steer_func}'
        csv_path = f'{config.logging_dir}/checkpoint-0/cache.csv'
        if not config.overwrite and os.path.isfile(csv_path):
            print(f'{csv_path} exists, skipping...')
            continue
        
        os.makedirs(config.logging_dir, exist_ok=True)

        df = dataset[split]
        kwargs = dataset['kwargs']
        layers_to_control = controller.hidden_layers
        judge = score_judges[concept]
        
        prompts = df["x"].tolist()
        judge_inputs = df["base_input"].tolist()
        judge_labels = df["base_label"].tolist()
        
        gs = GridSearch(controller, control_coefs, layers_to_control, judge, **kwargs)
        hist, cache = gs.fit(prompts, judge_inputs, judge_labels)
        gs.save(hist, cache, config.logging_dir)
        print(config.logging_dir)
        
        ################################################################################################################################################

if __name__ == "__main__":
    main()