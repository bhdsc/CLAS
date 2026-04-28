import __import__
print(__file__)

import os
from tqdm import tqdm

import torch
from transformers import TrainingArguments

import _config as config
from _controller import from_pretrained
from _tune import dataset_completion_only, fit, trainable_params, update_layer, unfreeze_control_vec
from _peft import fit as fit_peft, update_layer as update_layer_peft
from _utils import format_prompt, format_eos, set_seed; set_seed(config.seed)

def get_datasets(tokenizer):
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

        kwargs = {}
        kwargs['use_chat_template'] = dataset.use_chat_template
        
        df = dataset.dataset
        df["x"] = df["x"].apply(lambda prompt: format_prompt(tokenizer, prompt, **kwargs))
        df["y"] = dataset.replace_eos_token(df["y"].tolist(), eos_token=format_eos(tokenizer))
        dataset.dataset = df

        df = {}
        df['id'] = dataset_id 
        df['train'] = dataset.get_train()
        df['valid'] = dataset.get_valid()
        df['test'] = dataset.get_test()
        dfs.append(df)

    return dfs

def update_model(controller, config, steer_func):
    model = controller.model
    r = config.r
    target_modules = config.target_modules
    hidden_layers = controller.hidden_layers
    directions = controller.directions

    fn = lambda x: None if config.random_basis else x
    directions = {k: fn(v) for k, v in directions.items()}

    if "lora" in steer_func:
        return update_layer_peft(model, r, target_modules), config.learning_rate

    model = update_layer(
        model,
        hidden_layers,
        directions,
        coefficients=0,
        biases=0,
        unit_norm=True,
        steer_func=config.steer_func,
        r=r, target_modules=target_modules
    )
    step_size = float(os.getenv("step_size", 1e-1))

    if "reft" in steer_func:
        unfreeze_control_vec(model)
        return model, config.learning_rate

    return model, step_size

def main():
    ####################################################
    # Environment variables
    model_id = config.model_id
    cache_dir = config.cache_dir
    torch_dtype = "auto" # NOTE: Change to torch.float32 for higher precision

    model_name = model_id.split('/')[-1]
    config.steer_func, steer_func = config.steer_func.split('-', 1)
    method = os.getenv("method")
    ####################################################

    controller = from_pretrained(model_id, control_method=method, cache_dir=cache_dir, torch_dtype=torch_dtype)
    tokenizer = controller.tokenizer
    model = controller.model
    datasets = get_datasets(tokenizer)

    for dataset in tqdm(datasets):
        concept = dataset['id']
        path = __import__.path_directions
        controller.load(concept=concept, model_name=model_name, path=path)

        config.logging_dir = f'{__import__.path_coefficients}/{controller.name_or_path}/{steer_func}'
        config_path = f'{config.logging_dir}/config.json'
        if not config.overwrite and os.path.isfile(config_path):
            print(f'{config_path} exists, skipping...')
            continue

        model, step_size = update_model(controller, config, steer_func)

        from pprint import pprint
        print(model)
        pprint(list(trainable_params(model)))
        
        ################################################################################################################################################
    
        train_dataset = dataset_completion_only(tokenizer, dataset['train']['x'].tolist(), dataset['train']['y'].tolist(), max_length=config.max_length)
        eval_dataset  = dataset_completion_only(tokenizer, dataset['valid']['x'].tolist(), dataset['valid']['y'].tolist(), max_length=config.max_length)

        trainer_kwargs = {}
        trainer_kwargs['per_device_train_batch_size'] = 1
        trainer_kwargs['gradient_accumulation_steps'] = config.gradient_accumulation_steps

        trainer_args = TrainingArguments(
            output_dir=f'{__import__.path_coefficients}/{controller.name_or_path}',
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            lr_scheduler_type="constant_with_warmup",
            save_only_model=True,
            save_strategy=config.save_strategy,
            save_steps=config.logging_steps / config.num_train_epochs,
            logging_strategy=config.logging_strategy,
            logging_steps=config.logging_steps / config.num_train_epochs,
            eval_strategy=config.eval_strategy,
            eval_steps=config.logging_steps / config.num_train_epochs,
            debug="underflow_overflow",
            logging_dir=config.logging_dir,
            eval_on_start=config.eval_strategy != "no",
            remove_unused_columns=False,
            **trainer_kwargs,
        )
        # cleanup for lora
        if hasattr(model, "unload"):
            trainer = fit_peft(model, tokenizer, trainer_args, train_dataset, eval_dataset, step_size)
            model = model.unload()
            controller.model = model
        else:
            trainer = fit(model, tokenizer, trainer_args, train_dataset, eval_dataset, step_size)

if __name__ == "__main__":
    main()