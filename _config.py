import os

model_id = os.getenv("model_id")
cache_dir = os.getenv("cache_dir")
data_dir = os.getenv("data_dir")
steer_csv = os.getenv("steer_csv")
steer_dir = os.getenv("steer_dir")
bench_csv = os.getenv("bench_csv")

# probe
n_probe_samples = int(os.getenv("n_probe_samples", 500))
n_steer_samples = int(os.getenv("n_steer_samples", 100))

# train
seed = int(os.getenv("seed", 42))
max_length = os.getenv("max_length")
if max_length is not None:
    max_length = int(max_length)
hidden_layers = [int(layer.strip()) for layer in 
os.getenv("hidden_layers", "").split(",") if layer.strip()]
steer_func = os.getenv("steer_func", "add")
num_train_samples = int(os.getenv("num_train_samples", 1))
num_eval_samples = int(os.getenv("num_eval_samples", 30))
num_train_epochs = int(os.getenv("num_train_epochs", 50))
logging_steps = float(os.getenv("logging_steps", 0.1))
learning_rate = float(os.getenv("learning_rate", 5e-1))
gradient_accumulation_steps = int(os.getenv("gradient_accumulation_steps", 1))
random_basis = os.getenv("random_basis", "false").lower() == "true"
control_coef = float(os.getenv("control_coef", "nan"))
control_coef = None if control_coef != control_coef else control_coef
control_vec = float(os.getenv("control_vec", "nan"))
control_vec = None if control_vec != control_vec else control_vec
unfreeze_control_vec = os.getenv("unfreeze_control_vec", "false").lower() == "true"
logging_dir = os.getenv("logging_dir")
save_strategy = os.getenv("save_strategy", "steps")
logging_strategy = os.getenv("logging_strategy", "steps")
eval_strategy = os.getenv("eval_strategy", "steps")
r = int(os.getenv("r", 1))
target_modules = (os.getenv("target_modules", "down_proj")).split()

# generate
max_new_tokens = int(os.getenv("max_new_tokens", 1024))
prompt_suffix = os.getenv("prompt_suffix", "")
filename_tag = os.getenv("filename_tag", "")
overwrite = os.getenv("overwrite", "false").lower() == "true"
eval_step = os.getenv("eval_step")
eval_split = os.getenv("eval_split", "test")
update_prompt = os.getenv("update_prompt", "true").lower() == "true"
answer_only = os.getenv("answer_only", "false").lower() == "true"

##########################################################

import types

def to_dict():
    return {
        name: value
        for name, value in globals().items()
        if not name.startswith("_")
        and not callable(value)
        and not isinstance(value, type)
        and not isinstance(value, types.ModuleType)
    }
print(to_dict())