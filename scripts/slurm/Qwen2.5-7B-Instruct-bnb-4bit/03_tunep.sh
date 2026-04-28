#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=JOB_NAME
#SBATCH --partition=PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --chdir=CHDIR
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EMAIL
#SBATCH --array=0-4

nvidia-smi
conda info --envs
pip show torch

export model_id="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
export cache_dir="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="0"

export max_new_tokens="1024"
export target_modules=""
export r="1"

export random_basis="false"
export unfreeze_control_vec="false"

export logging_steps="2"
export num_train_epochs="8"
export gradient_accumulation_steps="10"
export step_size="5e-1"
export control_coef="3.5"

#####################################################################################

jobs=(
  "method=rfm learning_rate=${step_size} steer_func=add-las_single"
  "method=rfm learning_rate=${step_size} steer_func=add-las_layerwise"
  "method=rfm learning_rate=3e-3 steer_func=add_dynamic-clas-r${r}"
  "method=rfm learning_rate=3e-3 steer_func=add_dynamic-reft-r${r} random_basis=true unfreeze_control_vec=true"
  "method=rfm learning_rate=3e-3 steer_func=add_dynamic-lora_mlp-r${r}"
)

eval "${jobs[$SLURM_ARRAY_TASK_ID]}"
export method learning_rate steer_func
python 03_tune.py