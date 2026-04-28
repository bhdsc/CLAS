#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=JOB_NAME
#SBATCH --partition=PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --chdir=CHDIR
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EMAIL
#SBATCH --array=0-7

nvidia-smi
conda info --envs
pip show torch

export model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
export cache_dir="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="0"

export max_new_tokens="1024"
export target_modules=""
export r="1"

#####################################################################################

export control_coef="nan"
export eval_split="test"

jobs=(
  "method=rfm steer_func=add-search-las_single"
  "method=rfm steer_func=add-las_single"
  "method=rfm steer_func=add-las_layerwise"
  "method=rfm steer_func=add_dynamic-clas-r${r}"
  "method=rfm steer_func=add_dynamic-reft-r${r}"
  "method=rfm steer_func=add_dynamic-lora_mlp-r${r}"
  "method=rfm steer_func=add-prompt_0 control_coef=0"
  "method=rfm steer_func=add-prompt_1 control_coef=0"
)

eval "${jobs[$SLURM_ARRAY_TASK_ID]}"
export method steer_func control_coef
python 04_generate.py