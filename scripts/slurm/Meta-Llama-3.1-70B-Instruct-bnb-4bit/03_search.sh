#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=JOB_NAME
#SBATCH --partition=PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --chdir=CHDIR
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EMAIL

nvidia-smi
conda info --envs
pip show torch

export model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
export cache_dir="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="0"

export max_new_tokens="1024"
export target_modules=""
export r="1"

export logging_steps="2"
export num_train_epochs="20"
export gradient_accumulation_steps="10"
export step_size="5e-2"

#####################################################################################

search() {
    export eval_split="train"
    steer_func="add-search-las_single" python 03_search.py
}
export method="rfm"
search
