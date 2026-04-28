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

nvidia-smi
conda info --envs
pip show torch

export model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
export cache_dir="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="0"

#####################################################################################

probe() {
    export compute_directions="true"
    python 01_probe.py

    export compute_directions="false"
    python 01_probe.py
}
export method="rfm"
probe
