#!/bin/bash
#SBATCH -p lem-gpu-normal
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=80GB
#SBATCH -t 6:00:00
#SBATCH -J pyscf_test
#SBATCH --gres=gpu:hopper:1

# Initialize conda (without loading any modules)
source /home/<user>/miniconda3/etc/profile.d/conda.sh  # Adjust path as needed
conda activate jaxsce_env

# Set JAX GPU options
export JAX_PLATFORM_NAME=gpu
export CUDA_VISIBLE_DEVICES=0

# Run your script
python -u star.py --density "ccsd(t)"
