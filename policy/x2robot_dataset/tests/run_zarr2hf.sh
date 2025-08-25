#!/bin/bash
#SBATCH --job-name=lazy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60 #
#SBATCH --array=0-2
## SBATCH --gres=gpu:0
#SBATCH -o task_%A_%a.log
#SBATCH -e task_%A_%a.err
#SBATCH --mem-per-cpu=8G

export NUM_TASKS=3
srun python zarr2hf.py





