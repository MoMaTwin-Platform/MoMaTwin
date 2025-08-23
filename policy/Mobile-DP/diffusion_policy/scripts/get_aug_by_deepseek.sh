#!/bin/bash
#SBATCH --job-name=aug_data_by_deepseek
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=50G
#SBATCH -o /x2robot/ganruyi/workspace/diffusion_policy/logs/%x-%j.log

export HYDRA_FULL_ERROR=1

srun python ~/code/get_aug_by_deepseek.py