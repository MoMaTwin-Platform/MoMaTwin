#!/bin/bash
#SBATCH --job-name=lazy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24 #
#SBATCH --gres=gpu:4
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err
#SBATCH --mem-per-cpu=10G

#python test_lerobot_dataset.py
#accelerate launch --num_processes 2 --main_process_port 29876 test_lerobot_dataset.py #test_map_dataset_buffer_acc.py
accelerate launch --num_processes 4 --main_process_port 28373 test_lazy_dataset.py





