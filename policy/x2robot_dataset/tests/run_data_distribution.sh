#!/bin/bash
#SBATCH --job-name=data_distribution_pickplace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40 #
#SBATCH --gres=gpu:0
#SBATCH -o %x-%j.log
#SBATCH --mem-per-cpu=8G

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
echo "GPU IDs: $CUDA_VISIBLE_DEVICES"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "Number of GPU: $num_gpus"
# set port range
PORT_RANGE_START=20000
PORT_RANGE_END=30000

# random port
PORT=$(shuf -i $PORT_RANGE_START-$PORT_RANGE_END -n 1)

echo "Randomly selected port: $PORT"
srun accelerate launch --num_processes=$num_gpus --main_process_port=$PORT show_data_distribute.py --dataset_file pickplace_dataset_paths.txt --output_dir ./output_pickplace
# srun accelerate launch --num_processes=$num_gpus --main_process_port=$PORT show_data_distribute.py --dataset_file leju.txt --parse_head_action --is_binocular --output_dir ./output_leju