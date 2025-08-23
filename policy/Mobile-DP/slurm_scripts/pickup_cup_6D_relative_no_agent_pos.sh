#!/bin/bash
#SBATCH --job-name=pr_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=124
#SBATCH --gres=gpu:8
#SBATCH --mem=1600G 
#SBATCH --exclude=master
#SBATCH -o /x2robot/xinyuanfang/projects/diffusion_policy/workspace/pickup_cup_raw/%x-%j.log

export WANDB_PROXY="http://10.7.145.219:3128"
export HTTPS_PROXY="http://10.7.145.219:3128"
export HTTP_PROXY="http://10.7.145.219:3128"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
# wandb setting
export WANDB_ENTITY='x2robot'

# Simplified NCCL settings for single node
export NCCL_DEBUG=INFO                    # Enable detailed logging
export NCCL_IB_DISABLE=1                  # Disable InfiniBand for single node
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

echo "GPU IDs: $CUDA_VISIBLE_DEVICES"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
export GPUS_PER_NODE=$num_gpus

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "head_node_ip: $head_node_ip"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo $((SLURM_NNODES * GPUS_PER_NODE))

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=10231

NNODES=$SLURM_NNODES
all_gpus=$((SLURM_NNODES * GPUS_PER_NODE))
echo "all_gpus: $all_gpus"

export LAUNCHER="accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $all_gpus \
    --num_machines $NNODES \
    --mixed_precision="no" \
    "
export SCRIPT="../train.py"
export SCRIPT_ARGS="--config-name=pickup_cup_6D_relative_no_agent_pos"

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
set -e 
srun --kill-on-bad-exit=1 --jobid $SLURM_JOBID bash -c "$CMD"
