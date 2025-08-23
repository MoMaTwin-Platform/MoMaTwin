#!/bin/bash
#SBATCH --job-name=pick_waste_exhibitation
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=124
#SBATCH --gres=gpu:8
#SBATCH --mem=1600G
#SBATCH --exclude=master
#SBATCH -o /x2robot_v2/wjm/prj/diffusion_policy_logs/%x-%j.log

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export NOW=$(date +"%Y.%m.%d/%H.%M.%S")
# wandb setting
export WANDB_PROXY="http://10.7.145.219:3128"
export HTTPS_PROXY="http://10.7.145.219:3128"
export HTTP_PROXY="http://10.7.145.219:3128"
# wandb setting
export WANDB_ENTITY='x2robot'
export WANDB_API_KEY=300b652d395da1ab9fe659414c443c4e9f8c7886
export NCCL_IB_DISABLE=0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3 
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_TC=184
export NCCL_IB_TIMEOUT=23
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Add FFmpeg library path for torchcodec
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "GPU IDs: $CUDA_VISIBLE_DEVICES"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
export GPUS_PER_NODE=$num_gpus

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#head_node_ip="10.60.77.151"
echo "head_node_ip: $head_node_ip"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo $((SLURM_NNODES * GPUS_PER_NODE))
######################

# Use a fixed port instead of a random one
export PORT=$((21000 + $RANDOM % 30000))

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=10231 # use 5 digits ports

NNODES=$SLURM_NNODES
all_gpus=$((SLURM_NNODES * GPUS_PER_NODE))
echo "all_gpus: $all_gpus"
# no config_file for ddp
export LAUNCHER="accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $all_gpus \
    --num_machines $NNODES \
    "
export SCRIPT="/x2robot_v2/wjm/prj/diffusion_policy/train.py"
export SCRIPT_ARGS="--config-name=pick_waste_exhibitation"

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
srun --jobid $SLURM_JOBID bash -c "$CMD"