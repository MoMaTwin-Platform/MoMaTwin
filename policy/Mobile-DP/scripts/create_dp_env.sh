#!/bin/bash

# Exit on any error
set -e

echo "=== Creating and Configuring Diffusion Policy Environment ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if environment file exists
ENV_FILE="/x2robot_v2/wjm/prj/MoMaTwin/policy/Mobile-DP/scripts/dp_move.yml"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file not found: $ENV_FILE"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "dp_move"; then
    echo "Warning: Environment 'dp_move' already exists"
    read -p "Do you want to remove the existing environment and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n dp_move -y
    else
        echo "Skipping environment creation, installing torchcodec directly..."
        goto_install_torchcodec=true
    fi
fi

# Create environment (if not skipped)
if [ "$goto_install_torchcodec" != "true" ]; then
    echo "Creating conda environment 'dp_move'..."
    conda env create -f "$ENV_FILE" -n dp_move
    
    if [ $? -eq 0 ]; then
        echo "✓ Environment created successfully"
    else
        echo "✗ Environment creation failed"
        exit 1
    fi
fi

# Activate environment and install torchcodec
echo "Activating environment and installing torchcodec..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dp_move

# Check CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 | cut -d. -f1)
    echo "Detected CUDA driver version: $CUDA_VERSION"
    
    if [ "$CUDA_VERSION" -ge 12 ]; then
        TORCHCODEC_URL="https://download.pytorch.org/whl/cu121"
        echo "Using CUDA 12.1 compatible torchcodec"
    else
        TORCHCODEC_URL="https://download.pytorch.org/whl/cu118"
        echo "Using CUDA 11.8 compatible torchcodec"
    fi
else
    echo "No NVIDIA GPU detected, using CPU version"
    TORCHCODEC_URL="https://download.pytorch.org/whl/cpu"
fi

# Install torchcodec
echo "Installing torchcodec..."
pip install torchcodec --index-url="$TORCHCODEC_URL"

if [ $? -eq 0 ]; then
    echo "✓ torchcodec installed successfully"
else
    echo "✗ torchcodec installation failed"
    exit 1
fi

# Verify installation
echo "Verifying installation..."
python -c "import torchcodec; print('torchcodec version:', torchcodec.__version__)"

echo ""
echo "=== Environment Configuration Complete ==="
echo "Usage:"
echo "  conda activate dp_move"
echo "  python your_script.py"
echo ""
echo "Environment location: $(conda info --envs | grep dp_move | awk '{print $2}')"