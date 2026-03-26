#!/bin/bash
# cloud GPU setup for crystal battle transformer training
# run on a Vast.ai / RunPod instance with PyTorch template
#
# Usage: cd crystal-battle && bash cloud_setup.sh

set -e

echo "========================================"
echo "  Crystal Battle - Cloud GPU Setup"
echo "========================================"

# create venv if not exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip -q
pip install gymnasium sb3-contrib stable-baselines3 tensorboard numpy -q

# verify GPU
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Setup complete. Options:"
echo ""
echo "  1) Pre-train + evaluate simple transformer:"
echo "     .venv/bin/python training/imitation.py --generate --n-games 5000"
echo "     .venv/bin/python pretrain_simple.py --pretrain --device cuda --epochs 30"
echo "     .venv/bin/python pretrain_simple.py --evaluate --device cuda --n-games 200"
echo ""
echo "  2) Full transformer PPO training (after pre-train):"
echo "     .venv/bin/python training/train_transformer.py \\"
echo "       --total-steps 5000000 --device cuda --seq-len 32 --n-layers 3"
echo ""
