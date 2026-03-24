#!/bin/bash
# cloud GPU setup script for transformer training
# run on a fresh GPU instance (Vast.ai, RunPod, Lambda)
#
# Prerequisites: NVIDIA GPU with CUDA, Python 3.11+
#
# Usage:
#   scp -r crystal-battle/ user@instance:~/
#   ssh user@instance
#   cd crystal-battle && bash cloud_setup.sh

set -e

echo "========================================"
echo "  Crystal Battle - Cloud GPU Setup"
echo "========================================"

# create venv
python3 -m venv .venv
source .venv/bin/activate

# install PyTorch with CUDA (auto-detects CUDA version)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# install project deps
pip install gymnasium sb3-contrib stable-baselines3 tensorboard numpy

# verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected!')
"

echo ""
echo "Setup complete. Run transformer training with:"
echo ""
echo "  .venv/bin/python training/train_transformer.py \\"
echo "    --total-steps 5000000 \\"
echo "    --device cuda \\"
echo "    --n-envs 8 \\"
echo "    --seq-len 32 \\"
echo "    --n-layers 3 \\"
echo "    --eval-freq 50000 \\"
echo "    --eval-games 100"
echo ""
echo "Or with imitation pre-training first:"
echo ""
echo "  # 1. generate imitation data"
echo "  .venv/bin/python training/imitation.py --generate --n-games 5000"
echo ""
echo "  # 2. pre-train transformer on Smart agent sequences"
echo "  .venv/bin/python training/train_transformer.py \\"
echo "    --pretrain --data expert_sequences.pkl --device cuda"
echo ""
echo "  # 3. fine-tune with PPO"
echo "  .venv/bin/python training/train_transformer.py \\"
echo "    --total-steps 5000000 --device cuda --resume transformer_pretrained"
echo ""
