#!/bin/bash
# One-shot installer for ElemeNet (GNN + EGNN). Run from the repo root after `git clone https://github.com/hjkgrp/ElemeNet && cd ElemeNet`: bash install/install.sh

# It creates the conda env, installs PyTorch and the ElemeNet package, and the torch_cluster PyG extension the EGNN models need.
# If the prebuilt torch_cluster wheel can't be imported (typically a GLIBC mismatch on older clusters), it rebuilds it from source automatically.

# Configuration (export before running to override):
#   CUDA: PyTorch/PyG CUDA build matching your driver, or "cpu" for CPU-only. Pick the tag from https://pytorch.org/get-started/locally/ . Default cu126.
#   TORCH_CUDA_ARCH_LIST: GPU arch(s) for the source-build fallback only, e.g. "8.0" A100 | "8.9" L40S/RTX40xx | "9.0" H100/H200 | "12.0+PTX" Blackwell.

# The source-build fallback needs a CUDA toolkit (nvcc) on PATH. On module-based clusters, this is usually `module load cuda/<ver>`.
# Submit through your scheduler if desired, e.g.: sbatch --gres=gpu:1 -c 8 --mem=32G -t 4:00:00 install/install.sh

set -eo pipefail

CUDA="${CUDA:-cu126}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0+PTX}"

conda env create --file=install/ElemeNet.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ElemeNet

# PyTorch wheels bundle their own CUDA runtime, so only a matching NVIDIA driver is required (no separate CUDA toolkit install).
pip install torch==2.8.0 --index-url "https://download.pytorch.org/whl/${CUDA}"

# ElemeNet itself (also pulls torch_geometric and the remaining dependencies).
pip install -e .
pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel tensorflow-base

# torch_cluster (EGNN radius graphs): try the prebuilt wheel, then fall back to a source build against the system GLIBC if the wheel can't be imported.
pip install torch_cluster -f "https://data.pyg.org/whl/torch-2.8.0+${CUDA}.html" || true
if ! python -c "import torch_cluster" 2>/dev/null; then
    echo ">> Prebuilt torch_cluster unusable (likely a GLIBC mismatch). Building from source..."
    export FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-8}"
    pip uninstall -y torch-cluster
    pip cache purge
    pip install torch-cluster==1.6.3 --no-binary :all: --no-build-isolation --no-cache-dir
fi

python - <<'PY'
import torch, torch_cluster
print(f"ElemeNet install OK: torch {torch.__version__}, "
      f"CUDA available: {torch.cuda.is_available()}, "
      f"torch_cluster {torch_cluster.__version__}")
PY
