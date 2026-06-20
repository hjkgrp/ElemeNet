#!/bin/bash

#TODO: update to your .bashrc path
source /home/jwt/.bashrc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ElemeNet

python - <<'PY'
import torch, torch_cluster, torch_geometric, elemenet
from molSimplify.Informatics.MOF.PBC_functions import readcif
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("torch_geometric", torch_geometric.__version__, "| torch_cluster", torch_cluster.__version__)
# exercise a GPU op + a torch_cluster call (EGNN radius graph)
x = torch.randn(2000, 3, device="cuda")
d = (x @ x.t()).sum()
from torch_cluster import radius_graph
ei = radius_graph(x, r=1.0, batch=torch.zeros(2000, dtype=torch.long, device="cuda"))
print("GPU matmul + radius_graph OK; edges:", ei.shape[1])
PY
echo "=== CUDA TEST COMPLETE ==="

