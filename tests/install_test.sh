#!/bin/bash

#TODO: update to your .bashrc path
source /home/jwt/.bashrc

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ElemeNet

echo "=== extra import checks ==="
python -c "import elemenet; from elemenet.trainer import canonical_loss_type; from molSimplify.Informatics.MOF.PBC_functions import readcif; print('elemenet + readcif import OK; alias:', canonical_loss_type('mop_binary_classification'))"
which elemenet_train elemenet_inference
echo "=== INSTALL TEST COMPLETE ==="

