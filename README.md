# ElemeNet
ElemeNet is a suite of end-to-end machine learning workflows for molecular property prediction across the periodic table. Built upon PyTorch Geometric, we offer advanced 2D and 3D architectures, representations, and loss functions for state-of-the-art performance in predicting atom, bond, molecule, and moiety properties.

### Installation
PyTorch ships its own bundled CUDA runtime, so you only need a recent NVIDIA driver. There is no need to install a system CUDA toolkit.

**Quick install** Clone the repo and run the installer:
```
git clone https://github.com/hjkgrp/ElemeNet
cd ElemeNet
bash install/install.sh
```
The above command defaults to CUDA 12.6. For a different CUDA build (or a CPU-only build), set `CUDA` to the
tag matching your driver (from the [PyTorch selector](https://pytorch.org/get-started/locally/)).
For example, to install for CUDA 12.9:
```
git clone https://github.com/hjkgrp/ElemeNet
cd ElemeNet
CUDA=cu129 bash install/install.sh
```

or to install for CPU only:
```
git clone https://github.com/hjkgrp/ElemeNet
cd ElemeNet
CUDA=cpu bash install/install.sh
```

**Manual install.** Equivalent to running the install.sh script:
1. `git clone https://github.com/hjkgrp/ElemeNet`
2. `cd ElemeNet`
3. `conda env create --file=install/ElemeNet.yml`
4. `conda activate ElemeNet`
5. `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`
6. `pip install -e .`
7. `pip install torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu126.html`
8. `pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel tensorflow-base`

### Testing
To ensure ElemeNet was successfully installed, simply run the provided test scripts:
```
cd ElemeNet
bash tests/install_test.sh
bash tests/cuda_test.sh
```

### Tutorials
Worked examples of the core workflows live in [`tutorials/`](tutorials/) as Jupyter
notebooks (run them with the `ElemeNet` environment as the kernel):
- [`01_training.ipynb`](tutorials/01_training.ipynb) — train a model via the `training_pipeline` Python API and the `elemenet_train` CLI.
- [`02_finetuning.ipynb`](tutorials/02_finetuning.ipynb) — continue training from a checkpoint.
- [`03_inference.ipynb`](tutorials/03_inference.ipynb) — run a trained model on new data with `elemenet_inference`.

### Troubleshooting
- **Different GPU / driver, or no GPU.** Match the `cuXXX` (or `cpu`) tag in the torch and
  torch_cluster commands to your driver via the
  [PyTorch selector](https://pytorch.org/get-started/locally/) and the
  [CUDA compatibility tables](https://en.wikipedia.org/wiki/CUDA#GPUs_supported). The CUDA
  build only has to be supported by your installed driver; no separate CUDA toolkit is needed.
- **`torch_cluster` import error mentioning GLIBC.** The prebuilt wheel was compiled against
  a newer GLIBC than your system. `install/install.sh` rebuilds it from source automatically;
  to do it by hand (with a CUDA toolkit / `nvcc` on PATH):
  ```
  export TORCH_CUDA_ARCH_LIST="8.9;9.0"   # your GPU architecture(s)
  pip uninstall -y torch-cluster && pip cache purge
  FORCE_CUDA=1 pip install torch-cluster==1.6.3 --no-binary :all: --no-build-isolation --no-cache-dir
  ```
- **PyG installation reference:** https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
