# ElemeNet Tutorials

Short walkthroughs of the three core workflows. Each assumes you have already installed ElemeNet (see the top-level [README](../README.md)) and activated the conda environment:

```bash
conda activate ElemeNet
```

| Tutorial | Content |
|----------|----------------|
| [Tutorial 1: Training](01_training.ipynb) | Train a model two ways: the `training_pipeline` Python API and the `elemenet_train` CLI. |
| [Tutorial 2: Fine-tuning](02_finetuning.ipynb) | Continue training from a checkpoint. Explains the difference between resuming (`--resume`) and transfer-learning (`--transfer_learn`). |
| [Tutorial 3: Inference](03_inference.ipynb) | Run a trained model on new data with `elemenet_inference`. |

Note that the paths used in the examples should be adapted to your own workflows.