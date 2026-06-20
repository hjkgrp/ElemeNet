import numpy as np
import os
import pandas as pd
import random
import time
import torch
from torch.utils.data import get_worker_info
import ast
from typing import List, Optional
import json
from dataclasses import is_dataclass, asdict


def wait_for_file(path, poll_interval=5):
    """Poll until a file appears on disk. Used to sync ranks without holding an NCCL connection."""
    while not os.path.exists(path):
        time.sleep(poll_interval)


class DataclassEncoder(json.JSONEncoder):
    """JSON encoder that serializes dataclass instances as dicts."""

    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy scalars and arrays to native Python types."""

    def default(self, obj):
        # numpy scalars
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        # numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def print_metrics(
    metrics: List[dict], target_columns: List[str], loss: float, split: str
):
    """Print per-target metrics to stdout for a given data split.

    Parameters
    ----------
    metrics : List[dict]
        List of metric dictionaries, one per target column.
    target_columns : List[str]
        Names of target columns, used as labels in output.
    loss : float
        Aggregate loss value for the split.
    split : str
        Name of the split (e.g. 'train', 'val', 'test').
    """
    print(f"\nMetrics for {split} set (loss: {loss}):")
    for i, metric in enumerate(metrics):
        print(f"Metrics for target {target_columns[i]}:")
        for key, value in metric.items():
            print(f"  {key}: {value}")


def normalize_to_float_list(x):
    """Coerce a scalar, numeric string, or list-like value into a list of floats.

    Handles int, float, list, and string representations of lists. Raises
    ``ValueError`` for NaN inputs or values that cannot be converted.

    Parameters
    ----------
    x : int, float, str, or list
        Value to normalize.

    Returns
    -------
    list of float
    """
    # treat NaN as empty or placeholder
    if pd.isna(x):
        raise ValueError(
            "Encountered NaN value when expecting float or list of floats."
        )

    # if already a list, ensure all elements are float
    if isinstance(x, list):
        return [float(v) for v in x]

    # if a number, wrap in list
    if isinstance(x, (int, float)):
        return [float(x)]

    # if a string representing a list, try parsing
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)  # safely parse list-like strings
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
            else:
                return [float(parsed)]
        except:
            # try converting to float directly
            try:
                return [float(x)]
            except Exception as e:
                raise ValueError(f"Cannot convert string to float list: {x}") from e

    # fallback to empty list
    raise ValueError(f"Cannot convert value to float list: {x}")


def reconstruct_list_columns(
    original_df: pd.DataFrame, target_columns: List[str], flat_array: np.ndarray
) -> pd.DataFrame:
    """
    Reconstruct list-of-lists columns from a flattened array.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original dataframe containing list-of-lists in the target columns.
    target_columns : List[str]
        The columns to reconstruct.
    flat_array : np.ndarray
        The flattened array containing concatenated elements from all rows.
        Shape should be (sum(len(l) for row in all rows), n_columns)

    Returns
    -------
    pd.DataFrame
        DataFrame with the same shape as original_df[target_columns], with lists restored.
    """
    reconstructed = pd.DataFrame(index=original_df.index, columns=target_columns)

    for col_idx, col in enumerate(target_columns):
        arr = flat_array[:, col_idx]
        idx = 0
        col_list = []
        for row_list in original_df[col]:
            l = len(row_list)
            col_list.append(arr[idx : idx + l].tolist())
            idx += l
        reconstructed[col] = col_list

    return reconstructed


def set_seed(random_seed):
    """Set random seed for Python, NumPy, and PyTorch (including all CUDA devices).

    Parameters
    ----------
    random_seed : int
        Seed value to apply globally.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_workers(worker_id):
    """Seed a DataLoader worker subprocess for reproducible data loading.

    Designed for use as the ``worker_init_fn`` argument of
    ``torch.utils.data.DataLoader``. Each worker derives its seed from the
    global worker seed set by PyTorch, so workers remain independent while
    being deterministic across runs.

    Parameters
    ----------
    worker_id : int
        Worker index passed automatically by DataLoader (unused directly;
        seed is read from ``get_worker_info()``).
    """
    worker_info = get_worker_info()
    seed = worker_info.seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def estimate_size(model):
    """Estimate the memory footprint of model parameters in megabytes.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters to size.

    Returns
    -------
    float
        Total parameter memory in MB.
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024**2)


def estimate_training_memory(model_config, batch_size, param_bytes=4):
    """Estimate peak GPU memory required during training.

    Accounts for encoder and readout parameter storage, optimizer state,
    activation memory, and EGNN message/coordinate buffers, plus a 2 GB
    fixed buffer. Also queries the total memory of GPU 0 for comparison.

    Parameters
    ----------
    model_config : dict
        Model configuration containing ``encoder_config`` and ``readout_config``
        sub-dicts with keys such as ``layers``, ``neurons``, ``inv_sublayers``,
        ``max_nodes``.
    batch_size : int
        Number of graphs per training batch.
    param_bytes : int, optional
        Bytes per parameter (default 4 for float32).

    Returns
    -------
    total_gb : float
        Estimated peak memory requirement in GB.
    gpu_ram : float
        Total memory of GPU 0 in GB, or 0 if CUDA is unavailable.
    """
    # encoder parameters
    encoder = model_config.get("encoder_config", {})
    enc_layers = encoder.get("layers", 0)
    enc_neurons = encoder.get("neurons", 0)
    enc_sublayers = encoder.get("inv_sublayers", 0)
    # graph size
    max_nodes = encoder.get("max_nodes", 0)
    nodes_per_batch = max_nodes * batch_size
    max_edges = max_nodes**2
    edges_per_batch = max_edges * batch_size
    # estimate as layer weight matrices + biases + inv_sublayers
    param_mem = enc_layers * enc_neurons**2 * param_bytes * (1 + enc_sublayers)

    # readout parameters
    readout = model_config.get("readout_config", {})
    read_layers = readout.get("layers", 0)
    read_neurons = readout.get("neurons", 0)
    # estimate for fully connected layers
    param_mem += read_layers * read_neurons**2 * param_bytes

    # optimizer memory
    optimizer_mem = 2 * param_mem

    # activation memory
    encoder_act_mem = (
        enc_layers
        * edges_per_batch
        * enc_neurons
        * param_bytes
        * 2
        * (1 + enc_sublayers)
    )
    readout_act_mem = read_layers * nodes_per_batch * read_neurons * param_bytes * 2

    # message/coordinate memory
    message_mem = (
        enc_layers * edges_per_batch * enc_neurons * param_bytes * (1 + enc_sublayers)
    )

    # finalize estimate
    buffer_gb = 2.0
    total_bytes = (
        param_mem + optimizer_mem + encoder_act_mem + readout_act_mem + message_mem
    )
    total_gb = total_bytes / (1024**3) + buffer_gb

    # estimate available memory
    if torch.cuda.is_available():
        gpu_id = 0  # adjust if using multiple GPUs
        gpu_ram = torch.cuda.get_device_properties(gpu_id).total_memory
        gpu_ram = gpu_ram / (1024**3)  # convert bytes to GB
    else:
        gpu_ram = 0

    return total_gb, gpu_ram


def save_predictions(
    targets: np.ndarray,
    preds: np.ndarray,
    preds_uncertainties: np.ndarray,
    label_column: str,
    labels: np.ndarray,
    target_columns: list,
    split: str,
    task: str,
    num_classes: int,
    save_dir: str,
    metrics: Optional[dict] = None,
    uncertainty_metrics: Optional[dict] = None,
    preds_logit_std: Optional[np.ndarray] = None,
):
    """Save predictions (and optionally metrics) for a given split to disk.

    Writes a CSV of true labels, predicted values, and uncertainty columns to
    ``<save_dir>/<split>_predictions.csv``. If ``metrics`` is provided, also
    writes ``<save_dir>/<split>_metrics.json``. Supports multi-target regression
    and binary/multi-class classification.

    Parameters
    ----------
    targets : np.ndarray
        Ground-truth target values, shape ``(N,)`` or ``(N, T)``.
    preds : np.ndarray
        Model predictions, shape ``(N,)`` or ``(N, T)`` for regression/binary
        classification, or ``(N, C)`` for multi-class classification.
    preds_uncertainties : np.ndarray or None
        Prediction uncertainties (e.g. ensemble std). Pass ``None`` to omit.
    label_column : str
        Name of the identifier column in the output CSV.
    labels : np.ndarray
        Sample identifiers aligned with predictions.
    target_columns : list of str
        Names of the target property columns.
    split : str
        Data split name (e.g. ``'train'``, ``'val'``, ``'test'``).
    task : str
        Learning task, either ``'regression'`` or ``'classification'``.
    num_classes : int
        Number of classes (used to distinguish binary vs. multi-class output).
    save_dir : str
        Directory in which to write output files.
    metrics : dict or None, optional
        Per-target metric dictionaries to serialize as JSON.
    uncertainty_metrics : dict or None, optional
        Additional uncertainty metrics (e.g. latent-space distances, entropy)
        keyed by metric name. Values are 1-D or 2-D arrays aligned with
        predictions.

    Returns
    -------
    pd.DataFrame
        DataFrame of all saved columns.
    """

    # Prediction-only output: no ground-truth targets, and possibly no target
    # column names. Synthesize generic, prediction-width-sized names so the
    # saved CSV still has one ``*_predicted`` column per output (works for
    # single- and multi-target regression / binary classification).
    has_targets = targets is not None
    if target_columns is None:
        if task == "classification" and num_classes is not None and num_classes > 2:
            target_columns = ["target_0"]
        else:
            n_cols = preds.shape[1] if preds.ndim > 1 else 1
            target_columns = [f"target_{i}" for i in range(n_cols)]

    results = pd.DataFrame({label_column: labels})

    # Naming convention:
    #   {col}_true       -- ground-truth target
    #   {col}_predicted  -- ensemble-mean prediction in original space
    #                       (real-valued for regression, probability for
    #                       binary classification)
    #   {col}_std        -- ensemble std in the same space
    # For multi-class, per-class statistics are suffixed:
    #   {col}_predicted_class_{c} / {col}_std_class_{c}
    # For classification, an optional logit-space std (sigma_l, the ensemble
    # std before sigmoid/softmax) is written when ``preds_logit_std`` is
    # provided: {col}_logit_std (binary) or {col}_logit_std_class_{c} (multi).
    # sigma_l retains ensemble disagreement information that sigma_p loses to
    # sigmoid saturation.

    if task == "regression":
        for i, col in enumerate(target_columns):
            if has_targets:
                results[f"{col}_true"] = targets[:, i]
            results[f"{col}_predicted"] = preds[:, i]
            if preds_uncertainties is not None and preds_uncertainties.size > 0:
                if preds_uncertainties.ndim == 1:
                    results[f"{col}_std"] = preds_uncertainties
                else:
                    results[f"{col}_std"] = preds_uncertainties[:, i]

    elif task == "classification":
        if has_targets and targets.ndim == 1:
            targets = targets[:, None]
        if preds.ndim == 1:
            preds = preds[:, None]
        for ti, col in enumerate(target_columns):
            # true labels
            if has_targets:
                results[f"{col}_true"] = targets[:, ti]

            if num_classes > 2 and len(target_columns) > 1:
                raise NotImplementedError(
                    "Multi-class classification does not support multi-target prediction."
                )
            if num_classes > 2:
                for c in range(num_classes):
                    results[f"{col}_predicted_class_{c}"] = preds[:, c]
                    if preds_uncertainties is not None and preds_uncertainties.size > 0:
                        results[f"{col}_std_class_{c}"] = preds_uncertainties[:, c]
                    if preds_logit_std is not None and preds_logit_std.size > 0:
                        results[f"{col}_logit_std_class_{c}"] = preds_logit_std[:, c]
            else:
                results[f"{col}_predicted"] = preds[:, ti]
                if preds_uncertainties is not None:
                    if preds_uncertainties.ndim == 1:
                        results[f"{col}_std"] = preds_uncertainties
                    else:
                        results[f"{col}_std"] = preds_uncertainties[:, ti]
                if preds_logit_std is not None and preds_logit_std.size > 0:
                    if preds_logit_std.ndim == 1:
                        results[f"{col}_logit_std"] = preds_logit_std
                    else:
                        results[f"{col}_logit_std"] = preds_logit_std[:, ti]
    else:
        raise ValueError(f"Unknown task type: {task}")

    # add uncertainty metrics (latent space distances, entropy, confidence, etc.)
    if uncertainty_metrics is not None:
        for key, values in uncertainty_metrics.items():
            if values.ndim == 1:
                results[key] = values
            elif values.ndim == 2:
                for i, col in enumerate(target_columns):
                    results[f"{col}_{key}"] = values[:, i]

    # save
    save_path = f"{save_dir}/{split}_predictions.csv"
    results.to_csv(save_path, index=False)

    if metrics is not None:
        metrics_path = f"{save_dir}/{split}_metrics.json"
        combined_metrics = {}
        for ti, col in enumerate(target_columns):
            combined_metrics[col] = metrics[ti]
        with open(metrics_path, "w") as f:
            json.dump(combined_metrics, f, cls=NumpyEncoder, indent=4)
        print(f"Saved metrics to {metrics_path}")

    print(f"Saved {task} predictions to {save_path}")
    print("Thank you for choosing ElemeNet!")

    return results
