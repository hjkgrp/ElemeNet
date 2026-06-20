import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch
from tqdm import tqdm
from typing import Optional


def _compute_batch_graph_ids(data, scope: str) -> torch.Tensor:
    """Graph index per prediction within a single PyG batch.

    Mirrors the masking applied in ``Model.forward`` so the returned tensor is
    aligned 1:1 with ``prediction_mean`` after ``subselect_node_embeddings`` /
    ``subselect_edge_embeddings``.
    """
    if scope == "graph":
        return torch.arange(int(data.num_graphs), device=data.batch.device)
    if scope == "node":
        mask = getattr(data, "node_mask", None)
        if mask is None:
            return data.batch
        return data.batch[mask.squeeze(-1)]
    if scope == "edge":
        edge_graph_ids = data.batch[data.edge_index[0]]
        mask = getattr(data, "center_edge_mask", None)
        if mask is None:
            mask = getattr(data, "edge_mask", None)
        if mask is not None:
            edge_graph_ids = edge_graph_ids[mask]
        return edge_graph_ids
    raise ValueError(f"Unknown scope: {scope}")


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    return_graph_ids: bool = False,
):
    """
    Load a trained model and make predictions

    INPUTS
        model: torch.nn.Module
            Trained model.
        loader: DataLoader
            DataLoader object for the dataset.
            default=None
        return_graph_ids: bool
            If True, also return a 1-D tensor of graph indices aligned with
            each prediction (used for graph-grouped metrics like
            molecular_accuracy on node/edge tasks). Default False.

    OUTPUTS
        outputs: torch.Tensor
            ``model.prediction_mean`` per sample, shape ``(N, D)``. For an
            ensemble this is the per-head mean of raw outputs (logit space
            for classification, real space for regression).
        uncertainties: torch.Tensor
            ``model.prediction_std`` per sample, shape ``(N, D)`` (per-head
            std in raw output space).
        raw: torch.Tensor
            ``model.prediction_raw`` per sample, shape ``(N, D, n_ens)``;
            the per-head outputs needed by ensemble-aware losses (MoP) and
            for probability-space aggregation at evaluation time.
        embeddings: torch.Tensor
            Latent space embeddings.
        targets: torch.Tensor
            Stacked ground-truth targets aligned with predictions.
        graph_ids: torch.Tensor (only if return_graph_ids=True)
            Global graph index per prediction.
    """
    # use GPU if available
    was_training = model.training
    model.eval()
    _m = model.module if hasattr(model, 'module') else model
    device = next(_m.parameters()).device
    scope = getattr(_m, "scope", "graph")

    outputs_list, uncertainty_list, raw_list = [], [], []
    embeddings_list, targets_list = [], []
    graph_ids_list = []
    graph_offset = 0
    for batch in tqdm(loader, desc="Making Predictions"):
        batch = batch.to(device)
        out = model(batch)
        outputs_list.append(out.prediction_mean.detach())
        uncertainty_list.append(out.prediction_std.detach())
        raw_list.append(out.prediction_raw.detach())
        embeddings_list.append(out.embeddings.detach())
        targets_list.append(batch.y.detach())
        if return_graph_ids:
            local_ids = _compute_batch_graph_ids(out, scope)
            graph_ids_list.append(local_ids + graph_offset)
            graph_offset += int(out.num_graphs)

    outputs = torch.cat(outputs_list, dim=0)
    uncertainties = torch.cat(uncertainty_list, dim=0)
    raw = torch.cat(raw_list, dim=0)
    embeddings = torch.cat(embeddings_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    # restore training mode
    if was_training:
        model.train()

    if return_graph_ids:
        graph_ids = torch.cat(graph_ids_list, dim=0)
        return outputs, uncertainties, raw, embeddings, targets, graph_ids
    return outputs, uncertainties, raw, embeddings, targets


def evaluate(
    outputs: torch.Tensor,
    uncertainties: torch.Tensor,
    raw: torch.Tensor,
    targets: torch.Tensor,
    task: str,
    loss_function: torch.nn.Module,
    scaler=None,
    num_classes: Optional[int] = None,
    scope: str = "graph",
    graph_ids: Optional[torch.Tensor] = None,
    has_targets: bool = True,
):
    """Evaluate performance of a trained model.

    For classification, predictions and uncertainties are aggregated in
    probability space (mean-of-probabilities): ``p_bar = mean(sigmoid(raw))``
    or ``mean(softmax(raw))`` per sample, with ensemble std taken in the
    same probability space. This matches the MoP loss training objective.
    For regression, ``outputs`` and ``uncertainties`` (already in real
    space) are used directly.

    INPUTS
        outputs: torch.Tensor
            Per-sample mean of raw model outputs across ensemble heads,
            shape ``(N, D)``. Logits for classification, real-space
            predictions for regression.
        uncertainties: torch.Tensor
            Per-sample std of raw model outputs across ensemble heads,
            shape ``(N, D)``.
        raw: torch.Tensor
            Per-head model outputs, shape ``(N, D, n_ens)``. Required for
            both ensemble-aware losses (MoP) and probability-space
            aggregation at evaluation time.
        targets: torch.Tensor
            Ground-truth targets.
        task: str
            ``'regression'`` or ``'classification'``.
        loss_function: torch.nn.Module
            Loss function (a ``LossWrapper``).
        scaler: sklearn StandardScaler, optional
            Inverse-transform for regression predictions. ``None`` skips
            inverse-scaling.
        num_classes: int, optional
            Number of classes (required for classification).

    OUTPUTS
        predictions_transformed: np.ndarray
            Predictions in original space (probabilities for classification,
            inverse-scaled values for regression).
        uncertainties_transformed: np.ndarray
            Predictive std in the same space as ``predictions_transformed``.
        transformed_targets: np.ndarray
            Targets in original space.
        loss: float
            Per-sample mean loss in scaled / logit space.
        metrics: list of dict
            Per-target performance metrics.
    """
    if task == "classification":
        assert (
            num_classes is not None
        ), "Number of classes must be specified for classification tasks."

    # build a Data-like object so the LossWrapper can route to MoP losses
    # (which need prediction_raw) as well as the older NLL/BCE/MSE paths.
    # Skipped entirely for prediction-only runs (no ground-truth targets).
    if has_targets:
        from torch_geometric.data import Data
        batch_for_loss = Data()
        batch_for_loss.prediction_mean = outputs
        batch_for_loss.prediction_std = uncertainties
        batch_for_loss.prediction_raw = raw
        batch_for_loss.y = targets
        loss = loss_function(batch_for_loss).item()
    else:
        loss = None

    # get metrics in original space (and on CPU)
    if task == "regression":
        predictions_transformed = outputs.detach().cpu().numpy()
        uncertainties_transformed = uncertainties.detach().cpu().numpy()
        transformed_targets = targets.detach().cpu().numpy() if has_targets else None
        # inverse scale predictions and uncertainties
        if scaler is not None:
            predictions_transformed = scaler.inverse_transform(predictions_transformed)
            uncertainties_transformed = uncertainties_transformed * scaler.scale_
            if has_targets:
                transformed_targets = scaler.inverse_transform(transformed_targets)
        # For regression, the model output IS the prediction (no nonlinearity),
        # so there is no separate "logit space" uncertainty.
        logit_uncertainties_transformed = None

    elif task == "classification":
        transformed_targets = targets.detach().cpu().numpy() if has_targets else None
        # Mean-of-probabilities aggregation: p_bar = mean_i(sigma(l_i)) and
        # sigma_p = std_i(sigma(l_i)). This is the calibrated probability
        # space estimate, and is correct for MoP-trained ensembles. For
        # ensemble_size=1 the mean is the single head's probability and the
        # std is defined as zero (single sample -> no spread).
        #
        # We also expose the logit-space std (sigma_l = std_i(l_i)) under
        # logit_uncertainties_transformed. sigma_p is correct for calibrated
        # probability uncertainty but is structurally crushed by sigmoid /
        # softmax saturation (sigma_p ≈ [p(1-p)]^2 sigma_l^2 by the delta
        # method). sigma_l retains ensemble disagreement information even in
        # the saturated regime, which is useful for OOD-detection analysis.
        if num_classes == 2:
            p_per_head = torch.sigmoid(raw)               # (N, D, n_ens)
            p_bar = p_per_head.mean(dim=-1)               # (N, D)
        else:
            p_per_head = torch.softmax(raw, dim=1)        # (N, K, n_ens)
            p_bar = p_per_head.mean(dim=-1)               # (N, K)
        if raw.shape[-1] > 1:
            sigma_p = p_per_head.std(dim=-1)              # Bessel-corrected std
        else:
            sigma_p = torch.zeros_like(p_bar)             # ensemble_size=1: no spread
        predictions_transformed = p_bar.detach().cpu().numpy()
        uncertainties_transformed = sigma_p.detach().cpu().numpy()
        # logit-space std is already the model's per-head std (`uncertainties`
        # passed in here is model.prediction_std = std_i(raw_i)). Keep it
        # alongside sigma_p for downstream calibration / OOD analysis.
        logit_uncertainties_transformed = uncertainties.detach().cpu().numpy()

    # Prediction-only path: without ground-truth targets there is no loss or
    # metrics to compute, so return the (correctly transformed) predictions and
    # leave targets/metrics as None.
    if not has_targets:
        return (
            predictions_transformed,
            uncertainties_transformed,
            logit_uncertainties_transformed,
            None,
            loss,
            None,
        )

    if task == "regression":
        diff = predictions_transformed - transformed_targets
        mae = np.mean(np.abs(diff), axis=0)
        mse = np.mean(diff**2, axis=0)
        rmse = np.sqrt(mse)

        metrics = []
        for i in range(predictions_transformed.shape[1]):
            metrics.append(
                {"mae": mae[i].item(), "mse": mse[i].item(), "rmse": rmse[i].item()}
            )

    elif task == "classification":
        # precompute graph grouping for molecular_accuracy on node/edge tasks
        compute_mol_acc = scope in ("node", "edge") and graph_ids is not None
        if compute_mol_acc:
            gid_np = (
                graph_ids.detach().cpu().numpy()
                if isinstance(graph_ids, torch.Tensor)
                else np.asarray(graph_ids)
            )
            _, gid_inverse = np.unique(gid_np, return_inverse=True)
            n_per_graph = np.bincount(gid_inverse)

        if num_classes == 2:
            metrics = []
            for i in range(predictions_transformed.shape[1]):
                round_preds = np.round(predictions_transformed[:, i])
                accuracy = np.mean(round_preds == transformed_targets[:, i])
                balanced_accuracy = balanced_accuracy_score(
                    transformed_targets[:, i], round_preds
                )
                precision = precision_score(
                    transformed_targets[:, i], round_preds, zero_division=0
                )
                recall = recall_score(
                    transformed_targets[:, i], round_preds, zero_division=0
                )
                f1 = f1_score(transformed_targets[:, i], round_preds, zero_division=0)
                roc_auc = roc_auc_score(
                    transformed_targets[:, i], predictions_transformed[:, i]
                )
                metric_i = {
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc,
                }
                if compute_mol_acc:
                    correct_per_graph = np.bincount(
                        gid_inverse,
                        weights=(round_preds == transformed_targets[:, i]).astype(np.float64),
                    )
                    metric_i["molecular_accuracy"] = float(
                        np.mean(correct_per_graph >= n_per_graph)
                    ) if len(n_per_graph) > 0 else float("nan")
                metrics.append(metric_i)
        else:
            if predictions_transformed.shape[1] != num_classes:
                raise NotImplementedError(
                    "Multi-class classification does not support multi-task prediction."
                )
            max_preds = np.argmax(predictions_transformed, axis=1)
            transformed_targets = transformed_targets.squeeze()  # (N,1) -> (N,)
            accuracy = np.mean(max_preds == transformed_targets)
            # 'macro average' computes metrics separately for each class then averages the scores
            # useful for cases of class imbalance
            precision_macro = precision_score(
                transformed_targets, max_preds, average="macro", zero_division=0
            )
            recall_macro = recall_score(
                transformed_targets, max_preds, average="macro", zero_division=0
            )
            f1_macro = f1_score(
                transformed_targets, max_preds, average="macro", zero_division=0
            )
            # 'weighted average' computes metrics separately for each class then averages, weighted by frequency in dataset
            # sensitive to class imbalance
            precision_weighted = precision_score(
                transformed_targets, max_preds, average="weighted", zero_division=0
            )
            recall_weighted = recall_score(
                transformed_targets, max_preds, average="weighted", zero_division=0
            )
            f1_weighted = f1_score(
                transformed_targets, max_preds, average="weighted", zero_division=0
            )
            # one-vs-rest macro ROC-AUC: requires all classes to be present in targets
            try:
                roc_auc_ovr = roc_auc_score(
                    transformed_targets,
                    predictions_transformed,
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError:
                roc_auc_ovr = float("nan")
            report = classification_report(
                transformed_targets, max_preds, output_dict=True
            )
            multi_metric = {
                "accuracy": accuracy,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
                "f1_weighted": f1_weighted,
                "roc_auc_ovr": roc_auc_ovr,
                "per_class": report,
            }
            if compute_mol_acc:
                correct_per_graph = np.bincount(
                    gid_inverse,
                    weights=(max_preds == transformed_targets).astype(np.float64),
                )
                multi_metric["molecular_accuracy"] = float(
                    np.mean(correct_per_graph >= n_per_graph)
                ) if len(n_per_graph) > 0 else float("nan")
            metrics = [multi_metric]

    return (
        predictions_transformed,
        uncertainties_transformed,
        logit_uncertainties_transformed,
        transformed_targets,
        loss,
        metrics,
    )


def predict_and_evaluate(
    model,
    loader,
    task,
    loss_function,
    num_classes=None,
    scaler=None,
    has_targets=True,
):
    """
    Make predictions and evaluate performance of trained model.
    INPUTS
        model: torch.nn.Module
            Trained model.
        loader: DataLoader
            DataLoader object for the dataset.
        targets: torch.Tensor
            Target tensor.
        task: str
            Task type: 'regression' or 'classification'.
        scaler: sklearn StandardScaler
            Scaler to invert regression predictions.
            default=None
        num_classes: int or None
            Number of classes for classification tasks.
            default=None
    OUTPUTS
        preds: np.array
            Rescaled predictions (regression) or class probabilities (classification).
        outputs: np.array
            Raw model outputs.
        loss: float
            Loss function evaluation.
        metrics: dict
            Dictionary of key performance metrics.
        embeddings: np.array or None
            Latent space embeddings if return_embeddings=True, else None.
    """
    if task == "classification":
        assert (
            num_classes is not None
        ), "Number of classes must be specified for classification tasks."

    # get predictions (in scaled / logit space) along with per-prediction graph
    # ids so we can compute graph-grouped metrics (molecular_accuracy) when the
    # model is node- or edge-level.
    _m = model.module if hasattr(model, "module") else model
    scope = getattr(_m, "scope", "graph")
    outputs, uncertainties, raw, embeddings, targets, graph_ids = predict(
        model, loader, return_graph_ids=True
    )

    # evaluate predictions (in original space)
    (
        predictions_transformed,
        uncertainties_transformed,
        logit_uncertainties_transformed,
        transformed_targets,
        loss,
        metrics,
    ) = evaluate(
        outputs, uncertainties, raw, targets, task, loss_function, scaler, num_classes,
        scope=scope, graph_ids=graph_ids, has_targets=has_targets,
    )
    return (
        predictions_transformed,
        uncertainties_transformed,
        logit_uncertainties_transformed,
        transformed_targets,
        loss,
        metrics,
        embeddings,
    )


def uncertainty(preds, task, embeddings=None, train_embeddings=None, num_classes=None):
    """
    Quantify uncertainty of model predictions.
    Return euclidean distance and cosine similarity of latent space embeddings when
    embeddings are provided (graph-level only; infeasible for node/edge-level tasks).
    For classification tasks, also return heuristic "confidence" uncertainty and Shannon entropy.
    Theoretical bounds summarized as follows:
        euclidean distance: min=0, max=inf
        cosine similarity: min=-1, max=1
        confidence (binary): min=0.5, max=1
        confidence (multiclass, K classes): min=1/K, max=1
        normalized confidence (multiclass): min=0, max=1
        Shannon entropy: min=0, max=log(K)
        normalized Shannon entropy: min=0, max=1
    INPUTS
        preds
        task
        embeddings: np.ndarray or None
            Latent space embeddings for test data. None for node/edge-level tasks.
        train_embeddings: np.ndarray or None
            Latent space embeddings for training data. None for node/edge-level tasks.
    OUTPUTS
        uncertainty_summary: dict
            Summary of different uncertainty metrics defined for a given task
    """
    uncertainty_summary = {}

    # latent space distance metrics (only for graph-level tasks where embeddings
    # are available). We compute three k-NN variants: the minimum (k=1) is
    # retained under its original column name for backward compatibility, and
    # averages over the k=10 and k=200 nearest training points are added as
    # `..._kN` columns. Averaging over multiple neighbours generally tracks
    # error better than the minimum (which is just one nearest point and is
    # therefore noisier).
    if embeddings is not None and train_embeddings is not None:
        latent_ks = (1, 10, 200)
        n_train = len(train_embeddings)
        distances = euclidean_distances(embeddings, train_embeddings)
        similarities = cosine_similarity(embeddings, train_embeddings)
        # Sort distances ascending; sort similarities descending (largest first).
        dist_sorted = np.sort(distances, axis=1)
        sim_sorted = -np.sort(-similarities, axis=1)
        # Retain the existing single-NN columns under their original names so
        # downstream code that reads them keeps working.
        uncertainty_summary["euclidean_distance"] = dist_sorted[:, 0]
        uncertainty_summary["cosine_similarity"] = sim_sorted[:, 0]
        # Add additional k-NN averages. If the training set is smaller than
        # the requested k, clip k to the available number of points.
        for k in latent_ks:
            k_eff = min(k, n_train)
            uncertainty_summary[f"euclidean_distance_k{k}"] = dist_sorted[:, :k_eff].mean(axis=1)
            uncertainty_summary[f"cosine_similarity_k{k}"]  = sim_sorted[:, :k_eff].mean(axis=1)

    if task == "classification":
        eps = 1e-12
        if num_classes == 2:
            preds = preds.squeeze()
            # confidence: max(p, 1-p), range [0.5, 1]
            confidence = np.maximum(preds, 1 - preds)
            # entropy
            entropy = -(
                preds * np.log(preds + eps) + (1 - preds) * np.log(1 - preds + eps)
            )
            norm_entropy = entropy / np.log(2)
            uncertainty_summary.update(
                {
                    "confidence": confidence,
                    "entropy": entropy,
                    "normalized_entropy": norm_entropy,
                }
            )
        else:
            K = preds.shape[1]
            # confidence: max(p_k), range [1/K, 1]
            max_prob = np.max(preds, axis=1)
            confidence = max_prob
            norm_confidence = (confidence - 1 / K) / (1 - 1 / K)
            # entropy
            entropy = -np.sum(preds * np.log(preds + eps), axis=1)
            norm_entropy = entropy / np.log(K)
            uncertainty_summary.update(
                {
                    "confidence": confidence,
                    "normalized_confidence": norm_confidence,
                    "entropy": entropy,
                    "normalized_entropy": norm_entropy,
                }
            )
    return uncertainty_summary
