import os
import numpy as np
import pandas as pd
import torch
import yaml
from elemenet.data import DataModule
from elemenet.evaluate import predict_and_evaluate, uncertainty
from elemenet.trainer import build_loss_fn, canonical_loss_type, extract_loss_config
from elemenet.utils import save_predictions


def _patch_legacy_mlp_attrs(module, scope):
    """Backfill attributes that newer ElemeNet MLP/readout code reads in
    ``forward`` but that older pickled modules may lack.

    Operates on a single module and is a no-op for any attribute already
    present, so modules built by the current code are unaffected. ``scope``
    is only used as the fallback for modules missing it.
    """
    if module is None:
        return
    # ``scope`` was added to MLP / Transformer readouts at some point.
    if not hasattr(module, "scope"):
        module.scope = scope
    # ``ensemble_size`` defaults to 1 for pre-ensemble checkpoints.
    if not hasattr(module, "ensemble_size"):
        module.ensemble_size = 1
    # ``output_size`` can be inferred from the final linear layer when
    # the field itself is absent.
    if not hasattr(module, "output_size"):
        if hasattr(module, "output_layer"):
            ens = getattr(module, "ensemble_size", 1)
            module.output_size = module.output_layer.out_features // ens
        else:
            module.output_size = 1
    # Graph-attribute support fields — None / 0 for modules that didn't use them.
    for attr, default in (
        ("graph_attr_dim", 0),
        ("graph_attr_hidden_dim", None),
        ("graph_attr_mlp", None),
        ("edge_dim", 0),
    ):
        if not hasattr(module, attr):
            setattr(module, attr, default)


def _backfill_legacy_model_attrs(model, train_args):
    """Set sensible defaults for attributes that newer ElemeNet code expects on
    readout / encoder modules but that older saved checkpoints don't have.

    The model object inside ``checkpoints/best.pt`` is a *pickled instance*,
    so it carries whatever attributes were assigned to its modules at
    training time. If we add a new attribute to a readout class later
    (e.g., ``scope``), legacy checkpoints will be missing it and the forward
    pass will raise ``AttributeError``. This helper patches those gaps based
    on ``train_args`` (which is reliable: it was written at training time
    and rides alongside the checkpoint) and reasonable defaults — no
    behaviour change for models trained against the current code, since
    those models already have these attributes.
    """
    scope = train_args["model_config"].get("scope", "graph")

    readout = getattr(model, "readout", None)
    if readout is None:
        return
    _patch_legacy_mlp_attrs(readout, scope)
    # The readout's optional graph-attribute MLP is itself an MLP module that
    # reads ``self.scope`` (and friends) in its own forward. It always operates
    # in graph mode — it embeds pooled, graph-level attributes and never takes
    # the edge-gather path — so patch any missing attrs with scope='graph'
    # regardless of the model's prediction scope.
    _patch_legacy_mlp_attrs(getattr(readout, "graph_attr_mlp", None), "graph")

    # ``Model.scope`` (top-level) is read by evaluate.predict via getattr() so
    # missing values fall through to "graph" by default — no patch needed.


def inference_pipeline(
    model_path,
    new_data_path,
    mol_column,
    target_column=None,
    save_dir="inference_results",
    graph_format="mol2",
    edge_invariant=False,
    batch_size=128,
    device="cuda",
):
    """Run inference with a trained ElemeNet model on new (unlabelled) data.

    Loads a saved model checkpoint, preprocesses the new dataset using the
    same settings as training (including feature columns and graph format),
    overwrites the target scaler with the one fitted on training data to
    prevent data leakage, and saves predictions and uncertainty metrics to
    ``<save_dir>/stats/``.

    Latent-space uncertainty metrics (e.g. distances to training embeddings)
    are computed only for graph-scope (pooled) models, where a single embedding
    vector per molecule is meaningful.

    Parameters
    ----------
    model_path : str
        Path to the training output directory containing ``config.yaml``,
        ``data/y_scaler.pkl``, and ``checkpoints/best.pt``.
    new_data_path : str
        Path to the CSV file containing new molecules for inference.
    mol_column : str
        Column in the CSV containing molecular graph strings.
    target_column : str, list of str, or None, optional
        Name(s) of the ground-truth column(s) in ``new_data_path``. Used
        **only** to compute evaluation metrics and to write ``*_true`` columns
        alongside the predictions. Default ``None`` — in which case inference
        runs prediction-only: no labels are required (or read) from the input,
        no metrics are computed, and the output CSV contains predictions
        without ``*_true`` columns. Predictions themselves never depend on the
        target column; they are produced from the model outputs and inverse-
        scaled with the training-time scaler regardless.
    save_dir : str, optional
        Root directory for inference outputs. Default ``'inference_results'``.
    graph_format : str, optional
        Molecular graph format: ``'mol2'``, ``'smiles'``, ``'mol'``, or
        ``'sdf'``. Default ``'mol2'``.
    edge_invariant : bool, optional
        Build graphs with distance-based (invariant) edge features only.
        Default False.
    batch_size : int, optional
        Number of molecules per inference batch. Default 128.
    device : str or torch.device, optional
        Device for model inference. Default ``'cuda'``.

    Returns
    -------
    pd.DataFrame
        DataFrame of predictions and uncertainty metrics saved to
        ``<save_dir>/stats/test_predictions.csv``.
    """
    data_path = os.path.join(save_dir, "data")
    stats_path = os.path.join(save_dir, "stats")
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(stats_path, exist_ok=True)

    # When no target column is given, run prediction-only: don't require labels
    # in the input, skip loss/metrics, and omit the ``*_true`` output columns.
    has_targets = target_column is not None

    # load model, args, and scaler from training data
    device = torch.device(device)
    with open(f"{model_path}/config.yaml") as f:
        train_args = yaml.safe_load(f)
    train_scaler = (
        pd.read_pickle(f"{model_path}/data/y_scaler.pkl")
        if train_args["task"] == "regression"
        else None
    )

    # process new data
    dataset = DataModule(task=train_args["task"])

    # use same feature columns as training to ensure consistent graph representations
    implicit_Hs = train_args.get("implicit_Hs", False)
    # Pass the training-time ``y_scaler`` into ``preprocess`` so target scaling
    # at inference uses the same scaler the model was trained against. Without
    # this, ``preprocess`` would silently re-fit a fresh scaler on the inference
    # data and produce per-row shifts in the saved ``*_true`` values (predictions
    # and latent distances are unaffected). For classification, ``train_scaler``
    # is None and no target scaling is performed regardless.
    dataset.preprocess(
        raw_data_path=[
            new_data_path,  # dummy
            new_data_path,  # dummy
            new_data_path,  # inference data
        ],
        mol_column=mol_column,
        target_column=target_column,
        task=train_args["task"],
        feature_columns=train_args["feature_columns"],
        representation="learned",
        graph_format=graph_format,
        train_val_test_split=None,
        data_path=data_path,
        y_scaler=train_scaler if has_targets else None,
    )

    dataset.process(
        encoder_type=train_args["model_config"]["encoder_type"],
        feature_columns=train_args["feature_columns"],
        target_column=target_column,
        mol_column=mol_column,
        scope=train_args["model_config"]["scope"],
        graph_format=graph_format,
        edge_invariant=edge_invariant,
        data_path=data_path,
        implicit_Hs=implicit_Hs,
        rdkit_features=train_args.get("rdkit_features", False),
    )

    # Reassign for safety: ``preprocess`` will have written the supplied scaler
    # to disk and ``process`` may load it back, but pinning the reference here
    # guarantees ``evaluate()`` inverse-transforms with the training-time scaler.
    dataset.scaler = train_scaler

    # load model
    checkpoint = torch.load(
        f"{model_path}/checkpoints/best.pt", map_location=device, weights_only=False
    )
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    _backfill_legacy_model_attrs(model, train_args)
    model.to(device)
    model.eval()

    # create dataloader
    test_loader = dataset.test_dataloader(batch_size=batch_size, num_workers=0)

    # define loss — loss_type may be at top level (old checkpoints) or in model_config (new)
    if "loss_type" not in train_args:
        train_args["loss_type"] = train_args["model_config"]["loss_type"]
    loss_config = extract_loss_config(params=train_args)
    loss_fn = build_loss_fn(**loss_config)

    # inference
    (
        preds,
        uncertainties,
        logit_uncertainties,
        targets,
        loss,
        metrics,
        embeddings,
    ) = predict_and_evaluate(
        model=model,
        loader=test_loader,
        task=dataset.task,
        loss_function=loss_fn,
        num_classes=dataset.num_classes,
        scaler=dataset.scaler,
        has_targets=has_targets,
    )

    # compute uncertainty metrics
    # latent space distances only feasible for graph-level (pooled) embeddings,
    # and only when the training embeddings were saved (save_embeddings=True at
    # train time). Skip gracefully when they're absent rather than requiring them.
    is_graph_scope = model.scope == "graph"
    train_embeddings_path = os.path.join(model_path, "data", "train_embeddings.npy")
    if is_graph_scope and os.path.exists(train_embeddings_path):
        train_embeddings = np.load(train_embeddings_path)
        embeddings_np = embeddings.cpu().numpy()
        uncertainty_metrics = uncertainty(
            preds=preds,
            task=dataset.task,
            embeddings=embeddings_np,
            train_embeddings=train_embeddings,
            num_classes=dataset.num_classes,
        )
    else:
        if is_graph_scope:
            print(
                f"Note: {train_embeddings_path} not found "
                "(model trained with save_embeddings=False); "
                "skipping latent-space uncertainty metrics."
            )
        uncertainty_metrics = uncertainty(
            preds=preds,
            task=dataset.task,
            embeddings=None,
            train_embeddings=None,
            num_classes=dataset.num_classes,
        )

    # save uncertainty std for losses that produce a meaningful per-sample
    # uncertainty (the ensemble regression / classification variants).
    # Mirror the list used in trainer.py at train-time evaluation.
    loss_type = train_args.get("loss_type", train_args["model_config"].get("loss_type"))
    is_uq_loss = canonical_loss_type(loss_type) in (
        "ensemble_regression",
        "ensemble_binary_classification", "ensemble_multi_classification",
    )
    save_uncertainties = uncertainties if is_uq_loss else None
    save_logit_std = logit_uncertainties if is_uq_loss else None

    # Align identifiers with the prediction rows. For graph-scope models these
    # match; fall back to a positional index if they don't (e.g. node/edge
    # scope with dummy targets), so the output CSV is always well-formed.
    labels = dataset.test_labels
    if labels is None or len(labels) != len(preds):
        labels = list(range(len(preds)))

    results = save_predictions(
        targets=targets,
        preds=preds,
        preds_uncertainties=save_uncertainties,
        preds_logit_std=save_logit_std,
        label_column=dataset.label_column,
        labels=labels,
        target_columns=dataset.target_column,
        split="test",
        task=dataset.task,
        num_classes=dataset.num_classes,
        save_dir=stats_path,
        metrics=metrics,
        uncertainty_metrics=uncertainty_metrics,
    )
    
    # clean up unnecessary outputs
    to_remove = [
        "y_scaler.pkl",
        "X_data/X_train.pkl",
        "X_data/X_val.pkl",
        "y_data/y_train.pkl",
        "y_data/y_train_scaled.pkl",
        "y_data/y_val.pkl",
        "y_data/y_val_scaled.pkl",
    ]
    for f in to_remove:
        path = os.path.join(data_path, f)
        if os.path.exists(path):
            os.remove(path)

    return results
