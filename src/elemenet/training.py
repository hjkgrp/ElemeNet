import os
import time
import warnings
import torch
import torch.distributed as dist
import yaml
from elemenet.data import DataModule
from elemenet.hypersearch import optimize, hypersearch_worker_loop
from elemenet.model import extract_model_config
from elemenet.trainer import (
    Trainer,
    extract_loss_config,
    extract_optimizer_config,
    extract_scheduler_config,
)
from elemenet.utils import wait_for_file


def _merge_model_config(base, override):
    """Deep-merge ``override`` into ``base`` in-place.

    Nested dicts are merged recursively; all other values are overwritten.
    Modifies ``base`` directly and returns nothing.

    Parameters
    ----------
    base : dict
        Target dict to merge into.
    override : dict
        Values to merge on top of ``base``.
    """
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v


def training_pipeline(
    target_column,
    task,
    model_config,
    label_column=None,
    graph_format="mol2",
    mol_column="mol2_string",
    epochs=100,
    early_stopping=True,
    save_dir=None,
    max_trials=25,
    quicksearch=False,
    epochs_hypersearch=100,
    edge_invariant=False,
    use_xtb=False,
    use_bulk=False,
    batch_size=64,
    num_workers=0,
    random_seed=0,
    center_column=None,
    feature_columns=None,
    k_hops=1,
    stratify=None,
    group_by=None,
    bond_scale_factor=1.0,
    charge_spin_override=False,
    preprocessing_kwargs=None,
    seed_model_state_path=None,
    seed_optimizer_state_path=None,
    device=None,
    implicit_Hs=False,
    rdkit_features=False,
    save_embeddings=True,
    grad_clip_norm=None,
    max_grad_skips=0,
):
    """End-to-end training pipeline: preprocess, (optionally) hyperparameter-search, train, and evaluate.

    Supports single-GPU, CPU, and multi-GPU distributed data-parallel (DDP)
    training launched via ``torchrun``. In DDP mode, rank 0 handles all
    preprocessing and disk I/O; non-main ranks wait on a sentinel file before
    loading preprocessed data, so no NCCL connection is held during
    potentially long preprocessing steps.

    If ``seed_model_state_path`` and ``seed_optimizer_state_path`` both point
    to the same checkpoint file, training resumes from that checkpoint.
    If only ``seed_model_state_path`` is provided, the checkpoint's weights are
    loaded for transfer-learning (optimizer state is not restored).

    Hyperparameter search (Optuna) is triggered when ``max_trials`` is not
    ``None`` and ``readout_config`` does not already specify ``neurons``. In
    DDP mode, rank 0 drives the Optuna study and broadcasts each trial's
    parameters to worker ranks.

    Parameters
    ----------
    target_column : str or list of str
        Column name(s) of the target property in the data CSV.
    task : str
        Learning task: ``'regression'`` or ``'classification'``.
    model_config : dict
        Model architecture and training hyperparameters. Required keys include
        ``encoder_type``, ``readout_type``, ``loss_type``, and sub-dicts
        ``encoder_config`` / ``readout_config``. Edge-level prediction
        (``scope='edge'``) is supported by ``readout_type='mlp'``,
        ``'transformer'``, or ``'edge_predictor'`` — these differ in whether
        endpoint-pair fusion happens before, after, or via attention over the
        readout trunk.
    label_column : str or None, optional
        Column used as sample identifier in saved outputs.
    graph_format : str, optional
        Molecular graph format: ``'mol2'``, ``'smiles'``, ``'mol'``, or
        ``'sdf'``. Default ``'mol2'``.
    mol_column : str, optional
        Column containing molecular graph strings. Default ``'mol2_string'``.
    epochs : int, optional
        Maximum number of training epochs. Default 100.
    early_stopping : bool, optional
        Save and restore the best checkpoint by validation loss. Default True.
    save_dir : str
        Root directory for all outputs (config, checkpoints, data cache, stats).
        Must be provided; the pipeline always writes preprocessed graphs to disk.
    max_trials : int or None, optional
        Number of Optuna hyperparameter search trials. Pass ``None`` to skip
        hyperparameter search and use ``model_config`` values directly.
        Default 25.
    quicksearch : bool, optional
        Use a reduced search space for faster hyperparameter optimization.
        Default False.
    epochs_hypersearch : int, optional
        Training epochs per hyperparameter search trial. Default 100.
    edge_invariant : bool, optional
        Build graphs with only distance-based (invariant) edge features.
        Default False.
    use_xtb : bool, optional
        Augment atom features with xTB-derived electronic properties.
        Default False.
    use_bulk : bool, optional
        Augment atom features with bulk electronic structure descriptors.
        Default False.
    batch_size : int, optional
        Number of graphs per batch per GPU. Default 64.
    num_workers : int, optional
        DataLoader worker processes. Default 0.
    random_seed : int, optional
        Global random seed for reproducibility. Default 0.
    center_column : str or list of str or None, optional
        Column(s) specifying center atoms for subgraph selection.
    feature_columns : list of str or None, optional
        Extra tabular feature columns to concatenate with graph embeddings.
    k_hops : int, optional
        Neighborhood depth for subgraph extraction around center atoms.
        Default 1.
    stratify : str or None, optional
        Column by which to stratify train/val/test splits.
    group_by : str or list of str or None, optional
        Column(s) by which to group samples so entire groups land in the same
        split (prevents data leakage across structurally related molecules).
    bond_scale_factor : float, optional
        Scaling factor applied to bond lengths during graph construction.
        Default 1.0.
    charge_spin_override : bool, optional
        Use explicit charge/spin multiplicity columns from the CSV instead of
        inferring them from the molecular graph. Default False.
    preprocessing_kwargs : dict or None, optional
        Additional keyword arguments forwarded to ``DataModule.preprocess``.
    seed_model_state_path : str or None, optional
        Path to a ``.pt`` checkpoint. Used to resume training (when combined
        with ``seed_optimizer_state_path``) or to initialize weights for
        transfer-learning (when provided alone).
    seed_optimizer_state_path : str or None, optional
        Must equal ``seed_model_state_path`` when provided. Signals that
        optimizer state should also be restored (i.e. full resume).
    device : str or torch.device or None, optional
        Target device (e.g. ``'cuda'``, ``'cpu'``). Auto-detected if None.
    implicit_Hs : bool, optional
        Include implicit hydrogen atoms when building molecular graphs.
        Default False.
    save_embeddings : bool, optional
        Save training-set embeddings to disk for inference-time uncertainty
        quantification. Default True.

    Returns
    -------
    dict
        Training result containing keys: ``'model'``, ``'dataset'``,
        ``'loss_function'``, ``'train_loss'``, ``'val_loss'``, ``'test_loss'``,
        ``'test_metrics'``, and (when ``early_stopping=True``) ``'best_val_loss'``
        and ``'best_epoch'``.
    """
    if save_dir is None:
        raise ValueError(
            "save_dir is required. The pipeline always writes preprocessed graphs, "
            "checkpoints, and stats to disk and cannot operate without a save directory."
        )

    # detect distributed launch from environment variables set by torchrun.
    # we intentionally do NOT call dist.init_process_group here and instead
    # wait until all data is on disk so non-main ranks never hold an NCCL
    # connection while waiting for rank 0 to finish expensive preprocessing/graph building.
    _distributed = os.environ.get('LOCAL_RANK') is not None
    _local_rank = int(os.environ.get('LOCAL_RANK', 0))
    _global_rank = int(os.environ.get('RANK', 0))
    if _distributed:
        torch.cuda.set_device(_local_rank)
        device = f'cuda:{_local_rank}'
    _is_main = not _distributed or _global_rank == 0

    # save all input parameters to a yaml file
    input_params = {k: v for k, v in locals().items() if not k.startswith("_")}

    # save training hyperparams before model_config may be replaced by a checkpoint's
    # architecture-only model_config (which does not store loss_type/optimizer_type).
    _loss_type = model_config.get("loss_type")
    _optimizer_type = model_config.get("optimizer_type", "adamw")

    # possibly resume training from checkpoint or use a pretrained model for finetuning.
    seed_model_state = None
    seed_optimizer_state = None
    seed_scheduler_state = None
    start_epoch = 0
    best_val_loss = float("inf")

    if seed_model_state_path is not None and seed_optimizer_state_path is not None:
        assert (
            seed_model_state_path == seed_optimizer_state_path
        ), "seed_model_state_path and seed_optimizer_state_path must be the same checkpoint file if both are provided"
        if not os.path.isfile(seed_model_state_path):
            raise ValueError(f"Resume checkpoint not found at {seed_model_state_path}")
        print(f"\nResuming training from checkpoint at '{seed_model_state_path}'\n")
        ckpt = torch.load(
            seed_model_state_path, map_location=device, weights_only=False
        )
        seed_model_state = ckpt["state_dict"]
        model_config = ckpt["model_config"]
        seed_optimizer_state = ckpt["optimizer"]
        # .get() so old checkpoints predating the scheduler feature (no key) load cleanly.
        seed_scheduler_state = ckpt.get("scheduler")
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

        # Warn if resuming a stateful epoch-derived scheduler with a different
        # epochs value than the original run — the LR trajectory will be wrong
        # because schedulers like CosineAnnealingLR(T_max=...) and OneCycleLR
        # encode the total-length assumption into their step counter. Old
        # checkpoints lacking "epochs" skip the check (None != epochs short-circuits
        # to True but ckpt_scheduler_type will be None for those too).
        ckpt_scheduler_type = model_config.get("scheduler_type")
        ckpt_epochs = ckpt.get("epochs")
        if (
            ckpt_scheduler_type in {"cosine", "onecycle"}
            and ckpt_epochs is not None
            and ckpt_epochs != epochs
        ):
            warnings.warn(
                f"Resuming a {ckpt_scheduler_type} scheduler with epochs={epochs}, "
                f"but the original run used epochs={ckpt_epochs}. Stateful schedulers "
                f"tied to total epochs will produce an incorrect LR trajectory. Re-run "
                f"with the original epochs value, or pass seed_model_state_path alone "
                f"to start a fresh scheduler.",
                RuntimeWarning,
            )

    if seed_model_state_path is not None and seed_optimizer_state_path is None:
        if not os.path.isfile(seed_model_state_path):
            raise ValueError(f"Seed model not found at {seed_model_state_path}")
        ckpt = torch.load(
            seed_model_state_path, map_location=device, weights_only=False
        )
        seed_model_state = ckpt["state_dict"]
        model_config = ckpt["model_config"]
        print(
            f"Load model at '{seed_model_state_path}' with model config:\n {model_config}"
        )

    if _is_main and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        params_path = os.path.join(save_dir, "config.yaml")
        with open(params_path, "w") as f:
            yaml.dump(input_params, f)

    data_path = os.path.join(save_dir, "data")
    dataset = DataModule(task=task)

    # If a non-empty processed-graphs cache already exists, skip preprocess()
    # entirely — process() short-circuits to the .pt cache and never reads
    # the X_data/y_data pickles in that case, so re-running preprocess() is
    # wasted work (30-60 min on multi-million-row datasets per resume).
    _target_cols = [target_column] if isinstance(target_column, str) else list(target_column)
    _cache_path = os.path.join(data_path, f"processed_graphs_{'_'.join(_target_cols)}.pt")
    _cache_exists = os.path.isfile(_cache_path) and os.path.getsize(_cache_path) > 0

    if preprocessing_kwargs is not None:
        preprocessing_kwargs.update(
            {
                "mol_column": mol_column,
                "target_column": target_column,
                "label_column": label_column,
                "task": task,
                "data_path": data_path,
                "graph_format": graph_format,
                "random_seed": random_seed,
                "center_columns": center_column,
                "feature_columns": feature_columns,
                "stratify": stratify,
                "group_by": group_by,
                "bond_scale_factor": bond_scale_factor,
            }
        )
        # only rank 0 preprocesses; non-main ranks wait for the .pt cache below.
        if _is_main and not _cache_exists:
            dataset.preprocess(**preprocessing_kwargs)
        elif _is_main:
            print(
                f"Found processed-graphs cache at {_cache_path}. "
                f"Skipping preprocess() — graphs will be loaded from cache."
            )

    # process data: rank 0 builds (and saves) the .pt cache; non-main ranks wait for
    # that file to appear on disk before loading it.  No NCCL connection is needed here,
    # so there is no risk of a timeout no matter how long preprocessing takes.
    process_kwargs = dict(
        encoder_type=model_config["encoder_type"],
        target_column=target_column,
        mol_column=mol_column,
        scope=model_config["scope"],
        label_column=label_column,
        graph_format=graph_format,
        edge_invariant=edge_invariant,
        center_column=center_column,
        feature_columns=feature_columns,
        k_hops=k_hops,
        use_xtb=use_xtb,
        use_bulk=use_bulk,
        bond_scale_factor=bond_scale_factor,
        charge_spin_override=charge_spin_override,
        data_path=data_path,
        implicit_Hs=implicit_Hs,
        rdkit_features=rdkit_features,
    )
    # a sentinel written *after* process() returns guarantees torch.save is complete,
    # so non-main ranks can safely load the cache without seeing a truncated file.
    _cache_sentinel = _cache_path + ".ready"
    if _is_main:
        dataset.process(**process_kwargs)
        if _distributed:
            open(_cache_sentinel, 'w').close()
    elif _distributed:
        wait_for_file(_cache_sentinel)
        dataset.process(**process_kwargs)

    # all ranks have data ready — now initialize the NCCL process group.
    # the watchdog timer starts here, so it only runs during actual training.
    # init_process_group is collective, so when it returns every rank has already
    # passed wait_for_file; rank 0 can then safely remove the sentinel.
    if _distributed:
        dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{_local_rank}'))
        if _is_main and os.path.exists(_cache_sentinel):
            os.remove(_cache_sentinel)

    # infer model input and output size given the data
    inferred = dataset.infer_model_input_output_size()
    fixed_params = inferred
    inferred_graph_attr_dim = inferred["readout_config"]["graph_attr_dim"]
    fixed_params["loss_type"] = model_config.get("loss_type") or _loss_type
    fixed_params["optimizer_type"] = model_config.get("optimizer_type") or _optimizer_type
    # deep-merge nested dicts (encoder_config / readout_config) so a partial user-provided
    # config does not drop inferred values like input_size / edge_dim. Top-level scalars
    # behave identically to dict.update; user values still override inferred ones.
    _merge_model_config(fixed_params, model_config)

    # in case no model parameters are provided, perform hyperparameter optimization.
    # in distributed mode, rank 0 drives the Optuna study and broadcasts each trial's
    # parameters to all other ranks so they can participate in DDP training together.
    # non-main ranks run a worker loop that mirrors each trial; once hypersearch finishes,
    # rank 0 broadcasts the best params to all ranks.
    readout_config = model_config.get("readout_config", None)
    if max_trials is None:
        print("max_trials=None: skipping hyperparameter optimization, using model_config values and system defaults for unspecified parameters")
    if max_trials is not None and (readout_config is None or "neurons" not in readout_config):
        if _is_main:
            fixed_params = optimize(
                dataset=dataset,
                fixed_params=fixed_params,
                epochs=epochs_hypersearch,
                max_trials=max_trials,
                early_stopping=early_stopping,
                save_dir=save_dir,
                quicksearch=quicksearch,
                batch_size=batch_size,
                num_workers=num_workers,
                random_seed=random_seed,
                device=device,
                grad_clip_norm=grad_clip_norm,
                max_grad_skips=max_grad_skips,
                _distributed=_distributed,
            )
        else:
            hypersearch_worker_loop(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                grad_clip_norm=grad_clip_norm,
                max_grad_skips=max_grad_skips,
            )
        if _distributed:
            objects = [fixed_params if _is_main else None]
            dist.broadcast_object_list(objects, src=0)
            fixed_params = objects[0]
    else:
        _merge_model_config(fixed_params, model_config)

    # always use the data-inferred graph_attr_dim — the user cannot know this without inspecting the data
    fixed_params["readout_config"]["graph_attr_dim"] = inferred_graph_attr_dim

    model_config = extract_model_config(fixed_params)
    loss_config = extract_loss_config(fixed_params)
    optimizer_config = extract_optimizer_config(fixed_params)
    scheduler_config = extract_scheduler_config(fixed_params)

    trainer = Trainer(
        model_config=model_config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        device=device,
        grad_clip_norm=grad_clip_norm,
        max_grad_skips=max_grad_skips,
    )

    result = trainer.train(
        dataset=dataset,
        epochs=epochs,
        early_stopping=early_stopping,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=random_seed,
        save_dir=save_dir,
        seed_model_state=seed_model_state,
        seed_optimizer_state=seed_optimizer_state,
        seed_scheduler_state=seed_scheduler_state,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        save_embeddings=save_embeddings,
    )

    if _distributed:
        dist.destroy_process_group()

    return result
