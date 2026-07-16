import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adagrad, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch_geometric.data import Data
from tqdm import tqdm
from elemenet.evaluate import predict, predict_and_evaluate, uncertainty
from elemenet.model import build_model, invert_encoder_config, invert_readout_config
from elemenet.utils import (
    estimate_size,
    save_predictions,
    set_seed,
    print_metrics,
)


class NaNLossError(Exception):
    """Raised when a NaN validation loss is detected during training."""

    pass


class Trainer:
    """Manages the full training lifecycle: model construction, optimization, and evaluation.

    Supports single-device and distributed data-parallel (DDP) training.
    In DDP mode the class expects ``dist.init_process_group`` to have already
    been called; it pins each rank to the GPU specified by ``LOCAL_RANK`` and
    wraps the model with ``DistributedDataParallel`` inside ``train()``.

    Parameters
    ----------
    model_config : dict
        Resolved model config dict as returned by ``extract_model_config``,
        containing typed encoder/readout config dataclasses.
    optimizer_config : dict
        Optimizer settings as returned by ``extract_optimizer_config``
        (keys: ``optimizer_type``, ``lr``, ``weight_decay``).
    loss_config : dict
        Loss function settings as returned by ``extract_loss_config``
        (keys: ``loss_type`` and any loss-specific parameters).
    scheduler_config : dict or None, optional
        LR scheduler settings as returned by ``extract_scheduler_config``
        (keys: ``scheduler_type``, ``scheduler_config``). Default ``None``
        produces a no-op scheduler (constant LR), preserving pre-feature
        behavior byte-for-byte.
    device : torch.device or str or None, optional
        Target device. Overridden by ``LOCAL_RANK`` in DDP mode, or
        auto-detected if None. Default None.
    process_group : torch.distributed.ProcessGroup or None, optional
        DDP process group. Uses the default group when None. Default None.
    """

    def __init__(self, model_config, optimizer_config, loss_config,
                 scheduler_config=None, device=None, process_group=None,
                 grad_clip_norm=None, max_grad_skips=0):
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.loss_config = loss_config
        # max global gradient norm for clipping after backward(); None disables
        # clipping (preserves the original behaviour). Useful for stabilising
        # heavy-tailed losses (e.g. Gaussian NLL) whose gradients can spike.
        self.grad_clip_norm = grad_clip_norm
        # if > 0, skip (rather than apply) an optimizer step whose loss/gradient
        # is non-finite, up to this many CONSECUTIVE skips before aborting with a
        # NaNLossError. 0 disables skipping (a non-finite step propagates and is
        # caught by the end-of-epoch NaN check, preserving the original behaviour).
        self.max_grad_skips = max_grad_skips
        self._consec_grad_skips = 0
        self.scheduler_config = scheduler_config or {
            "scheduler_type": None, "scheduler_config": {},
        }

        # save clean model_config which is used to set up the advanced model config
        self.clean_model_config = model_config.copy()
        self.clean_model_config.update(
            invert_encoder_config(
                model_config["encoder_config"], model_config["encoder_type"]
            )
        )
        self.clean_model_config.update(
            invert_readout_config(
                model_config["readout_config"], model_config["readout_type"]
            )
        )

        # detect distributed environment (requires dist.init_process_group already called).
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main = (self.rank == 0)
        self.process_group = process_group

        # in DDP each rank is pinned to its own GPU; otherwise use the caller-supplied
        # device or auto-detect cuda/cpu (preserving the original single-device behaviour).
        if self.distributed:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = torch.device(f'cuda:{local_rank}')
        elif device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def training_loop(self, train_loader, model, optimizer, loss_fn, epoch=0, scheduler=None):
        """Run one epoch of forward passes, loss computation, and backpropagation.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training set.
        model : torch.nn.Module
            Model to train (may be DDP-wrapped).
        optimizer : torch.optim.Optimizer
            Optimizer to step.
        loss_fn : LossWrapper
            Loss function callable.
        epoch : int, optional
            Current epoch index (0-indexed). Used to set the epoch on the
            DistributedSampler for proper shuffling in DDP mode. Default 0.
        scheduler : SchedulerWrapper or None, optional
            LR scheduler whose ``step_batch()`` is called after every
            optimizer step. ``None`` is treated as a no-op scheduler.

        Returns
        -------
        float
            Mean training loss for the epoch.
        """
        if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            # call the model
            outputs = model(batch)
            # compute the loss
            loss = loss_fn(outputs)
            # backpropagation and optimization step
            loss.backward()
            if self.grad_clip_norm is not None:
                # clip_grad_norm_ returns the (pre-clip) total norm; non-finite here
                # means a non-finite gradient that clipping cannot sanitise.
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.grad_clip_norm
                )
                step_is_finite = bool(torch.isfinite(total_norm))
            else:
                step_is_finite = bool(torch.isfinite(loss))
            # optionally skip a non-finite step instead of writing NaN/Inf into the
            # weights; abort after max_grad_skips consecutive skips.
            if not step_is_finite and self.max_grad_skips > 0:
                self._consec_grad_skips += 1
                if self.is_main:
                    print(
                        f"  [skip {self._consec_grad_skips}/{self.max_grad_skips}] "
                        f"non-finite loss/gradient at epoch {epoch}, batch {batch_idx} "
                        f"— skipping optimizer step",
                        flush=True,
                    )
                if self._consec_grad_skips >= self.max_grad_skips:
                    raise NaNLossError(
                        f"{self._consec_grad_skips} consecutive non-finite gradient "
                        f"steps (max_grad_skips={self.max_grad_skips}) at epoch {epoch}"
                    )
                continue  # weights untouched; grads cleared by next zero_grad()
            optimizer.step()
            self._consec_grad_skips = 0
            if scheduler is not None:
                scheduler.step_batch()
            running_loss += loss.item()
        train_epoch_loss = running_loss / len(train_loader)
        return train_epoch_loss

    @torch.no_grad()
    def validation_loop(self, val_loader, model, loss_fn):
        """Evaluate the model on the validation set without gradient computation.

        Parameters
        ----------
        val_loader : DataLoader
            DataLoader for the validation set.
        model : torch.nn.Module
            Model to evaluate.
        loss_fn : LossWrapper
            Loss function callable.

        Returns
        -------
        float
            Mean validation loss for the epoch.
        """
        model.eval()
        val_running_loss = 0
        for batch in val_loader:
            batch = batch.to(self.device)
            # call the model
            outputs = model(batch)
            # compute the loss
            loss = loss_fn(outputs)
            val_running_loss += loss.item()
        val_epoch_loss = val_running_loss / len(val_loader)
        return val_epoch_loss

    def train(
        self,
        dataset,
        epochs=100,
        early_stopping=True,
        batch_size=64,
        num_workers=0,
        random_seed=0,
        save_dir=None,
        seed_model_state=None,
        seed_optimizer_state=None,
        seed_scheduler_state=None,
        start_epoch=0,
        best_val_loss=float("inf"),
        save_embeddings=True,
    ):
        """Build, train, evaluate, and optionally save a model.

        Orchestrates the full training run: constructs the model and
        optimizer, optionally loads seed weights/optimizer state for
        resume or transfer-learning, runs the training/validation loop with
        optional early stopping, evaluates on the test set, and writes
        checkpoints and predictions to ``save_dir``.

        In DDP mode the model is wrapped with ``DistributedDataParallel``;
        only rank 0 writes to disk and evaluates on the full test set.

        Parameters
        ----------
        dataset : DataModule
            Preprocessed dataset object providing train/val/test dataloaders.
        epochs : int, optional
            Maximum number of training epochs. Default 100.
        early_stopping : bool, optional
            Save and restore the best-validation-loss checkpoint. Default True.
        batch_size : int, optional
            Graphs per batch per GPU. Default 64.
        num_workers : int, optional
            DataLoader worker processes. Default 0.
        random_seed : int, optional
            Global random seed for reproducibility. Default 0.
        save_dir : str or None, optional
            Root directory for checkpoints, stats, and embeddings. If None,
            nothing is written to disk.
        seed_model_state : dict or None, optional
            State dict to load into the model before training starts.
        seed_optimizer_state : dict or None, optional
            Optimizer state dict to restore (for resuming training).
        seed_scheduler_state : dict or None, optional
            LR scheduler state dict to restore (for resuming training).
            Ignored when no scheduler is configured.
        start_epoch : int, optional
            First epoch index (non-zero when resuming). Default 0.
        best_val_loss : float, optional
            Starting best validation loss (for resume). Default inf.
        save_embeddings : bool, optional
            Save training-set embeddings for inference-time uncertainty
            quantification (graph-scope only). Default True.

        Returns
        -------
        dict
            Keys: ``'model'``, ``'dataset'``, ``'loss_function'``,
            ``'train_loss'``, ``'val_loss'``, ``'test_loss'``,
            ``'test_metrics'``, and (when ``early_stopping=True``)
            ``'best_val_loss'`` and ``'best_epoch'``.
        """
        if start_epoch >= epochs:
            if self.is_main:
                print(
                    f"Start epoch ({start_epoch}) is greater than or equal to total epochs ({epochs})"
                )
            return

        if start_epoch > 0 and self.is_main:
            print(
                f"Resuming training from epoch {start_epoch} out of {epochs} total epochs"
            )

        # seed for reproducibility
        set_seed(random_seed=random_seed)

        # build model
        model = build_model(**self.model_config, device=self.device)

        # seed model if provided
        if seed_model_state is not None:
            model.load_state_dict(seed_model_state)

        # wrap with DDP if running in a distributed context; each rank holds a full
        # model replica, gradients are all-reduced automatically during backward
        if self.distributed:
            model = DDP(model, device_ids=[self.device.index], process_group=self.process_group)
            model._set_static_graph()

        # build optimizer
        optimizer = build_optimizer(**self.optimizer_config, params=model.parameters())

        # seed optimizer if provided
        if seed_optimizer_state is not None:
            optimizer.load_state_dict(seed_optimizer_state)

        # build loss function and move weight tensors to device
        loss_fn = build_loss_fn(**self.loss_config)
        lf = loss_fn.loss_function
        if hasattr(lf, 'to'):
            loss_fn.loss_function = lf.to(self.device)
        for attr in ('pos_weight', 'class_weight'):
            val = getattr(lf, attr, None)
            if isinstance(val, torch.Tensor):
                setattr(lf, attr, val.to(self.device))

        if self.is_main:
            print(
                f"Training {self.model_config['encoder_type'].upper()} model with "
                f"{sum(param.numel() for param in model.parameters())} parameters"
            )
            print(f"Using device: {self.device}"
                  + (f" ({self.world_size} GPUs)" if self.distributed else ""))
            print(f"Batch size: {batch_size}"
                  + (f" per GPU, {batch_size * self.world_size} total" if self.distributed else ""))
            print(f"Number of workers: {num_workers}")
            print(f"Random seed: {random_seed}")
            print(f"Estimated model size: {estimate_size(model):.1f} MB")

        if save_dir:
            checkpoint_dir = os.path.join(save_dir, "checkpoints")
            stats_dir = os.path.join(save_dir, "stats")
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(stats_dir, exist_ok=True)

        # get dataloaders; training data is sharded across ranks via DistributedSampler
        # when running in a distributed context; val/test are not sharded
        train_loader = dataset.train_dataloader(
            batch_size, num_workers, random_seed,
            rank=self.rank if self.distributed else None,
            world_size=self.world_size if self.distributed else None,
        )
        val_loader = dataset.val_dataloader(batch_size, num_workers)
        test_loader = dataset.test_dataloader(batch_size, num_workers)

        # build LR scheduler now that train_loader exists (needed for steps_per_epoch).
        # When scheduler_type is None, returns a no-op wrapper so the training loop's
        # unconditional step_batch()/step_epoch() calls become free.
        scheduler = build_scheduler(
            **self.scheduler_config,
            optimizer=optimizer,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )
        if seed_scheduler_state is not None:
            scheduler.load_state_dict(seed_scheduler_state)

        if self.is_main:
            stype = self.scheduler_config["scheduler_type"]
            if stype is not None:
                print(f"Scheduler: {stype} (config: {self.scheduler_config['scheduler_config']})")
            else:
                print("Scheduler: none (constant LR)")

        # training loop
        best_epoch = -1
        best_checkpoint_path = None
        train_loss_list = []
        val_loss_list = []
        with tqdm(
            total=(epochs - start_epoch), desc="Training Progress", unit="epoch",
            disable=not self.is_main,
        ) as pbar:
            for epoch in range(start_epoch, epochs):
                loss_fn.update_epoch(epoch)
                train_loss = self.training_loop(train_loader, model, optimizer, loss_fn, epoch, scheduler)
                val_loss = self.validation_loop(val_loader, model, loss_fn)
                scheduler.step_epoch(val_loss)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                pbar.set_postfix({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                })
                pbar.update(1)

                # prune if loss is NaN (skip epoch 0 to allow for unusual initializations).
                # in distributed mode, all-reduce the NaN flag so all ranks make the same
                # decision at the same epoch — NCCL NaN propagation through all-reduce can
                # be asymmetric across GPUs, causing ranks to diverge on which epoch they
                # prune, which desyncs the outer broadcast_object_list control loop.
                if epoch > 0:
                    val_is_nan = float(np.isnan(val_loss))
                    if self.distributed:
                        nan_flag = torch.tensor([val_is_nan], device=self.device)
                        dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX, group=self.process_group)
                        val_is_nan = nan_flag.item()
                    if val_is_nan:
                        raise NaNLossError(
                            f"NaN validation loss detected at epoch {epoch + 1}"
                        )

                # early stopping logic
                if early_stopping and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    if save_dir:
                        best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
                        if self.is_main:
                            # unwrap DDP to save a portable, non-distributed checkpoint
                            raw_model = model.module if self.distributed else model
                            torch.save(
                                {
                                    "model": raw_model,
                                    "model_config": self.clean_model_config,
                                    "state_dict": raw_model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "epoch": epoch,
                                    "epochs": epochs,
                                    "train_loss": train_loss,
                                    "val_loss": val_loss,
                                    "best_val_loss": best_val_loss,
                                },
                                best_checkpoint_path,
                            )

        # in distributed mode, ensure rank 0 has finished writing the best checkpoint
        # before all ranks attempt to read it back.
        if self.distributed:
            dist.barrier(group=self.process_group)

        # return early-stopped model
        if early_stopping and best_checkpoint_path is not None:
            checkpoint = torch.load(
                best_checkpoint_path, map_location=self.device, weights_only=False
            )
            # checkpoint always holds the raw (non-DDP) state_dict; load into the
            # underlying module so the same file works for single- and multi-GPU runs.
            if self.distributed:
                model.module.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["state_dict"])
            model.to(self.device)
            if self.is_main:
                print(
                    f"Loaded best model from epoch {checkpoint['epoch'] + 1} with validation loss {checkpoint['val_loss']:.4f}"
                )

        # unwrap DDP so the rest of the code (evaluation, saving) sees a plain nn.Module.
        eval_model = model.module if self.distributed else model
        eval_model.eval()

        # only the main rank evaluates and writes outputs to disk.
        # val_loss / train_loss are available on all ranks (needed by hypersearch trials).
        is_graph_scope = self.model_config["scope"] == "graph"
        # losses that produce a meaningful per-sample uncertainty worth saving
        is_uq_loss = loss_fn.loss_type in [
            "ensemble_regression",
            "ensemble_binary_classification", "ensemble_multi_classification",
        ]
        test_loss = None
        test_metrics = None

        if self.is_main:
            (
                test_preds,
                test_uncertainties,
                test_logit_uncertainties,
                test_targets,
                test_loss,
                test_metrics,
                test_embeddings,
            ) = predict_and_evaluate(
                model=eval_model,
                loader=test_loader,
                task=dataset.task,
                scaler=dataset.scaler,
                num_classes=dataset.num_classes,
                loss_function=loss_fn,
            )
            print_metrics(test_metrics, dataset.target_column, test_loss, "test")

            # compute uncertainty metrics
            # latent space distances only feasible for graph-level (pooled) embeddings,
            # and only when save_embeddings=True (pairwise distance over millions of
            # training points is otherwise intractable in memory).
            if is_graph_scope and save_embeddings:
                # use a fresh non-sharded loader so rank 0 sees the full training set
                # (train_loader may be sharded via DistributedSampler in multi-GPU runs)
                full_train_loader = dataset.train_dataloader(
                    batch_size, num_workers, random_seed, shuffle=False
                )
                _, _, _, train_embeddings, _ = predict(eval_model, full_train_loader)
                train_embeddings_np = train_embeddings.cpu().numpy()
                test_embeddings_np = test_embeddings.cpu().numpy()
                uncertainty_metrics = uncertainty(
                    preds=test_preds,
                    task=dataset.task,
                    embeddings=test_embeddings_np,
                    train_embeddings=train_embeddings_np,
                    num_classes=dataset.num_classes,
                )
            else:
                # latent space UQ skipped: either node/edge-level task, or save_embeddings=False
                uncertainty_metrics = uncertainty(
                    preds=test_preds,
                    task=dataset.task,
                    embeddings=None,
                    train_embeddings=None,
                    num_classes=dataset.num_classes,
                )

            save_uncertainties = test_uncertainties if is_uq_loss else None
            save_logit_std = test_logit_uncertainties if is_uq_loss else None

            # save the weights, architecture, scalers, loss, and test predictions
            if save_dir:
                last_checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
                # save model weights
                torch.save(
                    {
                        "model": eval_model,
                        "model_config": self.clean_model_config,
                        "state_dict": eval_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "epochs": epochs,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                    },
                    last_checkpoint_path,
                )
                print(f"Trained model saved to {last_checkpoint_path}")

                # save training and validation loss
                num_epochs_run = len(train_loss_list)
                loss_summary = pd.DataFrame(
                    {
                        "epoch": list(range(num_epochs_run)),
                        "training_loss": train_loss_list,
                        "validation_loss": val_loss_list,
                    }
                )
                loss_summary.to_csv(
                    os.path.join(stats_dir, "loss_summary.csv"), index=False
                )
                print(f"Training and validation loss saved to {stats_dir}/loss_summary.csv")

                # save training embeddings for inference-time uncertainty (graph-level only)
                if is_graph_scope and save_embeddings:
                    data_dir = os.path.join(save_dir, "data")
                    os.makedirs(data_dir, exist_ok=True)
                    np.save(
                        os.path.join(data_dir, "train_embeddings.npy"), train_embeddings_np
                    )

                # save training, validation, and test predictions
                save_predictions(
                    targets=test_targets,
                    preds=test_preds,
                    preds_uncertainties=save_uncertainties,
                    preds_logit_std=save_logit_std,
                    label_column=dataset.label_column,
                    labels=dataset.test_labels,
                    target_columns=dataset.target_column,
                    split="test",
                    task=dataset.task,
                    num_classes=dataset.num_classes,
                    save_dir=stats_dir,
                    metrics=test_metrics,
                    uncertainty_metrics=uncertainty_metrics,
                )

        output = {
            "model": eval_model,
            "dataset": dataset,
            "loss_function": loss_fn,
            "train_loss": train_loss_list[-1],
            "val_loss": val_loss_list[-1],
            "test_loss": test_loss,       # None on non-main ranks
            "test_metrics": test_metrics,  # None on non-main ranks
        }
        if early_stopping:
            output.update(
                {
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch + 1,
                }
            )

        return output


class Ensemble_binary_classification:
    """Mean-of-probabilities ensemble loss for binary classification.

    Each ensemble head outputs a logit l_i; the aggregated probability is

        p_bar = (1 / n_ens) * sum_i sigmoid(l_i)

    and the loss is binary cross-entropy on p_bar.

    This is NOT the same as BCE on sigmoid(mean(l_i)). The nonlinear
    aggregation gives the loss a direct gradient to every head, and ensemble
    disagreement propagates into p_bar via Jensen's inequality (more spread
    -> p_bar pulled toward 0.5), so the heads' variance is implicitly used
    by the primary loss without requiring an auxiliary KL term.

    For inference, the calibrated uncertainty is the probability-space std:

        sigma_p = std_i(sigmoid(l_i))

    Parameters
    ----------
    reduction : {"mean", "sum", "none"}, default "mean"
    pos_weight : float or torch.Tensor, optional
        Positive-class weight applied to the y * log(p_bar) term, matching
        the convention of ``BCEWithLogitsLoss``.
    anti_collapse_weight : float, default 0.0
        Weight on an optional log-variance bonus that discourages head
        collapse when the BCE gradient alone is insufficient. Set to 0 to
        disable; try small values (e.g. 0.01-0.1) only if the empirical
        probability-space variance is near zero.
    anti_collapse_eps : float, default 1e-4
        Numerical floor inside the log of the anti-collapse term.
    """

    def __init__(
        self,
        reduction="mean",
        pos_weight=None,
        anti_collapse_weight=0.0,
        anti_collapse_eps=1e-4,
        **kwargs,
    ):
        assert reduction in ["mean", "sum", "none"], (
            f"Unknown reduction type: '{reduction}'"
            " (choose from 'mean', 'sum', or 'none')"
        )
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.anti_collapse_weight = anti_collapse_weight
        self.anti_collapse_eps = anti_collapse_eps

    def __call__(self, raw, target):
        # raw    : per-head logits, shape (N, D, n_ens)
        # target : binary labels {0, 1},        shape (N, D)
        target = target.float()

        p_per_head = torch.sigmoid(raw)              # (N, D, n_ens)
        p_bar = p_per_head.mean(dim=-1)              # (N, D)

        eps = 1e-7
        p_bar = torch.clamp(p_bar, eps, 1.0 - eps)

        if self.pos_weight is not None:
            pw = self.pos_weight
            if not isinstance(pw, torch.Tensor):
                pw = torch.tensor(pw, dtype=p_bar.dtype, device=p_bar.device)
            else:
                pw = pw.to(dtype=p_bar.dtype, device=p_bar.device)
            bce = -(
                pw * target * torch.log(p_bar)
                + (1.0 - target) * torch.log(1.0 - p_bar)
            )
        else:
            bce = -(
                target * torch.log(p_bar)
                + (1.0 - target) * torch.log(1.0 - p_bar)
            )

        # Optional anti-collapse: rewards probability-space spread; disabled
        # by default (weight=0). Only activated when the ensemble has >1 head.
        if self.anti_collapse_weight > 0 and raw.shape[-1] > 1:
            p_var = p_per_head.var(dim=-1)           # (N, D)
            bce = bce - self.anti_collapse_weight * torch.log(
                p_var + self.anti_collapse_eps
            )

        if self.reduction == "sum":
            return bce.sum()
        if self.reduction == "none":
            return bce
        return bce.mean()


class Ensemble_multi_classification:
    """Mean-of-probabilities ensemble loss for multi-class classification.

    Direct generalisation of ``Ensemble_binary_classification``: per-head softmax,
    averaged across heads, NLL on the true class:

        p_bar = (1 / n_ens) * sum_i softmax(l_i)
        L     = -log p_bar[y]

    As in the binary case, the nonlinear aggregation gives every head a
    direct gradient and routes ensemble disagreement into p_bar without an
    auxiliary KL regulariser. The calibrated uncertainty at inference is the
    per-class probability-space std (or any scalar summary thereof).

    Parameters
    ----------
    reduction : {"mean", "sum", "none"}, default "mean"
    class_weight : torch.Tensor or sequence, optional
        Per-class weights applied to the per-sample NLL. Same shape semantics
        as ``CrossEntropyLoss``'s ``weight`` argument (length C).
    anti_collapse_weight : float, default 0.0
    anti_collapse_eps : float, default 1e-4
    """

    def __init__(
        self,
        reduction="mean",
        class_weight=None,
        anti_collapse_weight=0.0,
        anti_collapse_eps=1e-4,
        **kwargs,
    ):
        assert reduction in ["mean", "sum", "none"], (
            f"Unknown reduction type: '{reduction}'"
            " (choose from 'mean', 'sum', or 'none')"
        )
        self.reduction = reduction
        self.class_weight = class_weight
        self.anti_collapse_weight = anti_collapse_weight
        self.anti_collapse_eps = anti_collapse_eps

    def __call__(self, raw, target):
        # raw    : per-head logits,       shape (N, K, n_ens)
        # target : class indices in [0,K), shape (N,) or (N, 1)
        if target.dim() > 1:
            target = target.squeeze(-1)
        target = target.long()

        p_per_head = torch.softmax(raw, dim=1)       # (N, K, n_ens)
        p_bar = p_per_head.mean(dim=-1)              # (N, K)

        eps = 1e-7
        p_bar = torch.clamp(p_bar, eps, 1.0)

        N = p_bar.shape[0]
        nll = -torch.log(p_bar[torch.arange(N, device=p_bar.device), target])

        if self.class_weight is not None:
            cw = self.class_weight
            if not isinstance(cw, torch.Tensor):
                cw = torch.tensor(cw, dtype=p_bar.dtype, device=p_bar.device)
            else:
                cw = cw.to(dtype=p_bar.dtype, device=p_bar.device)
            nll = nll * cw[target]

        if self.anti_collapse_weight > 0 and raw.shape[-1] > 1:
            # Sum per-class probability variances; encourages disagreement
            # in any direction on the simplex.
            p_var = p_per_head.var(dim=-1).sum(dim=-1)  # (N,)
            nll = nll - self.anti_collapse_weight * torch.log(
                p_var + self.anti_collapse_eps
            )

        if self.reduction == "sum":
            return nll.sum()
        if self.reduction == "none":
            return nll
        return nll.mean()


class Ensemble_regression:
    """Negative log-likelihood loss for regression using a Gaussian likelihood.

    Models each target as a Gaussian whose mean is ``prediction`` and whose
    standard deviation is ``uncertainty`` (the ensemble spread). The loss is:

        NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)

    When ``ensemble_size=1``, ``uncertainty`` is identically zero and the loss
    degenerates to MSE (the variance term is clamped to avoid log(0)).

    Parameters
    ----------
    reduction : str, optional
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``. Default
        ``'mean'``.
    """

    def __init__(self, reduction="mean", **kwargs):
        assert reduction in ["mean", "sum", "none"], (
            f"Unknown reduction type: '{reduction}'"
            " (choose from 'mean', 'sum', or 'none')"
        )
        self.reduction = reduction

    def __call__(self, prediction, target, uncertainty):
        variance = torch.clip(torch.square(uncertainty), min=1e-6)
        l1 = torch.log(variance)
        l2 = nn.functional.mse_loss(prediction, target, reduction="none") / variance
        nll = 0.5 * (l1 + l2)
        if self.reduction == "sum":
            loss = nll.sum()
        elif self.reduction == "none":
            loss = nll
        else:
            loss = nll.mean()
        return loss


LOSS_FN = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
    "ensemble_regression": Ensemble_regression,
    "ensemble_binary_classification": Ensemble_binary_classification,
    "ensemble_multi_classification": Ensemble_multi_classification,
}

# legacy loss_type names kept for backward compatibility: configs/checkpoints
# from before the rename store these strings, and canonical_loss_type() maps them
# onto the current keys so old baselines load unchanged.
_LOSS_TYPE_ALIASES = {
    "nll_regression": "ensemble_regression",
    "mop_binary_classification": "ensemble_binary_classification",
    "mop_multi_classification": "ensemble_multi_classification",
}


def canonical_loss_type(loss_type):
    """Map a possibly-legacy loss_type string onto its current canonical name."""
    return _LOSS_TYPE_ALIASES.get(loss_type, loss_type)

OPTIMIZER = {
    "adam": Adam,
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
    "adamw": AdamW,
}


SCHEDULER = {
    "cosine":   {"cls": CosineAnnealingLR,  "granularity": "epoch", "needs_val_loss": False},
    "plateau":  {"cls": ReduceLROnPlateau,  "granularity": "epoch", "needs_val_loss": True},
    "onecycle": {"cls": OneCycleLR,         "granularity": "batch", "needs_val_loss": False},
}


class SchedulerWrapper:
    """Thin wrapper that exposes a uniform step interface for any LR scheduler.

    Two motivations:

    1. The training loop calls ``step_batch()`` after every optimizer step and
       ``step_epoch(val_loss)`` after every validation pass, unconditionally.
       The wrapper dispatches based on its declared ``granularity`` and
       silently no-ops when ``scheduler is None`` (the "no scheduler" case),
       so the training loop never needs to branch on scheduler type or
       presence.
    2. ``ReduceLROnPlateau`` takes ``val_loss`` as a step argument while every
       other scheduler does not. The wrapper hides that asymmetry via the
       ``needs_val_loss`` flag baked into the SCHEDULER registry.

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler.LRScheduler or None
        The underlying scheduler (or None for the no-op case).
    granularity : {"batch", "epoch", None}
        When the underlying scheduler expects ``.step()`` to be called.
    needs_val_loss : bool
        Whether ``.step()`` consumes ``val_loss`` as an argument
        (true for ReduceLROnPlateau, false otherwise).
    """

    def __init__(self, scheduler, granularity, needs_val_loss):
        self.scheduler = scheduler
        self.granularity = granularity
        self.needs_val_loss = needs_val_loss

    def step_batch(self):
        if self.scheduler is not None and self.granularity == "batch":
            self.scheduler.step()

    def step_epoch(self, val_loss):
        if self.scheduler is not None and self.granularity == "epoch":
            if self.needs_val_loss:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

    def state_dict(self):
        return None if self.scheduler is None else self.scheduler.state_dict()

    def load_state_dict(self, sd):
        if self.scheduler is not None and sd is not None:
            self.scheduler.load_state_dict(sd)


class LossWrapper:
    """Unified callable wrapping any supported loss function.

    Normalizes the interface between standard PyTorch losses (MSE, MAE,
    BCE, CrossEntropy) and the custom NLL losses so the training loop can
    call a single object regardless of loss type.

    Extracts ``prediction_mean``, ``prediction_std``, and ``y`` from a
    ``torch_geometric.data.Data`` batch, or unpacks a ``(prediction, target,
    uncertainty)`` tuple directly.

    Parameters
    ----------
    loss_type : str
        One of the keys in ``LOSS_FN``: ``'mse'``, ``'mae'``,
        ``'cross_entropy'``, ``'bce'``, ``'ensemble_regression'``,
        ``'ensemble_binary_classification'``, or ``'ensemble_multi_classification'``.
    **kwargs
        Additional keyword arguments forwarded to the underlying loss
        constructor (e.g. ``pos_weight``, ``class_weight``, ``kl_weight``).
    """

    def __init__(self, loss_type, **kwargs):
        loss_type = canonical_loss_type(loss_type)
        assert loss_type in LOSS_FN, f"Unknown loss type: '{loss_type}'"
        self.loss_type = loss_type

        # convert pos_weight to tensor if provided
        if "pos_weight" in kwargs and kwargs["pos_weight"] is not None:
            if not isinstance(kwargs["pos_weight"], torch.Tensor):
                kwargs["pos_weight"] = torch.tensor(kwargs["pos_weight"])

        # convert class_weight to tensor; rename to 'weight' for CrossEntropyLoss
        # (NLL multiclass accepts class_weight directly)
        if "class_weight" in kwargs:
            class_weight = kwargs.pop("class_weight")
            if class_weight is not None:
                if not isinstance(class_weight, torch.Tensor):
                    class_weight = torch.tensor(class_weight, dtype=torch.float)
            if loss_type == "cross_entropy":
                if class_weight is not None:
                    kwargs["weight"] = class_weight
            else:
                kwargs["class_weight"] = class_weight

        self.loss_function = LOSS_FN[loss_type](**kwargs, reduction="mean")

    def update_epoch(self, epoch: int) -> None:
        """Forward epoch updates to the inner loss function (e.g. for KL annealing)."""
        if hasattr(self.loss_function, "update_epoch"):
            self.loss_function.update_epoch(epoch)

    def __call__(self, output):
        raw = None
        if isinstance(output, Data):
            # prediction_mean/std are already edge-masked by the model's subselect_edge_embeddings.
            # targets are stored per graph as subgraph-only values, so no further masking needed.
            prediction = output.prediction_mean
            uncertainty = output.prediction_std
            target = output.y
            # per-head outputs for MoP losses (may be absent for older models)
            raw = getattr(output, "prediction_raw", None)

        elif isinstance(output, (tuple, list)):
            prediction, target, uncertainty = output
        else:
            raise TypeError(f"Unknown output type: {type(output)}")

        # cross_entropy expects 1D target (class indices), not 2D
        if self.loss_type == "cross_entropy" and target.dim() > 1:
            target = target.squeeze(-1)

        if self.loss_type in ["ensemble_binary_classification", "ensemble_multi_classification"]:
            if raw is None:
                raise ValueError(
                    f"Loss '{self.loss_type}' requires per-head ensemble outputs"
                    " (Data.prediction_raw); pass a Data batch from a model that"
                    " attaches prediction_raw."
                )
            return self.loss_function(raw, target)

        if self.loss_type == "ensemble_regression":
            return self.loss_function(prediction, target, uncertainty)
        else:
            return self.loss_function(prediction, target)


def extract_optimizer_config(params):
    """Extract optimizer settings from a flat parameter dict.

    Parameters
    ----------
    params : dict
        Parameter dict with optional keys ``optimizer_type``,
        ``learning_rate``, and ``weight_decay``.

    Returns
    -------
    dict
        Keys: ``'optimizer_type'``, ``'lr'``, ``'weight_decay'``.
    """
    return {
        "optimizer_type": params.get("optimizer_type", "adamw"),
        "lr": params.get("learning_rate", 0.0001),
        "weight_decay": params.get("weight_decay", 0),
    }


def extract_loss_config(params):
    """Extract loss function settings from a flat parameter dict.

    Returns only the parameters relevant to the specified ``loss_type``
    (e.g. ``pos_weight`` for BCE, ``anti_collapse_weight`` for ensemble classification).

    Parameters
    ----------
    params : dict
        Parameter dict containing at least ``'loss_type'`` and any
        loss-specific keys.

    Returns
    -------
    dict
        Dict with ``'loss_type'`` and loss-specific keyword arguments.
    """
    loss_type = canonical_loss_type(params["loss_type"])
    if loss_type == "bce":
        return {
            "loss_type": loss_type,
            "pos_weight": params.get("pos_weight", None),
        }
    elif loss_type == "ensemble_binary_classification":
        return {
            "loss_type": loss_type,
            "pos_weight": params.get("pos_weight", None),
            "anti_collapse_weight": params.get("anti_collapse_weight", 0.0),
            "anti_collapse_eps": params.get("anti_collapse_eps", 1e-4),
        }
    elif loss_type == "ensemble_multi_classification":
        return {
            "loss_type": loss_type,
            "class_weight": params.get("class_weight", None),
            "anti_collapse_weight": params.get("anti_collapse_weight", 0.0),
            "anti_collapse_eps": params.get("anti_collapse_eps", 1e-4),
        }
    elif loss_type == "cross_entropy":
        return {
            "loss_type": loss_type,
            "class_weight": params.get("class_weight", None),
        }
    else:
        return {"loss_type": loss_type}


def build_loss_fn(loss_type, **loss_kwargs):
    """Construct and return a ``LossWrapper`` for the specified loss type.

    Parameters
    ----------
    loss_type : str
        Loss function name (must be a key in ``LOSS_FN``).
    **loss_kwargs
        Additional arguments forwarded to the underlying loss constructor.

    Returns
    -------
    LossWrapper
    """
    assert loss_type in LOSS_FN, f"Unknown loss function: '{loss_type}'"
    return LossWrapper(loss_type=loss_type, **loss_kwargs)


def build_optimizer(optimizer_type, **optimiter_kwargs):
    """Construct and return an optimizer for the specified type.

    Parameters
    ----------
    optimizer_type : str
        Optimizer name (must be a key in ``OPTIMIZER``): ``'adam'``,
        ``'sgd'``, ``'rmsprop'``, ``'adagrad'``, or ``'adamw'``.
    **optimiter_kwargs
        Arguments forwarded to the optimizer constructor (e.g. ``lr``,
        ``weight_decay``, ``params``).

    Returns
    -------
    torch.optim.Optimizer
    """
    assert optimizer_type in OPTIMIZER, f"Unknown optimizer: '{optimizer_type}'"
    return OPTIMIZER[optimizer_type](**optimiter_kwargs)


def extract_scheduler_config(params):
    """Extract LR scheduler settings from a flat parameter dict.

    Both keys are optional. Absence of ``scheduler_type`` means "no
    scheduler" and yields a no-op SchedulerWrapper at build time, which
    preserves byte-identical behavior for configs predating this feature.

    Parameters
    ----------
    params : dict
        Parameter dict with optional keys ``scheduler_type`` and
        ``scheduler_config``.

    Returns
    -------
    dict
        Keys: ``'scheduler_type'`` (str or None), ``'scheduler_config'`` (dict).
    """
    return {
        "scheduler_type": params.get("scheduler_type"),
        "scheduler_config": params.get("scheduler_config") or {},
    }


def build_scheduler(scheduler_type, scheduler_config, optimizer, epochs, steps_per_epoch):
    """Construct a ``SchedulerWrapper`` for the specified scheduler type.

    Auto-derives the per-scheduler params that depend on training length
    (``T_max`` for cosine, ``total_steps`` / ``max_lr`` for onecycle) when
    the user does not supply them. User-supplied values in
    ``scheduler_config`` always take precedence via ``dict.setdefault``.

    Parameters
    ----------
    scheduler_type : str or None
        One of the keys in ``SCHEDULER``, or ``None`` for "no scheduler".
    scheduler_config : dict
        Per-scheduler kwargs forwarded to the underlying scheduler's
        constructor.
    optimizer : torch.optim.Optimizer
        Optimizer the scheduler will modify.
    epochs : int
        Total training epochs (used to derive ``T_max`` / ``total_steps``).
    steps_per_epoch : int
        Batches per epoch (used to derive ``total_steps`` for per-batch
        schedulers).

    Returns
    -------
    SchedulerWrapper
    """
    if scheduler_type is None:
        return SchedulerWrapper(scheduler=None, granularity=None, needs_val_loss=False)

    assert scheduler_type in SCHEDULER, (
        f"Unknown scheduler: '{scheduler_type}' "
        f"(choose from {sorted(SCHEDULER.keys())} or None)"
    )

    meta = SCHEDULER[scheduler_type]
    cfg = dict(scheduler_config)  # don't mutate caller's dict

    if scheduler_type == "cosine":
        cfg.setdefault("T_max", epochs)
    elif scheduler_type == "onecycle":
        cfg.setdefault("total_steps", epochs * steps_per_epoch)
        cfg.setdefault("max_lr", optimizer.param_groups[0]["lr"])
    elif scheduler_type == "plateau":
        cfg.setdefault("mode", "min")

    scheduler = meta["cls"](optimizer, **cfg)
    return SchedulerWrapper(
        scheduler=scheduler,
        granularity=meta["granularity"],
        needs_val_loss=meta["needs_val_loss"],
    )
