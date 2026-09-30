from elemenet.model import extract_model_config
from elemenet.trainer import (
    extract_loss_config,
    extract_optimizer_config,
    extract_scheduler_config,
    Trainer,
    NaNLossError,
)
from elemenet.utils import set_seed, estimate_training_memory
import yaml
import optuna
import os
import signal
import torch
import torch.distributed as dist
import gc
from datetime import datetime, timedelta


class NCCLWatchdogError(RuntimeError):
    """NCCL watchdog fired SIGABRT — collective timeout or OOM in a peer rank."""
    pass


def _sigabrt_to_exception(signum, frame):
    # raising here prevents abort() from reaching _Exit(), converting the
    # otherwise-unrecoverable SIGABRT into a catchable Python exception.
    # this works for any rank executing Python code at signal delivery time.
    # ranks blocked inside NCCL C++ may still be killed before this fires.
    raise NCCLWatchdogError(
        "NCCL watchdog fired (SIGABRT): collective timeout or peer OOM"
    )


# suppress Optuna's internal INFO logs (study creation, trial results)
# our own print statements provide cleaner, more relevant output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# timeout for per-trial DDP process groups.  short enough to surface a worker
# OOM quickly (instead of waiting the default 10-minute NCCL watchdog), but
# long enough not to fire spuriously on slow forward/backward passes.
_TRIAL_PG_TIMEOUT = timedelta(seconds=120)


# tunable keys per search-space slice — used only by _summarize_pinned to
# decide which user-fixed values to log as "pinned in search". Kept in sync
# with the dict_key arguments passed to _maybe_suggest in the search_space_*
# functions below; out-of-search keys (e.g. inferred input_size) are skipped.
_TUNABLE_ENCODER_KEYS = {
    "layers", "neurons", "dropout", "activation", "convolution", "shape",
    "inv_sublayers", "num_gaussians", "attention", "distance_embedding",
    "aggregation_method",
}
_TUNABLE_READOUT_KEYS = {
    "layers", "neurons", "dropout", "activation", "shape", "use_norm",
    "graph_attr_hidden_dim", "expansion", "num_heads",
}
_TUNABLE_TOP_KEYS = {"pooling", "learning_rate", "weight_decay"}


def _maybe_suggest(trial, fixed, dict_key, suggest_fn):
    """Skip Optuna's search dimension for any param the user has pinned.

    If ``dict_key`` is present in ``fixed`` (even with value ``None``), return
    the pinned value without calling ``trial.suggest_*``. Otherwise call
    ``suggest_fn()`` so Optuna samples the value normally.

    Bypassing ``trial.suggest_*`` is what makes pinning meaningful: a pinned
    key never enters ``trial.params``, so TPE never builds a surrogate over
    a dimension that has no effect on the trained model.
    """
    if fixed is not None and dict_key in fixed:
        return fixed[dict_key]
    return suggest_fn()


def _summarize_pinned(fixed_params):
    """Flatten user-pinned tunable params for one-shot logging at hypersearch start."""
    pinned = {}
    for k, v in (fixed_params.get("encoder_config") or {}).items():
        if k in _TUNABLE_ENCODER_KEYS:
            pinned[f"encoder_config.{k}"] = v
    for k, v in (fixed_params.get("readout_config") or {}).items():
        if k in _TUNABLE_READOUT_KEYS:
            pinned[f"readout_config.{k}"] = v
    for k in _TUNABLE_TOP_KEYS:
        if k in fixed_params:
            pinned[k] = fixed_params[k]
    return pinned


def optimize(
    dataset,
    fixed_params,
    epochs=100,
    max_trials=25,
    early_stopping=True,
    save_dir=None,
    quicksearch=False,
    batch_size=64,
    num_workers=0,
    random_seed=0,
    device=None,
    grad_clip_norm=None,
    max_grad_skips=0,
    _distributed=False,
):
    """
    Performs hyperparameter optimization to identify best model architecture. Validation data used in place of test data.
    INPUTS
        X_splits: tuple
            Length 3 tuple containing training and validation splits of X data.
        y_splits: tuple
            Length 3 tuple containing training and validation splits of y data.
        target_column: str
            String indicating which column of y data contains target property.
        task: str
            Learning task, either 'regression' or 'classification'.
        label_column: str
            Column with which to label test prediction.
        epochs: int
            Number of epochs to use during model training.
            default=100
        max_trials: int
            Maximum number of hyperparameter configurations considered during optimization.
            default=25
        save_dir: str
            Directory where model results are saved.
            default=os.path.join(path, 'model')
        quicksearch: boolean
            Flag to perform more a quick, less rigorous hyperparameter search.
            default=False
        edge_invariant: boolean
            Flag to drop all GNN features based on bond order, allowing for resonant-invariance.
            default=False
        batch_size: int
            Number of training samples processed at once. In practice, users should often set this as high as possible while remaining within memory limits.
            default=64
        num_workers: int
            Number of parallel subprocesses to use when loading data. Higher numbers increase speed but at the expense of memory usage.
            default=0
        random_seed: int
            Seed for reproducibility.
            default=0
    OUTPUTS
        best_params: dict
            Dictionary of optimal hyperparameters.
    """
    if not quicksearch:
        print("Starting hyperparameter optimization")
    else:
        print("Starting hyperparameter optimization with quicksearch")
    pinned = _summarize_pinned(fixed_params)
    if pinned:
        print(f"  Pinned (skipped in search): {pinned}")
    # seed for reproducibility
    set_seed(random_seed=random_seed)

    def objective(trial):
        """
        Objective function for use in hyperparameter optimization.
        """

        # seed for reproducibility
        current_trial = trial.number
        trial_seed = int(random_seed) + int(current_trial)
        set_seed(random_seed=trial_seed)

        # define parameters
        trial_params = search_space(
            trial=trial,
            encoder_type=fixed_params["encoder_type"],
            readout_type=fixed_params["readout_type"],
            scope=fixed_params["scope"],
            quicksearch=quicksearch,
            fixed_params=fixed_params,
        )

        # estimate memory in GB
        est_mem, gpu_ram = estimate_training_memory(trial_params, batch_size)
        if est_mem > gpu_ram:
            print(
                f"\nTrial {current_trial}: SKIPPED — estimated memory "
                f"{est_mem:.1f} GB exceeds GPU capacity {gpu_ram:.1f} GB"
            )
            if _distributed:
                dist.broadcast_object_list([{"action": "skip"}], src=0)
            raise optuna.TrialPruned(
                f"Estimated memory {est_mem:.1f} GB exceeds GPU capacity {gpu_ram:.1f} GB"
            )

        for key in fixed_params:
            if isinstance(trial_params.get(key), dict):
                trial_params[key].update(fixed_params[key])
            else:
                trial_params[key] = fixed_params[key]

        total_label = "inf" if max_trials == -1 else max_trials - 1
        print(
            f"\nStarting trial {current_trial}/{total_label} "
            f"(seed: {trial_seed})"
        )
        print(f"  Parameters: {trial.params}")

        # broadcast this trial's params to all worker ranks so they can
        # participate in DDP training alongside rank 0.
        if _distributed:
            dist.broadcast_object_list([{
                "action": "train",
                "params": trial_params,
                "seed": trial_seed,
                "epochs": epochs,
                "early_stopping": early_stopping,
            }], src=0)

        # create fresh NCCL process group for this trial's DDP.
        # using a dedicated group keeps the outer (signalling) process group clean: if
        # any rank OOMs mid-training and destroys this group, NCCL propagates
        # the abort to the remaining ranks quickly, and all ranks return to the
        # outer group for the next broadcast without a 10-minute watchdog hang.
        trial_pg = (
            dist.new_group(backend='nccl', timeout=_TRIAL_PG_TIMEOUT)
            if _distributed else None
        )

        # add fixed parameters to trial parameters (but in a way that possible fixed arguments are overwritten if also in trial params)
        model_config = extract_model_config(trial_params)
        loss_config = extract_loss_config(trial_params)
        optimizer_config = extract_optimizer_config(trial_params)
        scheduler_config = extract_scheduler_config(trial_params)

        trainer = Trainer(
            model_config=model_config,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            device=device,
            process_group=trial_pg,
            grad_clip_norm=grad_clip_norm,
            max_grad_skips=max_grad_skips,
        )

        _old_sigabrt = signal.signal(signal.SIGABRT, _sigabrt_to_exception)
        try:
            metrics = trainer.train(
                dataset=dataset,
                epochs=epochs,
                early_stopping=early_stopping,
                batch_size=batch_size,
                num_workers=num_workers,
                random_seed=trial_seed,
                save_dir=None,
            )
            # val_loss = metrics["val_loss"]
            val_loss = metrics.get("best_val_loss", metrics["val_loss"])
        except NaNLossError as e:
            print(f"Trial {current_trial}: NaN loss — pruning trial.")
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned(str(e))
        except NCCLWatchdogError as e:
            # NCCL watchdog fired SIGABRT because a collective timed out.
            # the most common cause is a peer rank running out of GPU memory
            # inside a CUDA/NCCL call (OOM is not always catchable before NCCL
            # kills the communicator).  Reduce batch_size or model complexity
            # if this trial keeps being pruned.
            print(
                f"Trial {current_trial}: NCCL watchdog timeout (SIGABRT) — "
                f"likely GPU OOM on one or more ranks. Pruning trial.\n"
                f"  Hint: if this recurs, try reducing batch_size or "
                f"constraining the encoder search space."
            )
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned(str(e))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(
                e, torch.cuda.OutOfMemoryError
            ):
                print(f"Trial {current_trial}: OOM during training — pruning trial.")
                # aggressively free memory
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"OOM during training: {e}")
            elif _distributed and (
                "nccl" in str(e).lower() or "timeout" in str(e).lower()
            ):
                # a worker rank OOM'd and destroyed the trial process group,
                # which caused NCCL to abort this rank's collective and surface
                # a timeout/system error here.  Treat it as a pruned trial so
                # hypersearch can continue with the next trial.
                print(
                    f"Trial {current_trial}: distributed communication error "
                    f"(likely worker OOM) — pruning trial."
                )
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"Distributed training error: {e}")
            else:
                raise  # re-raise non-OOM, non-NCCL RuntimeErrors
        finally:
            signal.signal(signal.SIGABRT, _old_sigabrt)
            # always destroy the trial process group so all ranks return to the
            # clean outer process group, whether the trial succeeded or failed.
            if trial_pg is not None:
                try:
                    dist.destroy_process_group(trial_pg)
                except Exception:
                    pass

        print(f"Trial {current_trial} completed — val_loss: {val_loss:.4f}")

        # clear cache
        del trainer
        del metrics
        gc.collect()
        torch.cuda.empty_cache()

        return val_loss

    def best_so_far_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best = study.best_trial
            print(
                f"  Best so far — Trial {best.number}, "
                f"val_loss: {study.best_value:.4f}, "
                f"params: {best.params}"
            )
            # incrementally save best params so a walltime kill doesn't lose all results
            if save_dir:
                try:
                    best_params_so_far = params_to_space(
                        best.params,
                        encoder_type=fixed_params["encoder_type"],
                        readout_type=fixed_params["readout_type"],
                        scope=fixed_params["scope"],
                        quicksearch=quicksearch,
                        fixed_params=fixed_params,
                    )
                    for key in fixed_params:
                        if isinstance(best_params_so_far.get(key), dict):
                            best_params_so_far[key].update(fixed_params[key])
                        else:
                            best_params_so_far[key] = fixed_params[key]
                    os.makedirs(save_dir, exist_ok=True)
                    with open(os.path.join(save_dir, "opt_params.yaml"), "w") as f:
                        yaml.dump(best_params_so_far, f)
                except Exception as e:
                    print(f"  Warning: could not save opt_params.yaml after trial {best.number}: {e}")

    # create study with a descriptive, unique name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"elemenet_hypersearch_{timestamp}_{random_seed}"
    study = optuna.create_study(direction="minimize", study_name=study_name)
    n_trials = None if max_trials == -1 else max_trials
    study.optimize(objective, n_trials=n_trials, callbacks=[best_so_far_callback])

    # signal worker ranks that hypersearch is finished.
    if _distributed:
        dist.broadcast_object_list([{"action": "stop"}], src=0)

    # summarize results
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(
        f"\nHyperparameter optimization complete: "
        f"{len(completed)} completed, {len(pruned)} pruned "
        f"(out of {len(study.trials)} total trials)"
    )
    if not completed:
        raise RuntimeError(
            "All hyperparameter optimization trials were pruned (OOM, NaN loss, or memory estimate exceeded). "
            "Consider reducing batch_size, model complexity, or using quicksearch=True."
        )
    print(f"Best validation loss: {study.best_value:.4f}")
    # save best hyperparameters
    best_params = params_to_space(
        study.best_params,
        encoder_type=fixed_params["encoder_type"],
        readout_type=fixed_params["readout_type"],
        scope=fixed_params["scope"],
        quicksearch=quicksearch,
        fixed_params=fixed_params,
    )

    for key in fixed_params:
        if isinstance(best_params.get(key), dict):
            best_params[key].update(fixed_params[key])
        else:
            best_params[key] = fixed_params[key]
    print(f"Optimal hyperparameters identified as: {best_params}")
    if save_dir:
        # final save — also written incrementally by best_so_far_callback after each trial,
        # so a walltime kill still leaves the best result found so far on disk.
        # note: opt_params.yaml records architecture/optimizer params only; training/data
        # settings (epochs, batch_size, graph_format, etc.) are in config.yaml, which is
        # the authoritative record for resuming or reproducing a full run.
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "opt_params.yaml"), "w") as f:
            yaml.dump(best_params, f)
        print(f"Optimal hyperparameters saved to {save_dir}/opt_params.yaml")
    return best_params


def hypersearch_worker_loop(dataset, batch_size, num_workers, device, grad_clip_norm=None,
                            max_grad_skips=0):
    """
    Worker loop for non-main ranks during distributed hyperparameter optimization.
    Waits for trial signals from rank 0 and participates in each DDP training run,
    then exits when rank 0 signals that hypersearch is complete.
    """
    while True:
        msg = [None]
        dist.broadcast_object_list(msg, src=0)
        action = msg[0]["action"]

        if action == "stop":
            break
        elif action == "skip":
            continue
        elif action == "train":
            trial_params = msg[0]["params"]
            trial_seed = msg[0]["seed"]
            trial_epochs = msg[0]["epochs"]
            trial_early_stopping = msg[0]["early_stopping"]

            # mirror the rank-0 new_group call so all ranks enter the
            # collective simultaneously and get the same trial process group.
            trial_pg = dist.new_group(backend='nccl', timeout=_TRIAL_PG_TIMEOUT)

            set_seed(random_seed=trial_seed)
            model_config = extract_model_config(trial_params)
            loss_config = extract_loss_config(trial_params)
            optimizer_config = extract_optimizer_config(trial_params)
            scheduler_config = extract_scheduler_config(trial_params)

            trainer = Trainer(
                model_config=model_config,
                loss_config=loss_config,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config,
                device=device,
                process_group=trial_pg,
                grad_clip_norm=grad_clip_norm,
                max_grad_skips=max_grad_skips,
            )
            _old_sigabrt = signal.signal(signal.SIGABRT, _sigabrt_to_exception)
            try:
                trainer.train(
                    dataset=dataset,
                    epochs=trial_epochs,
                    early_stopping=trial_early_stopping,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    random_seed=trial_seed,
                    save_dir=None,
                )
            except Exception:
                pass  # rank 0 handles trial outcome; worker just needs to stay in sync
            finally:
                signal.signal(signal.SIGABRT, _old_sigabrt)
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                # destroy the trial process group so this rank returns to the
                # clean outer group before the next broadcast_object_list call.
                # if this rank OOM'd and the group is already corrupted, the
                # destroy aborts NCCL on all other ranks, letting them surface
                # the error and clean up their own trial_pg promptly.
                try:
                    dist.destroy_process_group(trial_pg)
                except Exception:
                    pass


def params_to_space(params, encoder_type, readout_type, scope, quicksearch=False, fixed_params=None):
    """Converts the final best params to the desired parameter dictionary format.

    ``fixed_params`` must match what was passed to ``search_space`` during the
    search: pinned keys were never sampled, so they are not in ``params`` and
    the DummyTrial would otherwise KeyError on them.
    """

    class DummyTrial:
        def __init__(self, params):
            self.params = params

        def suggest_categorical(self, name, choices):
            return self.params[name]

        def suggest_float(self, name, low, high, **kwargs):
            return self.params[name]

    return search_space(
        DummyTrial(params), encoder_type, readout_type, scope, quicksearch,
        fixed_params=fixed_params,
    )


def search_space_readout(trial, quicksearch=False, readout_type="mlp", scope="graph", fixed=None):
    """
    Defines hyperparameter search space for readout models.
    INPUTS
        quicksearch: boolean
            Flag to perform a quick, reduced hyperparameter search.
            default=False
        scope: str
            Prediction granularity ('graph', 'node', or 'edge'). When 'edge',
            graph_attr_hidden_dim is fixed to None since graph_attr is not
            injected for edge-level prediction.
            default='graph'
    OUTPUTS
        space: dict
            Searchable hyperparameter space.
    """
    fixed = fixed or {}
    if quicksearch:
        neurons = _maybe_suggest(trial, fixed, "neurons",
            lambda: trial.suggest_categorical("readout_neurons", [128, 256]))
        dropout = _maybe_suggest(trial, fixed, "dropout",
            lambda: trial.suggest_float("readout_dropout", 0, 0.3))
        layers = _maybe_suggest(trial, fixed, "layers",
            lambda: trial.suggest_categorical("readout_layers", [1, 2, 3]))
        trial_params = {
            "layers": layers,
            "neurons": neurons,
            "dropout": dropout,
            "use_norm": True,
            "graph_attr_hidden_dim": None,
            "shape": "constant",
            "activation": "relu",
        }
    else:
        neurons = _maybe_suggest(trial, fixed, "neurons",
            lambda: trial.suggest_categorical("readout_neurons", [64, 128, 256, 512]))
        dropout = _maybe_suggest(trial, fixed, "dropout",
            lambda: trial.suggest_float("readout_dropout", 0, 0.4))
        layers = _maybe_suggest(trial, fixed, "layers",
            lambda: trial.suggest_categorical("readout_layers", [1, 2, 3, 4, 5]))
        use_norm = _maybe_suggest(trial, fixed, "use_norm",
            lambda: trial.suggest_categorical("readout_use_norm", [True, False]))
        # graph_attr is not injected for scope='edge', so the graph_attr sub-MLP
        # hidden size is irrelevant — fix it to None instead of wasting a search dim
        if scope == "edge":
            graph_attr_hidden_dim = None
        else:
            graph_attr_hidden_dim = _maybe_suggest(trial, fixed, "graph_attr_hidden_dim",
                lambda: trial.suggest_categorical(
                    "readout_graph_attr_hidden_dim", [8, 16, 32, 64, None]
                ))
        activation = _maybe_suggest(trial, fixed, "activation",
            lambda: trial.suggest_categorical(
                "readout_activation", ["relu", "tanh", "leakyrelu", "gelu", "silu"]
            ))
        if readout_type == "transformer":
            shape = "constant"
        else:
            shape = _maybe_suggest(trial, fixed, "shape",
                lambda: trial.suggest_categorical(
                    "readout_shape",
                    ["constant", "increasing", "decreasing", "hourglass", "pyramid"],
                ))
        trial_params = {
            "layers": layers,
            "neurons": neurons,
            "dropout": dropout,
            "use_norm": use_norm,
            "graph_attr_hidden_dim": graph_attr_hidden_dim,
            "shape": shape,
            "activation": activation,
        }

    if readout_type == "transformer":
        expansion = _maybe_suggest(trial, fixed, "expansion",
            lambda: trial.suggest_categorical("readout_expansion", [2, 4, 8]))
        num_heads = _maybe_suggest(trial, fixed, "num_heads",
            lambda: trial.suggest_categorical("readout_num_heads", [2, 4, 8]))
        trial_params.update(
            {
                "expansion": expansion,
                "num_heads": num_heads,
            }
        )

    return trial_params


def search_space_encoder(trial, quicksearch=False, encoder_type="gnn", fixed=None):
    fixed = fixed or {}
    if quicksearch:
        layers = _maybe_suggest(trial, fixed, "layers",
            lambda: trial.suggest_categorical("encoder_layers", [3, 5, 7]))
        neurons = _maybe_suggest(trial, fixed, "neurons",
            lambda: trial.suggest_categorical("encoder_neurons", [128, 256]))
        dropout = _maybe_suggest(trial, fixed, "dropout",
            lambda: trial.suggest_float("encoder_dropout", 0, 0.3))
    else:
        layers = _maybe_suggest(trial, fixed, "layers",
            lambda: trial.suggest_categorical("encoder_layers", [3, 4, 5, 6, 7, 8, 9]))
        neurons = _maybe_suggest(trial, fixed, "neurons",
            lambda: trial.suggest_categorical("encoder_neurons", [64, 128, 256, 512]))
        dropout = _maybe_suggest(trial, fixed, "dropout",
            lambda: trial.suggest_float("encoder_dropout", 0, 0.4))

    if encoder_type == "mlp":
        if quicksearch:
            trial_params = {
                "layers": layers,
                "neurons": neurons,
                "activation": "relu",
                "shape": "constant",
                "dropout": dropout,
            }
        else:
            activation = _maybe_suggest(trial, fixed, "activation",
                lambda: trial.suggest_categorical(
                    "encoder_activation", ["relu", "tanh", "leakyrelu", "gelu", "silu"]
                ))
            shape = _maybe_suggest(trial, fixed, "shape",
                lambda: trial.suggest_categorical(
                    "encoder_shape",
                    ["constant", "increasing", "decreasing", "hourglass", "pyramid"],
                ))
            trial_params = {
                "layers": layers,
                "neurons": neurons,
                "activation": activation,
                "shape": shape,
                "dropout": dropout,
            }
    elif encoder_type == "gnn":
        if quicksearch:
            convolution = _maybe_suggest(trial, fixed, "convolution",
                lambda: trial.suggest_categorical(
                    "encoder_convolution",
                    ["gcnconv", "graphconv", "gineconv", "nnconv", "gat"],
                ))
            trial_params = {
                "layers": layers,
                "neurons": neurons,
                "activation": "relu",
                "convolution": convolution,
                "shape": "constant",
                "dropout": dropout,
            }
        else:
            activation = _maybe_suggest(trial, fixed, "activation",
                lambda: trial.suggest_categorical(
                    "encoder_activation", ["relu", "tanh", "leakyrelu", "gelu", "silu"]
                ))
            convolution = _maybe_suggest(trial, fixed, "convolution",
                lambda: trial.suggest_categorical(
                    "encoder_convolution",
                    ["gcnconv", "graphconv", "gineconv", "nnconv", "gat"],
                ))
            shape = _maybe_suggest(trial, fixed, "shape",
                lambda: trial.suggest_categorical(
                    "encoder_shape",
                    ["constant", "increasing", "decreasing", "hourglass", "pyramid"],
                ))
            trial_params = {
                "layers": layers,
                "neurons": neurons,
                "activation": activation,
                "convolution": convolution,
                "dropout": dropout,
                "shape": shape,
            }

    elif encoder_type == "egnn":
        trial_params = {
            "layers": layers,
            "neurons": neurons,
            "shape": "constant",
            "dropout": dropout,
        }
        if quicksearch:
            trial_params["activation"] = "relu"
            trial_params["inv_sublayers"] = 2
            trial_params["num_gaussians"] = 64
        else:
            trial_params["activation"] = _maybe_suggest(trial, fixed, "activation",
                lambda: trial.suggest_categorical(
                    "encoder_activation", ["relu", "tanh", "leakyrelu", "gelu", "silu"]
                ))
            trial_params["inv_sublayers"] = _maybe_suggest(trial, fixed, "inv_sublayers",
                lambda: trial.suggest_categorical(
                    "gnn_inv_sublayers", [1, 2, 3]
                ))
            trial_params["num_gaussians"] = _maybe_suggest(trial, fixed, "num_gaussians",
                lambda: trial.suggest_categorical(
                    "encoder_num_gaussians", [32, 64, 128]
                ))
            trial_params["attention"] = _maybe_suggest(trial, fixed, "attention",
                lambda: trial.suggest_categorical(
                    "encoder_attention", [True, False]
                ))
            trial_params["distance_embedding"] = _maybe_suggest(trial, fixed, "distance_embedding",
                lambda: trial.suggest_categorical(
                    "encoder_distance_embedding", [True, False]
                ))
            trial_params["aggregation_method"] = _maybe_suggest(trial, fixed, "aggregation_method",
                lambda: trial.suggest_categorical(
                    "encoder_aggregation_method", ["sum", "mean"]
                ))
            trial_params["tanh"] = _maybe_suggest(trial, fixed, "tanh",
                lambda: trial.suggest_categorical(
                    "encoder_tanh", [True, False]
                ))
            # coords_range only affects the model when tanh bounding is on, so it
            # is suggested conditionally — Optuna only explores it within tanh=True
            # trials (and gets real val-loss signal from it there).
            if trial_params["tanh"]:
                trial_params["coords_range"] = _maybe_suggest(trial, fixed, "coords_range",
                    lambda: trial.suggest_float(
                        "encoder_coords_range", 1.0, 20.0
                    ))
    return trial_params


def search_space(trial, encoder_type, readout_type, scope, quicksearch=False, fixed_params=None):
    """
    Defines hyperparameter search space.
    INPUTS
        trial: optuna.Trial
            Instance of Optuna trial
        task: str
            Learning task, either 'regression' or 'classification'.
        encoder_type: str
            Model architecture to use. Either 'mlp', 'gnn', or 'egnn'.
            default='mlp'
        quicksearch: boolean
            Flag to perform more a quick, less rigorous hyperparameter search.
            default=False
        fixed_params: dict, optional
            User-supplied values that should be pinned (skipped) in the
            search. Top-level keys (e.g. ``pooling``, ``learning_rate``,
            ``weight_decay``) override their suggest dimensions directly;
            nested ``encoder_config`` / ``readout_config`` sub-dicts are
            forwarded to the corresponding sub-search-spaces.
    OUTPUTS
        space: dict
            Searchable hyperparameter space.
    """
    fixed_params = fixed_params or {}
    fixed_encoder = fixed_params.get("encoder_config") or {}
    fixed_readout = fixed_params.get("readout_config") or {}

    if quicksearch:
        weight_decay = _maybe_suggest(trial, fixed_params, "weight_decay",
            lambda: trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True))
        learning_rate = _maybe_suggest(trial, fixed_params, "learning_rate",
            lambda: trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True))
    else:
        weight_decay = _maybe_suggest(trial, fixed_params, "weight_decay",
            lambda: trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True))
        learning_rate = _maybe_suggest(trial, fixed_params, "learning_rate",
            lambda: trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True))

    trial_params = {
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "encoder_type": encoder_type,
        "readout_type": readout_type,
        "scope": scope,
    }

    if encoder_type is None:
        # encoder-free (tabular) model: there is no encoder to tune, so none of
        # its dimensions are suggested and no trial budget is spent on them.
        trial_params["encoder_config"] = {}
    else:
        trial_params["encoder_config"] = search_space_encoder(
            trial, quicksearch=quicksearch, encoder_type=encoder_type, fixed=fixed_encoder
        )

    trial_params["readout_config"] = search_space_readout(
        trial, quicksearch=quicksearch, readout_type=readout_type, scope=scope, fixed=fixed_readout
    )

    if scope == "graph":
        if quicksearch:
            trial_params["pooling"] = "mean"
        else:
            trial_params["pooling"] = _maybe_suggest(trial, fixed_params, "pooling",
                lambda: trial.suggest_categorical(
                    "pooling", ["mean", "sum", "max"]
                ))

    return trial_params
