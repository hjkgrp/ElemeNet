from typing import Optional
from elemenet.preprocess import (
    drop_invalid_smiles,
    featurize,
    get_extra_features,
    get_split,
    load_tabular_features,
    scale_and_encode_user_features,
    mol_to_graph,
    scale,
    validate_inputs,
)
from elemenet.utils import seed_workers, normalize_to_float_list
import copy
import os
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm
import numpy as np


def attach_node_features(graphs, values_path, offsets_path, labels_path, mode, split):
    """Seed each graph's node features with precomputed per-atom vectors.

    Lets a foundation-model embedding (e.g. UMA per-atom output) serve as the
    *initial* node representation that the GNN/EGNN encoder then refines by
    message passing, rather than being pooled and fed straight to a readout.

    ``values_path`` is a stacked ``(total_atoms, D)`` array and ``offsets_path``
    an ``(n_graphs + 1,)`` index, so graph *i* owns rows
    ``offsets[i]:offsets[i+1]`` -- the same layout the encoder-free path uses.

    ROW-ORDER CONTRACT: blocks are matched to graphs BY POSITION, and atom *k*
    of block *i* must be atom *k* of graph *i*. Graph construction preserves the
    source .xyz atom order, so this holds when the embeddings were extracted from
    the same file in the same order. Both the per-graph atom count and (when
    ``labels_path`` is given) the identifier are checked, so a mismatch raises
    rather than silently pairing atoms with the wrong embeddings.

    Parameters
    ----------
    graphs : list of Data
        Graphs for one split, already carrying ``x`` and ``label``.
    values_path, offsets_path : str
        ``.npy`` files holding the stacked per-atom features and their offsets.
    labels_path : str or None
        One-column CSV of identifiers, one per graph, for the order check.
    mode : {'concat', 'replace'}
        ``concat`` appends the vectors to the existing node features (keeping
        ElemeNet's atom typing); ``replace`` uses them alone, which isolates
        what message passing adds to the embedding itself.
    split : str
        Split name, for error messages.
    """
    if mode not in ("concat", "replace"):
        raise ValueError(f"node_feature_mode must be 'concat' or 'replace', got '{mode}'.")
    values = np.load(values_path, mmap_mode="r")
    offsets = np.asarray(np.load(offsets_path), dtype=np.int64)
    n = len(offsets) - 1
    if n != len(graphs):
        raise ValueError(
            f"{split}: node-feature file describes {n:,} graphs but the split has "
            f"{len(graphs):,}. Per-atom node features are matched to graphs by row "
            "order, so the files must cover the same samples in the same order."
        )
    if int(offsets[-1]) != values.shape[0]:
        raise ValueError(
            f"{split}: offsets end at {int(offsets[-1]):,} but '{values_path}' has "
            f"{values.shape[0]:,} atom rows."
        )
    labels = None
    if labels_path is not None:
        labels = pd.read_csv(labels_path).iloc[:, 0].tolist()
        if len(labels) != n:
            raise ValueError(
                f"{split}: {len(labels):,} labels for {n:,} graphs ({labels_path})."
            )
    for i, g in enumerate(graphs):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if g.x.shape[0] != hi - lo:
            raise ValueError(
                f"{split}: graph {i} has {g.x.shape[0]} atoms but its node-feature "
                f"block has {hi - lo}. The embeddings were extracted from a "
                "different structure or atom ordering than the graphs were built "
                "from; they cannot be aligned."
            )
        if labels is not None:
            gl = g.label[0] if isinstance(g.label, (list, tuple)) else g.label
            if gl != labels[i]:
                raise ValueError(
                    f"{split}: graph {i} is '{gl}' but the node-feature file has "
                    f"'{labels[i]}' at that position. The two are not row-aligned."
                )
        block = torch.tensor(np.asarray(values[lo:hi], dtype="float32"))
        g.x = block if mode == "replace" else torch.cat([g.x, block], dim=-1)
    return graphs


class GraphDataset(Dataset):
    """
    Dataset class to handle graph objects.
    """

    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


class DataModule:
    """Manages preprocessing, processing, and batched access to molecular graph datasets.

    The typical usage pattern is:
    1. Call ``preprocess()`` to featurize, split, and scale raw CSV data.
    2. Call ``process()`` to convert saved features into ``torch_geometric``
       ``Data`` objects and cache them to disk.
    3. Use ``train_dataloader()``, ``val_dataloader()``, and
       ``test_dataloader()`` during training and evaluation.

    Parameters
    ----------
    task : str
        Learning task: ``'regression'`` or ``'classification'``.
    """

    def __init__(self, task):
        self.task = task

    def preprocess(
        self,
        raw_data_path,
        mol_column,
        target_column,
        task,
        data_path=os.getcwd(),
        feature_data_path=None,
        feature_offsets_path=None,
        feature_labels_path=None,
        label_column=None,
        center_columns=None,
        feature_columns=None,
        graph_format="mol2",
        train_val_test_split=None,
        stratify=None,
        group_by=None,
        bond_scale_factor=1.0,
        random_seed=0,
        y_scaler=None,
    ):
        """
        Wrapper function for integrated preprocessing pipeline.
        Given a .csv file of molecular graphs, function featurizes, splits, and scales data.
        INPUTS:
            raw_data_path: str or list
                Path to data containing molecular graphs and other associated properties.
                For list inputs, assumes data is pre-split and provided as [train, val, test].
            mol_column: str
                Column containing molecular graphs.
            target_column: str
                String indicating which column of y data contains target property.
            task: str
                Learning task, either 'regression' or 'classification'.
            data_path: str
                Directory where results are saved.
                default=current working directory
            feature_data_path: list or None
                Paths to CSVs of precomputed features, ordered [train, val, test],
                for encoder-free models (``encoder_type=None``). When supplied, no
                molecular graph is built: features come from these files while
                targets and node/edge/graph attributes still come from
                ``raw_data_path``.

                IMPORTANT — rows are matched BY POSITION, not by joining on a key.
                Each feature file must contain exactly the same samples, in exactly
                the same order, as the corresponding ``raw_data_path`` split, one
                row each. ``label_column`` is compared elementwise between the two
                files and a mismatch raises rather than training on misaligned
                targets, but that check only catches disagreements it can see — keep
                the files in sync at the source.

                Every column is used as a feature except ``label_column`` and any
                column named in ``feature_columns`` (those are injected separately
                as graph attributes). Features must be numeric and finite.
                default=None
            label_column: str or None
                Identifier column (e.g. refcode) present in both ``raw_data_path``
                and ``feature_data_path``. Required when ``feature_data_path`` is
                given, so the positional alignment above can be verified.
                default=None
            center_columns: list
                Columns containing particularly relevant atoms upon which to center subgraphs.
                default=None
            feature_columns: list
                Columns containing extra features to process and return.
                default=None
            graph_format: str
                Format of molecular graph. Supported types are 'mol2', 'smiles', 'mol', and 'sdf'.
                default='mol2'
            train_val_test_split: list
                Percentages by which to split data into training, validation, and test sets. Must sum to 1.
                default=[0.8, 0.1, 0.1]
            stratify: str
                Column of y_data by which to stratify data splits. Useful for classification tasks on imbalanced datasets.
                default=None
            group_by: str or list
                Column(s) of y_data by which to group data so that all members of a group
                stay in the same split. Mutually exclusive with stratify.
                default=None
            random_seed: int
                Seed for reproducibility.
                default=0
        """
        # validate inputs
        presplit, task, center_columns, graph_format, drop_indices = (
            validate_inputs(
                raw_data_path,
                task,
                center_columns,
                graph_format,
                mol_column,
                train_val_test_split,
                stratify=stratify,
                group_by=group_by,
                bond_scale_factor=bond_scale_factor,
            )
        )

        # ensure compatibility for multi target prediction
        if isinstance(target_column, str):
            target_column = [target_column]

        if drop_indices:
            # drop any invalid SMILES
            if not presplit:
                drop_invalid_smiles(raw_data_path, drop_indices)
            else:
                for data_split, drop_ind in zip(raw_data_path, drop_indices):
                    drop_invalid_smiles(data_split, drop_ind)

        # default: every split is independent. Overridden in the user-supplied
        # splits branch below, where the same source file may be reused for
        # multiple splits (e.g. inference passes one file as all three). Lets
        # get_processed_features skip rebuilding graphs for duplicated splits.
        self._split_source_map = [0, 1, 2]
        if feature_data_path is not None:
            # encoder-free (tabular) models: features come from user-supplied CSVs
            # instead of being derived from molecular graphs, so no molecule is
            # parsed here. Targets and graph attributes still come from
            # raw_data_path; the feature files are matched to it by row order.
            if not presplit or len(feature_data_path) != 3:
                raise ValueError(
                    "feature_data_path must be a list of three feature files ordered "
                    "as [train, val, test], matching a presplit raw_data_path."
                )
            for name, paths in (("feature_offsets_path", feature_offsets_path),
                                ("feature_labels_path", feature_labels_path)):
                if paths is not None and len(paths) != 3:
                    raise ValueError(
                        f"{name} must be a list of three files ordered as "
                        f"[train, val, test], got {len(paths)}."
                    )
            offsets_paths = feature_offsets_path or [None] * 3
            labels_paths = feature_labels_path or [None] * 3
            target_cols = (
                [target_column] if isinstance(target_column, str) else list(target_column or [])
            )
            X_splits, y_splits, extra_features_splits = [], [], []
            atom_offsets = []
            for feat_path, raw_path, off_path, lab_path in zip(
                feature_data_path, raw_data_path, offsets_paths, labels_paths
            ):
                X, offsets = load_tabular_features(
                    feat_path, raw_path, label_column, feature_columns,
                    offsets_path=off_path, labels_path=lab_path,
                )
                X_splits.append(X)
                atom_offsets.append(offsets)
                keep = [label_column] + [c for c in target_cols if c != label_column]
                raw = pd.concat(
                    pd.read_csv(raw_path, chunksize=20000), ignore_index=True
                )
                y_splits.append(raw[[c for c in keep if c in raw.columns]].copy())
                extra = get_extra_features(data=raw, feature_columns=feature_columns)
                extra, _ = scale_and_encode_user_features(extra)
                extra_features_splits.append(extra)
            per_atom = atom_offsets[0] is not None
            kind = "atom rows" if per_atom else "rows"
            print(
                f"Loaded precomputed {'per-atom' if per_atom else 'graph-level'} "
                f"features: {X_splits[0].shape[1]} columns, "
                f"{[len(x) for x in X_splits]} {kind} per split."
            )
        elif not presplit:
            # featurize
            X_data, y_data, extra_features = featurize(
                data_path=raw_data_path,
                mol_column=mol_column,
                save_dir=data_path,
                center_columns=center_columns,
                feature_columns=feature_columns,
                graph_format=graph_format,
            )
            # split
            X_splits, y_splits, extra_features_splits = get_split(
                X_data=X_data,
                y_data=y_data,
                train_val_test_split=train_val_test_split,
                random_seed=random_seed,
                stratify=stratify,
                group_by=group_by,
                extra_features=extra_features,
            )
        # accept user-defined data splits
        else:
            # detect splits that share an identical source path so each unique
            # source is featurized only once; duplicate splits reuse an
            # independent deep copy (kept independent so the in-place scaling
            # below stays isolated per split). Distinct paths (the training case)
            # all map to themselves, leaving behavior unchanged.
            self._split_source_map = []
            _seen_sources = {}
            for idx, data_split in enumerate(raw_data_path):
                key = data_split if isinstance(data_split, str) else id(data_split)
                self._split_source_map.append(_seen_sources.setdefault(key, idx))

            X_splits, y_splits, extra_features_splits = [], [], []
            for idx, data_split in enumerate(raw_data_path):
                src = self._split_source_map[idx]
                if src != idx:
                    # identical source already featurized above; deep-copy it
                    # rather than recomputing the features.
                    X_data = copy.deepcopy(X_splits[src])
                    y_data = copy.deepcopy(y_splits[src])
                    extra_features = copy.deepcopy(extra_features_splits[src])
                else:
                    # featurize
                    X_data, y_data, extra_features = featurize(
                        data_path=data_split,
                        mol_column=mol_column,
                        save_dir=data_path,
                        center_columns=center_columns,
                        feature_columns=feature_columns,
                        graph_format=graph_format,
                    )
                X_splits.append(X_data)
                y_splits.append(y_data)
                extra_features_splits.append(extra_features)

        # convert targets to list(float) per column (skipped when no target
        # column is supplied, e.g. prediction-only inference on unlabelled data)
        if target_column is not None:
            for i in range(len(y_splits)):
                y_splits[i][target_column] = y_splits[i][target_column].map(
                    normalize_to_float_list
                )

        # scale (do for regression tasks only). ``y_scaler`` is None by default,
        # which preserves training-pipeline behaviour (fit a fresh scaler on
        # ``y_train``). ``inference_pipeline`` passes the training-time scaler
        # so that ``y_train``/``y_val``/``y_test`` are all transformed with the
        # same scaler the model was trained against, rather than a fresh one
        # fitted on the inference data.
        scale(
            X_splits,
            y_splits,
            target_column=target_column,
            scale_features=feature_data_path is not None,
            scale=(task == "regression") and (target_column is not None),
            save_dir=data_path,
            y_scaler=y_scaler,
        )

        # per-atom feature blocks: offsets ride alongside the feature tables so
        # process() can slice the stacked (total_atoms, D) matrix back into samples
        if feature_data_path is not None and atom_offsets[0] is not None:
            os.makedirs(os.path.join(data_path, "X_data"), exist_ok=True)
            for name, offsets in zip(["train", "val", "test"], atom_offsets):
                np.save(
                    os.path.join(data_path, "X_data", f"atom_offsets_{name}.npy"),
                    offsets,
                )

        # save extra features
        if feature_columns:
            for name, ef in zip(["train", "val", "test"], extra_features_splits):
                # node (atom) features: list of dicts of dicts (one per molecule)
                # feature names are preserved as dict keys
                if not ef["node"].empty:
                    node_path = os.path.join(
                        data_path, "X_data", f"extra_node_features_{name}.pkl"
                    )
                    pd.to_pickle(ef["node"], node_path)

                # edge (bond) features: list of dicts of dicts (one per molecule)
                # feature names are preserved as dict keys
                if not ef["edge"].empty:
                    edge_path = os.path.join(
                        data_path, "X_data", f"extra_edge_features_{name}.pkl"
                    )
                    pd.to_pickle(ef["edge"], edge_path)

                # graph (molecule) features: DataFrame
                # column names are preserved in the DataFrame
                if not ef["graph"].empty:
                    graph_path = os.path.join(
                        data_path, "X_data", f"extra_graph_features_{name}.pkl"
                    )
                    pd.to_pickle(ef["graph"], graph_path)

        print("Preprocessing pipeline completed.")

    def infer_number_of_classes(self, y_data, target_column):
        """Infer the number of classes from training labels and store sorted class array.

        Parameters
        ----------
        y_data : pd.DataFrame
            Training target DataFrame with list-valued entries in each
            ``target_column``.
        target_column : list of str
            Names of target columns to examine.

        Returns
        -------
        int
            Number of unique classes (must be identical across all targets).

        Raises
        ------
        AssertionError
            If targets have different numbers of classes.
        NotImplementedError
            If multi-target multi-class (>2 classes) classification is requested.
        """
        all_labels = np.column_stack(
            [np.concatenate(y_data[col].values) for col in target_column]
        )
        # use unique values so num_classes is correct regardless of label indexing (0-based, 1-based, etc.)
        classes_per_col = [np.unique(all_labels[:, i]) for i in range(all_labels.shape[1])]
        num_classes_per_col = np.array([len(c) for c in classes_per_col])

        assert all(
            num_classes_per_col[0] == num_classes_per_col
        ), "All targets must have the same number of classes for classification tasks."
        num_classes = int(num_classes_per_col[0])

        # check for multi-target multi-class classification
        if len(target_column) > 1 and num_classes > 2:
            raise NotImplementedError(
                "Multi-target multi-class classification is not currently supported."
            )

        # store sorted unique classes so we can re-index labels to 0-based
        self.classes = classes_per_col[0]
        return num_classes

    def get_processed_targets(
        self, data_path, target_column, X_graphs_split, scope, label_column=None
    ):
        """Load scaled targets for each split and attach them to graph objects.

        For classification tasks, infers the number of classes on the training
        split and builds a 0-based class-index mapping for multi-class targets.
        Attaches a ``y`` tensor and a ``label`` list to each graph in
        ``X_graphs_split``.

        Parameters
        ----------
        data_path : str
            Directory containing ``y_data/y_{split}.pkl`` and optionally
            ``y_data/y_{split}_scaled.pkl`` files.
        target_column : list of str
            Target column names.
        X_graphs_split : list of list of Data
            Graph lists for [train, val, test] splits; modified in-place.
        label_column : str or None, optional
            Column used as a sample identifier. If None, integer indices are
            used. Default None.

        Returns
        -------
        list of list of Data
            ``X_graphs_split`` with ``y`` and ``label`` attributes attached
            to each graph.
        """
        processed_graphs = []
        self.num_classes = None
        self.classes = None
        self.class_to_idx = None
        for i, split in enumerate(["train", "val", "test"]):
            default_path = os.path.join(data_path, "y_data", "y_" + split + ".pkl")
            scaled_path = os.path.join(
                data_path, "y_data", "y_" + split + "_scaled.pkl"
            )
            target_path = scaled_path if os.path.exists(scaled_path) else default_path

            # read target csv
            y_data = pd.read_pickle(target_path)

            y_labels = (
                y_data[label_column].values
                if label_column is not None
                else range(0, y_data.shape[0])
            )

            X_graph = X_graphs_split[i]

            # Prediction-only inference: no target column. Attach a dummy
            # per-graph ``y`` so the prediction loop (which reads ``batch.y``)
            # runs; it is never used for loss/metrics — callers gate on target
            # availability and discard these values.
            if target_column is None:
                assert len(X_graph) == y_data.shape[0], (
                    f"Mismatch between number of graphs and rows in {split} split."
                )
                for j in range(len(X_graph)):
                    X_graph[j].y = torch.zeros((1, 1))
                    X_graph[j].label = [y_labels[j]]
                processed_graphs.append(X_graph)
                continue

            assert len(X_graph) == len(
                y_data[target_column].values
            ), f"Mismatch between number of graphs and targets in {split} split."

            # for classification, store the number of classes per task
            if self.task == "classification" and split == "train":
                self.num_classes = self.infer_number_of_classes(y_data, target_column)
                # build 0-indexed mapping so CrossEntropyLoss always receives labels in [0, num_classes)
                if self.num_classes > 2:
                    self.class_to_idx = {int(c): i for i, c in enumerate(self.classes)}

            for i, value in enumerate(y_data[target_column].values):
                if self.task == "classification" and self.num_classes > 2:
                    # reindex class labels to 0-based before converting to long tensor
                    raw = np.column_stack(value)
                    remapped = np.searchsorted(self.classes, raw)
                    y_tensor = torch.tensor(remapped).long()
                else:
                    # for binary classification or regression, convert to float tensor
                    y_tensor = torch.tensor(np.column_stack(value)).float()

                # filter node/edge-level targets to match heavy atoms when implicit_Hs was used.
                # use scope to pick the correct filter — using length alone could spuriously
                # fire the bond filter on a node target whose length coincidentally equalled
                # the original bond count (e.g., O=S(=O)([O-])O, 5 heavy atoms == 5 bonds
                # post-AddHs).
                heavy_atom_idx = getattr(X_graph[i], "heavy_atom_indices", None)
                heavy_bond_idx = getattr(X_graph[i], "heavy_bond_indices", None)
                n_orig_atoms = getattr(X_graph[i], "original_n_atoms", None)
                n_orig_bonds = getattr(X_graph[i], "original_n_bonds", None)
                target_len = y_tensor.shape[0]
                if scope == "node":
                    if (
                        heavy_atom_idx is not None
                        and target_len > 1
                        and n_orig_atoms is not None
                        and target_len == n_orig_atoms
                    ):
                        y_tensor = y_tensor[heavy_atom_idx]
                elif scope == "edge":
                    if (
                        heavy_bond_idx is not None
                        and target_len > 1
                        and n_orig_bonds is not None
                        and target_len == n_orig_bonds
                    ):
                        y_tensor = y_tensor[heavy_bond_idx]

                # validate that implicit_Hs filtering produced a target aligned with the graph.
                # only checked when implicit_Hs was used — without implicit_Hs, moiety/subgraph
                # targets are legitimately shorter than num_nodes/num_edges (the model
                # subselects via node_mask/edge_mask/center_edge_mask in Model.forward).
                if heavy_atom_idx is not None or heavy_bond_idx is not None:
                    num_nodes = X_graph[i].x.shape[0]
                    num_edges = X_graph[i].edge_index.shape[1]
                    target_len = y_tensor.shape[0]
                    if target_len > 1 and target_len != num_nodes and target_len != num_edges:
                        raise ValueError(
                            f"Graph {i} in {split} split: target length ({target_len}) "
                            f"does not match num_nodes ({num_nodes}) or num_edges "
                            f"({num_edges}). With implicit_Hs=True, the target vector length "
                            f"must match either the original total atom/bond count (in which "
                            f"case it is auto-filtered) or the heavy atom/bond count "
                            f"(pre-filtered)."
                        )

                X_graph[i].y = y_tensor

                # labels are defined per graph, but targets are per graph/node/edge
                X_graph[i].label = [y_labels[i]] * X_graph[i].y.shape[0]

            processed_graphs.append(X_graph)

        return processed_graphs

    def get_processed_features(
        self,
        data_path,
        encoder_type,
        mol_column,
        graph_format,
        edge_invariant,
        center_column=None,
        feature_columns=None,
        k_hops=1,
        use_xtb=False,
        use_bulk=False,
        bond_scale_factor=1.0,
        charge_spin_override=False,
        implicit_Hs=False,
        rdkit_features=False,
    ):
        """Load or construct graph/feature objects for each split.

        For MLP encoder types, loads pre-scaled feature arrays from
        ``X_data/X_{split}.pkl``. For GNN/EGNN encoder types, loads molecular
        strings from ``X_data/X_{split}.pkl`` and converts each molecule to a
        ``torch_geometric`` ``Data`` object via ``mol_to_graph``, optionally
        attaching extra node, edge, and graph-level features.

        Parameters
        ----------
        data_path : str
            Directory containing ``X_data/`` sub-directories produced by
            ``preprocess()``.
        encoder_type : str
            Encoder architecture: ``'mlp'``, ``'gnn'``, or ``'egnn'``.
        mol_column : str
            Column containing molecular graph strings.
        graph_format : str
            Molecular graph format (``'mol2'``, ``'smiles'``, etc.).
        edge_invariant : bool
            Use only distance-based edge features.
        center_column : str or None, optional
            Column specifying the center atom index. Default None.
        feature_columns : list of str or None, optional
            Extra feature columns whose pickled files are loaded from
            ``X_data/``. Default None.
        k_hops : int, optional
            Neighborhood depth for subgraph extraction. Default 1.
        use_xtb : bool, optional
            Augment atom features with xTB properties. Default False.
        use_bulk : bool, optional
            Augment atom features with bulk descriptors. Default False.
        bond_scale_factor : float, optional
            Scaling factor for bond lengths. Default 1.0.
        charge_spin_override : bool, optional
            Use explicit charge/spin columns instead of inference. Default False.
        implicit_Hs : bool, optional
            Include implicit hydrogen atoms. Default False.

        Returns
        -------
        list of list of Data
            Graph/feature lists for [train, val, test] splits.
        """
        X_splits = []
        # splits whose source was identical to an earlier split (e.g. inference
        # feeds one file as all three) are deep-copied from the already-built
        # split instead of re-running graph construction. Defaults to no reuse
        # when the map is absent (process called without preprocess), and
        # training's distinct splits each map to themselves.
        source_map = getattr(self, "_split_source_map", None)
        # when every split shares one source (the inference case: a single dataset
        # fed as all three splits), label it generically rather than as "train".
        single_source = source_map is not None and len(set(source_map)) == 1
        for idx, split in enumerate(["train", "val", "test"]):
            if source_map is not None and source_map[idx] != idx:
                X_splits.append(copy.deepcopy(X_splits[source_map[idx]]))
                continue
            feature_desc = (
                "Processing features" if single_source else f"Processing {split} features"
            )
            if encoder_type is None:
                default_path = os.path.join(data_path, "X_data", "X_" + split + ".pkl")
                scaled_path = os.path.join(
                    data_path, "X_data", "X_" + split + "_scaled.pkl"
                )
                features_path = (
                    scaled_path if os.path.exists(scaled_path) else default_path
                )
                features = torch.tensor(
                    pd.read_pickle(features_path).to_numpy(dtype="float32")
                )
                # graph-level attributes (charge, spinmult, ...) are injected the
                # same way as for graph models, so feature_columns keeps working
                graph_attr = None
                if feature_columns:
                    graph_path = os.path.join(
                        data_path, "X_data", f"extra_graph_features_{split}.pkl"
                    )
                    if os.path.exists(graph_path):
                        graph_attr = torch.tensor(
                            pd.read_pickle(graph_path).to_numpy(dtype="float32")
                        )
                # per-atom features arrive as a stacked (total_atoms, D) table plus
                # offsets; sample i owns rows offsets[i]:offsets[i+1]. Without
                # offsets each row is one sample, x is (1, D), and no pooling is
                # needed. With offsets x is (n_atoms, D) and the readout pools (MLP)
                # or attends over atoms (transformer).
                offsets_path = os.path.join(
                    data_path, "X_data", f"atom_offsets_{split}.npy"
                )
                offsets = (
                    np.load(offsets_path) if os.path.exists(offsets_path) else None
                )
                if offsets is not None:
                    if len(features) != int(offsets[-1]):
                        raise ValueError(
                            f"{split}: offsets end at {int(offsets[-1])} but the "
                            f"feature table has {len(features)} atom rows."
                        )
                    n_samples = len(offsets) - 1
                    if graph_attr is not None and len(graph_attr) != n_samples:
                        raise ValueError(
                            f"{split}: {len(graph_attr)} graph-attribute rows for "
                            f"{n_samples} samples."
                        )
                else:
                    n_samples = len(features)
                    if graph_attr is not None and len(graph_attr) != n_samples:
                        raise ValueError(
                            f"{split}: {len(graph_attr)} graph-attribute rows for "
                            f"{n_samples} feature rows."
                        )
                data_ = []
                for i in tqdm(range(n_samples), total=n_samples, desc=feature_desc):
                    if offsets is None:
                        x = features[i].unsqueeze(0)
                    else:
                        x = features[int(offsets[i]) : int(offsets[i + 1])]
                    d = Data(x=x)
                    if graph_attr is not None:
                        d.graph_attr = graph_attr[i].unsqueeze(0)
                    data_.append(d)
            elif encoder_type in ["gnn", "egnn"]:
                mols = pd.read_pickle(
                    os.path.join(data_path, "X_data", f"X_{split}.pkl")
                )

                # load extra features if they exist
                extra_features = {
                    "node": pd.DataFrame(),
                    "edge": pd.DataFrame(),
                    "graph": pd.DataFrame(),
                }
                if feature_columns:
                    node_path = os.path.join(
                        data_path, "X_data", f"extra_node_features_{split}.pkl"
                    )
                    edge_path = os.path.join(
                        data_path, "X_data", f"extra_edge_features_{split}.pkl"
                    )
                    graph_path = os.path.join(
                        data_path, "X_data", f"extra_graph_features_{split}.pkl"
                    )

                    if os.path.exists(node_path):
                        extra_features["node"] = pd.read_pickle(node_path)
                    if os.path.exists(edge_path):
                        extra_features["edge"] = pd.read_pickle(edge_path)
                    if os.path.exists(graph_path):
                        extra_features["graph"] = pd.read_pickle(graph_path)

                data_ = []
                for i, mol in tqdm(
                    enumerate(mols[mol_column]),
                    total=len(mols),
                    desc=feature_desc,
                ):
                    center_idx = (
                        int(mols[center_column].iloc[i])
                        if center_column is not None
                        else None
                    )

                    # extract node and edge features for this molecule
                    extra_node_dict = (
                        extra_features["node"].iloc[i].to_dict()
                        if not extra_features["node"].empty
                        else None
                    )
                    extra_edge_dict = (
                        extra_features["edge"].iloc[i].to_dict()
                        if not extra_features["edge"].empty
                        else None
                    )
                    extra_graph_dict = (
                        extra_features["graph"].iloc[i].to_dict()
                        if not extra_features["graph"].empty
                        else None
                    )

                    data_.append(
                        mol_to_graph(
                            mol,
                            graph_format,
                            edge_invariant,
                            center_idx=center_idx,
                            extra_node_features=extra_node_dict,
                            extra_edge_features=extra_edge_dict,
                            extra_graph_features=extra_graph_dict,
                            k_hops=k_hops,
                            use_xtb=use_xtb,
                            use_bulk=use_bulk,
                            bond_scale_factor=bond_scale_factor,
                            charge_spin_override=charge_spin_override,
                            implicit_Hs=implicit_Hs,
                            rdkit_features=rdkit_features,
                        )
                    )
            X_splits.append(data_)
            if feature_columns and encoder_type in ("gnn", "egnn"):
                if not extra_features["node"].empty:
                    print(
                        f"Included extra node features: {list(extra_features['node'].columns)}"
                    )
                if not extra_features["edge"].empty:
                    print(
                        f"Included extra edge features: {list(extra_features['edge'].columns)}"
                    )
                if not extra_features["graph"].empty:
                    print(
                        f"Included extra graph features: {list(extra_features['graph'].columns)}"
                    )
        return X_splits

    def process(
        self,
        encoder_type,
        target_column,
        mol_column,
        scope,
        label_column=None,
        graph_format="mol2",
        edge_invariant=False,
        center_column: Optional[str] = None,
        feature_columns=False,
        k_hops=1,
        use_xtb=False,
        use_bulk=False,
        bond_scale_factor=1.0,
        charge_spin_override=False,
        data_path=os.getcwd(),
        implicit_Hs=False,
        rdkit_features=False,
        node_feature_path=None,
        node_feature_offsets_path=None,
        node_feature_labels_path=None,
        node_feature_mode="concat",
    ):
        """Build graph datasets and cache them to disk, or load from cache if available.

        If a ``.pt`` cache file already exists at
        ``<data_path>/processed_graphs_<target_columns>.pt``, loads it
        directly. Otherwise calls ``get_processed_features`` and
        ``get_processed_targets`` to construct all splits, then saves the
        result. Also loads the target scaler if present.

        Parameters
        ----------
        encoder_type : str
            Encoder architecture: ``'mlp'``, ``'gnn'``, or ``'egnn'``.
        target_column : str or list of str
            Target column name(s).
        mol_column : str
            Column containing molecular graph strings.
        label_column : str or None, optional
            Column used as sample identifiers. Default None.
        graph_format : str, optional
            Molecular graph format. Default ``'mol2'``.
        edge_invariant : bool, optional
            Use only distance-based edge features. Default False.
        center_column : str or None, optional
            Column specifying center atom indices. Default None.
        feature_columns : list of str or bool, optional
            Extra feature columns to include. Default False.
        k_hops : int, optional
            Neighborhood depth for subgraph extraction. Default 1.
        use_xtb : bool, optional
            Augment with xTB properties. Default False.
        use_bulk : bool, optional
            Augment with bulk descriptors. Default False.
        bond_scale_factor : float, optional
            Bond length scaling factor. Default 1.0.
        charge_spin_override : bool, optional
            Use explicit charge/spin columns. Default False.
        data_path : str, optional
            Directory for cached ``.pt`` files. Default current directory.
        implicit_Hs : bool, optional
            Include implicit hydrogen atoms. Default False.
        """
        if isinstance(target_column, str):
            target_column = [target_column]
        scaler_path = os.path.join(data_path, "y_scaler.pkl")
        self.scaler = None
        if os.path.exists(scaler_path):
            self.scaler = pd.read_pickle(scaler_path)
        # ``target_column`` is None for prediction-only inference on unlabelled data
        cache_name = "_".join(target_column) if target_column else "no_target"
        # graphs seeded with precomputed node features are a different dataset, so
        # they must not share a cache with the plain-featured build
        if node_feature_path is not None:
            cache_name += f"_nodefeat-{node_feature_mode}"
        save_path = os.path.join(
            data_path, f"processed_graphs_{cache_name}.pt"
        )
        if encoder_type == "egnn" and graph_format == "smiles":
            raise ValueError(
                "SMILES do not encode 3D coordinates and are not supported for EGNN models."
            )
        if os.path.exists(save_path):
            print(f"Found processed graphs at {save_path}. Loading...")
            data = torch.load(save_path, weights_only=False)
            self.graphs_splits = data["graphs_splits"]
            self.target_column = data["target_column"]
            self.label_column = data["label_column"]
            self.num_classes = data["num_classes"]
            self.max_nodes = data["max_nodes"]
            self.max_edges = data["max_edges"]
            self.classes = data.get("classes", None)
            self.class_to_idx = (
                {int(c): i for i, c in enumerate(self.classes)}
                if self.classes is not None
                else None
            )
            print(f"Loaded processed graphs for target: {self.target_column}.")
            print(f"Number of classes: {self.num_classes}")
            print(f"Train size: {len(self.graphs_splits[0])}")
            print(f"Val size: {len(self.graphs_splits[1])}")
            print(f"Test size: {len(self.graphs_splits[2])}")
        else:
            # read processed feature data (from preprocessing_pipeline)
            X_splits = self.get_processed_features(
                data_path,
                encoder_type,
                mol_column,
                graph_format,
                edge_invariant,
                center_column,
                feature_columns,
                k_hops,
                use_xtb,
                use_bulk,
                bond_scale_factor,
                charge_spin_override,
                implicit_Hs,
                rdkit_features,
            )

            # read target data (from preprocessing_pipeline) and add to graphs
            self.graphs_splits = self.get_processed_targets(
                data_path, target_column, X_splits, scope=scope, label_column=label_column
            )

            if node_feature_path is not None:
                if node_feature_offsets_path is None:
                    raise ValueError(
                        "node_feature_offsets_path is required alongside "
                        "node_feature_path: without offsets there is no way to know "
                        "which atom rows belong to which graph."
                    )
                labels = node_feature_labels_path or [None, None, None]
                for idx, split in enumerate(["train", "val", "test"]):
                    self.graphs_splits[idx] = attach_node_features(
                        self.graphs_splits[idx],
                        node_feature_path[idx],
                        node_feature_offsets_path[idx],
                        labels[idx],
                        node_feature_mode,
                        split,
                    )
                print(
                    f"Seeded node features from precomputed embeddings "
                    f"(mode={node_feature_mode}); node feature dim is now "
                    f"{self.graphs_splits[0][0].x.shape[1]}."
                )

            self.target_column = target_column
            self.label_column = label_column
            self.max_nodes = self.get_max_num_nodes()
            self.max_edges = self.get_max_num_edges()

            torch.save(
                {
                    "graphs_splits": self.graphs_splits,
                    "target_column": self.target_column,
                    "label_column": self.label_column,
                    "num_classes": self.num_classes,
                    "max_nodes": self.max_nodes,
                    "max_edges": self.max_edges,
                    "classes": self.classes,
                },
                save_path,
            )
            print(f"Saved processed graphs to {save_path}")

    @property
    def y_train(self):
        return torch.cat([g.y for g in self.graphs_splits[0]], dim=0)

    @property
    def y_val(self):
        return torch.cat([g.y for g in self.graphs_splits[1]], dim=0)

    @property
    def y_test(self):
        return torch.cat([g.y for g in self.graphs_splits[2]], dim=0)

    @property
    def x_train(self):
        return torch.cat([g.x for g in self.graphs_splits[0]], dim=0)

    @property
    def x_val(self):
        return torch.cat([g.x for g in self.graphs_splits[1]], dim=0)

    @property
    def x_test(self):
        return torch.cat([g.x for g in self.graphs_splits[2]], dim=0)

    @property
    def train_labels(self):
        return [l for g in self.graphs_splits[0] for l in g.label]

    @property
    def val_labels(self):
        return [l for g in self.graphs_splits[1] for l in g.label]

    @property
    def test_labels(self):
        return [l for g in self.graphs_splits[2] for l in g.label]

    def get_max_num_nodes(self):
        """Return the maximum number of nodes across all graphs in all splits."""
        max_num_nodes = 0
        for split in self.graphs_splits:
            for graph in split:
                num_nodes = graph.x.shape[0]
                if num_nodes > max_num_nodes:
                    max_num_nodes = num_nodes
        return max_num_nodes

    def get_max_num_edges(self):
        """Return the maximum number of edges across all graphs in all splits."""
        max_num_edges = 0
        for split in self.graphs_splits:
            for graph in split:
                edge_index = getattr(graph, "edge_index", None)
                num_edges = 0 if edge_index is None else edge_index.shape[1]
                if num_edges > max_num_edges:
                    max_num_edges = num_edges
        return max_num_edges

    def infer_model_input_output_size(self):
        """Infer encoder/readout input and output sizes from the processed training data.

        Returns a nested dict suitable for merging into a model config. The
        readout ``input_size`` is the number of node features (overwritten
        later by the encoder's output size for GNN/EGNN models). The readout
        ``output_size`` is the number of regression targets or classification
        outputs.

        Returns
        -------
        dict
            Nested dict with keys ``'readout_config'`` (sub-keys:
            ``input_size``, ``output_size``, ``graph_attr_dim``, ``edge_dim``)
            and ``'encoder_config'`` (sub-keys: ``input_size``, ``edge_dim``,
            ``max_nodes``, ``max_edges``).
        """
        # in case that gnn is used as encoder, mlp_input_size will be overwritten later.
        X_train = self.graphs_splits[0][0]
        mlp_input_size = X_train.x.shape[1]  # set mlp input size to node feature size
        gnn_input_size = X_train.x.shape[1]  # set gnn input size to node feature size
        edge_attr = getattr(X_train, "edge_attr", None)
        edge_dim = edge_attr.shape[1] if edge_attr is not None else 0
        graph_attr = getattr(X_train, "graph_attr", None)
        graph_attr_dim = graph_attr.shape[1] if graph_attr is not None else 0
        if self.task == "regression":
            mlp_output_size = len(self.target_column)
        elif self.task == "classification":
            # binary classification is treated as single output with sigmoid activation
            # only single task multi-class classification is currently supported
            mlp_output_size = (
                len(self.target_column) if self.num_classes == 2 else self.num_classes
            )
        return {
            "readout_config": {
                "input_size": mlp_input_size,
                "output_size": mlp_output_size,
                "graph_attr_dim": graph_attr_dim,
                "edge_dim": edge_dim,
            },
            "encoder_config": {
                "input_size": gnn_input_size,
                "edge_dim": edge_dim,
                "max_nodes": self.max_nodes,
                "max_edges": self.max_edges,
            },
        }

    def train_dataloader(self, batch_size, num_workers, random_seed=None, shuffle=True,
                         rank=None, world_size=None):
        """Return a DataLoader for the training set.

        In distributed data-parallel mode (``world_size > 1``), wraps the
        dataset with a ``DistributedSampler`` so each rank receives a unique
        shard, which is re-shuffled each epoch.

        Parameters
        ----------
        batch_size : int
            Number of graphs per batch.
        num_workers : int
            Number of DataLoader worker processes.
        random_seed : int or None, optional
            Seed for the DataLoader's generator and DistributedSampler.
        shuffle : bool, optional
            Shuffle the dataset each epoch (ignored when using
            DistributedSampler). Default True.
        rank : int or None, optional
            Current process rank for distributed sampling.
        world_size : int or None, optional
            Total number of processes; enables DistributedSampler when > 1.

        Returns
        -------
        torch_geometric.loader.DataLoader
        """
        train_ds = GraphDataset(self.graphs_splits[0])
        sampler = None
        if world_size is not None and world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                train_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=random_seed if random_seed is not None else 0,
            )
            shuffle = False  # mutually exclusive with sampler
        generator = None
        if random_seed is not None and sampler is None:
            generator = torch.Generator().manual_seed(random_seed)
        train_loader = GraphDataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=seed_workers,
            generator=generator,
        )
        return train_loader

    def val_dataloader(self, batch_size, num_workers):
        """Return a DataLoader for the validation set (no shuffle).

        Parameters
        ----------
        batch_size : int
            Number of graphs per batch.
        num_workers : int
            Number of DataLoader worker processes.

        Returns
        -------
        torch_geometric.loader.DataLoader
        """
        val_ds = GraphDataset(self.graphs_splits[1])
        val_loader = GraphDataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return val_loader

    def test_dataloader(self, batch_size, num_workers):
        """Return a DataLoader for the test set (no shuffle).

        Parameters
        ----------
        batch_size : int
            Number of graphs per batch.
        num_workers : int
            Number of DataLoader worker processes.

        Returns
        -------
        torch_geometric.loader.DataLoader
        """
        test_ds = GraphDataset(self.graphs_splits[2])
        test_loader = GraphDataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return test_loader
