import csv
from elemenet import globalvars
from elemenet.utils import reconstruct_list_columns
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Informatics.autocorrelation import (
    generate_atomonly_autocorrelations,
    generate_atomonly_deltametrics,
    generate_full_complex_autocorrelations,
)
from molSimplify.Informatics.MOF.PBC_functions import (
    fractional2cart,
    mkcell,
    readcif,
)
import networkx as nx
import numpy as np
import os
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
import pickle
from rdkit import Chem
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from tblite.interface import Calculator
import torch
from torch_geometric.data import Data
from typing import Optional


def get_mol3D(data, mol_column, graph_format="mol2"):
    """
    Helper function to streamline generation of mol3D objects.
    INPUTS:
        data: DataFrame
            Data containing molecular graphs and other associated properties.
        mol_column: str
            Column of data containing molecular graphs.
        graph_format: str
            Format of molecular graph. Supported types are 'mol2', 'xyz', 'smiles', 'mol', and 'sdf'.
            default='mol2'
    OUTPUTS:
        mols: list
            List of mol3D objects.
    """

    def molsimplify_mol2(mol_string):
        mol3d = mol3D()
        mol3d.readfrommol2(filename=mol_string, readstring=True)
        check_deuterated(mol3d)
        return mol3d

    def molsimplify_xyz(mol_string):
        mol3d = mol3D()
        mol3d.readfromxyz(filename=mol_string, readstring=True)
        mol3d.convert2OBMol()
        mol3d.graph = mol3d.populateBOMatrix(bonddict=True)
        mol3d.bo_graph_trun = mol3d.graph
        check_deuterated(mol3d)
        return mol3d

    def molsimplify_smiles(mol_string):
        mol3d = mol3D()
        mol3d.read_smiles(smiles=mol_string)
        mol3d.convert2OBMol()
        mol3d.graph = mol3d.populateBOMatrix(bonddict=True)
        mol3d.bo_graph_trun = mol3d.graph
        check_deuterated(mol3d)
        return mol3d

    def check_deuterated(mol3d):
        if "D" in mol3d.symvect():
            print("Warning: deuterium present in molecule, treating as hydrogen")
            atoms = mol3d.atoms
            natoms = mol3d.natoms
            [atoms[idx].mutate("H") for idx in range(natoms) if atoms[idx].sym == "D"]
        return

    parsers = {
        "mol2": molsimplify_mol2,
        "xyz": molsimplify_xyz,
        "smiles": molsimplify_smiles,
    }

    assert graph_format in (
        "mol2",
        "xyz",
        "smiles",
    ), "only mol2, xyz, and smiles formats supported for RACs MLP models"

    mols = [
        parsers[graph_format](mol_string=mol_string) for mol_string in data[mol_column]
    ]

    return mols


def get_racs(data, mols, save_dir, depth=4, center_columns=None, prefix=""):
    """
    Generate revised autocorrelations (RACs) features from molecular graph inputs.
    INPUTS:
        data: DataFrame
            Data containing molecular graphs and other associated properties.
        mols: list
            List of mol3D objects.
        save_dir: str
            Directory where results are saved.
        depth: int
            Maximum depth of graph searched when generating RACs.
            default=4
        center_columns: list
            Columns containing zero-indexed atom indices upon which to center RACs vectors. If None, RACs vectors are generated across every atom and averaged.
            default=None
        prefix: str
            Optional prefix for RACs features. Useful when generating multiple RACs vectors for single prediction task.
            default=''
    OUTPUTS:
        racs_data: DataFrame
            DataFrame containing RACs for all input molecular graphs.
        data: DataFrame
            DataFrame containing original provided dataset.
    """
    racs_list = []
    colnames = None
    for idx, mol in enumerate(mols):
        # generate racs
        all_racs = []
        all_colnames = []
        if center_columns:
            # generate product racs centered on specific atoms
            for column in center_columns:
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not in dataset")
                center_idx = int(data[column][idx])
                racs_product = generate_atomonly_autocorrelations(
                    mol=mol, atomIdx=center_idx, depth=depth, oct=False
                )
                # generate delta racs centered on specific atoms
                racs_delta = generate_atomonly_deltametrics(
                    mol=mol, atomIdx=center_idx, depth=depth, oct=False
                )
                # combine product and delta vectors
                racs_product_flattened = [
                    item for sublist in racs_product["results"] for item in sublist
                ]
                racs_delta_flattened = [
                    item for sublist in racs_delta["results"] for item in sublist
                ]
                all_racs.extend(racs_product_flattened + racs_delta_flattened)
                # name column so they are distinguishable
                if not colnames:
                    colnames_product = [
                        f"{prefix}product_{item}_{column}"
                        for sublist in racs_product["colnames"]
                        for item in sublist
                    ]
                    colnames_delta = [
                        f"{prefix}delta_{item}_{column}"
                        for sublist in racs_delta["colnames"]
                        for item in sublist
                    ]
                    all_colnames.extend(colnames_product + colnames_delta)

        else:
            # generate full complex racs averaged across the entire graph
            racs = generate_full_complex_autocorrelations(
                mol=mol, depth=depth, oct=False
            )
            all_racs = [item for sublist in racs["results"] for item in sublist]
            # name columns
            all_colnames = [
                prefix + item for sublist in racs["colnames"] for item in sublist
            ]

        racs_list.append(all_racs)
        if not colnames:
            colnames = all_colnames

    racs_data = pd.DataFrame(racs_list, columns=colnames)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Length of generated RACs feature vector: {racs_data.shape[1]}")
    return racs_data


def get_extra_features(data, feature_columns):
    """
    Processes additional features provided by user.
    Assumes all features are graph-level unless otherwise specified. Node and edge-level features are only defined for GNNs (i.e., representation=='learned').
    INPUTS:
        data: DataFrame
            Data containing molecular graphs and other associated properties.
        feature_columns: dict or list
            Column names containing extra features to process and return.
    OUTPUTS:
        extra_features: dict
            Dictionary containing extra features organized by scope ('node', 'edge', 'graph').
            All are DataFrames where rows = molecules and columns = feature names.
            For 'node' and 'edge': cells contain lists with one value per node/edge.
            For 'graph': cells contain scalar values.
    """
    if isinstance(feature_columns, list):
        print("Assuming user-inputs are graph-level features.")
        print(
            "To specify node, edge, or graph-level features, input a dictionary formatted as:"
        )
        print(
            "feature_columns = {'node': [node_feature_columns], 'edge': [edge_feature_columns], 'graph': [graph_feature_columns]}"
        )
        feature_columns = {"node": [], "edge": [], "graph": feature_columns}
    elif isinstance(feature_columns, dict):
        for key in ("node", "edge", "graph"):
            if key in feature_columns.keys() and not isinstance(
                feature_columns[key], list
            ):
                raise ValueError(
                    f"feature_columns dict contains type {type(feature_columns[key])} at key {key}, expected list."
                )
    else:
        raise ValueError(
            f"feature_columns must be a list or dictionary. Detected type: {type(feature_columns)}"
        )

    extra_features = {}

    # process atom and bond features (node and edge level)
    for scope in ("node", "edge"):
        columns = feature_columns.get(scope, [])
        if columns:
            scope_data = {}
            for column in columns:
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not in dataset.")

                def parse_list(x):
                    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
                        return [float(i) for i in x.strip("[]").split(",")]
                    elif isinstance(x, list):
                        return x
                    else:
                        raise ValueError(
                            f"Unsupported format of '{x}' in column '{column}', expected list or string representation of list."
                        )

                scope_data[column] = data[column].apply(parse_list)

            # create DataFrame where each row is a molecule, each column is a feature
            extra_features[scope] = pd.DataFrame(scope_data)
        else:
            extra_features[scope] = pd.DataFrame()

    # process graph-level features
    mol_data = []
    for column in feature_columns.get("graph", []):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not in dataset.")

        def parse_list(x):
            if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
                return [float(i) for i in x.strip("[]").split(",")]
            elif isinstance(x, list):
                return x
            elif isinstance(x, (int, float, np.number)):
                return x
            else:
                raise ValueError(
                    f"Unsupported format of '{x}' in column '{column}', expected list, numeric value, or string representation of list."
                )

        parsed_column = data[column].apply(parse_list)
        dtype = parsed_column.dtype

        if (
            is_object_dtype(dtype)
            or is_string_dtype(dtype)
            or is_categorical_dtype(dtype)
        ):
            mol_data.append(pd.get_dummies(parsed_column, prefix=column, dtype=int))
        elif is_numeric_dtype(dtype):
            mol_data.append(parsed_column.to_frame())
        else:
            raise ValueError(f"Column '{column}' has unsupported data type '{dtype}'")

    # finalize graph-level features
    extra_features["graph"] = (
        pd.concat(mol_data, axis=1) if mol_data else pd.DataFrame()
    )

    return extra_features


def _flatten_series_of_lists(values):
    """
    Flatten a pandas Series where each element is a list (or scalar) into a single flat Python list of scalars.
    """
    flat = []
    for item in values:
        if isinstance(item, (list, tuple, np.ndarray)):
            for sub in item:
                if isinstance(sub, (list, tuple, np.ndarray)):
                    # handle unexpected double-nesting
                    flat.extend(sub)
                else:
                    flat.append(sub)
        else:
            flat.append(item)
    return flat


def _is_feature_continuous(values, feature_type):
    """
    Detect whether a feature column contains continuous or categorical values.

    INPUTS:
        values: Series
            Pandas Series containing feature values (lists for node/edge, scalars for graph).
        feature_type: str
            Type of feature: "node", "edge", or "graph".
    OUTPUTS:
        is_continuous: bool
            True if feature is continuous, False if categorical.
    """
    # flatten values for inspection
    if feature_type in ["node", "edge"]:
        flattened = _flatten_series_of_lists(values)
    else:
        # values are scalars
        flattened = values.dropna().tolist()

    if len(flattened) == 0:
        return True  # default to continuous if empty

    # try to convert all values to float
    try:
        for val in flattened:
            float(val)
        return True  # all values are numeric --> continuous feature
    except (ValueError, TypeError):
        return False  # at least one non-numeric --> categorical feature


def scale_and_encode_user_features(extra_features_dict):
    """Automatically detect and scale or encode user-provided feature columns.

    Numeric columns are treated as continuous and StandardScaler-normalized.
    Non-numeric columns are treated as categorical and one-hot encoded.
    Graph-level ``charge`` and ``spinmult`` columns are passed through
    unscaled because they are normalized later inside ``mol_fingerprint``.

    .. note::
        Do not pass pre-constructed one-hot encoded columns (binary ``{0, 1}``
        arrays) as ``feature_columns``; they will be detected as continuous and
        incorrectly StandardScaler-normalized. Pass the original categorical
        column instead and let this function encode it.

    Parameters
    ----------
    extra_features_dict : dict
        Dictionary with keys ``'node'``, ``'edge'``, ``'graph'`` mapping to
        DataFrames of user features.

    Returns
    -------
    extra_features_dict : dict
        Modified in-place with scaled/encoded features.
    scalers_dict : dict
        Maps ``(feature_type, column_name)`` to a ``('continuous', scaler)``
        tuple or a ``('categorical', choices)`` tuple.
    """
    scalers_dict = {}

    for feature_type, df in extra_features_dict.items():
        if df is None or df.empty:
            continue

        for col in df.columns:
            # do not scale charge or spin; these get scaled later in mol_fingerprint
            if feature_type == "graph" and col in ["charge", "spinmult"]:
                continue

            values = df[col]

            if _is_feature_continuous(values, feature_type):
                # continuous feature: fit and apply StandardScaler
                # flatten all values across all rows
                if feature_type in ["node", "edge"]:
                    # list of lists --> flatten to 1D
                    flattened = np.array(_flatten_series_of_lists(values), dtype=float)
                else:
                    # molecule features are scalars
                    flattened = np.array(values, dtype=float)

                # fit scaler on flattened data
                scaler = StandardScaler()
                scaler.fit(flattened.reshape(-1, 1))
                scalers_dict[(feature_type, col)] = ("continuous", scaler)

                # apply scaler to each row
                if feature_type in ["node", "edge"]:
                    df[col] = df[col].apply(
                        lambda val_list, s=scaler: s.transform(
                            np.array(val_list, dtype=float).reshape(-1, 1)
                        )
                        .flatten()
                        .tolist()
                    )
                else:
                    df[col] = scaler.transform(
                        np.array(values, dtype=float).reshape(-1, 1)
                    ).flatten()

            else:
                # categorical feature: one-hot encode
                # first, get all unique values by flattening
                if feature_type in ["node", "edge"]:
                    all_values = set(_flatten_series_of_lists(values))
                else:
                    all_values = set(values.dropna().tolist())

                choices = sorted(list(all_values))
                scalers_dict[(feature_type, col)] = ("categorical", choices)

                # apply one-hot encoding to each row
                if feature_type in ["node", "edge"]:
                    df[col] = df[col].apply(
                        lambda val_list, c=choices: [
                            one_hot(v, c)
                            for v in (
                                val_list
                                if isinstance(val_list, (list, tuple))
                                else [val_list]
                            )
                        ]
                    )
                else:
                    df[col] = df[col].apply(lambda val, c=choices: one_hot(val, c))

    return extra_features_dict, scalers_dict


def featurize(
    data_path,
    mol_column,
    save_dir,
    representation="learned",
    depth=4,
    center_columns=None,
    prefix="",
    feature_columns=None,
    graph_format="mol2",
):
    """
    Helper function to featurize molecules either with RACs or additional user-provided features.
    INPUTS:
        data_path: str
            Path to data containing molecular graphs and other associated properties.
        mol_column: str
            Column containing molecular graphs.
        save_dir: str
            Directory where results are saved.
        representation: str
            Representation scheme to use, either RACs for an MLP or a learned representation for a GNN
            default='learned'
        depth: int
            Maximum depth of graph searched when generating RACs.
            default=4
        center_columns: list
            Columns containing particularly relevant atoms upon which to center RACs.
            default=None
        prefix: str
            Optional prefix for RACs features. Useful when generating multiple RACs vectors for single prediction task.
            default=''
        feature_columns: list
            Columns containing extra features to process and return.
            default=None
        graph_format: str
            Format of molecular graph. Supported types are 'mol2', 'smiles', 'mol', and 'sdf'.
            default='mol2'
    OUTPUTS:
        X_data: DataFrame
            DataFrame containing feature data.
        all_data: DataFrame
            DataFrame containing all property data.

    """
    # Initialize feature_columns as dict if not provided
    if feature_columns is None:
        feature_columns = {"node": [], "edge": [], "graph": []}
    elif isinstance(feature_columns, list):
        feature_columns = {"node": [], "edge": [], "graph": feature_columns}

    # read in original data
    if isinstance(data_path, str):
        all_data = pd.read_csv(data_path)
    elif isinstance(data_path, pd.DataFrame):
        all_data = data_path
    else:
        raise ValueError(
            f"Unsupported format of '{data_path}', must be either str or pd.DataFrame."
        )

    extra_features = get_extra_features(data=all_data, feature_columns=feature_columns)

    # automatically scale/encode user-provided features
    extra_features, _ = scale_and_encode_user_features(extra_features)

    if representation == "auto":
        mols = get_mol3D(
            data=all_data, mol_column=mol_column, graph_format=graph_format
        )
        # full-complex RACs
        racs_data_full = get_racs(
            data=all_data,
            mols=mols,
            save_dir=save_dir,
            depth=depth,
            center_columns=None,
            prefix=prefix,
        )
        # metal-centered RACs
        if center_columns:
            racs_data_centered = get_racs(
                data=all_data,
                mols=mols,
                save_dir=save_dir,
                depth=depth,
                center_columns=center_columns,
                prefix=prefix,
            )
        else:
            racs_data_centered = None
        # combine data
        X_data = pd.concat(
            [
                data
                for data in [racs_data_full, racs_data_centered, extra_features]
                if data is not None
            ],
            axis=1,
        )

    elif representation == "learned":
        columns = [mol_column]
        if center_columns:
            columns.extend(center_columns)
        X_data = all_data[columns]
        if feature_columns and not extra_features["graph"].empty:
            X_data = pd.concat([X_data, extra_features["graph"]], axis=1)

    return X_data, all_data, extra_features


def get_split(
    X_data,
    y_data,
    train_val_test_split=[0.8, 0.1, 0.1],
    stratify=None,
    group_by=None,
    random_seed=0,
    extra_features=None,
):
    """
    Split feature and target data into training, validation, and test sets.
    INPUTS:
        X_data: DataFrame
            Pandas DataFrame containing feature data.
        y_data: DataFrame
            Pandas DataFrame containing target data.
        train_val_test_split: list
            Percentages by which to split data into training, validation, and test sets. Must sum to 1.
            default=[0.8, 0.1, 0.1]
        stratify: str
            Column of y_data by which to stratify data splits. For use with classification tasks on imbalanced datasets.
            default=None
        group_by: str or list
            Column(s) of y_data by which to group data so that all members of a group
            stay in the same split. Mutually exclusive with stratify.
            default=None
        random_seed: int
            Seed for reproducibility.
            default=0
        extra_features: dict
            Dictionary containing extra features to split. Keys are "node", "edge", "graph".
            default=None
    OUTPUTS:
        X_splits: tuple
            Length 3 tuple containing training, validation, and test splits of X data.
        y_splits: tuple
            Length 3 tuple containing training, validation, and test splits of y data.
        extra_features_splits: tuple (optional)
            Length 3 tuple containing training, validation, and test splits of extra features.
    """
    if stratify and group_by:
        raise ValueError(
            "stratify and group_by are mutually exclusive. Please specify only one."
        )

    train_size, val_size, test_size = train_val_test_split

    if group_by is not None:
        # Build a composite group key from one or more columns
        if isinstance(group_by, str):
            group_by = [group_by]
        groups = y_data[group_by].astype(str).agg("_".join, axis=1)

        # First split: separate test set
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_seed
        )
        trainval_idx, test_idx = next(gss_test.split(X_data, y_data, groups))

        X_trainval = X_data.iloc[trainval_idx]
        y_trainval = y_data.iloc[trainval_idx]
        groups_trainval = groups.iloc[trainval_idx]

        # Second split: separate validation from training
        relative_val_size = val_size / (train_size + val_size)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=relative_val_size, random_state=random_seed
        )
        train_idx_rel, val_idx_rel = next(
            gss_val.split(X_trainval, y_trainval, groups_trainval)
        )

        X_train = X_trainval.iloc[train_idx_rel]
        X_val = X_trainval.iloc[val_idx_rel]
        X_test = X_data.iloc[test_idx]
        y_train = y_trainval.iloc[train_idx_rel]
        y_val = y_trainval.iloc[val_idx_rel]
        y_test = y_data.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_data,
            test_size=test_size,
            random_state=random_seed,
            stratify=y_data[stratify] if stratify else None,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size / (train_size + val_size),
            random_state=random_seed,
            stratify=y_train[stratify] if stratify else None,
        )
    X_splits = (X_train, X_val, X_test)
    y_splits = (y_train, y_val, y_test)

    # split extra features if provided
    extra_features_splits = None
    if extra_features:
        # get indices for each split from the resulting DataFrames
        # train_test_split returns subsets with their original indices preserved
        train_idx = X_train.index.tolist()
        val_idx = X_val.index.tolist()
        test_idx = X_test.index.tolist()

        extra_features_splits = []
        for split_idx in [train_idx, val_idx, test_idx]:
            split_features = {}

            # split node features
            if not extra_features["node"].empty:
                # node features are a DataFrame, use loc with the original indices
                split_features["node"] = extra_features["node"].loc[split_idx]
            else:
                split_features["node"] = pd.DataFrame()

            # split edge features
            if not extra_features["edge"].empty:
                # edge features are a DataFrame, use loc with the original indices
                split_features["edge"] = extra_features["edge"].loc[split_idx]
            else:
                split_features["edge"] = pd.DataFrame()

            # split graph features
            if not extra_features["graph"].empty:
                # graph features are a DataFrame, use loc with the original indices
                split_features["graph"] = extra_features["graph"].loc[split_idx]
            else:
                split_features["graph"] = pd.DataFrame()

            extra_features_splits.append(split_features)

    print(f"Data partitioned according to {train_size}-{val_size}-{test_size} split.")
    if stratify:
        print(
            f"Splits stratified according to distribution of {stratify} in training data."
        )
    if group_by:
        print(
            f"Splits grouped by {group_by} — all members of each group are in the same split."
        )

    if extra_features_splits is None:
        # no extra features: return empty-DataFrame splits so callers can always unpack 3 values
        empty = {"node": pd.DataFrame(), "edge": pd.DataFrame(), "graph": pd.DataFrame()}
        extra_features_splits = [empty, empty, empty]
    return X_splits, y_splits, extra_features_splits


def save(save_dir, X_data, y_data, X_data_scaled, y_data_scaled, X_scaler, y_scaler):
    """
    Helper function to save data splits.
    """
    splits = ("train", "val", "test")
    os.makedirs(os.path.join(save_dir, "X_data"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "y_data"), exist_ok=True)
    for X, y, X_scaled, y_scaled, split in zip(
        X_data, y_data, X_data_scaled, y_data_scaled, splits
    ):
        # save feature and target data as .pkl for efficiency
        X.to_pickle(os.path.join(save_dir, f"X_data/X_{split}.pkl"))
        y.to_pickle(os.path.join(save_dir, f"y_data/y_{split}.pkl"))
        # save scaled features if defined
        if X_scaled is not None:
            with open(
                os.path.join(save_dir, "X_data", f"X_{split}_scaled.pkl"), "wb"
            ) as f:
                pickle.dump(X_scaled, f)
            with open(save_dir + "/X_scaler.pkl", "wb") as f:
                pickle.dump(X_scaler, f)
        # save scaled targets if defined
        if y_scaled is not None:
            with open(
                os.path.join(save_dir, "y_data", f"y_{split}_scaled.pkl"), "wb"
            ) as f:
                pickle.dump(y_scaled, f)
            with open(save_dir + "/y_scaler.pkl", "wb") as f:
                pickle.dump(y_scaler, f)
    print(
        f"Data saved to {os.path.join(save_dir, 'X_data')} and {os.path.join(save_dir, 'y_data')}"
    )
    return


def scale(
    X_splits,
    y_splits,
    target_column,
    save_dir,
    scale: bool = False,
    representation: str = "learned",
    y_scaler=None,
):
    """
    Scale features and targets according to distributions in training set only.
    Only for use with continuous features and targets (i.e., not for categorical values).
    INPUTS:
        X_splits: tuple
            Length 3 tuple containing training, validation, and test splits of X data.
        y_splits: tuple
            Length 3 tuple containing training, validation, and test splits of y data.
        target_column: str
            String indicating which column of y data contains target property.
        save_dir: str
            Directory where results are saved.
        representation: str
            Representation scheme to use, either RACs for an MLP or a learned representation for a GNN.
            default='learned'
        y_scaler: sklearn.preprocessing.StandardScaler or None, default None
            If provided AND ``scale`` is True, the function will use this pre-fitted
            scaler to ``transform`` all three splits (without re-fitting). This is
            used by ``inference_pipeline`` to apply the training-time scaler to new
            data, avoiding the silent test-set re-fit that otherwise corrupts
            saved targets. When ``None`` (the default, used by all training code
            paths), a fresh ``StandardScaler`` is fit on the training split as
            before — behavior is unchanged for training.
    """
    X_train, X_val, X_test = X_splits
    y_train, y_val, y_test = y_splits
    # NOTE: do not clobber the ``y_scaler`` argument here — when this function is
    # called from inference, a pre-fitted training-time scaler is passed in and
    # must be preserved. ``X_scaler`` is always fit fresh, so we initialise it
    # here; ``y_scaler`` keeps whatever the caller provided (None by default,
    # in which case the regression branch below fits a fresh ``StandardScaler``).
    X_scaler = None
    # process RACs features for MLPs
    if representation == "auto":
        # drop any invariant columns (i.e., feature with same value for all samples in training data)
        drop_columns = X_train.columns[X_train.nunique() == 1].tolist()
        X_train.drop(columns=drop_columns, inplace=True)
        X_val.drop(columns=drop_columns, inplace=True)
        X_test.drop(columns=drop_columns, inplace=True)
        if len(drop_columns) >= 1:
            print(f"Dropped {len(drop_columns)} invariant columns: {drop_columns}")
        # drop any redundant columns (i.e., multiple features with same value for all samples in training data)
        duplicate_mask = X_train.T.duplicated()
        drop_columns = X_train.columns[duplicate_mask].tolist()
        X_train.drop(columns=drop_columns, inplace=True)
        X_val.drop(columns=drop_columns, inplace=True)
        X_test.drop(columns=drop_columns, inplace=True)
        if len(drop_columns) >= 1:
            print(f"Dropped {len(drop_columns)} redundant columns: {drop_columns}")
        # fit X scaler to training data, transform all three data splits
        X_scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            X_scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_val_scaled = pd.DataFrame(
            X_scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            X_scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
    # process learned representations for GNNs
    # elif representation == "learned":
    # if there are extra features:
    #     # fit X scaler to training data, transform all three data splits
    #     X_scaler = StandardScaler()
    #     X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test
    #     X_train_scaled[feature_cols] = X_scaler.fit_transform(X_train[feature_cols])
    #     X_val_scaled[feature_cols] = X_scaler.transform(X_val[feature_cols])
    #     X_test_scaled[feature_cols] = X_scaler.transform(X_test[feature_cols])
    # for regression tasks, scale targets (if they are scalar)
    if scale:
        # Per column, the targets are defined as lists. To scale them, all lists are
        # concatenated into a 1 array, scaled, then reconstructed back into list columns.
        #
        # If a pre-fitted ``y_scaler`` was supplied (e.g., by inference_pipeline
        # passing the training-time scaler), reuse it via ``transform`` for all
        # three splits. Otherwise fit a fresh ``StandardScaler`` on ``y_train``,
        # which matches the historical training-pipeline behaviour.
        if y_scaler is None:
            y_scaler = StandardScaler()
            _train_op = y_scaler.fit_transform
        else:
            _train_op = y_scaler.transform
        y_train_scaled, y_val_scaled, y_test_scaled = (
            y_train.copy(),
            y_val.copy(),
            y_test.copy(),
        )
        y_train_scaled[target_column] = reconstruct_list_columns(
            y_train_scaled,
            target_column,
            _train_op(
                np.column_stack(
                    [np.concatenate(y_train[col].values) for col in target_column]
                )
            ),
        )
        y_val_scaled[target_column] = reconstruct_list_columns(
            y_val_scaled,
            target_column,
            y_scaler.transform(
                np.column_stack(
                    [np.concatenate(y_val[col].values) for col in target_column]
                )
            ),
        )
        y_test_scaled[target_column] = reconstruct_list_columns(
            y_test_scaled,
            target_column,
            y_scaler.transform(
                np.column_stack(
                    [np.concatenate(y_test[col].values) for col in target_column]
                )
            ),
        )
    # save data and scalers
    X_data_scaled = (None, None, None)
    y_data_scaled = (None, None, None)
    if X_scaler is not None:
        X_data_scaled = (X_train_scaled, X_val_scaled, X_test_scaled)
    if y_scaler is not None:
        y_data_scaled = (y_train_scaled, y_val_scaled, y_test_scaled)

    save(
        save_dir=save_dir,
        X_data=(X_train, X_val, X_test),
        y_data=(y_train, y_val, y_test),
        X_data_scaled=X_data_scaled,
        y_data_scaled=y_data_scaled,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
    )
    return


def parse_mol_string(mol_string, graph_format="mol2", bond_scale_factor=1.0):
    """
    Parse a molecular structure for atomic symbols and bonds.
    INPUTS:
        mol_string: str
            Molecular graph stored as text.
        graph_format: str
            Format of original molecular graph. Supported types are 'mol2', 'xyz', 'smiles', 'mol', 'sdf', 'cml', and 'cif'.
            default='mol2'
    OUTPUTS:
        syms: list
            List of symbols for each atom.
        coords: np.array
            Array of atomic coordinates
        bonds: list
            List of tuples storing atom indices of each bonded pair.
    """

    parsers = {
        "mol2": read_mol2,
        "xyz": read_xyz,
        "smiles": read_smiles,
        "mol": read_mol_sdf,
        "sdf": read_mol_sdf,
        "cml": read_cml,
        "cif": read_cif,
    }

    if graph_format == "xyz":
        syms, coords, bonds = parsers[graph_format](
            mol_string=mol_string, bond_scale_factor=bond_scale_factor
        )
    else:
        syms, coords, bonds = parsers[graph_format](mol_string=mol_string)

    return syms, coords, bonds


def read_mol2(mol_string):
    """
    Helper to read .mol2 formats
    Args:
        mol_string (str): string representing molecular structure
    """

    lines = [line.strip() for line in mol_string.splitlines()]
    counts = lines[2].split()
    # parse atom info
    n_atoms, atom_start = int(counts[0]), lines.index("@<TRIPOS>ATOM") + 1
    atom_lines = lines[atom_start : atom_start + n_atoms]
    syms = [line.split()[5].split(".")[0] for line in atom_lines]
    coords = np.array([line.split()[2:5] for line in atom_lines], dtype=float)
    # parse bond info
    n_bonds, bond_start = int(counts[1]), lines.index("@<TRIPOS>BOND") + 1
    bond_lines = lines[bond_start : bond_start + n_bonds]
    bonds = [
        (int(line.split()[1]) - 1, int(line.split()[2]) - 1, line.split()[3])
        for line in bond_lines
    ]
    return syms, coords, bonds


def read_xyz(mol_string, bond_scale_factor=1.0):
    """
    Helper to read .xyz formats
    Args:
        mol_string (str): string representing molecular structure
        bond_scale_factor (float): scaling factor for covalent radii sum when determining bonds.
            default=1.0
    """

    lines = [line.strip() for line in mol_string.splitlines()]
    # parse atom info
    n_atoms, atom_start = int(lines[0]), 2
    atom_lines = lines[atom_start : atom_start + n_atoms]
    syms = [line.split()[0] for line in atom_lines]
    coords = np.asarray([line.split()[1:] for line in atom_lines], dtype=float)
    # estimate bonds from sum of covalent radii
    if n_atoms > 1:
        pair_dists = squareform(pdist(coords))
        covr_arr = np.array([globalvars.covalent_radius[s] for s in syms])
        idx_i, idx_j = np.triu_indices(n_atoms, k=1)
        bonded = pair_dists[idx_i, idx_j] <= bond_scale_factor * (
            covr_arr[idx_i] + covr_arr[idx_j]
        )
        bonds = [(int(i), int(j), "1") for i, j in zip(idx_i[bonded], idx_j[bonded])]
    else:
        bonds = []

    return syms, coords, bonds


def read_smiles(mol_string):
    """
    Helper to read SMILES strings
    Args:
        mol_string (str): string representing molecular structure
    """

    mol_rdk = Chem.MolFromSmiles(mol_string, sanitize=True)
    mol_rdk = Chem.AddHs(mol_rdk)
    Chem.SanitizeMol(mol_rdk)
    # parse atom info
    syms = [atom.GetSymbol() for atom in mol_rdk.GetAtoms()]
    num_atoms = len(syms)
    coords = np.array([0, 0, 0])  # smiles do not define 3D structure

    # parse bond info
    bond_map = {
        Chem.rdchem.BondType.SINGLE: "1",
        Chem.rdchem.BondType.DOUBLE: "2",
        Chem.rdchem.BondType.TRIPLE: "3",
        Chem.rdchem.BondType.AROMATIC: "ar",
    }
    bonds = [
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_map.get(bond.GetBondType(), "0"),
        )
        for bond in mol_rdk.GetBonds()
    ]

    return syms, coords, bonds


def extract_rdkit_features(mol_string):
    """
    Compute additional per-atom and per-bond RDKit features from a SMILES string.
    Atom and bond ordering matches read_smiles() (post-AddHs; heavy atoms keep their
    original indices and Hs are appended), so the returned dicts can be passed
    directly through extra_node_features / extra_edge_features and will be
    correctly remapped by the implicit_Hs filter in mol_to_graph.

    Returns
    -------
    atom_extras : dict[str, list[int]]
        One-hot expanded per-atom features (formal charge, hybridization,
        aromaticity, chirality).
    bond_extras : dict[str, list[int]]
        One-hot expanded per-bond features (conjugation, ring membership, stereo).
    """
    mol = Chem.MolFromSmiles(mol_string, sanitize=True)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)

    HYB = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    FORMAL = [-2, -1, 0, 1, 2]
    CHIRAL = [0, 1, 2, 3]
    STEREO = list(range(6))

    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())

    atom_extras = {}
    for v in FORMAL:
        atom_extras[f"rdkit_fc_{v}"] = [int(a.GetFormalCharge() == v) for a in atoms]
    atom_extras["rdkit_fc_other"] = [
        int(a.GetFormalCharge() not in FORMAL) for a in atoms
    ]
    for h in HYB:
        atom_extras[f"rdkit_hyb_{h.name}"] = [
            int(a.GetHybridization() == h) for a in atoms
        ]
    atom_extras["rdkit_hyb_other"] = [
        int(a.GetHybridization() not in HYB) for a in atoms
    ]
    atom_extras["rdkit_is_aromatic"] = [int(a.GetIsAromatic()) for a in atoms]
    for c in CHIRAL:
        atom_extras[f"rdkit_chiral_{c}"] = [
            int(int(a.GetChiralTag()) == c) for a in atoms
        ]

    bond_extras = {
        "rdkit_is_conjugated": [int(b.GetIsConjugated()) for b in bonds],
        "rdkit_is_in_ring": [int(b.IsInRing()) for b in bonds],
    }
    for s in STEREO:
        bond_extras[f"rdkit_stereo_{s}"] = [int(int(b.GetStereo()) == s) for b in bonds]

    return atom_extras, bond_extras


def read_mol_sdf(mol_string):
    """
    Helper to read .mol and .sdf formats
    Args:
        mol_string (str): string representing molecular structure
    """

    lines = [line.strip() for line in mol_string.splitlines()]
    counts = lines[3].split()
    # parse atom info
    n_atoms, atom_start = int(counts[0]), 4
    atom_lines = lines[atom_start : atom_start + n_atoms]
    syms = [line.split()[3] for line in atom_lines]
    coords = np.array([line.split()[0:3] for line in atom_lines], dtype=float)
    # parse bond info
    n_bonds, bond_start = int(counts[1]), atom_start + n_atoms
    bond_lines = lines[bond_start : bond_start + n_bonds]
    # label aromatic bond types ('4') as 'ar'
    bonds = [
        (
            int(line.split()[0]) - 1,
            int(line.split()[1]) - 1,
            "ar" if line.split()[2] == "4" else line.split()[2],
        )
        for line in bond_lines
    ]

    return syms, coords, bonds


def read_cml(mol_string):
    """
    Helper to read .cml formats
    Args:
        mol_string (str): string representing molecular structure
    """
    lines = [line.strip() for line in mol_string.splitlines()]
    # parse atom info
    atom_start = lines.index("<atomArray>") + 1
    atom_stop = lines.index("</atomArray>")
    atom_lines = lines[atom_start:atom_stop]
    syms = [
        line.split()[2].split("=")[1].replace('"', "").replace("/>", "")
        for line in atom_lines
    ]
    coords = np.array(
        [
            [
                xyz.split("=")[1].replace('"', "").replace("/>", "")
                for xyz in line.split()[3:]
            ]
            for line in atom_lines
        ],
        dtype=float,
    )
    # parse bond info
    bond_start = lines.index("<bondArray>") + 1
    bond_stop = lines.index("</bondArray>")
    bond_lines = lines[bond_start:bond_stop]
    bonds = [
        (
            int(line.split()[1].split("=")[1].replace('"', "").replace("a", "")) - 1,
            int(line.split()[2].replace('"', "").replace("a", "")) - 1,
            line.split()[3].split("=")[1].replace('"', "").replace("/>", ""),
        )
        for line in bond_lines
    ]

    return syms, coords, bonds


def read_cif(mol_string, bond_scale_factor=1.0):
    """
    Helper to read .cif formats
    Args:
        mol_string (str): string representing molecular structure
        bond_scale_factor (float): scaling factor for covalent radii sum when determining bonds.
            default=1.0
    """
    
    # parse cif file using molSimplify functions
    cell_params, syms, frac_coords = readcif(name=mol_string, readstring=True)
    cell = mkcell(cell_params)
    coords = fractional2cart(frac_coords, cell)
    coords = np.asarray(coords, dtype=float)
    n_atoms = len(syms)
    # estimate bonds from sum of covalent radii
    if n_atoms > 1:
        pair_dists = squareform(pdist(coords))
        covr_arr = np.array([globalvars.covalent_radius[s] for s in syms])
        idx_i, idx_j = np.triu_indices(n_atoms, k=1)
        bonded = pair_dists[idx_i, idx_j] <= bond_scale_factor * (
            covr_arr[idx_i] + covr_arr[idx_j]
        )
        bonds = [(int(i), int(j), "1") for i, j in zip(idx_i[bonded], idx_j[bonded])]
    else:
        bonds = []

    return syms, coords, bonds
    

def one_hot(value, choices):
    """
    One-hot encoder.
    """
    size = len(choices)
    encoding = [0] * size

    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        pass  # return all zeros if value not in choices

    return encoding


def flatten_features(node, feature_list):
    values = []
    for feat in feature_list:
        val = node[feat]
        # extend if one-hot list
        if isinstance(val, (list, tuple)):
            values.extend(val)
        else:
            values.append(val)

    return values


def atomic_fingerprint(sym, num_bonds, extra_feature_dict=None, num_Hs=None):
    """
    Fingerprint nodes with atomic properties.
    INPUTS:
        sym: str
            Atomic symbol specifying element.
        num_bonds: int
            Number of bonds to other atoms in the molecule.
        extra_feature_dict: dict
            Additional user-provided node features.
            default=None
        num_Hs: int or None
            Number of implicit hydrogen atoms bonded to this atom.
            Only set when implicit_Hs=True; adds a 'num_Hs' node feature.
            default=None
    OUTPUTS:
        node_features: dict
            Dictionary of atomic features.
    """
    node_features = {
        # continuous features
        "atomic_mass": globalvars.atomic_mass[sym],
        "covalent_radius": globalvars.covalent_radius[sym],
        "vdw_radius": globalvars.vdw_radius[sym],
        # "atomic_volume": 4 * np.pi / 3 * globalvars.covalent_radius[sym] ** 3,
        "electronegativity": globalvars.electronegativity[sym],
        "polarizability": globalvars.polarizability[sym],
        "ionization_potential": globalvars.ionization_potential[sym],
        "electron_affinity": globalvars.electron_affinity[sym],
        # discrete features
        "coordination_number": num_bonds,
        # categorical features (redundant features like atomic_number and block removed)
        "valence_s": one_hot(
            globalvars.valence_s[sym], globalvars.choices["valence_s"]
        ),
        "valence_p": one_hot(
            globalvars.valence_p[sym], globalvars.choices["valence_p"]
        ),
        "valence_d": one_hot(
            globalvars.valence_d[sym], globalvars.choices["valence_d"]
        ),
        "valence_f": one_hot(
            globalvars.valence_f[sym], globalvars.choices["valence_f"]
        ),
        "period": one_hot(globalvars.period[sym], globalvars.choices["period"]),
        "group": one_hot(globalvars.group[sym], globalvars.choices["group"]),
    }

    # add implicit hydrogen count if using implicit_Hs mode
    if num_Hs is not None:
        node_features["num_Hs"] = num_Hs

    # add user-provided features if necessary
    if extra_feature_dict is not None and isinstance(extra_feature_dict, dict):
        for key, val in extra_feature_dict.items():
            if key in node_features:
                raise ValueError(
                    f"User-provided feature '{key}' conflicts with existing feature name. Please choose a different name for this feature."
                )
            node_features[key] = val
    # NOTE: User-provided continuous features should ideally be pre-normalized to [0,1] or [-1,1]
    # by the user. If needed, per-feature normalization can be added to the preprocessing pipeline
    # by computing mean/std across the entire dataset and applying StandardScaler-like normalization.

    # scale node features as needed
    scale_dict = globalvars.atom_scales
    to_scale = [
        "atomic_mass",
        "covalent_radius",
        "vdw_radius",
        "electronegativity",
        "polarizability",
        "ionization_potential",
        "electron_affinity",
        "coordination_number",
    ]
    if num_Hs is not None:
        to_scale.append("num_Hs")
    for key in to_scale:
        min_val, max_val = scale_dict[key]
        node_features[key] = (node_features[key] - min_val) / (max_val - min_val)

    return node_features


def bond_fingerprint(
    bonded_atoms, bond_type, edge_invariant=False, extra_feature_dict=None
):
    """
    Fingerprint edges with bond properties.
    INPUTS:
        bonded_atoms: tuple
            Tuple containing bonded atom symbols
        bond_type: str
            String indicating bond type, i.e., single, double, triple, or aromatic.
        edge_invariant: boolean
            Flag to drop all GNN features based on bond order, allowing for resonant-invariance.
            default=False
        extra_feature_dict: dict
            Additional user-provided edge features.
            default=None
    OUTPUTS:
        edge_features: dict
            Dictionary of bond features.
    """
    # unpack bond info
    atom1, atom2 = bonded_atoms
    has_metal = not {atom1, atom2}.isdisjoint(globalvars.transition_metals)

    # define bond type and order: single (1), double (2), triple (3), or aromatic (1.5)
    bond_type_mapping = {
        "1": [1, 0, 0, 0],
        "2": [0, 1, 0, 0],
        "3": [0, 0, 1, 0],
        "ar": [0, 0, 0, 1],
    }
    bond_type_onehot = bond_type_mapping.get(bond_type, [0, 0, 0, 0])

    # estimate bond length as sum of covalent radii
    bond_length = globalvars.covalent_radius[atom1] + globalvars.covalent_radius[atom2]
    if not edge_invariant:
        edge_features = {
            "is_single": bond_type_onehot[0],
            "is_double": bond_type_onehot[1],
            "is_triple": bond_type_onehot[2],
            "is_aromatic": bond_type_onehot[3],
            "bond_length": bond_length,
            "has_metal": 1 if has_metal else 0,
        }
    else:
        edge_features = {"bond_length": bond_length, "has_metal": 1 if has_metal else 0}

    # add user-provided features if necessary
    if extra_feature_dict is not None and isinstance(extra_feature_dict, dict):
        for key, val in extra_feature_dict.items():
            if key in edge_features:
                raise ValueError(
                    f"User-provided feature '{key}' conflicts with existing feature name. Please choose a different name for this feature."
                )
            edge_features[key] = val

    # scale edge features as needed
    scale_dict = globalvars.bond_scales
    to_scale = ["bond_length"]
    for key in to_scale:
        min_val, max_val = scale_dict[key]
        edge_features[key] = (edge_features[key] - min_val) / (max_val - min_val)

    return edge_features


def mol_fingerprint(
    charge,
    spinmult,
    num_atoms,
    num_bonds,
    node_features_list,
    extra_feature_dict=None,
):
    """
    Fingerprint graph with molecular properties.
    INPUTS:
        charge: int
            Net charge of molecule.
        spinmult: int
            Spin multiplicity of molecule. singlet=1, doublet=2, etc.
            spinmult=n+1, where n=number of unpaired electrons.
        num_atoms: int
            Number of atoms in molecule.
        num_bonds: int
            Number of bonds in molecule.
        node_features_list: list
            List of node feature dictionaries for each atom in molecule.
        extra_feature_dict: dict
            Additional user-provided graph features.
            default=None
    OUTPUTS:
        graph_features: dict
            Dictionary of molecular features.
    """
    graph_features = {"charge": charge, "spinmult": spinmult}

    # add user-provided features if necessary
    if extra_feature_dict is not None:
        for key, val in extra_feature_dict.items():
            if key in graph_features:
                raise ValueError(
                    f"User-provided feature '{key}' conflicts with existing feature name. Please choose a different name for this feature."
                )
            graph_features[key] = val
    # NOTE: User-provided continuous features should ideally be pre-normalized to [0,1] or [-1,1]
    # by the user. If needed, per-feature normalization can be added to the preprocessing pipeline
    # by computing mean/std across the entire dataset and applying StandardScaler-like normalization.

    # scale graph features as needed
    scale_dict = globalvars.molecule_scales
    to_scale = ["charge", "spinmult"]
    for key in to_scale:
        min_val, max_val = scale_dict[key]
        graph_features[key] = (graph_features[key] - min_val) / (max_val - min_val)
    return graph_features


def determine_charge_and_spin(
    syms, extra_node_features, extra_graph_features, charge_spin_override=False,
    num_implicit_Hs=0,
):
    """
    Determine molecular charge and spin multiplicity from atomic charges and/or user-provided values.

    INPUTS:
        syms: list
            List of atomic symbols.
        extra_node_features: dict or None
            Dictionary containing extra node features, may include 'atomic_charges'.
        extra_graph_features: dict or None
            Dictionary containing extra graph features, may include 'charge' and/or 'spinmult'.
        num_implicit_Hs: int
            Number of implicit hydrogen atoms (removed from syms). Their electrons
            are added back when computing the total electron count.
            default=0

    OUTPUTS:
        charge: int
            Net charge of molecule.
        spinmult: int
            Spin multiplicity of molecule.

    LOGIC:
        - Default: charge=0, spinmult=1 or 2 (low spin based on electron parity)
        - If atomic_charges provided: charge = sum(atomic_charges)
        - If graph charge provided: use it (validates against atomic_charges if both present)
        - If graph spinmult provided: use it (validates physical consistency)
        - All three can be provided: validates atomic_charges sum == charge and spinmult is physical
    """
    # check if user provided atomic charges
    has_atomic_charges = (
        extra_node_features is not None and "atomic_charges" in extra_node_features
    )
    atomic_charge_sum = (
        sum(float(x) for x in extra_node_features["atomic_charges"])
        if has_atomic_charges
        else None
    )

    # check if user provided graph-level charge or spin
    user_charge = (
        extra_graph_features.get("charge") if extra_graph_features is not None else None
    )
    user_spinmult = (
        extra_graph_features.get("spinmult")
        if extra_graph_features is not None
        else None
    )

    # determine final charge
    if has_atomic_charges and user_charge is not None:
        # both atomic charges and graph charge provided, validate they match
        if not np.isclose(atomic_charge_sum, user_charge):
            msg = (
                f"Mismatch between atom-level charges and molecule charge.\n"
                f"sum(atomic_charges) = {atomic_charge_sum}\n"
                f"molecule charge = {user_charge}"
            )
            if charge_spin_override:
                import warnings

                warnings.warn(f"WARNING (charge_spin_override=True): {msg}")
            else:
                raise ValueError(msg)
        charge = user_charge
    elif has_atomic_charges:
        # only atomic charges provided, sum them to get molecular charge
        charge = atomic_charge_sum
    elif user_charge is not None:
        # only graph charge provided
        charge = user_charge
    else:
        # no charge information provided, default to neutral molecule
        charge = 0

    # validate charge is an integer
    if not np.isclose(charge, round(charge)):
        raise ValueError(f"Charge must be an integer. Received charge={charge}")
    charge = int(round(charge))

    # calculate number of electrons (include implicit hydrogens)
    num_electrons = sum([globalvars.atomic_number[sym] for sym in syms]) + num_implicit_Hs - charge

    # determine final spin multiplicity
    if user_spinmult is not None:
        # user provided spin, validate it's physical
        if not (np.isclose(user_spinmult, int(user_spinmult)) and user_spinmult >= 1):
            raise ValueError(
                f"Spin multiplicity must be a positive integer. Received spinmult={user_spinmult}"
            )
        # check parity: even electrons requires odd spin multiplicity (and vice versa)
        if num_electrons % 2 == user_spinmult % 2:
            msg = (
                f"Invalid charge/spin multiplicity.\n"
                f"number of electrons: {num_electrons}\n"
                f"total charge: {charge}\n"
                f"spin multiplicity: {user_spinmult}\n"
                f"(even electrons requires odd spin multiplicity and vice versa)"
            )
            if charge_spin_override:
                import warnings

                warnings.warn(f"WARNING (charge_spin_override=True): {msg}")
            else:
                raise ValueError(msg)
        spinmult = user_spinmult
    else:
        # no user spin, assume low spin
        spinmult = 1 if num_electrons % 2 == 0 else 2

    return charge, spinmult


def bulk_fingerprint(sym):
    """
    Fingerprint nodes with atomic properties determined from bulk structures.
    INPUTS:
        sym: str
            Atomic symbol specifying element.
    OUTPUTS:
        bulk_features: dict
            Dictionary of bulk atomic features.
    """
    bulk_features = {
        # continuous features
        "melting_temperature": globalvars.melting_temperature[sym],
        "boiling_temperature": globalvars.boiling_temperature[sym],
        "heat_fusion": globalvars.heat_fusion[sym],
        "heat_vaporization": globalvars.heat_vaporization[sym],
        "specific_heat": globalvars.specific_heat[sym],
        "thermal_conductivity": globalvars.thermal_conductivity[sym],
        "electrical_conductivity": globalvars.electrical_conductivity[sym],
        "lattice_angles": globalvars.lattice_angles[sym],
        "lattice_constants": globalvars.lattice_constants[sym],
        # categorical features
        "phase": one_hot(globalvars.phase[sym], globalvars.choices["phase"]),
        "crystal_structure": one_hot(
            globalvars.crystal_structure[sym], globalvars.choices["crystal_structure"]
        ),
        "space_group": one_hot(
            globalvars.space_group[sym], globalvars.choices["space_group"]
        ),
    }

    # scale node features as needed
    scale_dict = globalvars.bulk_scales
    to_scale = [
        "melting_temperature",
        "boiling_temperature",
        "heat_fusion",
        "heat_vaporization",
        "specific_heat",
        "thermal_conductivity",
        "electrical_conductivity",
    ]
    for key in to_scale:
        min_val, max_val = scale_dict[key]
        bulk_features[key] = (bulk_features[key] - min_val) / (max_val - min_val)

    return bulk_features


def xtb_fingerprint(syms, coords, charge, spinmult):
    """
    Fingerprint graph with electronic properties calculated by xTB.
    INPUTS:
        syms: list
            List of atomic symbols.
        coords: list
            Numpy array of atomic coordinates.
        charge: int
            Net charge of molecule.
        spinmult: int
            Spin multiplicity of molecule. singlet=1, doublet=2, etc.
            spinmult=n+1, where n=number of unpaired electrons.
    OUTPUTS:
        electronic_features: tuple
            Length 3 tuple containing dictionaries of electronic features for atoms, bonds, and molecule.
    """
    numbers = [globalvars.atomic_number[sym] for sym in syms]
    num_atoms = len(syms)
    uhf = spinmult - 1 if spinmult is not None else None

    calc = Calculator(
        "GFN2-xTB",
        numbers=numbers,
        positions=coords,
        charge=charge,
        uhf=uhf,
    )
    try:
        res = calc.singlepoint()
        # graph features
        energy = float(res.get("energy"))
        dipole = np.linalg.norm(res.get("dipole"))
        eps = res.get("orbital-energies")
        occ = res.get("orbital-occupations")
        homo_index = np.where(occ > 0)[0].max()
        lumo_index = homo_index + 1
        homo = eps[homo_index]
        lumo = eps[lumo_index]
        gap = lumo - homo
        # node features
        atom_charges = res.get("charges")
        atom_energies = res.get("energies")
        # edge features
        bond_orders = res.get("bond-orders")
    except:
        print("excepted")
        energy = 0
        dipole = 0
        gap = 0
        atom_charges = np.array([0] * num_atoms)
        atom_energies = np.array([0] * num_atoms)
        bond_orders = np.array([np.array([0] * num_atoms)] * num_atoms)

    electronic_features_mol = {
        "energy": energy,
        "dipole": dipole,
        "gap": gap,
    }
    electronic_features_atom = {
        "atom_charges": atom_charges,
        "atom_energies": atom_energies,
    }
    electronic_features_bond = {"bond_orders": bond_orders}

    return (electronic_features_mol, electronic_features_atom, electronic_features_bond)


def mol_to_graph(
    mol_string,
    graph_format="mol2",
    edge_invariant=False,
    center_idx: Optional[int] = None,
    extra_node_features=None,
    extra_edge_features=None,
    extra_graph_features=None,
    k_hops=1,
    use_xtb=False,
    use_bulk=False,
    bond_scale_factor=1.0,
    charge_spin_override=False,
    implicit_Hs=False,
    rdkit_features=False,
):
    """
    Featurizes a molecular graph and converts to PyTorch Geometric.
    INPUTS:
        mol_string: str
            Molecular graph stored as text.
        graph_format: str
            Format of original molecular graph. Supported types are 'mol2', 'xyz', 'smiles', 'mol', 'sdf', 'cml', and 'cif'.
            default='mol2'
        edge_invariant: boolean
            Flag to drop all GNN features based on bond order, allowing for resonant-invariance.
            default=False
        charge: int
            Net charge of molecule.
            default=0
        spinmult: int
            Spin multiplicity of molecule. singlet=1, doublet=2, etc.
            spinmult=n+1, where n=number of unpaired electrons.
            default=1
        subgraph: boolean
            Flag to indicate subgraph-level predictions
            default=False
        center_idx: int
            Index upon which to center subgraph. Only used for subgraph=True.
            default=0
        k_hops: int
            Maximum number of edges away from center_idx to search when defining subgraph. Only used for subgraph=True.
            default=1
        use_xtb: boolean
            Flag to include electronic features calculated via xTB.
            default=False
        use_bulk: boolean
            Flag to include bulk atomic features for each node.
            default=False
        implicit_Hs: boolean
            If True, hydrogen atoms are not included as explicit nodes/edges.
            Instead, a 'num_Hs' feature is added to each heavy atom node counting
            the number of bonded hydrogens. Reduces graph size for 2D GNNs.
            default=False
    OUTPUTS:
        data: torch_geometric.data.Data
            Molecular graph stored in PyTorch Geometric Data format.
    """

    # instantiate graph
    mol = nx.Graph()
    syms, coords, bonds = parse_mol_string(
        mol_string=mol_string,
        graph_format=graph_format,
        bond_scale_factor=bond_scale_factor,
    )

    # capture original (pre-filter) atom and bond counts so data.py can correctly
    # discriminate node-level vs edge-level targets when applying implicit_Hs filtering
    original_n_atoms = len(syms)
    original_n_bonds = len(bonds)

    # optionally augment with RDKit-derived organic-chemistry features (SMILES only).
    # uses the existing extras plumbing so the implicit_Hs filter remaps correctly.
    if rdkit_features and graph_format == "smiles":
        rdkit_atom_x, rdkit_bond_x = extract_rdkit_features(mol_string)
        if extra_node_features is None:
            extra_node_features = {}
        if extra_edge_features is None:
            extra_edge_features = {}
        for k, v in rdkit_atom_x.items():
            if k in extra_node_features:
                raise ValueError(
                    f"rdkit_features collision: '{k}' already in extra_node_features"
                )
            extra_node_features[k] = v
        for k, v in rdkit_bond_x.items():
            if k in extra_edge_features:
                raise ValueError(
                    f"rdkit_features collision: '{k}' already in extra_edge_features"
                )
            extra_edge_features[k] = v

    # optionally collapse explicit hydrogens into a num_Hs node feature
    heavy_indices = None
    kept_edge_indices = None
    if implicit_Hs:
        # identify hydrogen atom indices to strip. For SMILES, only strip Hs added by
        # Chem.AddHs (true implicit Hs). Explicit Hs written in the SMILES — e.g. [H-]
        # hydride ions or [H+] protons — are semantically meaningful and kept as graph
        # nodes. read_smiles guarantees that post-AddHs atoms at indices [0, n_explicit)
        # preserve the original SMILES atom ordering, while indices [n_explicit, ...)
        # are AddHs-appended implicit hydrogens.
        if graph_format == "smiles":
            n_explicit = Chem.MolFromSmiles(mol_string, sanitize=True).GetNumAtoms()
            h_indices = {
                i for i, sym in enumerate(syms) if sym == "H" and i >= n_explicit
            }
        else:
            h_indices = {i for i, sym in enumerate(syms) if sym == "H"}
        # count H neighbors for each heavy atom
        h_counts = {i: 0 for i in range(len(syms)) if i not in h_indices}
        for bond in bonds:
            a1, a2, _ = bond
            if a1 in h_indices and a2 not in h_indices:
                h_counts[a2] += 1
            elif a2 in h_indices and a1 not in h_indices:
                h_counts[a1] += 1
        # build old-to-new index mapping for heavy atoms
        heavy_indices = sorted(h_counts.keys())
        idx_map = {old: new for new, old in enumerate(heavy_indices)}
        # filter atoms, coords, and bonds
        syms = [syms[i] for i in heavy_indices]
        if coords.ndim > 1:
            coords = coords[heavy_indices]
        # remap extra edge features before filtering bonds (need original indices)
        kept_edge_indices = [
            j
            for j, (a1, a2, _) in enumerate(bonds)
            if a1 not in h_indices and a2 not in h_indices
        ]
        if extra_edge_features is not None:
            extra_edge_features = {
                key: [values[j] for j in kept_edge_indices]
                for key, values in extra_edge_features.items()
            }
        bonds = [
            (idx_map[a1], idx_map[a2], bt)
            for a1, a2, bt in bonds
            if a1 not in h_indices and a2 not in h_indices
        ]
        # remap extra node features to heavy atoms only
        if extra_node_features is not None:
            extra_node_features = {
                key: [values[i] for i in heavy_indices]
                for key, values in extra_node_features.items()
            }
        # remap center_idx if provided
        if center_idx is not None:
            if center_idx in h_indices:
                raise ValueError(
                    f"center_idx {center_idx} refers to a hydrogen atom, which is "
                    "removed when implicit_Hs=True."
                )
            center_idx = idx_map[center_idx]
        # store H counts keyed by new indices
        h_counts_remapped = {idx_map[old]: h_counts[old] for old in heavy_indices}

    # featurize each atom (node)
    node_features_list = []
    for atom_idx, sym in enumerate(syms):
        num_bonds = sum(1 for bond in bonds if atom_idx in bond[0:2])
        # extract features for this specific atom from the dict of lists
        extra_feature_dict = None
        if extra_node_features is not None:
            extra_feature_dict = {
                key: values[atom_idx]
                for key, values in extra_node_features.items()
                if atom_idx < len(values)
            }
        node_features = atomic_fingerprint(
            sym=sym,
            num_bonds=num_bonds,
            extra_feature_dict=extra_feature_dict,
            num_Hs=h_counts_remapped.get(atom_idx) if implicit_Hs else None,
        )
        mol.add_node(atom_idx, symbol=sym, **node_features)
        node_features_list.append(node_features)
    feature_names = list(node_features.keys())
    node_attr = [
        flatten_features(node, feature_names) for _, node in mol.nodes(data=True)
    ]
    x = torch.tensor(node_attr, dtype=torch.float)

    # featurize each bond (edge)
    for bond_idx, bond in enumerate(bonds):
        atom1, atom2, bond_type = bond
        # extract features for this specific bond from the dict of lists
        extra_feature_dict = None
        if extra_edge_features is not None:
            extra_feature_dict = {
                key: values[bond_idx]
                for key, values in extra_edge_features.items()
                if bond_idx < len(values)
            }
        edge_features = bond_fingerprint(
            bonded_atoms=(syms[atom1], syms[atom2]),
            bond_type=bond_type.lower(),
            edge_invariant=edge_invariant,
            extra_feature_dict=extra_feature_dict,
        )
        mol.add_edge(atom1, atom2, **edge_features)

    # Handle edge_index creation, accounting for molecules with no edges (disconnected atoms)
    # This is valid for sparse/disconnected molecular graphs
    edges = list(mol.edges)
    if len(edges) > 0:
        edge_index = torch.tensor(edges).t().contiguous()
        edge_attr = [
            [edge[feature] for feature in edge_features.keys()]
            for _, _, edge in mol.edges(data=True)
        ]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # Molecules with no edges (all atoms disconnected) - valid for GNNs with proper message passing
        # Create template edge_features to determine the number of columns. Pass a synthetic
        # extras dict so the template includes any rdkit_features / user-provided extras and
        # matches the dimensionality of bonded molecules in the same batch.
        synthetic_extras = (
            {key: 0 for key in extra_edge_features.keys()}
            if extra_edge_features is not None
            else None
        )
        template_edge_features = bond_fingerprint(
            bonded_atoms=(
                "C",
                "C",
            ),  # Use dummy atoms to get template feature dimensionality
            bond_type="1",
            edge_invariant=edge_invariant,
            extra_feature_dict=synthetic_extras,
        )
        num_edge_features = len(template_edge_features)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, num_edge_features), dtype=torch.float)

    # determine charge and spin multiplicity from atomic charges and/or user inputs
    # count total implicit hydrogens for correct electron counting
    total_implicit_Hs = sum(h_counts_remapped.values()) if implicit_Hs else 0
    charge, spinmult = determine_charge_and_spin(
        syms,
        extra_node_features,
        extra_graph_features,
        charge_spin_override=charge_spin_override,
        num_implicit_Hs=total_implicit_Hs,
    )
    # filter out charge and spinmult from extra_graph_features to avoid duplicates
    filtered_graph_features = None
    if extra_graph_features is not None:
        filtered_graph_features = {
            key: val
            for key, val in extra_graph_features.items()
            if key not in ("charge", "spinmult")
        }
    # featurize entire molecule (graph)
    graph_features = mol_fingerprint(
        charge=charge,
        spinmult=spinmult,
        num_atoms=len(syms),
        num_bonds=len(bonds),
        node_features_list=node_features_list,
        extra_feature_dict=filtered_graph_features,
    )
    graph_attr = torch.tensor(
        [float(v) for v in graph_features.values()], dtype=torch.float
    ).unsqueeze(
        0
    )  # (1, graph_attr_dim) so PyG batching stacks to (batch_size, graph_attr_dim)

    # store coordinates (only used for 3D GNN)
    coords = torch.tensor(coords, dtype=torch.float)

    # store electronic features (optional)
    if use_xtb:
        electronic_features = xtb_fingerprint(
            syms=syms,
            coords=coords.numpy(),
            charge=charge,
            spinmult=spinmult,
        )
        elec_mol, elec_atom, elec_bond = electronic_features
        # save electronic features as graph attributes
        elec_mol_tensor = torch.tensor(
            list(elec_mol.values()), dtype=torch.float
        ).unsqueeze(0)
        graph_attr = torch.cat(
            [graph_attr, elec_mol_tensor], dim=1
        )  # concat along feature dim
        # save electronic features as node attributes
        elec_atom_tensor = torch.stack(
            [torch.tensor(elec_atom[k], dtype=torch.float) for k in elec_atom.keys()],
            dim=1,
        )
        x = torch.cat([x, elec_atom_tensor], dim=1)
        # save electronic features as edge attributes
        bo_matrix = elec_bond["bond_orders"]
        # only extract elements of BO matrix corresponding to chemical bonds in molecular graph
        bo_list = []
        for a, b in mol.edges():
            bo_list.append([bo_matrix[a, b]])
        elec_bond_tensor = torch.tensor(bo_list, dtype=torch.float)
        edge_attr = torch.cat([edge_attr, elec_bond_tensor], dim=1)

    # store bulk features (optional)
    if use_bulk:
        bulk_features_list = []
        for idx, sym in enumerate(syms):
            bulk_features = bulk_fingerprint(sym=sym)
            bulk_features_list.append(bulk_features)
        bulk_feature_names = list(bulk_features.keys())
        bulk_attr = [
            flatten_features(node, bulk_feature_names) for node in bulk_features_list
        ]
        bulk_attr = torch.tensor(bulk_attr, dtype=torch.float)
        x = torch.cat([x, bulk_attr], dim=1)

    # define subgraph
    node_mask = None
    edge_mask = None
    if center_idx is not None:
        assert center_idx < len(
            syms
        ), f"center_idx {center_idx} exceeds number of atoms {len(syms)}"
        distances = nx.single_source_shortest_path_length(
            mol, center_idx, cutoff=k_hops
        )
        subgraph_idx = torch.tensor(list(distances.keys()), dtype=torch.long)

        # node mask
        node_mask = torch.zeros(len(syms), dtype=torch.bool)
        node_mask[subgraph_idx] = True
        node_mask = node_mask.unsqueeze(1)

        # edge mask: all edges where both endpoints are in the k-hop subgraph
        edges = torch.tensor(list(mol.edges()), dtype=torch.long)
        edge_mask = torch.isin(edges[:, 0], subgraph_idx) & torch.isin(
            edges[:, 1], subgraph_idx
        )

        # center edge mask: only edges directly incident to the center node
        # used for edge-level prediction to align with center-incident targets
        center_tensor = torch.tensor([center_idx], dtype=torch.long)
        center_edge_mask = torch.isin(edges[:, 0], center_tensor) | torch.isin(
            edges[:, 1], center_tensor
        )

    # save to PyG Data object
    data = Data(
        x=x,
        coords=coords,
        edge_index=edge_index,
        edge_attr=edge_attr,
        graph_attr=graph_attr,
        node_mask=node_mask,
        edge_mask=edge_mask,
        center_edge_mask=center_edge_mask if center_idx is not None else None,
        center_idx=center_idx,
    )

    # store original heavy-atom/bond indices for node/edge-level target filtering.
    # original_n_atoms / original_n_bonds let data.py disambiguate whether a target
    # is node-aligned or edge-aligned (both heavy_*_indices exist with implicit_Hs).
    if implicit_Hs:
        data.heavy_atom_indices = torch.tensor(heavy_indices, dtype=torch.long)
        data.original_n_atoms = original_n_atoms
        data.original_n_bonds = original_n_bonds
        if kept_edge_indices is not None:
            data.heavy_bond_indices = torch.tensor(kept_edge_indices, dtype=torch.long)

    return data


def validate_inputs(
    data_path,
    task,
    representation,
    center_columns,
    graph_format,
    mol_column,
    train_val_test_split,
    stratify=None,
    group_by=None,
    bond_scale_factor=1.0,
):
    """
    Checks validity of user-provided inputs and converts to lowercase where applicable.
    """

    if isinstance(data_path, str):
        if not data_path.endswith(".csv"):
            raise ValueError(f"Data '{data_path}' must be in .csv file format")
        presplit = False
    elif isinstance(data_path, list):
        assert (
            len(data_path) == 3
        ), "List inputs must include all three data splits ordered as [train_path, val_path, test_path]"
        for data_split in data_path:
            if not data_split.endswith(".csv"):
                raise ValueError(f"Data '{data_split}' must be in .csv file format")
            header = get_csv_header(csv_path=data_split)
            if mol_column not in header:
                raise ValueError(f"Column '{mol_column}' not found in '{data_split}'")
        presplit = True
    else:
        raise ValueError(f"Data '{data_path} provided in unsupported format")

    task = task.lower()
    if task not in ["regression", "classification"]:
        raise ValueError(
            f"Task '{task}' must be either 'regression' or 'classification'"
        )

    representation = representation.lower()
    if representation not in ["auto", "learned"]:
        raise ValueError(
            f"Representation '{representation}' must be either 'auto' or 'learned'"
        )

    if center_columns is not None and not isinstance(center_columns, list):
        center_columns = [center_columns]
        for center_column in center_columns:
            if not presplit:
                header = get_csv_header(csv_path=data_path)
                if center_column not in header:
                    raise ValueError(
                        f"Column '{center_column}' not found in '{data_path}'"
                    )
            else:
                for data_split in data_path:
                    header = get_csv_header(csv_path=data_split)
                    if center_column not in header:
                        raise ValueError(
                            f"Column '{center_column}' not found in '{data_split}'"
                        )

    graph_format = graph_format.lower()
    if graph_format not in ["mol2", "xyz", "smiles", "mol", "sdf", "cml", "cif"]:
        raise ValueError(
            f"Molecular graph format '{graph_format}' must be either 'mol2', 'xyz', 'smiles', 'mol', 'sdf', 'cml', or 'cif'"
        )

    drop_indices = None
    if graph_format == "smiles":
        if not presplit:
            smiles_list = pd.read_csv(data_path)[mol_column].tolist()
            drop_indices = validate_smiles(smiles_list)
        else:
            smiles_lists = [
                pd.read_csv(data_split)[mol_column].tolist() for data_split in data_path
            ]
            drop_indices = [
                validate_smiles(smiles_list) for smiles_list in smiles_lists
            ]

    if train_val_test_split is not None and sum(train_val_test_split) != 1:
        raise ValueError(f"Data splits '{train_val_test_split}' must sum to 1")

    # validate bond_scale_factor
    if bond_scale_factor <= 0:
        raise ValueError(
            f"bond_scale_factor must be a positive number. Received bond_scale_factor={bond_scale_factor}"
        )

    # validate stratify and group_by columns exist in the data
    _validate_column_arg = []
    if stratify is not None:
        _validate_column_arg.append(("stratify", stratify))
    if group_by is not None:
        cols = [group_by] if isinstance(group_by, str) else group_by
        for col in cols:
            _validate_column_arg.append(("group_by", col))

    for arg_name, col_name in _validate_column_arg:
        if not presplit:
            header = get_csv_header(csv_path=data_path)
            if col_name not in header:
                raise ValueError(
                    f"{arg_name} column '{col_name}' not found in '{data_path}'"
                )
        else:
            for data_split in data_path:
                header = get_csv_header(csv_path=data_split)
                if col_name not in header:
                    raise ValueError(
                        f"{arg_name} column '{col_name}' not found in '{data_split}'"
                    )

    return presplit, task, representation, center_columns, graph_format, drop_indices


def get_csv_header(csv_path):
    """
    Gets header row of .csv without reading entire file.
    """
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
    return header


def validate_smiles(smiles_list):
    """
    Checks if SMILES is valid.
    """
    invalid_indices = []
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if not mol:
            print(f"WARNING: Skipping invalid SMILES {smiles}")
            valid = False
            invalid_indices.append(idx)
    return invalid_indices


def drop_invalid_smiles(data_path, drop_indices):
    """
    Drops invalid SMILES rows and saves a backup.
    """
    if not drop_indices:
        return
    data = pd.read_csv(data_path)
    backup_path = data_path.replace(".csv", "_with_invalid_smiles.csv")
    data.to_csv(backup_path, index=False)
    data.drop(index=drop_indices, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.to_csv(data_path, index=False)
    print(f"Dropped {len(drop_indices)} invalid SMILES from {data_path}.")
