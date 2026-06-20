import torch
import torch.nn as nn
from typing import Optional, Union
from elemenet.mlp import POOL_MAP
from elemenet.readout import (
    MLP,
    EdgePredictor,
    Transformer,
    MLPConfig,
    EdgePredictorConfig,
    TransformerConfig,
)
from elemenet.encoder import GNN_Encoder, EGNN_Encoder, GNNConfig, EGNNConfig
from dataclasses import asdict, is_dataclass


ENCODER_CONFIG_MAP = {
    "gnn": GNNConfig,
    "egnn": EGNNConfig,
}
ENCODER_CONFIGS = Union[GNNConfig, EGNNConfig]

READOUT_CONFIG_MAP = {
    "mlp": MLPConfig,
    "edge_predictor": EdgePredictorConfig,
    "transformer": TransformerConfig,
}
READOUT_CONFIGS = Union[MLPConfig, EdgePredictorConfig, TransformerConfig]

ENCODER_MAP = {
    "gnn": GNN_Encoder,
    "egnn": EGNN_Encoder,
}

READOUT_MAP = {
    "mlp": MLP,
    "edge_predictor": EdgePredictor,
    "transformer": Transformer,
}


def build_encoder_config(encoder_type: str, params: dict):
    """Construct a typed encoder config dataclass from a flat parameter dict.

    Reads layer structure from ``params['encoder_config']`` (keys: ``layers``,
    ``neurons``, ``shape``, ``activation``, ``dropout``, ``convolution``, etc.)
    and expands scalar values into per-layer lists via ``make_layer_dims`` and
    ``make_list``. Returns a ``GNNConfig`` or ``EGNNConfig`` depending on
    ``encoder_type``.

    Parameters
    ----------
    encoder_type : str
        Encoder architecture: ``'gnn'`` or ``'egnn'``.
    params : dict
        Full model parameter dict containing an ``'encoder_config'`` sub-dict.

    Returns
    -------
    GNNConfig or EGNNConfig
    """
    if encoder_type not in ENCODER_CONFIG_MAP:
        raise ValueError(
            f"Unknown encoder type '{encoder_type}'. Expected one of {list(ENCODER_CONFIG_MAP.keys())}."
        )

    # get encoder settings
    encoder_config = params["encoder_config"]

    # shared base settings
    input_size = encoder_config["input_size"]
    edge_dim = encoder_config["edge_dim"]

    # layer structure
    gnn_layers = encoder_config.get("layers", 3)
    gnn_shape = encoder_config.get("shape", "constant")
    hidden_sizes = make_layer_dims(
        encoder_config.get("neurons", 128), gnn_layers, gnn_shape
    )
    activations = make_list(encoder_config.get("activation", "relu"), gnn_layers)
    convolutions = make_list(encoder_config.get("convolution", "graphconv"), gnn_layers)
    dropouts = make_list(encoder_config.get("dropout", 0.1), gnn_layers)

    if encoder_type == "gnn":
        return GNNConfig(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            activations=activations,
            dropouts=dropouts,
            convolutions=convolutions,
            edge_dim=edge_dim,
            gat_heads=encoder_config.get("gat_heads"),
            gat_concat=encoder_config.get("gat_concat"),
        )

    elif encoder_type == "egnn":
        cutoff = encoder_config.get("cutoff", 10.0)
        if cutoff <= 0:
            raise ValueError(
                f"EGNN cutoff must be a positive number. Received cutoff={cutoff}"
            )
        return EGNNConfig(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            edge_dim=edge_dim,
            activations=activations,
            attention=encoder_config.get("attention", False),
            distance_embedding=encoder_config.get("distance_embedding", True),
            cutoff=cutoff,
            normalization_factor=encoder_config.get("normalization_factor", 100),
            aggregation_method=encoder_config.get("aggregation_method", "sum"),
            num_gaussians=encoder_config.get("num_gaussians", 64),
            inv_sublayers=encoder_config.get("inv_sublayers", 2),
        )


def build_readout_config(readout_type: str, params: dict):
    """Construct a typed readout config dataclass from a flat parameter dict.

    Reads layer structure and output settings from ``params['readout_config']``
    and expands them into per-layer lists. Returns the appropriate config
    dataclass for the given readout type.

    Parameters
    ----------
    readout_type : str
        Readout architecture: ``'mlp'``, ``'edge_predictor'``, or
        ``'transformer'``.
    params : dict
        Full model parameter dict containing a ``'readout_config'`` sub-dict.

    Returns
    -------
    MLPConfig, EdgePredictorConfig, or TransformerConfig
    """
    if readout_type not in READOUT_CONFIG_MAP:
        raise ValueError(
            f"Unknown readout type '{readout_type}'. Expected one of {list(READOUT_CONFIG_MAP.keys())}."
        )

    # get encoder settings
    readout_config = params["readout_config"]

    readout_input_size = readout_config.get("input_size")
    layers = readout_config.get("layers", 3)
    output_size = readout_config.get("output_size")
    shape = readout_config.get("shape", "constant")
    hidden_sizes = make_layer_dims(readout_config.get("neurons", 128), layers, shape)
    activations = make_list(readout_config.get("activation", "relu"), layers)
    dropouts = make_list(readout_config.get("dropout", 0.1), layers)
    ensemble_size = params.get("ensemble_size", 1)
    graph_attr_dim = readout_config.get("graph_attr_dim", 0)
    graph_attr_hidden_dim = readout_config.get("graph_attr_hidden_dim", None)

    if readout_type == "edge_predictor":
        return EdgePredictorConfig(
            input_size=readout_input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activations=activations,
            dropouts=dropouts,
            edge_dim=readout_config.get("edge_dim", 0),
            ensemble_size=ensemble_size,
        )
    elif readout_type == "mlp":
        return MLPConfig(
            input_size=readout_input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activations=activations,
            dropouts=dropouts,
            use_norm=readout_config.get("use_norm", True),
            ensemble_size=ensemble_size,
            graph_attr_dim=graph_attr_dim,
            graph_attr_hidden_dim=graph_attr_hidden_dim,
            scope=params.get("scope", "graph"),
            edge_dim=readout_config.get("edge_dim", 0),
        )
    elif readout_type == "transformer":
        return TransformerConfig(
            input_size=readout_input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activations=activations,
            dropouts=dropouts,
            ensemble_size=ensemble_size,
            max_nodes=readout_config.get("max_nodes"),
            num_heads=readout_config.get("num_heads", 8),
            expansion=readout_config.get("expansion", 4),
            graph_attr_dim=graph_attr_dim,
            graph_attr_hidden_dim=graph_attr_hidden_dim,
            scope=params.get("scope", "graph"),
            edge_dim=readout_config.get("edge_dim", 0),
        )


def _collapse_if_constant(lst):
    """Return scalar if list elements are all identical."""
    if isinstance(lst, (list, tuple)) and len(set(lst)) == 1:
        return lst[0]
    return lst


def invert_encoder_config(config_obj, encoder_type: str):
    """Convert a ``GNNConfig`` or ``EGNNConfig`` dataclass back to the compact dict format.

    Reverses ``build_encoder_config``: infers ``neurons``, ``shape``, and
    ``layers`` from the expanded ``hidden_sizes`` list, and collapses uniform
    per-layer lists to scalars.

    Parameters
    ----------
    config_obj : GNNConfig or EGNNConfig
        Typed encoder config dataclass.
    encoder_type : str
        Encoder type string: ``'gnn'`` or ``'egnn'``.

    Returns
    -------
    dict
        Dict with keys ``'encoder_type'`` and ``'encoder_config'`` matching
        the format expected by ``build_encoder_config``.
    """
    if not is_dataclass(config_obj):
        raise TypeError("Expected a dataclass config object.")

    cfg = asdict(config_obj)

    hidden_sizes = cfg["hidden_sizes"]

    # Iinfer neurons, shape, and layers from hidden_sizes
    inferred = infer_shape_and_base(hidden_sizes)

    encoder_config = {
        "input_size": cfg["input_size"],
        "edge_dim": cfg["edge_dim"],
        **inferred,
    }

    # collapse list-derived parameters
    if "activations" in cfg:
        encoder_config["activation"] = _collapse_if_constant(cfg["activations"])

    if "dropouts" in cfg:
        encoder_config["dropout"] = _collapse_if_constant(cfg["dropouts"])

    if encoder_type == "gnn":
        if "convolutions" in cfg:
            encoder_config["convolution"] = _collapse_if_constant(cfg["convolutions"])
        encoder_config["gat_heads"] = cfg.get("gat_heads")
        encoder_config["gat_concat"] = cfg.get("gat_concat")

    elif encoder_type == "egnn":
        encoder_config["attention"] = cfg.get("attention", False)
        encoder_config["distance_embedding"] = cfg.get("distance_embedding", True)
        encoder_config["cutoff"] = cfg.get("cutoff", 10.0)
        encoder_config["normalization_factor"] = cfg.get("normalization_factor", 100)
        encoder_config["aggregation_method"] = cfg.get("aggregation_method", "sum")
        encoder_config["num_gaussians"] = cfg.get("num_gaussians", 64)
        encoder_config["inv_sublayers"] = cfg.get("inv_sublayers", 2)

    return {
        "encoder_type": encoder_type,
        "encoder_config": encoder_config,
    }


def invert_readout_config(config_obj, readout_type: str):
    """Convert a readout config dataclass back to the compact dict format.

    Reverses ``build_readout_config``: infers ``neurons``, ``shape``, and
    ``layers`` from ``hidden_sizes`` and collapses uniform per-layer lists
    to scalars.

    Parameters
    ----------
    config_obj : MLPConfig, EdgePredictorConfig, or TransformerConfig
        Typed readout config dataclass.
    readout_type : str
        Readout type string: ``'mlp'``, ``'edge_predictor'``, or
        ``'transformer'``.

    Returns
    -------
    dict
        Dict with keys ``'readout_type'``, ``'readout_config'``, and
        ``'ensemble_size'`` matching the format expected by
        ``build_readout_config``.
    """

    if not is_dataclass(config_obj):
        raise TypeError("Expected a dataclass config object.")

    cfg = asdict(config_obj)

    hidden_sizes = cfg["hidden_sizes"]
    shape_info = infer_shape_and_base(hidden_sizes)

    ensemble_size = cfg.get("ensemble_size", 1)

    readout_config = {
        "input_size": cfg["input_size"],
        "output_size": cfg["output_size"],
        "layers": shape_info["layers"],
        "shape": shape_info["shape"],
        "neurons": shape_info["neurons"],
        "activation": _collapse_if_constant(cfg["activations"]),
        "dropout": _collapse_if_constant(cfg["dropouts"]),
    }

    # type-specific reconstruction
    if readout_type == "edge_predictor":
        readout_config["edge_dim"] = cfg.get("edge_dim", 0)

        return {
            "readout_type": readout_type,
            "readout_config": readout_config,
            "ensemble_size": ensemble_size,
        }

    elif readout_type == "mlp":
        readout_config["use_norm"] = cfg.get("use_norm", True)
        readout_config["graph_attr_hidden_dim"] = cfg.get("graph_attr_hidden_dim")
        readout_config["graph_attr_dim"] = cfg.get("graph_attr_dim", 0)
        readout_config["edge_dim"] = cfg.get("edge_dim", 0)

        return {
            "readout_type": readout_type,
            "readout_config": readout_config,
            "ensemble_size": ensemble_size,
            "scope": cfg.get("scope", "graph"),
        }

    elif readout_type == "transformer":
        readout_config["max_nodes"] = cfg.get("max_nodes")
        readout_config["num_heads"] = cfg.get("num_heads", 8)
        readout_config["expansion"] = cfg.get("expansion", 4)
        readout_config["graph_attr_hidden_dim"] = cfg.get("graph_attr_hidden_dim", None)
        readout_config["graph_attr_dim"] = cfg.get("graph_attr_dim", 0)
        readout_config["edge_dim"] = cfg.get("edge_dim", 0)

        return {
            "readout_type": readout_type,
            "readout_config": readout_config,
            "ensemble_size": ensemble_size,
            "scope": cfg.get("scope", "graph"),
        }

    else:
        raise ValueError(f"Unknown readout_type: {readout_type}")


def infer_shape_and_base(hidden_sizes):
    """Infer the ``(shape, neurons, layers)`` parameters that would reproduce ``hidden_sizes``.

    Tries each shape supported by ``make_layer_dims`` and returns the first
    combination whose reconstructed list matches ``hidden_sizes`` exactly.

    Parameters
    ----------
    hidden_sizes : list of int
        Per-layer hidden dimensions to reverse-engineer.

    Returns
    -------
    dict
        Keys: ``'shape'`` (str), ``'neurons'`` (int), ``'layers'`` (int).

    Raises
    ------
    ValueError
        If no supported shape can reproduce ``hidden_sizes``.
    """
    num_layers = len(hidden_sizes)

    if num_layers == 1:
        return {
            "shape": "constant",
            "neurons": hidden_sizes[0],
            "layers": 1,
        }

    shapes = ["constant", "increasing", "decreasing", "hourglass", "pyramid"]

    for shape in shapes:
        # candidate base_neurons guesses
        candidates = set()

        if shape == "constant":
            if len(set(hidden_sizes)) == 1:
                candidates.add(hidden_sizes[0])

        elif shape == "increasing":
            candidates.add(hidden_sizes[-1])  # end == base_neurons

        elif shape == "decreasing":
            candidates.add(hidden_sizes[0])  # start == base_neurons

        elif shape in ["hourglass", "pyramid"]:
            candidates.add(max(hidden_sizes))  # base is max value

        for base in candidates:
            regenerated = make_layer_dims(base, num_layers, shape)
            if regenerated == hidden_sizes:
                return {
                    "shape": shape,
                    "neurons": base,
                    "layers": num_layers,
                }
    raise ValueError(
        f"Could not infer shape and base_neurons from hidden_sizes: {hidden_sizes}"
    )


def make_layer_dims(base_neurons, num_layers, shape):
    """Generate a list of hidden layer dimensions according to a named shape pattern.

    Parameters
    ----------
    base_neurons : int
        Reference neuron count. For ``'constant'``, all layers have this size.
        For ``'increasing'`` / ``'decreasing'``, it is the max end. For
        ``'hourglass'`` / ``'pyramid'``, it is the max value.
    num_layers : int
        Number of layers (length of the returned list).
    shape : str
        One of ``'constant'``, ``'increasing'``, ``'decreasing'``,
        ``'hourglass'``, or ``'pyramid'``.

        - ``'constant'``: all layers the same size.
        - ``'increasing'``: linearly increases from ``base_neurons // 2`` to
          ``base_neurons``.
        - ``'decreasing'``: linearly decreases from ``base_neurons`` to
          ``base_neurons // 2``.
        - ``'hourglass'``: decreases to a bottleneck then increases.
        - ``'pyramid'``: increases to a peak then decreases.

    Returns
    -------
    list of int
    """
    if num_layers == 1:
        return [base_neurons]
    if shape == "constant":
        layer_dims = [base_neurons] * num_layers
    elif shape == "increasing":
        start = base_neurons // 2
        end = base_neurons
        step = (end - start) / (num_layers - 1)
        layer_dims = [int(start + i * step) for i in range(num_layers)]
    elif shape == "decreasing":
        start = base_neurons
        end = base_neurons // 2
        step = (end - start) / (num_layers - 1)
        layer_dims = [int(start + i * step) for i in range(num_layers)]
    elif shape == "hourglass":
        if num_layers == 2:  # avoids divide by zero error
            return [base_neurons, base_neurons // 2]
        half = num_layers // 2
        down = [base_neurons - i * (base_neurons // 2) // half for i in range(half + 1)]
        up = [
            base_neurons // 2 + i * (base_neurons // 2) // (num_layers - half - 1)
            for i in range(num_layers - half)
        ]
        layer_dims = down[:-1] + up
    elif shape == "pyramid":
        if num_layers == 2:  # avoids divide by zero error
            return [base_neurons, base_neurons // 2]
        half = num_layers // 2
        up = [
            base_neurons // 2 + i * (base_neurons // 2) // half for i in range(half + 1)
        ]
        down = [
            base_neurons - i * (base_neurons // 2) // (num_layers - half - 1)
            for i in range(num_layers - half)
        ]
        layer_dims = up[:-1] + down
    return layer_dims


def make_list(param, layers):
    """Expand a scalar or string into a list of length ``layers``.

    If ``param`` is already a list, validates its length and returns it
    unchanged. Otherwise broadcasts the single value to every layer.

    Parameters
    ----------
    param : scalar or list
        Value to broadcast, or a pre-expanded list.
    layers : int
        Expected list length.

    Returns
    -------
    list
    """
    if isinstance(param, list):
        assert (
            len(param) == layers
        ), f"Expected list of length {layers}, got {len(param)}"
        return param
    else:
        return [param] * layers


def extract_model_config(params):
    """Build a fully-resolved model config dict from a flat parameter dict.

    Calls ``build_encoder_config`` and ``build_readout_config`` to construct
    typed dataclass configs, then sets ``readout_config.input_size`` to the
    encoder's final hidden dimension so the two components are aligned.

    Parameters
    ----------
    params : dict
        Flat parameter dict with keys including ``encoder_type``,
        ``readout_type``, ``scope``, ``pooling``, and sub-dicts
        ``encoder_config`` / ``readout_config``.

    Returns
    -------
    dict
        Keys: ``'scope'``, ``'encoder_type'``, ``'readout_type'``,
        ``'encoder_config'`` (dataclass), ``'readout_config'`` (dataclass),
        ``'pooling'``.
    """
    encoder_type = params["encoder_type"]
    readout_type = params["readout_type"]

    # convert to lists of per-layer parameters
    scope = params["scope"]

    # get pooling method
    pooling = params.get("pooling", "mean")

    # get readout settings
    readout_config = build_readout_config(readout_type, params)

    # optional get GNN encoder settings
    encoder_config = build_encoder_config(encoder_type, params)

    # readout input = encoder output
    readout_config.input_size = encoder_config.hidden_sizes[-1]

    return {
        "scope": scope,
        "encoder_type": encoder_type,
        "readout_type": readout_type,
        "encoder_config": encoder_config,
        "readout_config": readout_config,
        "pooling": pooling,
    }


def build_model(
    encoder_type: str,
    scope: str,
    readout_config: READOUT_CONFIGS,
    encoder_config: Optional[ENCODER_CONFIGS] = None,
    readout_type: str = "mlp",
    pooling: Optional[str] = None,
    device=None,
):
    """Instantiate and return a ``Model`` from typed config objects.

    Constructs the encoder (GNN or EGNN) and readout (MLP, EdgePredictor, or
    Transformer) modules, links the encoder's output size to the readout's
    input size, wraps them in a ``Model``, and moves everything to ``device``.

    Parameters
    ----------
    encoder_type : str
        Encoder architecture: ``'gnn'`` or ``'egnn'``. Pass ``None`` for a
        pure MLP model (no encoder).
    scope : str
        Prediction granularity: ``'graph'``, ``'node'``, or ``'edge'``.
    readout_config : MLPConfig, EdgePredictorConfig, or TransformerConfig
        Typed readout config dataclass (``input_size`` will be overwritten
        by the encoder's output size when an encoder is present).
    encoder_config : GNNConfig or EGNNConfig or None, optional
        Typed encoder config dataclass. Required when ``encoder_type`` is not
        None.
    readout_type : str, optional
        Readout architecture: ``'mlp'``, ``'edge_predictor'``, or
        ``'transformer'``. Default ``'mlp'``.
    pooling : str or None, optional
        Graph pooling method (``'mean'``, ``'max'``, or ``'sum'``). Required
        for graph-level prediction.
    device : torch.device or str or None, optional
        Target device. Auto-detected (CUDA if available, else CPU) if None.

    Returns
    -------
    Model
        Assembled model moved to ``device``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if scope == "edge" and encoder_type is None:
        raise ValueError(
            "Edge-level prediction (scope='edge') requires a graph encoder "
            "(encoder_type='gnn' or 'egnn'). A pure MLP encoder has no edge structure."
        )

    if encoder_type is None:
        encoder = None
        assert (
            hasattr(readout_config, "input_size")
            and readout_config.input_size is not None
        ), "For MLP models, must provide readout_input_size"
    else:
        assert (
            encoder_config is not None
        ), "Must provide encoder_config for GNN/EGNN models"
        encoder = ENCODER_MAP[encoder_type](**asdict(encoder_config))
        # readout input = encoder output
        readout_config.input_size = encoder.output_size

    readout = READOUT_MAP[readout_type](**asdict(readout_config))

    model = Model(encoder, readout, pooling=pooling, scope=scope)
    model = model.to(device)
    return model


class Model(nn.Module):
    """Deep learning model comprising an optional GNN encoder and a readout head.

    When an encoder is provided, node embeddings are produced by message-passing,
    then optionally pooled (for graph-level tasks) or sub-selected (for node/edge
    tasks) before being passed to the readout. When no encoder is provided, raw
    input features are passed directly to the readout.

    The forward pass attaches ``prediction_mean``, ``prediction_std``, and
    ``embeddings`` to the input ``Data`` object and returns it.

    Parameters
    ----------
    encoder : GNN_Encoder, EGNN_Encoder, or None
        Graph neural network encoder. Pass None for pure MLP operation.
    readout : MLP, EdgePredictor, or Transformer
        Prediction head.
    pooling : str or None
        Pooling method for graph-level tasks: ``'mean'``, ``'max'``, or
        ``'sum'``. Not used for node/edge-level tasks.
    scope : str
        Prediction granularity: ``'graph'``, ``'node'``, or ``'edge'``.
        Default ``'graph'``.
    """

    def __init__(self, encoder, readout, pooling: bool = None, scope: str = "graph"):
        super().__init__()
        self.encoder = encoder
        self.readout = readout
        self.pooling = pooling
        self.scope = scope
        assert scope in [
            "graph",
            "node",
            "edge",
        ], "Scope must be either 'graph', 'node', or 'edge'"
        if scope == "graph":
            assert (
                pooling is not None
            ), "For graph-level prediction, must provide a pooling method"

    def subselect_node_embeddings(self, x, data):
        """Return only the embeddings and batch indices for unmasked (subgraph) nodes.

        If ``data.node_mask`` is absent, all nodes and their batch indices are
        returned unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Node embeddings, shape ``(N, hidden_dim)``.
        data : torch_geometric.data.Data
            Batch object; optionally contains ``node_mask`` of shape ``(N, 1)``.

        Returns
        -------
        x : torch.Tensor
            Filtered node embeddings.
        batch : torch.Tensor
            Corresponding batch assignment vector.
        """
        mask = getattr(data, "node_mask", None)
        if mask is None:
            return x, data.batch
        else:
            mask = mask.squeeze(-1)  # node_mask is stored as [N,1]; flatten to [N] for indexing
            return x[mask], data.batch[mask]

    def subselect_edge_embeddings(self, mean, std, data):
        """Return only the predictions for edges selected by a mask.

        Prefers ``data.center_edge_mask`` (edges incident to the designated
        center atom), falling back to ``data.edge_mask`` (all intra-subgraph
        edges). If neither mask is present, all edge predictions are returned.

        Parameters
        ----------
        mean : torch.Tensor
            Per-edge prediction means, shape ``(E, output_size)``.
        std : torch.Tensor
            Per-edge prediction stds, shape ``(E, output_size)``.
        data : torch_geometric.data.Data
            Batch object; optionally contains ``center_edge_mask`` or
            ``edge_mask``.

        Returns
        -------
        mean : torch.Tensor
            Filtered prediction means.
        std : torch.Tensor
            Filtered prediction stds.
        """
        # for edge-level prediction, prefer center_edge_mask (center-incident edges,
        # aligned with center-incident targets). Fall back to edge_mask (all intra-subgraph
        # edges) when no center is defined.
        mask = getattr(data, "center_edge_mask", None)
        if mask is None:
            mask = getattr(data, "edge_mask", None)
        if mask is None:
            return mean, std
        return mean[mask], std[mask]

    def forward(self, data):
        pool_fn = None
        batch = data.batch
        if self.encoder is not None:
            x, embeddings = self.encoder(data)
            if self.scope == "graph":
                # in case only subgraph is of interest, take only unmasked nodes before pooling
                x, batch = self.subselect_node_embeddings(x, data)
                pool_fn = POOL_MAP[self.pooling]
            elif self.scope == "node":
                # for node-level prediction, restrict output to subgraph nodes if defined
                x, batch = self.subselect_node_embeddings(x, data)
        else:
            x, embeddings = data, data

        # extract graph-level attributes for concatenation after pooling
        graph_attr = getattr(data, "graph_attr", None)
        edge_attr = getattr(data, "edge_attr", None)
        edge_index = getattr(data, "edge_index", None)

        # output has dimension (batch_size, output_size, ensemble_size)
        output, embeddings = self.readout(
            x,
            batch=batch,
            pool_fn=pool_fn,
            edge_attr=edge_attr,
            edge_index=edge_index,
            graph_attr=graph_attr,
        )

        # in case shallow ensemble is used
        if output.shape[-1] > 1:
            mean, std = output.mean(dim=-1), output.std(dim=-1)
        else:
            mean, std = output.squeeze(dim=-1), torch.zeros_like(output).squeeze(dim=-1)

        # if scope is edge-level then subselect given the edge mask
        if self.scope == "edge":
            mean, std = self.subselect_edge_embeddings(mean, std, data)
            # apply the same edge mask to the per-head ensemble output so that
            # ensemble-aware losses (e.g. MoP) see only selected edges.
            edge_mask = getattr(data, "center_edge_mask", None)
            if edge_mask is None:
                edge_mask = getattr(data, "edge_mask", None)
            if edge_mask is not None:
                output = output[edge_mask]

        data.prediction_mean = mean
        data.prediction_std = std
        # Per-head outputs of shape (..., output_size, ensemble_size); used by
        # ensemble-aware losses that require nonlinear aggregation across heads.
        data.prediction_raw = output
        data.embeddings = embeddings
        return data
