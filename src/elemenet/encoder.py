import torch.nn as nn
import torch
from torch_geometric.nn import GATv2Conv, GINEConv, GraphConv, NNConv, radius_graph
from elemenet.mlp import (
    ACT_MAP,
    CONV_MAP,
    GaussianSmearing,
    coord2diff,
    EquivariantBlock,
)
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np


@dataclass
class GNNConfig:
    """Configuration for the GNN encoder.

    Attributes
    ----------
    input_size : int
        Dimensionality of raw atom feature vectors.
    hidden_sizes : List[int]
        Number of channels in each convolutional layer.
    activations : List[str]
        Activation function name for each layer.
    dropouts : List[float]
        Dropout rate for each layer (applied only to non-final layers).
    convolutions : List[str]
        Convolution type for each layer: one of ``'gcnconv'``,
        ``'graphconv'``, ``'gineconv'``, ``'nnconv'``, or ``'gat'``.
    edge_dim : Optional[int]
        Dimensionality of edge feature vectors. Required for ``gineconv``,
        ``nnconv``, and ``gat``; ignored otherwise. Default None.
    gat_heads : Optional[int]
        Number of attention heads for GAT layers. Broadcast to all layers if
        a single int. Default None (uses 1 head per layer).
    gat_concat : Optional[bool]
        If True, concatenate multi-head outputs; if False, average them.
        Broadcast to all layers if a single bool. Default None (True).
    graph_attr_dim : int
        Number of graph-level attributes (e.g. charge, spin) appended to every
        node's input features. Default 0 (none).
    bidirectional_edges : bool
        Pass messages both ways along every bond. Default True.
    """

    input_size: int
    hidden_sizes: List[int]
    activations: List[str]
    dropouts: List[float]
    convolutions: List[str]
    edge_dim: Optional[int] = None
    gat_heads: Optional[int] = None
    gat_concat: Optional[bool] = None
    graph_attr_dim: int = 0
    bidirectional_edges: bool = True


@dataclass
class EGNNConfig:
    """Configuration for the E(3)-equivariant GNN (EGNN) encoder.

    Attributes
    ----------
    input_size : int
        Dimensionality of raw atom feature vectors.
    hidden_sizes : List[int]
        Hidden channel dimension for each equivariant block. All values
        should be equal (EGNN requires a constant hidden size across layers).
    edge_dim : Optional[int]
        Dimensionality of pre-computed edge features. Ignored when ``cutoff``
        is set (radius graph edges carry no pre-computed attributes). Default None.
    activations : str or List[str]
        Activation function name(s). Broadcast to all layers if a single str.
        Default ``'silu'``.
    attention : bool
        Enable attention gating in GCL message-passing steps. Default False.
    norm_diff : bool
        Normalize displacement vectors before passing to equivariant update.
        Default True.
    tanh : bool
        Bound equivariant coordinate updates through tanh. Default False.
    coords_range : float
        Maximum coordinate update magnitude per block (used when ``tanh=True``).
        Default 15.
    norm_constant : float
        Additive constant in the coordinate normalization denominator. Default 1.
    inv_sublayers : int
        Number of GCL message-passing steps per equivariant block. Default 2.
    distance_embedding : bool
        Embed scalar distances using Gaussian basis functions. Default True.
    cutoff : float
        Cutoff radius (Å) for radius-graph construction. Overrides precomputed
        edges in the data object. Default 10.0.
    normalization_factor : float
        Divisor for message aggregation. Default 100.
    aggregation_method : str
        Aggregation strategy: ``'sum'`` or ``'mean'``. Default ``'sum'``.
    num_gaussians : int
        Number of Gaussian basis functions for distance embedding. Default 64.
    use_norm : bool
        Apply LayerNorm to the (invariant) node features after each equivariant
        block. Bounds the node-feature residual stream so it cannot compound to
        fp32 overflow in deep stacks. Coordinate-features are never normalized
        (that would break E(3) equivariance). Default False.
    graph_attr_dim : int
        Number of graph-level attributes (e.g. charge, spin) appended to every
        node's input features. Default 0 (none).
    max_num_neighbors : Optional[int]
        Most neighbours each atom keeps from the cutoff graph. Default 32.
        ``None`` keeps every atom inside the cutoff.
    nearest_neighbors : bool
        When the cap binds, keep each atom's ``max_num_neighbors`` nearest
        atoms (ties go to the lower atom index), identically on CPU and GPU.
        Default False keeps PyG's choice: the lowest-numbered atoms inside the
        cutoff on GPU, KD-tree search order on CPU.
    """

    input_size: int
    hidden_sizes: List[int]
    edge_dim: Optional[int] = None
    activations: Union[str, List[str]] = "silu"
    attention: bool = False
    norm_diff: bool = True
    tanh: bool = False
    coords_range: float = 15
    norm_constant: float = 1
    inv_sublayers: int = 2
    distance_embedding: bool = True
    cutoff: float = 10.0
    normalization_factor: float = 100
    aggregation_method: str = "sum"
    num_gaussians: int = 64
    use_norm: bool = False
    graph_attr_dim: int = 0
    max_num_neighbors: Optional[int] = 32
    nearest_neighbors: bool = False


def append_graph_attr(x, data):
    """Append each graph's attributes (e.g. charge, spin) to all of its nodes' features.

    Uses ``data.graph_attr`` as-is, so nodes see exactly the values the
    readout receives.
    """
    return torch.cat([x, data.graph_attr[data.batch]], dim=-1)


def cutoff_graph(x, r, batch, max_num_neighbors=32, nearest_neighbors=False):
    """Radius graph for the EGNN: ``edge_index[1]`` holds each centre atom and
    ``edge_index[0]`` the neighbours it keeps, at most ``max_num_neighbors``.

    ``nearest_neighbors=False`` is PyG's ``radius_graph``. ``True`` keeps each
    centre's nearest atoms inside the cutoff (ties go to the lower atom index),
    with distances in float64 so CPU and GPU keep exactly the same edges.
    """
    n = x.size(0)
    largest = int(torch.bincount(batch).max()) if batch is not None and n > 0 else n
    if not nearest_neighbors:
        cap = largest if max_num_neighbors is None else max_num_neighbors
        return radius_graph(x, r=r, batch=batch, max_num_neighbors=cap)
    # every pair inside a slightly larger radius, then the exact cutoff in float64
    edge_index = radius_graph(x, r=r * (1 + 1e-5), batch=batch, max_num_neighbors=largest)
    diff = x[edge_index[0]].double() - x[edge_index[1]].double()
    dist2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]
    inside = dist2 < float(r) ** 2
    edge_index, dist2 = edge_index[:, inside], dist2[inside]
    # order by centre, then distance, then neighbour index
    order = torch.argsort(edge_index[0], stable=True)
    order = order[torch.argsort(dist2[order], stable=True)]
    order = order[torch.argsort(edge_index[1][order], stable=True)]
    edge_index = edge_index[:, order]
    if max_num_neighbors is not None:
        centre = edge_index[1]
        counts = torch.bincount(centre, minlength=n)
        rank = torch.arange(centre.numel(), device=x.device) - (torch.cumsum(counts, 0) - counts)[centre]
        edge_index = edge_index[:, rank < max_num_neighbors]
    return edge_index


class GNN_Encoder(nn.Module):
    """
    Graph neural network encoder.
    INPUTS:
        input_size: int
            Number of input features.
        hidden_sizes: list of int
            Number of neurons in each hidden layer.
        activations: list of str
            Activation functions to use in each hidden layer.
        dropouts: list of float
            Dropout rate to use in each hidden layer.
        convolutions: list of str
            Convolution functions to use in each hidden layer.
        edge_dim: int
            Edge feature dimension.
            default=None
        gat_heads: list of int
            Number of attention heads to use in each hidden layer for GAT architecture.
            default=None
        gat_concat: list of bool
            Whether to concatenate or average multi-head-attentions in each hidden layer for GAT architecture.
            default=None
        graph_attr_dim: int
            Number of graph-level attributes appended to every node's input features.
            default=0
        bidirectional_edges: bool
            Pass messages both ways along every bond.
            default=True
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        activations,
        dropouts,
        convolutions,
        edge_dim=None,
        gat_heads=None,
        gat_concat=None,
        graph_attr_dim=0,
        bidirectional_edges=True,
        **kwargs,
    ):
        super().__init__()
        num_layers = len(hidden_sizes)
        assert all(
            len(lst) == num_layers for lst in [activations, dropouts, convolutions]
        ), "All per-layer configurations must have the same length!"

        self.convolutions = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.edge_dim = edge_dim
        self.graph_attr_dim = graph_attr_dim
        self.bidirectional_edges = bidirectional_edges
        in_dim = input_size + graph_attr_dim
        # configure GAT parameters
        if gat_heads is None:
            gat_heads = [1] * num_layers
        elif isinstance(gat_heads, int):
            gat_heads = [gat_heads] * num_layers
        else:
            assert (
                len(gat_heads) == num_layers
            ), "gat_heads must be an int or list of int with length=num_layers"
        if gat_concat is None:
            gat_concat = [True] * num_layers
        elif isinstance(gat_concat, bool):
            gat_concat = [gat_concat] * num_layers
        else:
            assert (
                len(gat_concat) == num_layers
            ), "gat_concat must be a bool or list of bool with length=num_layers"

        for idx in range(num_layers):
            if convolutions[idx] == "gineconv":
                nn_fn = nn.Sequential(
                    nn.Linear(in_dim, hidden_sizes[idx]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[idx], hidden_sizes[idx]),
                )
                conv = GINEConv(nn_fn, edge_dim=edge_dim)
            elif convolutions[idx] == "nnconv":
                small_hidden = int(
                    np.sqrt(edge_dim + in_dim)
                )  # reduce hidden size for memory restrictions
                nn_fn = nn.Sequential(
                    nn.Linear(edge_dim, small_hidden),
                    nn.ReLU(),
                    nn.Linear(small_hidden, in_dim * hidden_sizes[idx]),
                )
                conv = NNConv(
                    in_channels=in_dim, out_channels=hidden_sizes[idx], nn=nn_fn
                )
            elif convolutions[idx] == "gat":
                heads = gat_heads[idx]
                concat = gat_concat[idx]
                if concat:
                    if hidden_sizes[idx] % heads != 0:
                        raise ValueError(
                            f"Layer {idx}: hidden_sizes[{idx}]={hidden_sizes[idx]} must be divisible by heads={heads} when concat=True"
                        )
                    out_channels = hidden_sizes[idx] // heads
                else:
                    out_channels = hidden_sizes[idx]
                conv = GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_channels,
                    heads=heads,
                    concat=concat,
                    edge_dim=edge_dim,
                )
            else:
                conv = CONV_MAP[convolutions[idx]](in_dim, hidden_sizes[idx])
            self.convolutions.append(conv)
            self.activations.append(ACT_MAP[activations[idx]])

            # dropout only on non-final layers
            if idx < num_layers - 1 and dropouts[idx] > 0:
                self.dropouts.append(nn.Dropout(dropouts[idx]))
            else:
                self.dropouts.append(nn.Identity())
            in_dim = hidden_sizes[idx]
        self.output_size = hidden_sizes[-1]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.graph_attr_dim > 0:
            x = append_graph_attr(x, data)
        edge_attr = getattr(data, "edge_attr", None)
        # data.edge_index holds each bond once (readouts, masks and edge targets rely
        # on that), so add the reverse direction for message passing only.
        if getattr(self, "bidirectional_edges", False):
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        for idx, conv in enumerate(self.convolutions):
            # store residual
            res = x

            # message passing depending on convolution type
            if isinstance(conv, GraphConv) and edge_attr is not None:
                x = conv(x, edge_index)
            elif isinstance(conv, (GINEConv, NNConv)) and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            elif (
                isinstance(conv, GATv2Conv)
                and self.edge_dim is not None
                and edge_attr is not None
            ):
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            # activation
            x = self.activations[idx](x)

            # dropout
            x = self.dropouts[idx](x)

            # residual connection if dims match
            if x.shape == res.shape:
                x = x + res

        # embeddings returned twice for pooling + projection modules
        return x, x


class EGNN_Encoder(nn.Module):
    """
    E(3)-equivariant graph neural network encoder.
    INPUTS:
        input_size: int
            Number of input features.
        hidden_sizes: list of int
            Number of neurons in each hidden layer.
        cutoff: float
            Cutoff radius to use for graph construction.
            default=None
        activations: list of str
            Activation functions to use in each hidden layer.
        attention: bool
            Whether to use attention during equivariant message-passing.
            default=False
        norm_diff: bool
            Normalize displacement vectors before passing to the equivariant update.
            default=True
        tanh: bool
            Bound equivariant coordinate updates through tanh, capped at coords_range.
            default=False
        coords_range: float
            Maximum coordinate update magnitude per block when tanh=True.
            default=15
        norm_constant: int
            Normalization constant applied to coordinate updates.
            default=1
        inv_sublayers: int
            Number of inverse sublayers to use during atomic coordinate updates.
            default=2
        distance_embedding: bool
            Whether to use distance-based edge-embedding.
            default=True
        normalization_factor: int
            Normalization factor to use during equivariant message-passing.
            default=100
        aggregation_method: str
            Aggregation method to use during atomic coordinate updates.
            default="sum"
        num_gaussians: int
            Number of Gaussians to use during distance embedding projection.
            default=64
        dropouts: list of float
            Dropout rate to use in each hidden layer.
        graph_attr_dim: int
            Number of graph-level attributes appended to every node's input features.
            default=0
        max_num_neighbors: int or None
            Most neighbours each atom keeps from the cutoff graph; None keeps all.
            default=32
        nearest_neighbors: bool
            Keep each atom's nearest neighbours when the cap binds, identically on CPU and GPU.
            default=False
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        cutoff=None,
        edge_dim=None,
        activations="silu",
        attention=False,
        norm_diff=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        inv_sublayers=2,
        distance_embedding=True,
        normalization_factor=100,
        aggregation_method="sum",
        num_gaussians=64,
        use_norm=False,
        dropouts=None,
        graph_attr_dim=0,
        max_num_neighbors=32,
        nearest_neighbors=False,
        **kwargs,
    ):
        super().__init__()
        if edge_dim is None:
            edge_dim = 0
        hidden_size = hidden_sizes[0]
        out_node_nf = hidden_sizes[-1]
        self.output_size = out_node_nf
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.nearest_neighbors = nearest_neighbors
        if dropouts is None:
            dropouts = [0.0] * len(hidden_sizes)
        if self.cutoff is not None:
            print(
                f"EGNN Encoder: Using cutoff radius of {self.cutoff} Å for graph construction."
            )
            print("Ignoring precomputed edge attributes in the data object.")
            edge_dim = 0  # edge attributes not used with cutoff-based graph

        if not all(h == hidden_size for h in hidden_sizes):
            print(
                "!Warning: EGNN_Encoder only supports same hidden size for all layers. "
                "Using the first hidden size for all layers."
            )
        n_layers = len(hidden_sizes)
        if isinstance(activations, str):
            activations = [activations] * n_layers
        elif len(activations) != n_layers:
            raise ValueError("Length of activations list must match number of layers.")
        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if distance_embedding:
            self.distance_embedding = GaussianSmearing(0, cutoff, num_gaussians)
            edge_feat_nf = self.distance_embedding.num_output * 2 + edge_dim
        else:
            self.distance_embedding = None
            edge_feat_nf = 2 + edge_dim  # distance (x0 and xl) + edge_attr

        self.graph_attr_dim = graph_attr_dim
        self.embedding = nn.Linear(input_size + graph_attr_dim, hidden_size)
        self.embedding_out = nn.Linear(hidden_size, out_node_nf)
        # optional LayerNorm on the invariant node features after each block;
        # bounds the residual stream so |h| cannot compound to fp32 overflow.
        self.use_norm = use_norm
        self.h_norms = (
            nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers)])
            if use_norm
            else None
        )
        self.dropouts = nn.ModuleList()
        for i in range(0, n_layers):
            self.add_module(
                "e_block_%d" % i,
                EquivariantBlock(
                    hidden_size,
                    edge_feat_nf=edge_feat_nf,
                    act_fn=ACT_MAP[activations[i]],
                    n_layers=inv_sublayers,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=coords_range,
                    norm_constant=norm_constant,
                    edge_embedding=self.distance_embedding,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                ),
            )

            if i < n_layers - 1 and dropouts[i] > 0:
                self.dropouts.append(nn.Dropout(dropouts[i]))
            else:
                self.dropouts.append(nn.Identity())

    def forward(self, data):
        h, x = data.x, data.coords
        if self.graph_attr_dim > 0:
            h = append_graph_attr(h, data)

        if self.cutoff is not None:
            # encoders pickled before these options existed used a cap of 32 and PyG's choice
            edge_index = cutoff_graph(
                x,
                self.cutoff,
                data.batch,
                max_num_neighbors=getattr(self, "max_num_neighbors", 32),
                nearest_neighbors=getattr(self, "nearest_neighbors", False),
            )
            edge_attr = None
        else:
            edge_index = data.edge_index
            edge_attr = getattr(data, "edge_attr", None)

        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.distance_embedding is not None:
            distances = self.distance_embedding(distances)

        if edge_attr is not None:
            edge_attr = torch.cat([distances, edge_attr], dim=-1)
        else:
            edge_attr = distances

        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=edge_attr
            )
            h = self.dropouts[i](h)
            # normalize only the invariant node features h; never the coordinate
            # features x (LayerNorm on x would break E(3) equivariance).
            if self.use_norm:
                h = self.h_norms[i](h)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        return h, h
