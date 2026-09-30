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
    """

    input_size: int
    hidden_sizes: List[int]
    activations: List[str]
    dropouts: List[float]
    convolutions: List[str]
    edge_dim: Optional[int] = None
    gat_heads: Optional[int] = None
    gat_concat: Optional[bool] = None


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
        in_dim = input_size
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
        edge_attr = getattr(data, "edge_attr", None)

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
        **kwargs,
    ):
        super().__init__()
        if edge_dim is None:
            edge_dim = 0
        hidden_size = hidden_sizes[0]
        out_node_nf = hidden_sizes[-1]
        self.output_size = out_node_nf
        self.cutoff = cutoff
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

        self.embedding = nn.Linear(input_size, hidden_size)
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

        if self.cutoff is not None:
            edge_index = radius_graph(
                x, r=self.cutoff, batch=data.batch, max_num_neighbors=32
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

@dataclass
class OrbitalEGNNConfig(EGNNConfig):
    """Configuration for the orbital-aware EGNN encoder.

    Every field of :class:`EGNNConfig` is inherited unchanged, so an
    ``OrbitalEGNN_Encoder`` configured with only EGNN fields behaves exactly
    like ``EGNN_Encoder``.

    Attributes
    ----------
    shell_layers : int
        Rounds of shell-level refinement after the atom-level EGNN. ``0``
        disables the orbital path entirely and reproduces ``EGNN_Encoder``.
        Default 0.
    shell_hidden : Optional[int]
        Width of the shell feature vectors. Defaults to the atom hidden size.
    shell_embedding_dim : int
        Width of the (angular momentum, radial index) shell-identity embedding.
        Default 16.
    max_shell_am : int
        Largest angular momentum a shell may carry. Default 3 (f).
    max_shell_radial : int
        Largest radial index within one angular momentum. Default 8.
    """

    shell_layers: int = 0
    shell_hidden: Optional[int] = None
    shell_embedding_dim: int = 16
    max_shell_am: int = 3
    max_shell_radial: int = 8


class OrbitalEGNN_Encoder(nn.Module):
    """EGNN that additionally carries a learned state per BASIS-SET SHELL.

    Motivation.  In the plain EGNN every atom is a single invariant vector, and
    anything shell-resolved has to be produced in one shot by the readout.  For
    density-matrix prediction that was measured to be the binding constraint: a
    4.3x parameter increase lowered training error by only 6.6%, and the
    train/validation gap stayed flat at 0.0235, so the model was neither
    capacity- nor data-limited.  Giving each shell its own state, refined by
    message passing, is the targeted response.

    Two levels, deliberately:

    * ATOM level -- the unmodified EGNN, which carries long-range information
      and equivariant coordinate updates at ~35 nodes per complex.
    * SHELL level -- one node per basis-set shell (~150-400 per complex),
      initialised from its parent atom's features plus an embedding of
      (angular momentum, radial index), then refined along the SAME atom-level
      edges.  Messages travel atom-to-atom, so edge count is unchanged; only
      the per-node state is richer.  A shell-resolved all-pairs message would
      be ~121x more edges and is deliberately not done here.

    Shell features are INVARIANT scalars, exactly like the atom features h.
    Equivariance continues to come from the downstream head, so this encoder
    cannot break it.

    DEFAULTS ARE UNCHANGED.  With ``shell_layers=0`` (the default) no orbital
    machinery is constructed and ``forward`` returns exactly what
    ``EGNN_Encoder.forward`` returns.  The orbital path activates only when
    ``shell_layers > 0`` AND the batch actually carries shell assignments.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        cutoff=None,
        shell_layers=0,
        shell_hidden=None,
        shell_embedding_dim=16,
        max_shell_am=3,
        max_shell_radial=8,
        **kwargs,
    ):
        super().__init__()
        self.atom_encoder = EGNN_Encoder(
            input_size=input_size, hidden_sizes=hidden_sizes, cutoff=cutoff,
            **kwargs
        )
        self.output_size = self.atom_encoder.output_size
        self.cutoff = cutoff
        self.shell_layers = int(shell_layers)
        d_atom = hidden_sizes[-1]
        d_shell = int(shell_hidden or d_atom)
        self.shell_hidden = d_shell

        if self.shell_layers <= 0:
            return

        self.am_embedding = nn.Embedding(max_shell_am + 1, shell_embedding_dim)
        self.radial_embedding = nn.Embedding(max_shell_radial + 1,
                                             shell_embedding_dim)
        self.shell_init = nn.Sequential(
            nn.Linear(d_atom + 2 * shell_embedding_dim, d_shell), nn.SiLU(),
            nn.Linear(d_shell, d_shell),
        )
        self.distance_rbf = GaussianSmearing(
            0.0, cutoff if cutoff else 10.0, kwargs.get("num_gaussians", 64)
        )
        n_rbf = kwargs.get("num_gaussians", 64)
        self.shell_message = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_shell + n_rbf, d_shell), nn.SiLU(),
                nn.Linear(d_shell, d_shell),
            ) for _ in range(self.shell_layers)
        ])
        self.shell_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_shell + d_atom, d_shell), nn.SiLU(),
                nn.Linear(d_shell, d_shell),
            ) for _ in range(self.shell_layers)
        ])
        self.shell_norms = nn.ModuleList([
            nn.LayerNorm(d_shell) for _ in range(self.shell_layers)
        ])

    def forward(self, data):
        """Atom-level output, matching ``EGNN_Encoder.forward`` exactly.

        Returns ``(h, h)`` so this encoder is a drop-in wherever an
        ``EGNN_Encoder`` is used; the shell features are reached through
        :meth:`forward_shells`.
        """
        return self.atom_encoder(data)

    def forward_shells(self, data):
        """Return ``(h_atom, s_shell)``.

        ``s_shell`` is ``None`` when the orbital path is disabled or the batch
        carries no shell assignment, so callers can fall back cleanly.

        The batch supplies:
          ``data.shell_atom_index`` : (n_shell,) parent atom of each shell
          ``data.shell_am``         : (n_shell,) angular momentum
          ``data.shell_radial``     : (n_shell,) radial index within that am
        """
        h, _ = self.atom_encoder(data)
        if self.shell_layers <= 0:
            return h, None
        idx = getattr(data, "shell_atom_index", None)
        if idx is None:
            return h, None

        am = data.shell_am.long()
        radial = data.shell_radial.long().clamp_max(
            self.radial_embedding.num_embeddings - 1)
        s = self.shell_init(torch.cat(
            [h[idx], self.am_embedding(am.clamp_max(
                self.am_embedding.num_embeddings - 1)),
             self.radial_embedding(radial)], dim=-1))

        x = data.coords
        if self.cutoff is not None:
            edge_index = radius_graph(x, r=self.cutoff, batch=data.batch,
                                      max_num_neighbors=32)
        else:
            edge_index = data.edge_index
        row, col = edge_index
        dist = (x[row] - x[col]).norm(dim=-1, keepdim=True)
        edge_rbf = self.distance_rbf(dist.squeeze(-1))

        # Atom-level pooled shell state, so a shell sees its neighbours'
        # orbital content without paying for all-pairs shell edges.
        for layer in range(self.shell_layers):
            pooled = torch.zeros(h.shape[0], s.shape[1], device=s.device,
                                 dtype=s.dtype).index_add_(0, idx, s)
            msg = self.shell_message[layer](
                torch.cat([pooled[row], pooled[col], edge_rbf], dim=-1))
            agg = torch.zeros_like(pooled).index_add_(0, row, msg)
            s = s + self.shell_update[layer](
                torch.cat([s, agg[idx], h[idx]], dim=-1))
            s = self.shell_norms[layer](s)
        return h, s

