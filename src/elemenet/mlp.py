# shallow ensemble from: https://github.com/bananenpampe/DPOSE/blob/main/UCI_experiments/model/mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    GraphConv,
    NNConv,
)

ACT_MAP = {
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(negative_slope=0.01),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "none": nn.Identity(),
}

CONV_MAP = {
    "gcnconv": GCNConv,
    "graphconv": GraphConv,
    "gineconv": GINEConv,
    "nnconv": NNConv,
    "gat": GATv2Conv,
}

POOL_MAP = {"mean": global_mean_pool, "max": global_max_pool, "sum": global_add_pool}

class SwiGLU(torch.nn.Module):
    """SwiGLU feed-forward block used inside the Transformer readout.

    Projects the hidden dimension up by ``expansion``, splits into two halves,
    applies GELU to one half, multiplies element-wise with the other (gating),
    and projects back down. This is the FFN variant from Noam Shazeer (2020).

    Parameters
    ----------
    dim_h : int
        Input (and output) hidden dimension.
    expansion : float
        Expansion factor for the intermediate dimension.
    """

    def __init__(self, dim_h, expansion):
        super(SwiGLU, self).__init__()
        self.dim_h = dim_h
        self.inter_dim = int(dim_h * expansion)
        self.w1w3 = nn.Linear(dim_h, self.inter_dim * 2)
        self.w2 = nn.Linear(self.inter_dim, dim_h)

    def forward(self, x):
        x1, x3 = self.w1w3(x).view(*x.shape[:-1], 2, self.inter_dim).unbind(-2)
        return self.w2(F.gelu(x1) * x3)


class GCL(nn.Module):
    """Graph Convolutional Layer used in the EGNN encoder.

    Implements one message-passing step: edge features are updated from
    concatenated source/target node features (plus optional edge attributes),
    and node features are updated by aggregating the resulting messages.
    Optionally applies attention gating to edge messages.

    Parameters
    ----------
    input_nf : int
        Input node feature dimension.
    output_nf : int
        Output node feature dimension.
    hidden_nf : int
        Hidden dimension for edge and node MLPs.
    normalization_factor : float
        Normalization divisor applied when ``aggregation_method='sum'``.
    aggregation_method : str
        How to aggregate messages: ``'sum'`` or ``'mean'``.
    edges_in_d : int, optional
        Dimensionality of additional edge input features. Default 0.
    nodes_att_dim : int, optional
        Dimensionality of additional node attribute features. Default 0.
    act_fn : torch.nn.Module, optional
        Activation function. Default ``nn.SiLU()``.
    attention : bool, optional
        If True, learn a scalar attention weight per edge. Default False.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        attention=False,
    ):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(
            edge_attr,
            row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(
        self,
        h,
        edge_index,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    """Equivariant coordinate update layer for EGNN.

    Computes per-edge translation vectors from node and edge features, then
    aggregates them to produce an E(3)-equivariant update to atom coordinates.

    Parameters
    ----------
    hidden_nf : int
        Hidden node feature dimension.
    normalization_factor : float
        Normalization divisor for message aggregation.
    aggregation_method : str
        Aggregation method: ``'sum'`` or ``'mean'``.
    edges_in_d : int, optional
        Dimensionality of edge features passed to the coordinate MLP.
        Default 1.
    act_fn : torch.nn.Module, optional
        Activation function. Default ``nn.SiLU()``.
    tanh : bool, optional
        If True, scale coordinate updates through ``tanh`` bounded by
        ``coords_range``. Default False.
    coords_range : float, optional
        Bound on coordinate update magnitude when ``tanh=True``. Default 10.0.
    """

    def __init__(
        self,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=1,
        act_fn=nn.SiLU(),
        tanh=False,
        coords_range=10.0,
    ):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = (
                coord_diff
                * torch.tanh(self.coord_mlp(input_tensor))
                * self.coords_range
            )
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        edge_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    """One EGNN block: ``n_layers`` GCL message-passing steps followed by a single equivariant coordinate update.

    Computes pairwise distances and normalized displacement vectors from
    current atom coordinates at the start of each forward pass, optionally
    embeds distances via a ``GaussianSmearing`` module, then alternates node
    feature updates (GCL layers) with a final equivariant coordinate update.

    Parameters
    ----------
    hidden_nf : int
        Node feature dimension (constant across all layers in this block).
    edge_feat_nf : int, optional
        Dimensionality of incoming edge features. Default 2.
    act_fn : torch.nn.Module, optional
        Activation function. Default ``nn.SiLU()``.
    n_layers : int, optional
        Number of GCL message-passing layers per block. Default 2.
    attention : bool, optional
        If True, enable attention gating in GCL layers. Default True.
    norm_diff : bool, optional
        If True, normalize displacement vectors before passing to GCL.
        Default True.
    tanh : bool, optional
        Bound coordinate updates through tanh in the equivariant update.
        Default False.
    coords_range : float, optional
        Maximum coordinate update magnitude (used when ``tanh=True``).
        Default 15.
    norm_constant : float, optional
        Additive constant in the coordinate normalization denominator to
        prevent division by zero. Default 1.
    edge_embedding : torch.nn.Module or None, optional
        Module (e.g. ``GaussianSmearing``) applied to scalar distances before
        they are concatenated with edge features. Default None.
    normalization_factor : float, optional
        Divisor for message aggregation in GCL and equivariant update layers.
        Default 100.
    aggregation_method : str, optional
        Aggregation method for GCL and equivariant update: ``'sum'`` or
        ``'mean'``. Default ``'sum'``.
    """

    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        act_fn=nn.SiLU(),
        n_layers=2,
        attention=True,
        norm_diff=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        edge_embedding=None,
        normalization_factor=100,
        aggregation_method="sum",
    ):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.edge_embedding = edge_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=edge_feat_nf,
                    act_fn=act_fn,
                    attention=attention,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                ),
            )
        self.add_module(
            "gcl_equiv",
            EquivariantUpdate(
                hidden_nf,
                edges_in_d=edge_feat_nf,
                act_fn=nn.SiLU(),
                tanh=tanh,
                coords_range=self.coords_range_layer,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
            ),
        )

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.edge_embedding is not None:
            distances = self.edge_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](
                h,
                edge_index,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        x = self._modules["gcl_equiv"](
            h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GaussianSmearing(torch.nn.Module):
    """Expand scalar distances into a fixed basis of Gaussian functions.

    Maps each scalar distance ``d`` to a vector of ``num_gaussians`` values
    ``exp(coeff * (d - mu_i)^2)`` where the centers ``mu_i`` are linearly
    spaced between ``start`` and ``stop`` and ``coeff`` is derived from the
    spacing and ``basis_width_scalar``.

    Parameters
    ----------
    start : float, optional
        Center of the first Gaussian basis function. Default -5.0.
    stop : float, optional
        Center of the last Gaussian basis function. Default 5.0.
    num_gaussians : int, optional
        Number of basis functions (output dimensionality). Default 50.
    basis_width_scalar : float, optional
        Scales the width of each Gaussian; larger values give broader
        functions. Default 1.0.
    """

    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super(GaussianSmearing, self).__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def coord2diff(x, edge_index, norm_constant=1):
    """Compute squared distances and normalized displacement vectors for each edge.

    Parameters
    ----------
    x : torch.Tensor
        Atom coordinates, shape ``(N, 3)``.
    edge_index : torch.Tensor
        Edge indices, shape ``(2, E)``.
    norm_constant : float, optional
        Additive constant in the normalization denominator. Default 1.

    Returns
    -------
    radial : torch.Tensor
        Squared Euclidean distance for each edge, shape ``(E, 1)``.
    coord_diff : torch.Tensor
        Normalized displacement vector for each edge, shape ``(E, 3)``.
    """
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method: str
):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


class AttentionWithNodeMask(nn.Module):
    """Multi-head self-attention that respects a node padding mask.

    Standard scaled dot-product attention, extended to handle variable-length
    graphs padded to a fixed ``max_nodes`` size. The ``node_mask`` is expanded
    into a 2-D attention mask so padded nodes neither attend to nor receive
    attention from real nodes.

    Parameters
    ----------
    dim : int
        Total embedding dimension (must be divisible by ``num_head``).
    num_head : int, optional
        Number of attention heads. Default 8.
    qkv_bias : bool, optional
        Add bias to the QKV projection. Default False.
    qk_norm : bool, optional
        Apply LayerNorm to queries and keys before computing attention scores.
        Default False.
    attn_drop : float, optional
        Dropout probability applied to attention weights. Default 0.0.
    proj_drop : float, optional
        Dropout probability applied to the output projection. Default 0.0.
    norm_layer : torch.nn.Module, optional
        Normalization module used for QK norms. Default ``nn.LayerNorm``.
    """

    def __init__(
        self,
        dim,
        num_head=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_head == 0, "dim should be divisible by num_head"
        self.num_head = num_head
        self.head_dim = dim // num_head

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, node_mask):
        B, N, D = x.shape

        # B, head, N, head_dim
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_head, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # B, head, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        if node_mask is not None:
            # expect node_mask shape to be B, N
            attn_mask = (
                node_mask[:, None, :, None] & node_mask[:, None, None, :]
            ).expand(-1, self.num_head, N, N)
            attn_mask = attn_mask.clone()
            attn_mask[attn_mask.sum(-1) == 0] = True

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p,
            attn_mask=attn_mask,
        )

        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
