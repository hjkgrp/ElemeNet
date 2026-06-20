# shallow ensemble from: https://github.com/bananenpampe/DPOSE/blob/main/UCI_experiments/model/mlp.py
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional
from torch_geometric.utils import to_dense_batch
from elemenet.mlp import ACT_MAP, SwiGLU, AttentionWithNodeMask


@dataclass
class MLPConfig:
    """Configuration for a multi-layer perceptron (MLP) readout head.

    Attributes
    ----------
    input_size : int
        Dimensionality of the encoder's output embeddings passed to the MLP.
    hidden_sizes : List[int]
        Number of neurons in each hidden layer.
    activations : List[str]
        Activation function name for each hidden layer (e.g. ``'relu'``,
        ``'silu'``).
    dropouts : List[float]
        Dropout rate for each hidden layer.
    output_size : int
        Number of output neurons (number of targets).
    use_norm : bool
        Apply LayerNorm after each hidden layer. Default False.
    ensemble_size : int
        Number of shallow ensemble heads sharing the same trunk. Default 1.
    graph_attr_dim : int
        Dimensionality of graph-level attributes (e.g. charge, spin
        multiplicity) concatenated after pooling. Default 0.
    graph_attr_hidden_dim : Optional[int]
        If set, graph-level attributes are first processed by a small MLP of
        this hidden size before concatenation. Default None.
    scope : str
        Prediction granularity: ``'graph'``, ``'node'``, or ``'edge'``. When
        ``'edge'``, the trunk runs over per-node features (no pooling, no
        graph_attr injection) and the output layer gathers endpoint pairs
        from ``edge_index`` and optionally concatenates ``edge_attr`` before
        projection. Default ``'graph'``.
    edge_dim : int
        Dimensionality of pre-computed edge attributes concatenated with the
        gathered endpoint pairs when ``scope='edge'``. Ignored otherwise.
        Default 0.
    """

    input_size: int
    hidden_sizes: List[int]
    activations: List[str]
    dropouts: List[float]
    output_size: int
    use_norm: bool = False
    ensemble_size: int = 1
    graph_attr_dim: int = 0
    graph_attr_hidden_dim: Optional[int] = None
    scope: str = "graph"
    edge_dim: int = 0


@dataclass
class EdgePredictorConfig:
    """Configuration for an edge-level prediction readout head.

    Attributes
    ----------
    input_size : int
        Dimensionality of node embeddings used to construct edge features.
    hidden_sizes : List[int]
        Number of neurons in each hidden layer.
    activations : List[str]
        Activation function name for each hidden layer.
    dropouts : List[float]
        Dropout rate for each hidden layer.
    output_size : int
        Number of output neurons per edge.
    edge_dim : Optional[int]
        Dimensionality of pre-computed edge attributes concatenated with
        the node-pair features. Default 0.
    use_norm : bool
        Apply LayerNorm after each hidden layer. Default False.
    ensemble_size : int
        Number of shallow ensemble heads. Default 1.
    """

    input_size: int
    hidden_sizes: List[int]
    activations: List[str]
    dropouts: List[float]
    output_size: int
    edge_dim: Optional[int] = 0
    use_norm: bool = False
    ensemble_size: int = 1


@dataclass
class TransformerConfig:
    """Configuration for a Transformer-based readout head.

    Attributes
    ----------
    input_size : int
        Dimensionality of node embeddings entering the Transformer.
    hidden_sizes : List[int]
        Hidden dimension for each Transformer layer (all values must be equal).
    activations : List[str]
        Activation function name for each Transformer layer (used in SwiGLU
        feed-forward blocks).
    dropouts : List[float]
        Dropout rate for each Transformer layer.
    output_size : int
        Number of output neurons (number of targets).
    max_nodes : int
        Maximum number of nodes per graph; used for dense padding.
    num_heads : int
        Number of attention heads in each layer.
    expansion : float
        SwiGLU feed-forward intermediate dimension multiplier. Default 4.
    ensemble_size : int
        Number of shallow ensemble heads. Default 1.
    graph_attr_dim : int
        Dimensionality of graph-level attributes concatenated before the
        output layer. Default 0.
    graph_attr_hidden_dim : Optional[int]
        If set, graph-level attributes are processed by a small MLP of this
        hidden size before concatenation. Default None.
    scope : str
        Prediction granularity: ``'graph'``, ``'node'``, or ``'edge'``. When
        ``'edge'``, the attention trunk runs over per-node features and the
        output layer gathers endpoint pairs from ``edge_index`` and optionally
        concatenates ``edge_attr`` before projection. Default ``'graph'``.
    edge_dim : int
        Dimensionality of pre-computed edge attributes concatenated with the
        gathered endpoint pairs when ``scope='edge'``. Ignored otherwise.
        Default 0.
    """

    input_size: int
    hidden_sizes: List[int]
    activations: List[str]
    dropouts: List[float]
    output_size: int
    max_nodes: int
    num_heads: int
    expansion: float = 4
    ensemble_size: int = 1
    graph_attr_dim: int = 0
    graph_attr_hidden_dim: Optional[int] = None
    scope: str = "graph"
    edge_dim: int = 0


class MLP(nn.Module):
    """Multilayer perceptron readout head with optional pooling and graph-level attribute injection.

    Supports both graph-level (with pooling) and node/edge-level (without pooling)
    prediction. When ``graph_attr_dim > 0``, graph-level attributes (e.g. molecular
    charge, spin multiplicity) are concatenated with pooled node embeddings before the
    first hidden layer, either directly or via a small per-attribute sub-MLP.

    The output layer is a shallow ensemble: a single linear layer of size
    ``output_size * ensemble_size``, reshaped to ``(batch, output_size, ensemble_size)``.
    Use ``ensemble_size=1`` for a standard single-head MLP.

    Parameters
    ----------
    input_size : int
        Dimensionality of input embeddings (after pooling, if applicable).
    hidden_sizes : list of int
        Number of neurons in each hidden layer.
    output_size : int
        Number of output neurons (number of prediction targets).
    activations : list of str
        Activation function name for each hidden layer.
    dropouts : list of float
        Dropout rate for each hidden layer (applied only to non-final layers).
    use_norm : bool, optional
        Apply LayerNorm after each hidden layer. Default False.
    ensemble_size : int, optional
        Number of shallow ensemble output heads. Default 1.
    graph_attr_dim : int, optional
        Dimensionality of graph-level attributes to concatenate. Default 0.
    graph_attr_hidden_dim : int or None, optional
        If set, graph-level attributes are processed by a sub-MLP of this
        hidden dimension before concatenation. Default None.
    scope : str, optional
        Prediction granularity: ``'graph'``, ``'node'``, or ``'edge'``. When
        ``'edge'``, the trunk runs over per-node features (no pooling, no
        graph_attr injection) and the output layer gathers endpoint pairs
        from ``edge_index`` before projection. Default ``'graph'``.
    edge_dim : int, optional
        Dimensionality of pre-computed edge attributes concatenated with the
        gathered endpoint pairs when ``scope='edge'``. Default 0.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activations: List[str],
        dropouts: List[float],
        use_norm: bool = False,
        ensemble_size: int = 1,
        graph_attr_dim: int = 0,
        graph_attr_hidden_dim: Optional[int] = None,
        scope: str = "graph",
        edge_dim: int = 0,
    ):
        super().__init__()
        num_layers = len(hidden_sizes)
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = ensemble_size
        self.graph_attr_dim = graph_attr_dim
        self.graph_attr_hidden_dim = graph_attr_hidden_dim
        self.scope = scope
        self.edge_dim = edge_dim
        assert all(
            len(lst) == num_layers for lst in [activations, dropouts]
        ), "All per-layer configurations must have the same length!"

        self.graph_attr_mlp = None
        # graph_attr is not injected for scope='edge' (semantics for per-edge expansion
        # are ambiguous; can be added later if a use case appears)
        if graph_attr_dim > 0 and scope != "edge":
            if graph_attr_hidden_dim is None:
                input_size += graph_attr_dim
            else:
                # if we want to process graph-level attributes with a separate MLP before concatenation
                self.graph_attr_mlp = MLP(
                    input_size=graph_attr_dim,
                    hidden_sizes=[graph_attr_hidden_dim],
                    output_size=graph_attr_hidden_dim
                    * graph_attr_dim,  # each graph attribute gets its own hidden representation
                    activations=[activations[0]],
                    dropouts=[dropouts[0]],
                    use_norm=use_norm,
                    ensemble_size=1,
                )
                # update input size to account for processed graph attributes
                input_size += graph_attr_hidden_dim * graph_attr_dim

        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = input_size
        for idx in range(num_layers):
            self.linears.append(nn.Linear(in_dim, hidden_sizes[idx]))
            self.activations.append(ACT_MAP[activations[idx]])
            # dropout only on non-final layers
            if idx < num_layers - 1 and dropouts[idx] > 0:
                self.dropouts.append(nn.Dropout(dropouts[idx]))
            else:
                self.dropouts.append(nn.Identity())
            self.norms.append(
                nn.LayerNorm(hidden_sizes[idx]) if use_norm else nn.Identity()
            )
            in_dim = hidden_sizes[idx]

        # add the final output layer (using shallow ensemble)
        # for scope='edge', the output layer projects gathered endpoint pairs (and
        # optional edge_attr); for graph/node, it projects per-node trunk features
        if scope == "edge":
            self.output_layer = torch.nn.Linear(
                2 * in_dim + edge_dim, output_size * ensemble_size
            )
        else:
            self.output_layer = torch.nn.Linear(in_dim, output_size * ensemble_size)

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        pool_fn=None,
        graph_attr: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # pool before passing to MLP if needed (graph scope only)
        if pool_fn is not None:
            assert batch is not None, "batch must be provided for pooling."
            x = pool_fn(x, batch)

        # concatenate graph-level attributes (charge, spinmult, etc.) after pooling
        # skip for scope='edge' — pairs are gathered after the trunk instead
        if graph_attr is not None and self.scope != "edge":
            if self.graph_attr_mlp is None:
                graph_attr_embeddings = graph_attr
            else:
                graph_attr_embeddings = self.graph_attr_mlp(graph_attr)[0]
                if len(graph_attr_embeddings.shape) == 3:
                    graph_attr_embeddings = graph_attr_embeddings.squeeze(-1)
            # for node-level tasks (no pooling), expand graph attrs to per-node
            if (
                pool_fn is None
                and batch is not None
                and graph_attr_embeddings.size(0) != x.size(0)
            ):
                graph_attr_embeddings = graph_attr_embeddings[batch]
            x = torch.cat([x, graph_attr_embeddings], dim=-1)

        for idx, linear in enumerate(self.linears):
            x = linear(x)
            x = self.norms[idx](x)
            x = self.activations[idx](x)
            x = self.dropouts[idx](x)

        # for edge scope, gather endpoint pairs after the trunk and optionally
        # concatenate pre-computed edge_attr before the final projection
        if self.scope == "edge":
            assert edge_index is not None, "MLP with scope='edge' requires edge_index"
            i, j = edge_index
            x = torch.cat([x[i], x[j]], dim=-1)
            if edge_attr is not None and self.edge_dim > 0:
                x = torch.cat([x, edge_attr], dim=-1)
        embeddings = x

        # output has dimensions: (batch_size, output_size, ensemble_size)
        output = self.output_layer(x).reshape(-1, self.output_size, self.ensemble_size)

        return output, embeddings


class EdgePredictor(nn.Module):
    """MLP readout head for edge-level prediction.

    Constructs per-edge features by concatenating the embeddings of the two
    endpoint nodes (and optionally pre-computed edge attributes), then passes
    them through a shared MLP. Shares the same shallow-ensemble output design
    as ``MLP``.

    Parameters
    ----------
    input_size : int
        Dimensionality of node embeddings; edge input size will be
        ``2 * input_size + edge_dim``.
    hidden_sizes : list of int
        Number of neurons in each hidden layer.
    output_size : int
        Number of output neurons per edge.
    activations : list of str
        Activation function name for each hidden layer.
    dropouts : list of float
        Dropout rate for each hidden layer (applied only to non-final layers).
    use_norm : bool, optional
        Apply LayerNorm after each hidden layer. Default False.
    edge_dim : int or None, optional
        Dimensionality of pre-computed edge attributes to concatenate.
        Default 0.
    ensemble_size : int, optional
        Number of shallow ensemble output heads. Default 1.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activations: List[str],
        dropouts: List[float],
        use_norm: bool = False,
        edge_dim: Optional[int] = 0,
        ensemble_size: int = 1,
    ):
        super().__init__()
        num_layers = len(hidden_sizes)
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = ensemble_size
        assert all(
            len(lst) == num_layers for lst in [activations, dropouts]
        ), "All per-layer configurations must have the same length!"

        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.norms = nn.ModuleList()

        # to construct edge features we concatenate the embeddings of the two nodes and edge attributes
        in_dim = 2 * input_size + edge_dim
        for idx in range(num_layers):
            self.linears.append(nn.Linear(in_dim, hidden_sizes[idx]))
            self.activations.append(ACT_MAP[activations[idx]])
            # dropout only on non-final layers
            if idx < num_layers - 1 and dropouts[idx] > 0:
                self.dropouts.append(nn.Dropout(dropouts[idx]))
            else:
                self.dropouts.append(nn.Identity())
            self.norms.append(
                nn.LayerNorm(hidden_sizes[idx]) if use_norm else nn.Identity()
            )
            in_dim = hidden_sizes[idx]

        # add the final output layer (using shallow ensemble)
        self.output_layer = torch.nn.Linear(in_dim, output_size * ensemble_size)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # indices for edges
        i, j = edge_index

        # construct edge features (concat, diff, product)
        e_concat = torch.cat([x[i], x[j]], dim=-1)

        if edge_attr is not None:
            e_concat = torch.cat([e_concat, edge_attr], dim=-1)

        for idx, linear in enumerate(self.linears):
            e_concat = linear(e_concat)
            e_concat = self.norms[idx](e_concat)
            e_concat = self.activations[idx](e_concat)
            e_concat = self.dropouts[idx](e_concat)
        embeddings = e_concat

        # output has dimensions: (batch_size, output_size, ensemble_size)
        output = self.output_layer(e_concat).reshape(
            -1, self.output_size, self.ensemble_size
        )
        return output, embeddings


class Transformer(torch.nn.Module):
    """Transformer-based readout head for graph-level prediction.

    Pads node sequences to ``max_nodes`` via ``to_dense_batch``, applies
    ``num_layers`` pre-norm attention blocks (each with ``AttentionWithNodeMask``
    and a SwiGLU feed-forward sub-layer), then unpads and optionally pools
    before the output projection. Supports the same shallow-ensemble and
    graph-level attribute injection as ``MLP``.

    All hidden layer sizes must be identical (required by the residual
    connections between attention and feed-forward blocks).

    Parameters
    ----------
    input_size : int
        Dimensionality of incoming node embeddings.
    hidden_sizes : list of int
        Hidden dimension for each Transformer layer; all values must be equal.
    output_size : int
        Number of output neurons (number of targets).
    activations : list of str
        Activation function name for each layer's SwiGLU block.
    dropouts : list of float
        Dropout rate for each attention layer's output projection.
    num_heads : int
        Number of attention heads per layer.
    max_nodes : int
        Maximum graph size used to create dense padded batches.
    ensemble_size : int, optional
        Number of shallow ensemble output heads. Default 1.
    expansion : float, optional
        SwiGLU intermediate dimension multiplier. Default 4.
    graph_attr_dim : int, optional
        Dimensionality of graph-level attributes concatenated before the
        output layer. Default 0.
    graph_attr_hidden_dim : int or None, optional
        If set, graph-level attributes are processed by a sub-MLP before
        concatenation. Default None.
    scope : str, optional
        Prediction granularity: ``'graph'``, ``'node'``, or ``'edge'``. When
        ``'edge'``, the attention trunk runs over per-node features and the
        output layer gathers endpoint pairs from ``edge_index`` before
        projection. Default ``'graph'``.
    edge_dim : int, optional
        Dimensionality of pre-computed edge attributes concatenated with the
        gathered endpoint pairs when ``scope='edge'``. Default 0.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activations,
        dropouts,
        num_heads,
        max_nodes,
        ensemble_size=1,
        expansion=4,
        graph_attr_dim=0,
        graph_attr_hidden_dim: Optional[int] = None,
        scope: str = "graph",
        edge_dim: int = 0,
    ):
        assert all(
            [hidden_sizes[0] == h for h in hidden_sizes]
        ), "All hidden sizes must be the same for Transformer readout."
        super().__init__()
        num_layers = len(hidden_sizes)
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = ensemble_size
        self.max_nodes = max_nodes
        self.graph_attr_dim = graph_attr_dim
        self.graph_attr_hidden_dim = graph_attr_hidden_dim
        self.scope = scope
        self.edge_dim = edge_dim
        # proj_to_tf operates on node-level features only (no graph_attr)

        # graph_attr is not injected for scope='edge' (matches MLP behavior;
        # pairs are gathered after the trunk and projected directly)
        self.graph_attr_mlp = None
        add_graph_attr_dim = 0
        if graph_attr_dim > 0 and scope != "edge":
            if graph_attr_hidden_dim is None:
                add_graph_attr_dim = graph_attr_dim
            else:
                # if we want to process graph-level attributes with a separate MLP before concatenation
                self.graph_attr_mlp = MLP(
                    input_size=graph_attr_dim,
                    hidden_sizes=[graph_attr_hidden_dim],
                    output_size=graph_attr_hidden_dim
                    * graph_attr_dim,  # each graph attribute gets its own hidden representation
                    activations=[activations[0]],
                    dropouts=[dropouts[0]],
                    ensemble_size=1,
                )
                # update input size to account for processed graph attributes
                add_graph_attr_dim = graph_attr_hidden_dim * graph_attr_dim

        self.proj_to_tf = torch.nn.Linear(input_size, hidden_sizes[0])
        self.attn_layers = torch.nn.ModuleList(
            [
                AttentionWithNodeMask(
                    hidden_sizes[i],
                    num_heads,
                    qkv_bias=True,
                    qk_norm=True,
                    proj_drop=dropouts[i],
                )
                for i in range(num_layers)
            ]
        )

        self.proj_norm = torch.nn.LayerNorm(input_size)
        self.attn_norm = nn.ModuleList(
            [torch.nn.LayerNorm(hidden_sizes[i]) for i in range(num_layers)]
        )
        self.fc_layers = torch.nn.ModuleList(
            [SwiGLU(hidden_sizes[i], expansion) for i in range(num_layers)]
        )
        self.mlp_norm = nn.ModuleList(
            [torch.nn.LayerNorm(hidden_sizes[i]) for i in range(num_layers)]
        )

        # output layer takes transformer output + graph-level attributes and (using shallow ensemble).
        # for scope='edge', the output layer projects gathered endpoint pairs (and optional edge_attr)
        # produced after the attention trunk; graph_attr is not injected.
        if scope == "edge":
            in_dim = 2 * hidden_sizes[-1] + edge_dim
        else:
            in_dim = hidden_sizes[-1] + add_graph_attr_dim
        self.output_layer = torch.nn.Linear(in_dim, output_size * ensemble_size)

    def tf_forward(self, x, node_mask=None):
        for i, attn_layer in enumerate(self.attn_layers):
            attn = attn_layer(x, node_mask=node_mask)
            h = x + self.attn_norm[i](attn)
            fc = self.fc_layers[i](h)
            x = h + self.mlp_norm[i](fc)
        return x

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        pool_fn=None,
        graph_attr: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        assert batch is not None, "Transformer needs patching, which needs batch info."
        # patch to make transformer work
        x, node_mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)

        # pass through transformer
        x = self.proj_norm(x)
        x = self.proj_to_tf(x)
        x = self.tf_forward(x, node_mask=node_mask)

        # get back to original shape
        x = x[node_mask]

        # pool before passing to MLP if needed (graph scope only)
        if pool_fn is not None:
            assert batch is not None, "batch must be provided for pooling."
            x = pool_fn(x, batch)

        # concatenate graph-level attributes (charge, spinmult, etc.) after pooling.
        # skip for scope='edge' — pairs are gathered after the trunk instead
        if graph_attr is not None and self.scope != "edge":
            if self.graph_attr_mlp is None:
                graph_attr_embeddings = graph_attr
            else:
                graph_attr_embeddings = self.graph_attr_mlp(graph_attr)[0]
                if len(graph_attr_embeddings.shape) == 3:
                    graph_attr_embeddings = graph_attr_embeddings.squeeze(-1)
            # for node-level tasks (no pooling), expand graph attrs to per-node
            if (
                pool_fn is None
                and batch is not None
                and graph_attr_embeddings.size(0) != x.size(0)
            ):
                graph_attr_embeddings = graph_attr_embeddings[batch]
            x = torch.cat([x, graph_attr_embeddings], dim=-1)

        # for edge scope, gather endpoint pairs after the attention trunk and
        # optionally concatenate pre-computed edge_attr before the final projection
        if self.scope == "edge":
            assert edge_index is not None, "Transformer with scope='edge' requires edge_index"
            i, j = edge_index
            x = torch.cat([x[i], x[j]], dim=-1)
            if edge_attr is not None and self.edge_dim > 0:
                x = torch.cat([x, edge_attr], dim=-1)

        embeddings = x

        # output has dimensions: (batch_size, output_size, ensemble_size)
        output = self.output_layer(x).reshape(-1, self.output_size, self.ensemble_size)

        return output, embeddings
