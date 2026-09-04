#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GATv2 models for cell-level classification.

This classifier operates on frozen CellViT cell embeddings and uses
explicit spatial neighborhood information through GATv2 attention-based
message passing.

Architecture
------------
CellViT embedding (1280-D)
-> Linear projection (1280 -> hidden_dim)
-> ReLU
-> GATv2Conv (hidden_dim -> hidden_dim)
-> ReLU
-> Dropout
-> GATv2Conv (hidden_dim -> hidden_dim)
-> ReLU
-> Dropout
-> Linear classifier (hidden_dim -> num_classes)

The model performs node-level classification and returns one logit vector
per cell.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2Classifier(nn.Module):
    """
    Node-level GATv2 classifier for CellViT cell embeddings.

    Parameters
    ----------
    input_dim : int, default=1280
        Dimensionality of the CellViT cell embedding.

    hidden_dim : int, default=128
        Hidden dimensionality used by the projection and GATv2 layers.

    num_classes : int, default=3
        Number of output cell classes.

    dropout : float, default=0.4
        Dropout probability applied after each GATv2 layer.

    heads : int, default=1
        Number of attention heads.

    Notes
    -----
    The graph topology is supplied through edge_index and is assumed to
    have been constructed externally.

    Spatial edge distances are not used in this baseline GATv2 model,
    matching the GraphSAGE setup where edge_attr is also ignored.

    Using heads=1 preserves an architecture with similar dimensionality
    and capacity to the GraphSAGE baseline, allowing a cleaner comparison
    between mean-based aggregation and attention-based aggregation.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.4,
        heads: int = 1,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                f"'input_dim' must be > 0, received {input_dim}."
            )

        if hidden_dim <= 0:
            raise ValueError(
                f"'hidden_dim' must be > 0, received {hidden_dim}."
            )

        if num_classes <= 1:
            raise ValueError(
                f"'num_classes' must be > 1, received {num_classes}."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"'dropout' must satisfy 0 <= dropout < 1, "
                f"received {dropout}."
            )

        if heads <= 0:
            raise ValueError(
                f"'heads' must be > 0, received {heads}."
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)
        self.heads = int(heads)

        # Project the original CellViT embedding to the hidden GNN space.
        self.input_projection = nn.Linear(
            self.input_dim,
            self.hidden_dim,
        )

        # First attention-based message-passing layer.
        self.gat1 = GATv2Conv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.heads,
            concat=False,
            dropout=0.0,
        )

        # Second attention-based message-passing layer.
        self.gat2 = GATv2Conv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.heads,
            concat=False,
            dropout=0.0,
        )

        # Node-level TSN classifier.
        self.classifier = nn.Linear(
            self.hidden_dim,
            self.num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform node-level classification.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix with shape [N, input_dim].

        edge_index : torch.Tensor
            PyTorch Geometric graph connectivity with shape [2, E].

        Returns
        -------
        torch.Tensor
            Node-level logits with shape [N, num_classes].
        """

        if x.ndim != 2:
            raise ValueError(
                f"'x' must have shape [N, D], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected node features with dimension "
                f"{self.input_dim}, received {x.shape[1]}."
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"'edge_index' must have shape [2, E], "
                f"received {tuple(edge_index.shape)}."
            )

        # CellViT embedding projection.
        x = self.input_projection(x)
        x = F.relu(x)

        # First attention-based message-passing step.
        x = self.gat1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Second attention-based message-passing step.
        x = self.gat2(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # One three-class logit vector per cell.
        logits = self.classifier(x)

        return logits