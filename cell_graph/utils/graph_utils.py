#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph construction utilities for cell-level spatial graphs.

This module contains reusable functions to construct spatial graphs from
CellViT cell centroids.

Current baseline topology
-------------------------
- One cell = one node
- Thresholded k-nearest-neighbour graph
- Undirected biological graph
- Both edge directions stored for PyTorch Geometric message passing
- Edge attribute = Euclidean cell-to-cell distance in micrometres
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


def build_thresholded_knn(
    pos: torch.Tensor,
    k: int,
    radius_um: float,
    mpp: float,
):
    """
    Build an undirected thresholded k-nearest-neighbour graph.

    Two cells are connected if:
        1. One cell is among the k nearest neighbours of the other.
        2. Their Euclidean distance is <= radius_um.

    Both directions of every accepted edge are stored in ``edge_index``
    because PyTorch Geometric performs directed message passing.

    Parameters
    ----------
    pos : torch.Tensor
        Cell centroid coordinates with shape [N, 2], in pixels.

    k : int
        Maximum number of nearest neighbours considered per node.

    radius_um : float
        Maximum allowed cell-to-cell distance in micrometres.

    mpp : float
        Micrometres per pixel of the input patch.

    Returns
    -------
    edge_index : torch.Tensor
        Graph connectivity with shape [2, E].

    edge_attr : torch.Tensor
        Edge distances in micrometres with shape [E, 1].
    """

    # ----------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------

    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(
            f"'pos' must have shape [N, 2], "
            f"but received {tuple(pos.shape)}."
        )

    if k < 1:
        raise ValueError(
            f"'k' must be >= 1, but received {k}."
        )

    if radius_um <= 0:
        raise ValueError(
            f"'radius_um' must be > 0, "
            f"but received {radius_um}."
        )

    if mpp <= 0:
        raise ValueError(
            f"'mpp' must be > 0, but received {mpp}."
        )

    coords = (
        pos.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    n_nodes = len(coords)

    # ----------------------------------------------------------
    # Empty / single-node graph
    # ----------------------------------------------------------

    if n_nodes <= 1:
        return (
            torch.empty(
                (2, 0),
                dtype=torch.long,
            ),
            torch.empty(
                (0, 1),
                dtype=torch.float32,
            ),
        )

    # ----------------------------------------------------------
    # Convert biological radius from µm to pixels
    # ----------------------------------------------------------

    radius_px = (
        float(radius_um)
        / float(mpp)
    )

    effective_k = min(
        int(k),
        n_nodes - 1,
    )

    # ----------------------------------------------------------
    # k-NN search
    # ----------------------------------------------------------

    tree = cKDTree(
        coords
    )

    distances_px, indices = tree.query(
        coords,
        k=effective_k + 1,
    )

    # Store undirected edges only once before converting them
    # to the bidirectional PyG representation.
    undirected_edges = {}

    for source in range(
        n_nodes
    ):

        # Rank 0 corresponds to the node itself.
        for rank in range(
            1,
            effective_k + 1,
        ):

            target = int(
                indices[source, rank]
            )

            distance_px = float(
                distances_px[source, rank]
            )

            # Radius threshold.
            if distance_px > radius_px:
                continue

            a, b = sorted(
                (source, target)
            )

            if a == b:
                continue

            # If the same undirected pair is found from both nodes,
            # keep only one copy.
            undirected_edges[
                (a, b)
            ] = distance_px

    # ----------------------------------------------------------
    # Convert to PyG bidirectional representation
    # ----------------------------------------------------------

    edge_list = []
    edge_distances_um = []

    for (a, b), distance_px in sorted(
        undirected_edges.items()
    ):

        distance_um = (
            distance_px
            * float(mpp)
        )

        edge_list.extend(
            [
                (a, b),
                (b, a),
            ]
        )

        edge_distances_um.extend(
            [
                distance_um,
                distance_um,
            ]
        )

    # ----------------------------------------------------------
    # Handle graph with no accepted edges
    # ----------------------------------------------------------

    if len(edge_list) == 0:
        return (
            torch.empty(
                (2, 0),
                dtype=torch.long,
            ),
            torch.empty(
                (0, 1),
                dtype=torch.float32,
            ),
        )

    edge_index = torch.tensor(
        edge_list,
        dtype=torch.long,
    ).t().contiguous()

    edge_attr = torch.tensor(
        edge_distances_um,
        dtype=torch.float32,
    ).view(-1, 1)

    return edge_index, edge_attr