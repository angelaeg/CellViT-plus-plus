#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference backend for graph-based CellViT cell classification.

Supported architectures
-----------------------
- GraphSAGE
- GATv2

Supported inference modes
-------------------------
1. Single-checkpoint inference
2. Multi-checkpoint soft-voting ensemble

For ensemble inference, all checkpoints must describe the same graph
configuration and model architecture. The spatial graph is constructed
only once and reused by every model.

Ensemble probabilities are computed as:

    p_ensemble = mean(p_fold)

where p_fold is the softmax probability vector produced by each fold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from cell_graph.models.graphsage import GraphSAGEClassifier
from cell_graph.models.gatv2 import GATv2Classifier
from cell_graph.utils.graph_utils import build_thresholded_knn


TensorOrTensorList = Union[
    torch.Tensor,
    Sequence[torch.Tensor],
]


class GraphCellClassifier:
    """
    Common inference wrapper for GraphSAGE and GATv2 classifiers.

    The classifier can operate either with one checkpoint or with an
    ensemble of multiple compatible checkpoints.
    """

    SUPPORTED_ARCHITECTURES = {
        "graphsage",
        "gatv2",
    }

    def __init__(
        self,
        checkpoint_path: Optional[
            Union[str, Path]
        ] = None,
        checkpoint_paths: Optional[
            Sequence[Union[str, Path]]
        ] = None,
        device: Union[
            str,
            torch.device,
        ] = "cpu",
    ):
        # ------------------------------------------------------
        # Validate inference mode
        # ------------------------------------------------------

        if (
            checkpoint_path is None
            and checkpoint_paths is None
        ):
            raise ValueError(
                "Provide either 'checkpoint_path' "
                "or 'checkpoint_paths'."
            )

        if (
            checkpoint_path is not None
            and checkpoint_paths is not None
        ):
            raise ValueError(
                "Provide only one of 'checkpoint_path' "
                "or 'checkpoint_paths', not both."
            )

        if checkpoint_path is not None:
            paths = [
                Path(checkpoint_path)
            ]
        else:
            paths = [
                Path(p)
                for p in checkpoint_paths
            ]

        if len(paths) == 0:
            raise ValueError(
                "'checkpoint_paths' cannot be empty."
            )

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"GNN checkpoint not found:\n"
                    f"{path}"
                )

        self.checkpoint_paths = paths
        self.checkpoint_path = (
            paths[0]
            if len(paths) == 1
            else None
        )

        self.ensemble = (
            len(paths) > 1
        )

        self.device = torch.device(
            device
        )

        # ------------------------------------------------------
        # Load checkpoints
        # ------------------------------------------------------

        self.checkpoints = [
            torch.load(
                path,
                map_location="cpu",
            )
            for path in self.checkpoint_paths
        ]

        for path, checkpoint in zip(
            self.checkpoint_paths,
            self.checkpoints,
        ):
            if "model_state_dict" not in checkpoint:
                raise KeyError(
                    f"Checkpoint {path} does not contain "
                    "'model_state_dict'."
                )

            if "config" not in checkpoint:
                raise KeyError(
                    f"Checkpoint {path} does not contain "
                    "'config'."
                )

        # First checkpoint defines the reference configuration.
        self.checkpoint = self.checkpoints[0]
        self.config = self.checkpoint["config"]

        if "graph" not in self.config:
            raise KeyError(
                "Checkpoint config does not contain "
                "'graph'."
            )

        if "model" not in self.config:
            raise KeyError(
                "Checkpoint config does not contain "
                "'model'."
            )

        self.graph_config = self.config[
            "graph"
        ]

        self.model_config = self.config[
            "model"
        ]

        # ------------------------------------------------------
        # Parse reference configuration
        # ------------------------------------------------------

        self._parse_configuration()

        # ------------------------------------------------------
        # Validate ensemble compatibility
        # ------------------------------------------------------

        if self.ensemble:
            self._validate_ensemble()

        # ------------------------------------------------------
        # Build all models
        # ------------------------------------------------------

        self.models = []

        for checkpoint in self.checkpoints:

            model_config = checkpoint[
                "config"
            ][
                "model"
            ]

            model = self._build_model(
                model_config
            )

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            model.to(
                self.device
            )

            model.eval()

            self.models.append(
                model
            )

        # Backward compatibility for code expecting self.model.
        self.model = self.models[0]

    # ==========================================================
    # CONFIGURATION
    # ==========================================================

    def _parse_configuration(
        self,
    ) -> None:

        required_graph_keys = {
            "k",
            "radius_um",
            "mpp",
        }

        missing_graph_keys = (
            required_graph_keys
            - set(
                self.graph_config.keys()
            )
        )

        if missing_graph_keys:
            raise KeyError(
                "Missing graph configuration keys: "
                f"{sorted(missing_graph_keys)}"
            )

        self.k = int(
            self.graph_config["k"]
        )

        self.radius_um = float(
            self.graph_config[
                "radius_um"
            ]
        )

        self.mpp = float(
            self.graph_config["mpp"]
        )

        required_model_keys = {
            "architecture",
            "input_dim",
            "hidden_dim",
            "num_classes",
            "dropout",
        }

        missing_model_keys = (
            required_model_keys
            - set(
                self.model_config.keys()
            )
        )

        if missing_model_keys:
            raise KeyError(
                "Missing model configuration keys: "
                f"{sorted(missing_model_keys)}"
            )

        architecture_raw = str(
            self.model_config[
                "architecture"
            ]
        )

        self.architecture = (
            architecture_raw
            .strip()
            .lower()
        )

        if (
            self.architecture
            not in self.SUPPORTED_ARCHITECTURES
        ):
            raise ValueError(
                f"Unsupported GNN architecture "
                f"'{architecture_raw}'. "
                f"Supported architectures: "
                f"{sorted(self.SUPPORTED_ARCHITECTURES)}"
            )

        self.input_dim = int(
            self.model_config[
                "input_dim"
            ]
        )

        self.hidden_dim = int(
            self.model_config[
                "hidden_dim"
            ]
        )

        self.num_classes = int(
            self.model_config[
                "num_classes"
            ]
        )

        self.dropout = float(
            self.model_config[
                "dropout"
            ]
        )

    def _validate_ensemble(
        self,
    ) -> None:
        """
        Ensure that all checkpoints are compatible for soft voting.

        All models must use the same architecture, dimensions and graph
        topology. Otherwise averaging their probabilities would not
        represent the same inference model.
        """

        reference_model = self.model_config
        reference_graph = self.graph_config

        model_keys = [
            "architecture",
            "input_dim",
            "hidden_dim",
            "num_classes",
            "dropout",
        ]

        if self.architecture == "gatv2":
            model_keys.extend(
                [
                    "heads",
                    "concat",
                ]
            )

        graph_keys = [
            "k",
            "radius_um",
            "mpp",
            "position_system",
        ]

        for index, checkpoint in enumerate(
            self.checkpoints[1:],
            start=1,
        ):
            config = checkpoint[
                "config"
            ]

            if "model" not in config:
                raise KeyError(
                    f"Checkpoint {self.checkpoint_paths[index]} "
                    "does not contain config['model']."
                )

            if "graph" not in config:
                raise KeyError(
                    f"Checkpoint {self.checkpoint_paths[index]} "
                    "does not contain config['graph']."
                )

            current_model = config[
                "model"
            ]

            current_graph = config[
                "graph"
            ]

            for key in model_keys:

                if (
                    reference_model.get(key)
                    != current_model.get(key)
                ):
                    raise ValueError(
                        "Incompatible ensemble checkpoints. "
                        f"Model configuration '{key}' differs "
                        f"between checkpoint 0 and checkpoint "
                        f"{index}: "
                        f"{reference_model.get(key)} vs "
                        f"{current_model.get(key)}."
                    )

            for key in graph_keys:

                if (
                    reference_graph.get(key)
                    != current_graph.get(key)
                ):
                    raise ValueError(
                        "Incompatible ensemble checkpoints. "
                        f"Graph configuration '{key}' differs "
                        f"between checkpoint 0 and checkpoint "
                        f"{index}: "
                        f"{reference_graph.get(key)} vs "
                        f"{current_graph.get(key)}."
                    )

            reference_context = (
                self.config.get(
                    "context_size_cellvit"
                )
            )

            current_context = (
                config.get(
                    "context_size_cellvit"
                )
            )

            if (
                reference_context
                != current_context
            ):
                raise ValueError(
                    "Incompatible ensemble checkpoints. "
                    "'context_size_cellvit' differs between "
                    f"checkpoint 0 and checkpoint {index}: "
                    f"{reference_context} vs "
                    f"{current_context}."
                )

    # ==========================================================
    # MODEL CREATION
    # ==========================================================

    def _build_model(
        self,
        model_config: dict,
    ) -> torch.nn.Module:
        """
        Reconstruct one GNN from its stored model configuration.
        """

        architecture = str(
            model_config[
                "architecture"
            ]
        ).strip().lower()

        if architecture == "graphsage":

            return GraphSAGEClassifier(
                input_dim=int(
                    model_config[
                        "input_dim"
                    ]
                ),
                hidden_dim=int(
                    model_config[
                        "hidden_dim"
                    ]
                ),
                num_classes=int(
                    model_config[
                        "num_classes"
                    ]
                ),
                dropout=float(
                    model_config[
                        "dropout"
                    ]
                ),
            )

        if architecture == "gatv2":

            if "heads" not in model_config:
                raise KeyError(
                    "GATv2 checkpoint config does not "
                    "contain 'heads'."
                )

            return GATv2Classifier(
                input_dim=int(
                    model_config[
                        "input_dim"
                    ]
                ),
                hidden_dim=int(
                    model_config[
                        "hidden_dim"
                    ]
                ),
                num_classes=int(
                    model_config[
                        "num_classes"
                    ]
                ),
                dropout=float(
                    model_config[
                        "dropout"
                    ]
                ),
                heads=int(
                    model_config[
                        "heads"
                    ]
                ),
            )

        raise ValueError(
            f"Unsupported GNN architecture "
            f"'{architecture}'."
        )

    # ==========================================================
    # INPUT CONVERSION
    # ==========================================================

    def _prepare_tokens(
        self,
        cell_tokens: TensorOrTensorList,
    ) -> torch.Tensor:

        if isinstance(
            cell_tokens,
            torch.Tensor,
        ):
            x = cell_tokens
        else:
            if len(cell_tokens) == 0:
                raise ValueError(
                    "No cell tokens provided."
                )

            x = torch.stack(
                list(cell_tokens)
            )

        x = x.float()

        if x.ndim != 2:
            raise ValueError(
                f"Expected cell tokens with shape "
                f"[N, D], received {tuple(x.shape)}."
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Checkpoint expects embeddings with "
                f"dimension {self.input_dim}, "
                f"received {x.shape[1]}."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "Non-finite values found in cell embeddings."
            )

        return x

    def _prepare_positions(
        self,
        cell_positions: TensorOrTensorList,
    ) -> torch.Tensor:

        if isinstance(
            cell_positions,
            torch.Tensor,
        ):
            pos = cell_positions
        else:
            if len(cell_positions) == 0:
                raise ValueError(
                    "No cell positions provided."
                )

            pos = torch.stack(
                list(cell_positions)
            )

        pos = pos.float()

        if (
            pos.ndim != 2
            or pos.shape[1] != 2
        ):
            raise ValueError(
                f"Expected cell positions with shape "
                f"[N, 2], received {tuple(pos.shape)}."
            )

        if not torch.isfinite(pos).all():
            raise ValueError(
                "Non-finite values found in cell positions."
            )

        return pos

    # ==========================================================
    # GRAPH CREATION
    # ==========================================================

    def build_graph(
        self,
        cell_positions: TensorOrTensorList,
    ):
        """
        Construct the graph once using the common ensemble configuration.
        """

        pos = self._prepare_positions(
            cell_positions
        )

        edge_index, edge_attr = (
            build_thresholded_knn(
                pos=pos,
                k=self.k,
                radius_um=self.radius_um,
                mpp=self.mpp,
            )
        )

        return (
            pos,
            edge_index,
            edge_attr,
        )

    # ==========================================================
    # INFERENCE
    # ==========================================================

    @torch.inference_mode()
    def predict(
        self,
        cell_tokens: TensorOrTensorList,
        cell_positions: TensorOrTensorList,
    ) -> Dict[str, torch.Tensor]:
        """
        Run single-model or ensemble graph-based classification.

        In ensemble mode, the graph is constructed once and reused by all
        models. Softmax probabilities are averaged across checkpoints.
        """

        x = self._prepare_tokens(
            cell_tokens
        )

        (
            pos,
            edge_index,
            edge_attr,
        ) = self.build_graph(
            cell_positions
        )

        if x.shape[0] != pos.shape[0]:
            raise ValueError(
                "Number of cell embeddings and positions "
                "does not match: "
                f"{x.shape[0]} vs {pos.shape[0]}."
            )

        x = x.to(
            self.device
        )

        edge_index = edge_index.to(
            self.device
        )

        edge_attr = edge_attr.to(
            self.device
        )

        # ------------------------------------------------------
        # Run all folds over exactly the same graph
        # ------------------------------------------------------

        fold_logits = []
        fold_probabilities = []

        for model in self.models:

            logits = model(
                x,
                edge_index,
            )

            probabilities = F.softmax(
                logits.float(),
                dim=1,
            )

            fold_logits.append(
                logits
            )

            fold_probabilities.append(
                probabilities
            )

        stacked_probabilities = torch.stack(
            fold_probabilities,
            dim=0,
        )

        probabilities = (
            stacked_probabilities.mean(
                dim=0
            )
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        prediction_probabilities = (
            probabilities[
                torch.arange(
                    predictions.shape[0],
                    device=probabilities.device,
                ),
                predictions,
            ]
        )

        # For a single checkpoint this is exactly the original
        # model output. For an ensemble, mean logits are provided
        # only as a diagnostic field; predictions are ALWAYS
        # derived from mean softmax probabilities.
        stacked_logits = torch.stack(
            fold_logits,
            dim=0,
        )

        logits = stacked_logits.mean(
            dim=0
        )

        return {
            "logits":
                logits.detach(),

            "probabilities":
                probabilities.detach(),

            "predictions":
                predictions.detach(),

            "prediction_probabilities":
                prediction_probabilities.detach(),

            "edge_index":
                edge_index.detach(),

            "edge_attr":
                edge_attr.detach(),

            "fold_probabilities":
                stacked_probabilities.detach(),
        }

    # ==========================================================
    # CONVENIENCE
    # ==========================================================

    @property
    def is_graph_classifier(
        self,
    ) -> bool:
        return True

    @property
    def is_ensemble(
        self,
    ) -> bool:
        return self.ensemble

    def summary(
        self,
    ) -> dict:

        summary = {
            "architecture":
                self.architecture,

            "inference_mode":
                (
                    "ensemble"
                    if self.ensemble
                    else "single"
                ),

            "num_models":
                len(self.models),

            "checkpoints":
                [
                    str(p)
                    for p in self.checkpoint_paths
                ],

            "device":
                str(self.device),

            "input_dim":
                self.input_dim,

            "hidden_dim":
                self.hidden_dim,

            "num_classes":
                self.num_classes,

            "dropout":
                self.dropout,

            "k":
                self.k,

            "radius_um":
                self.radius_um,

            "mpp":
                self.mpp,
        }

        if self.architecture == "gatv2":
            summary["heads"] = int(
                self.model_config[
                    "heads"
                ]
            )

        if "context_size_cellvit" in self.config:
            summary[
                "training_context_size"
            ] = self.config[
                "context_size_cellvit"
            ]

        return summary