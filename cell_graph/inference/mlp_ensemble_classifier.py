#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference backend for the retrained CellViT TSN MLP classifier.

This module supports:
- Single-checkpoint inference
- Multi-checkpoint soft-voting ensemble inference

The classifier operates directly on frozen CellViT cell embeddings.

Architecture
------------
CellViT embedding (1280-D)
-> LinearClassifier
-> logits
-> softmax
-> TSN probabilities

For ensemble inference:

    p_ensemble = mean(p_fold)

The final class prediction is obtained from the averaged probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from cellvit.models.classifier.linear_classifier import (
    LinearClassifier,
)

from cellvit.utils.tools import (
    unflatten_dict,
)


TensorOrTensorList = Union[
    torch.Tensor,
    Sequence[torch.Tensor],
]


class MLPEnsembleClassifier:
    """
    Inference wrapper for retrained CellViT TSN classifiers.

    Supports either one checkpoint or an ensemble of multiple
    compatible checkpoints.
    """

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
                Path(path)
                for path in checkpoint_paths
            ]

        if len(paths) == 0:
            raise ValueError(
                "'checkpoint_paths' cannot be empty."
            )

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"MLP checkpoint not found:\n"
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

        # ------------------------------------------------------
        # Reference configuration from fold 0 / first model
        # ------------------------------------------------------

        self.checkpoint = self.checkpoints[0]

        self.run_conf = unflatten_dict(
            self.checkpoint["config"],
            ".",
        )

        self._parse_configuration()

        if self.ensemble:
            self._validate_ensemble()

        # ------------------------------------------------------
        # Reconstruct models
        # ------------------------------------------------------

        self.models = []

        for checkpoint in self.checkpoints:

            run_conf = unflatten_dict(
                checkpoint["config"],
                ".",
            )

            model = self._build_model(
                checkpoint=checkpoint,
                run_conf=run_conf,
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

        # Backward compatibility / convenience.
        self.model = self.models[0]

    # ==========================================================
    # CONFIGURATION
    # ==========================================================

    def _parse_configuration(
        self,
    ) -> None:
        """
        Parse architecture and class configuration from the first fold.
        """

        state_dict = self.checkpoint[
            "model_state_dict"
        ]

        required_state_keys = {
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
        }

        missing_keys = (
            required_state_keys
            - set(state_dict.keys())
        )

        if missing_keys:
            raise KeyError(
                "MLP checkpoint is missing required "
                f"state_dict keys: {sorted(missing_keys)}"
            )

        # Infer dimensions directly from trained weights.
        self.input_dim = int(
            state_dict[
                "fc1.weight"
            ].shape[1]
        )

        self.hidden_dim = int(
            state_dict[
                "fc1.weight"
            ].shape[0]
        )

        self.num_classes = int(
            state_dict[
                "fc2.weight"
            ].shape[0]
        )

        # Training configuration.
        self.drop_rate = float(
            self.run_conf[
                "training"
            ].get(
                "drop_rate",
                0.0,
            )
        )

        configured_hidden_dim = int(
            self.run_conf[
                "model"
            ].get(
                "hidden_dim",
                self.hidden_dim,
            )
        )

        configured_num_classes = int(
            self.run_conf[
                "data"
            ][
                "num_classes"
            ]
        )

        if (
            configured_hidden_dim
            != self.hidden_dim
        ):
            raise ValueError(
                "Checkpoint configuration and state_dict "
                "disagree on hidden_dim: "
                f"{configured_hidden_dim} vs "
                f"{self.hidden_dim}."
            )

        if (
            configured_num_classes
            != self.num_classes
        ):
            raise ValueError(
                "Checkpoint configuration and state_dict "
                "disagree on num_classes: "
                f"{configured_num_classes} vs "
                f"{self.num_classes}."
            )

        # Label map.
        label_map = (
            self.run_conf[
                "data"
            ][
                "label_map"
            ]
        )

        self.label_map = {
            int(key): value
            for key, value in label_map.items()
        }

    def _validate_ensemble(
        self,
    ) -> None:
        """
        Ensure all folds describe the same TSN classifier.
        """

        reference_state = self.checkpoint[
            "model_state_dict"
        ]

        reference_run_conf = self.run_conf

        reference_input_dim = int(
            reference_state[
                "fc1.weight"
            ].shape[1]
        )

        reference_hidden_dim = int(
            reference_state[
                "fc1.weight"
            ].shape[0]
        )

        reference_num_classes = int(
            reference_state[
                "fc2.weight"
            ].shape[0]
        )

        reference_label_map = {
            int(key): value
            for key, value in (
                reference_run_conf[
                    "data"
                ][
                    "label_map"
                ].items()
            )
        }

        for index, checkpoint in enumerate(
            self.checkpoints[1:],
            start=1,
        ):
            state_dict = checkpoint[
                "model_state_dict"
            ]

            run_conf = unflatten_dict(
                checkpoint[
                    "config"
                ],
                ".",
            )

            current_input_dim = int(
                state_dict[
                    "fc1.weight"
                ].shape[1]
            )

            current_hidden_dim = int(
                state_dict[
                    "fc1.weight"
                ].shape[0]
            )

            current_num_classes = int(
                state_dict[
                    "fc2.weight"
                ].shape[0]
            )

            current_label_map = {
                int(key): value
                for key, value in (
                    run_conf[
                        "data"
                    ][
                        "label_map"
                    ].items()
                )
            }

            if (
                current_input_dim
                != reference_input_dim
            ):
                raise ValueError(
                    "Incompatible ensemble checkpoints. "
                    f"input_dim differs in checkpoint {index}: "
                    f"{reference_input_dim} vs "
                    f"{current_input_dim}."
                )

            if (
                current_hidden_dim
                != reference_hidden_dim
            ):
                raise ValueError(
                    "Incompatible ensemble checkpoints. "
                    f"hidden_dim differs in checkpoint {index}: "
                    f"{reference_hidden_dim} vs "
                    f"{current_hidden_dim}."
                )

            if (
                current_num_classes
                != reference_num_classes
            ):
                raise ValueError(
                    "Incompatible ensemble checkpoints. "
                    f"num_classes differs in checkpoint {index}: "
                    f"{reference_num_classes} vs "
                    f"{current_num_classes}."
                )

            if (
                current_label_map
                != reference_label_map
            ):
                raise ValueError(
                    "Incompatible ensemble checkpoints. "
                    f"label_map differs in checkpoint {index}: "
                    f"{reference_label_map} vs "
                    f"{current_label_map}."
                )

            current_drop_rate = float(
                run_conf[
                    "training"
                ].get(
                    "drop_rate",
                    0.0,
                )
            )

            if (
                current_drop_rate
                != self.drop_rate
            ):
                raise ValueError(
                    "Incompatible ensemble checkpoints. "
                    f"drop_rate differs in checkpoint {index}: "
                    f"{self.drop_rate} vs "
                    f"{current_drop_rate}."
                )

    # ==========================================================
    # MODEL CREATION
    # ==========================================================

    def _build_model(
        self,
        checkpoint: dict,
        run_conf: dict,
    ) -> torch.nn.Module:
        """
        Reconstruct one retrained CellViT classifier.
        """

        state_dict = checkpoint[
            "model_state_dict"
        ]

        input_dim = int(
            state_dict[
                "fc1.weight"
            ].shape[1]
        )

        hidden_dim = int(
            state_dict[
                "fc1.weight"
            ].shape[0]
        )

        num_classes = int(
            state_dict[
                "fc2.weight"
            ].shape[0]
        )

        drop_rate = float(
            run_conf[
                "training"
            ].get(
                "drop_rate",
                0.0,
            )
        )

        return LinearClassifier(
            embed_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    # ==========================================================
    # INPUT
    # ==========================================================

    def _prepare_tokens(
        self,
        cell_tokens: TensorOrTensorList,
    ) -> torch.Tensor:
        """
        Convert embeddings to Tensor[N, D].
        """

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

        if (
            x.shape[1]
            != self.input_dim
        ):
            raise ValueError(
                f"Classifier expects embeddings with "
                f"dimension {self.input_dim}, "
                f"received {x.shape[1]}."
            )

        if not torch.isfinite(
            x
        ).all():
            raise ValueError(
                "Non-finite values found in cell embeddings."
            )

        return x

    # ==========================================================
    # INFERENCE
    # ==========================================================

    @torch.inference_mode()
    def predict(
        self,
        cell_tokens: TensorOrTensorList,
    ) -> Dict[str, torch.Tensor]:
        """
        Run single-model or ensemble MLP inference.
        """

        x = self._prepare_tokens(
            cell_tokens
        )

        x = x.to(
            self.device
        )

        fold_logits = []
        fold_probabilities = []

        for model in self.models:

            logits = model(
                x
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
        return False

    @property
    def is_mlp_ensemble_classifier(
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
        """
        Return the loaded classifier configuration.
        """

        return {
            "architecture":
                "mlp",

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
                    str(path)
                    for path in self.checkpoint_paths
                ],

            "device":
                str(self.device),

            "input_dim":
                self.input_dim,

            "hidden_dim":
                self.hidden_dim,

            "num_classes":
                self.num_classes,

            "drop_rate":
                self.drop_rate,

            "label_map":
                self.label_map,
        }