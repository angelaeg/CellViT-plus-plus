# Notice of modifications

This repository is a fork of [CellViT++](https://github.com/TIO-IKIM/CellViT-plus-plus) extended as part of an MSc thesis project at Universidad Politécnica de Madrid (UPM).

## Modifications

The modifications were developed by Ángela Esteban in 2026 and introduce context-aware Tumor–Stroma–Normal cellular-instance classification into the CellViT++ inference pipeline.

The principal additions are:

- MLP, GraphSAGE, and GATv2 classification backends.
- Dynamic construction of radius-thresholded k-nearest-neighbor cell graphs.
- Five-fold soft-voting inference for the three classifiers.
- Automatic recovery of architecture and graph parameters from compatible checkpoints.
- Selection of the integrated classifier through the CellViT++ command-line interface.
- Propagation of TSN labels and class probabilities through the standard JSON and GeoJSON outputs.
- Exposure of the internal CellViT++ patch size and overlap as command-line options for integrated inference.
- Documentation and dependency updates required by the graph-based classifiers.

## Modified upstream files

The following files from the original CellViT++ repository were modified:

- `cellvit/detect_cells.py`
- `cellvit/inference/cli.py`
- `cellvit/inference/inference_disk.py`
- `cellvit/inference/inference_memory.py`
- `cellvit/inference/postprocessing_cupy.py`
- `requirements.txt`
- `environment.yaml`
- `README.md`

## New integration files

The following files were added for the TSN classification extension:

- `cell_graph/inference/graph_classifier.py`
- `cell_graph/inference/mlp_ensemble_classifier.py`
- `cell_graph/models/graphsage.py`
- `cell_graph/models/gatv2.py`
- `cell_graph/utils/graph_utils.py`
- `docs/graph_classifier_integration.md`
- Package initialization files under `cell_graph/`

## Distribution scope

The repository does not contain patient data, cached cellular embeddings, graph indices, experimental outputs, or other derived clinical data.

Classifier checkpoints are not included unless their public distribution is separately authorized. The pretrained CellViT++ segmentation checkpoints must be obtained from the sources indicated by the original project.

## License and attribution

The original CellViT++ copyright, licensing conditions, attribution requirements, and citation requirements remain applicable. These modifications do not replace or alter the original license. See [`LICENSE`](LICENSE) and the citation section of [`README.md`](README.md).

This software is provided for research purposes and has not been clinically validated.
