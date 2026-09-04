# Context-aware TSN cell classification

## Overview

This fork extends the CellViT++ inference pipeline with three alternative classifiers for assigning Tumor, Stroma, or Normal (TSN) labels to detected cellular instances:

* **MLP:** classifies each cellular instance independently from its 1,280-dimensional CellViT++ embedding.
* **GraphSAGE:** propagates information between spatially neighboring cellular instances using mean neighborhood aggregation.
* **GATv2:** uses attention-based message passing to learn the relative contribution of neighboring instances.

CellViT++ segmentation and feature extraction remain unchanged. The extension replaces only the downstream cellular-instance classification stage.

The implementation was developed as part of an MSc thesis in computational pathology. It is intended for research use and does not constitute a clinically validated system.

## Installation

Follow the installation instructions in the main [CellViT++ README](../README.md). The graph classifiers additionally require PyTorch Geometric:

```bash
pip install torch-geometric==2.8.0.post1
```

The environment used for development and evaluation included:

* Python 3.10
* PyTorch 2.2.2 with CUDA 12.1
* PyTorch Geometric 2.8.0.post1

PyTorch Geometric has also been added to `requirements.txt` and `environment.yaml`.

## Classifier checkpoints

The named classifiers load five fold-specific checkpoints and combine their class probabilities through soft voting. The expected directory structure is:

```text
checkpoints/
└── cell_classifiers/
    ├── mlp_tsn/
    │   ├── fold_0.pth
    │   ├── fold_1.pth
    │   ├── fold_2.pth
    │   ├── fold_3.pth
    │   └── fold_4.pth
    ├── graphsage_context1024_k8/
    │   ├── fold_0.pt
    │   ├── fold_1.pt
    │   ├── fold_2.pt
    │   ├── fold_3.pt
    │   └── fold_4.pt
    └── gatv2_context1024_k6/
        ├── fold_0.pt
        ├── fold_1.pt
        ├── fold_2.pt
        ├── fold_3.pt
        └── fold_4.pt
```

Classifier checkpoints are not included in this repository. A single compatible MLP, GraphSAGE, or GATv2 checkpoint can instead be supplied using `--classifier_path`. The classifier architecture and graph parameters are recovered from the checkpoint metadata.

The pretrained CellViT segmentation checkpoint must be obtained separately according to the instructions in the original CellViT++ repository.

## Graph construction

Each detected cellular instance constitutes one graph node. Its frozen 1,280-dimensional CellViT++ embedding is used as the node feature, while its nuclear-centroid coordinates determine spatial connectivity.

Graphs are constructed dynamically using radius-thresholded k-nearest neighbors. The selected configurations are:

| Classifier | Neighborhood size | Maximum radius |
| ---------- | ----------------: | -------------: |
| GraphSAGE  |                 8 |          30 µm |
| GATv2      |                 6 |          30 µm |

The integrated classifiers were developed for a spatial resolution of 0.25 µm/px. Physical inter-instance distances determine graph topology but are not passed to the GNN as edge features.

Graph construction is performed once for each processed patch, and the same graph is supplied to the five fold-specific models. Softmax probabilities are averaged across models, and the class with the highest mean probability is assigned as the final TSN prediction.

## Inference

### Standard WSI inference

The standard CellViT++ configuration uses internal 1024 × 1024-pixel patches with 64-pixel overlap:

```bash
python cellvit/detect_cells.py \
  --model /path/to/CellViT-SAM-H-x40-AMP.pth \
  --classifier gatv2 \
  --resolution 0.25 \
  --patch_size 1024 \
  --overlap 64 \
  --outdir /path/to/output \
  --geojson \
  process_wsi \
  --wsi_path /path/to/slide.svs
```

Replace `gatv2` with `graphsage` or `mlp` to select another integrated five-model ensemble.

### Independent 256 × 256-pixel patches

The end-to-end comparison conducted in the associated MSc thesis processed independent 256 × 256-pixel image patches without overlap:

```bash
python cellvit/detect_cells.py \
  --model /path/to/CellViT-SAM-H-x40-AMP.pth \
  --classifier gatv2 \
  --resolution 0.25 \
  --patch_size 256 \
  --overlap 0 \
  --outdir /path/to/output \
  process_dataset \
  --wsi_folder /path/to/patches \
  --wsi_extension tif
```

### Loading a single checkpoint

A single compatible checkpoint can be loaded instead of a named five-model ensemble:

```bash
python cellvit/detect_cells.py \
  --model /path/to/CellViT-SAM-H-x40-AMP.pth \
  --classifier_path /path/to/checkpoint.pt \
  --resolution 0.25 \
  --outdir /path/to/output \
  process_wsi \
  --wsi_path /path/to/slide.svs
```

The options `--binary`, `--classifier`, and `--classifier_path` are mutually exclusive.

## Outputs

The predicted TSN label and the corresponding three-class probability vector are assigned to each CellViT++ cellular instance. The detailed `*_cells.json` output retains the TSN label, assigned-class probability, and complete three-class probability vector. The lightweight `*_cell_detection.json` output retains bounding boxes, centroids, and TSN labels. Optional GeoJSON exports contain the TSN class name and its visualization color.

The GeoJSON TSN palette is Tumor `[200, 0, 0]`, Stroma `[150, 200, 150]`, and Normal `[0, 174, 239]`.

The class order used by the integrated TSN classifiers is:

```text
0: Tumor
1: Stroma
2: Normal
```

## Spatial-context limitation

Message passing is restricted to cellular instances belonging to the same internal inference patch. No graph edges are created between adjacent patches. Consequently, standard integrated inference does not reproduce the reconstructed 1024 × 1024-pixel context-window procedure used during the offline model-selection experiment when its source data were independent 256 × 256-pixel patches.

The graph context available during deployment therefore depends on the selected internal patch size and the cellular instances detected within each patch.

## Attribution and license

Fork-specific modifications and authorship are documented in [NOTICE_TFM.md](../NOTICE_TFM.md).

This repository is derived from CellViT++. The original copyright, license, attribution requirements, and citations remain applicable. See [LICENSE](../LICENSE) and the citation section of the original [README](../README.md).
