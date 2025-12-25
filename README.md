# Influence Functions for Edge Edits in Non-Convex Graph Neural Networks

This repository is the official implementation of ["Influence Functions for Edge Edits in Non-Convex Graph Neural Networks"](https://arxiv.org/abs/2506.04694) accepted by NeurIPS 2025.

## Overview

This project provides a modular and easy-to-use framework for computing **influence functions** for edge edits (removal/insertion) in Graph Neural Networks (GNNs). The core module, `GraphInfluenceModule`, can be used as an analysis tool without deep understanding of the underlying implementation.

## Setup

### Environment Installation
```bash
conda create --name EIF python=3.9
conda activate EIF
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install matplotlib
pip install ogb
pip install tqdm
```

## Quick Start

The `tutorial.py` file provides a minimal example of how to use the `GraphInfluenceModule`:

```bash
# Run with default settings (Cora dataset, GCN model)
python tutorial.py

# Run with fewer epochs and candidates for quick testing
python tutorial.py --epochs 10 --num_removal_candidates 5 --num_insertion_candidates 5

# Run with different model and dataset configurations
python tutorial.py --model GCN --dataset Cora --num_layers 2 --hidden_dim 32
```

### Example Output

When you run `tutorial.py`, you'll see output like:

```
Best results, train loss: 1.9335, val loss: 1.9316, test_loss: 1.9298
Train acc: 18.82%, val acc: 19.19%, test_acc: 21.18%
-----------------------------------------------
Calculating influence for edge removal...
LiSSA converged. Norm: 0.0000001
Calculating influence for edge insertion...
Removal influence shape: torch.Size([5, 1])
Insertion influence shape: torch.Size([5, 1])
Average removal influence: -0.000007
Average insertion influence: 0.000025
```

The output shows:
1. Model training results (accuracy and loss)
2. LiSSA convergence status
3. Influence values for each edge candidate

## Usage Guide

### Basic Usage

Here's how to use `GraphInfluenceModule` to compute influence functions for your GNN:

```python
import torch
from src import GNN, DataLoader, make_metric_fns
from src.utils import set_seed, get_edge_removal_candidates, get_edge_insertion_candidates, get_eval_node_idxs
from src.graph_utils import find_k_hop_neighborhoods
from calculate_influence import GraphInfluenceModule

# 1. Load your dataset
dataset = DataLoader('Cora', root='datasets')
data = dataset[0]
data.edge_weight = torch.ones((data.edge_index.shape[1], ))

# 2. Train or load your GNN model
model = GNN(name='GCN', in_dim=dataset.num_node_features,
            hidden_dim=32, num_classes=dataset.num_classes,
            num_layers=2, linear=False, bias=False)
# ... train your model ...

# 3. Set up evaluation metric
eval_node_idxs = get_eval_node_idxs(data, 'mean_validation_loss', seed=42)
metric_fns = make_metric_fns(eval_node_idxs, None, data.edge_index)
metric_fn = metric_fns['mean_validation_loss']

# 4. Select candidate edges for analysis
# You can use helper functions to randomly select edges...
removal_candidates = get_edge_removal_candidates(data, num_candidates=100)
insertion_candidates = get_edge_insertion_candidates(data, num_candidates=100)
# Or specify your own edges of interest (see "Analyzing Specific Edges" section)

# Reshape to (num_candidates, 1, 2) format
removal_candidates = removal_candidates.view(-1, 1, 2)
insertion_candidates = insertion_candidates.view(-1, 1, 2)

# 5. Create influence module and compute influence
influence_module = GraphInfluenceModule(
    model=model,
    graph=data,
    args=args,  # See tutorial.py for args configuration
    eval_metric='mean_validation_loss',
    num_folds=1,
    eval_node_idxs=eval_node_idxs,
    metric_fn=metric_fn
)

# Compute influence for edge removal
total_inf, retrain_inf, perturb_inf, scale, inv_hvp_norm, avg_influenced = \
    influence_module.calculate_influence(removal_candidates, 'edge_removal')

# Compute influence for edge insertion
total_inf, retrain_inf, perturb_inf, scale, inv_hvp_norm, avg_influenced = \
    influence_module.calculate_influence(insertion_candidates, 'edge_insertion')
```

### Command-Line Arguments

Key arguments for `tutorial.py` (and `calculate_influence.py`):

**Model Configuration:**
- `--model`: GNN model type (`'GCN'`, `'SGC'`, `'GAT'`, `'ChebNet'`)
- `--num_layers`: Number of GNN layers (default: 2)
- `--hidden_dim`: Hidden dimension size (default: 32)
- `--linear`: Use linear layers (0 or 1)
- `--bias`: Use bias in layers (0 or 1)

**Training Configuration:**
- `--dataset`: Dataset name (default: `'Cora'`)
- `--lr`: Learning rate (default: 0.1)
- `--epochs`: Number of training epochs (default: 1000)
- `--weight_decay`: L2 regularization weight (default: 0.001)

**Influence Function Configuration:**
- `--hessian_type`: Hessian approximation type (`'hessian'` or `'GNH'`, default: `'GNH'`)
- `--damp`: Damping parameter for Hessian approximation (default: 0.1)
- `--scale`: Scaling factor for LiSSA (default: 1.0)
- `--lissa_iter`: Number of LiSSA iterations (default: 10000)
- `--eval_metric`: Evaluation metric type (default: `'mean_validation_loss'`)

## Project Structure

```
.
├── calculate_influence.py          # Full implementation with PBRF comparison
├── tutorial.py                      # Minimal usage example
├── src/
│   ├── __init__.py
│   ├── models.py                    # GNN model implementations (GCN, SGC, GAT, etc.)
│   ├── train.py                     # Training and evaluation functions
│   ├── metrics.py                   # Evaluation metric functions
│   ├── utils.py                     # Utility functions (edge selection, etc.)
│   ├── graph_utils.py               # Graph manipulation utilities
│   ├── dataset_loader.py            # Dataset loading
│   └── pbrf.py                      # PBRF baseline implementation
└── torch_influence/
    ├── base.py                      # Base classes for influence functions
    └── modules.py                   # LiSSA influence module
```

## Citation
Please cite our paper if you use the model or this code in your own work:
```
@article{heo2025influence,
  title={Influence Functions for Edge Edits in Non-Convex Graph Neural Networks},
  author={Heo, Jaeseung and Yun, Kyeongheung and Yoon, Seokwon and Park, MoonJeong and Ok, Jungseul and Kim, Dongwoo},
  journal={arXiv preprint arXiv:2506.04694},
  year={2025}
}
```