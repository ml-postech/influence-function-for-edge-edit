# Influence Functions for Edge Edits in Non-Convex Graph Neural Networks
This repository is the official implementation of ["Influence Functions for Edge Edits in Non-Convex Graph Neural Networks"](https://arxiv.org/abs/2506.04694) accepted by NeurIPS 2025.

## Setup
```
conda create --name {ENV_NAME} python=3.9
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install matplotlib
pip install ogb
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