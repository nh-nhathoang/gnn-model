# ML model for predicting elastic modulus of imperfect planar lattice structures

This repository contains the reproducible implementation for my **Bachelor’s Thesis**. The code focuses on predicting the elastic modulus of planar lattice structures using **Graph Neural Networks (GNNs)**.

## Repository Structure

```
├── saved_model/
│   └── epoch_200.pt          # Trained model checkpoint (optional / not required for reproduction)
├── utils/
│   ├── GNN_architecture.py   # GIN and Graph Transformer model definitions
│   ├── GNN_data.py           # Data loading and graph construction utilities
│   ├── train_model.py        # Training loop
│   ├── evaluate_model.py     # Evaluation utilities
│   └── __pycache__/
├── GIN.ipynb                 # GIN experiments and analysis
├── final-gnn-transformer-model.ipynb  # Final Transformer-based GNN experiments
├── README.md
└── .gitignore
```

## Models

* **GIN (Graph Isomorphism Network)** with 3 convolution layers and global pooling
* **Graph Transformer** using multi-head attention

Each node is represented by **two features** (x, y) coordinates. Graph connectivity encodes struts. The model predicts a **single scalar output** (elastic modulus).

## Dataset

* Data is generated from finite element (FE) simulations of planar lattice structures
* Each simulation corresponds to one `.inp` file and one elastic modulus E value
* The dataset itself is **not included** in this repository

## Reproducibility

The repository provides:

* Complete model architectures
* Training and evaluation pipelines
* Example notebooks reproducing the thesis methodology

Minor numerical differences compared to the thesis may occur due to random initialization or refactoring. 

## License

This repository is intended for **academic and educational use**.

## Reference 
```
@article{chung2024prediction,
title={Prediction of effective elastic moduli of rocks using Graph Neural Networks},
author={Chung, Jaehong and Ahmad, Rasool and Sun, WaiChing and Cai, Wei and Mukerji, Tapan},
journal={Computer Methods in Applied Mechanics and Engineering},
volume={421},
pages={116780},
year={2024},
publisher={Elsevier}
}
```
