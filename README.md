# GINE-Mol: Graph Isomorphism Networks for Molecular Property Prediction

[![Dataset: OGBG-MolHIV](https://img.shields.io/badge/OGB-molhiv-blue)](https://ogb.stanford.edu/docs/graphprop/#ogbg-molhiv)
[![Framework: PyTorch Geometric](https://img.shields.io/badge/Framework-PyTorch_Geometric-orange)](https://pytorch-geometric.readthedocs.io/)

This repository provides a benchmark submission for the **ogbg-molhiv** dataset using **Graph Isomorphism Network with Edge Features (GINE)**. The project introduces a specialized **Heterogeneous Encoder** (GINE+HE) that drastically reduces parameter count (by >3x) while maintaining competitive performance against standard GIN architectures.

## 🛠️ Methods & Architectures

1.  **GIN (Baseline Reproduction):** A reproduction of the current leaderboard GIN entry by William Bruns [4], optimized with `StepLR` and Batch Normalization for improved training stability.
2.  **GINE (Default Encoder):** Incorporates edge features into the message passing aggregation as proposed by Hu et al. [1]. It utilizes OGB's default encoders where all node and edge features are treated as categorical, projected to a common embedding space, and summed.
3.  **GINE+HE (Heterogeneous Encoder):** A specialized encoder designed to respect the underlying semantics and type of molecular data:

    - **Categorical Features** (e.g., Atom Type, Hybridization) are mapped via learnable embeddings with dimensions scaled to feature cardinality (e.g., Atom Type receives a larger latent space than Hybridization).
    - **Numerical Features** (e.g., Degree, Formal Charge) are scaled through a learnable weight to preserve ordinal relationships.
    - **Boolean Features** (e.g., Aromaticity, Conjugation) are processed as raw `0.0/1.0` float scalars.

    **Fusion Strategy:** All mapped features are **concatenated** into a single dense vector. This vector is then projected through a linear layer to the final embedding dimension.

## 🔬 Experimental Results (ogbg-molhiv)

| Method                     | Test AUC            | Validation AUC      | # Parameters | Hardware                      |
| :------------------------- | :------------------ | :------------------ | :----------- | :---------------------------- |
| **GINE (Default Encoder)** | **0.7901 ± 0.0104** | 0.7965 ± 0.0099     | 33,217       | CPU                           |
| **GINE+HE**                | 0.7878 ± 0.0057     | **0.8136 ± 0.0082** | **9,393**    | CPU                           |
| GIN (Baseline)\*           | 0.7850 ± 0.0104     | 0.8098 ± 0.0102     | 32,385       | CPU                           |
| _GIN (W. Bruns)_           | _0.7835 ± 0.0125_   | _0.8010 ± 0.0078_   | _32,385_     | _CPU; Colab L4 for HP search_ |

> **n.b.** _GINE+HE_ achieves nearly the same test accuracy as the best performing model while using **71% fewer parameters**, proving that intelligent feature encoding is more efficient than raw parameter scaling.

## 📦 Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable dependency management and reproducibility.

**1. Install uv** ([official instructions](https://docs.astral.sh/uv/getting-started/installation/))

```bash
pip install uv
```

**2. Clone and Sync**

```bash
git clone https://github.com/sergihrs/ogbg-molecular.git
cd ogbg-molecular
uv sync
```

## 🧪 Reproducing Experiments

To reproduce the results reported in the table, run the following commands. All experiments are configured to run with 10 random seeds for statistical significance.

### 1. Standard GINE (Best Accuracy)

Uses all node and edge features with the default OGB sum-aggregation encoding.

```bash
uv run python -m src.benchmark --encoder_type default --use_edge_features --emb_dim 64 --dropout 0.5
```

### 2. GINE+HE (Most Efficient)

Uses the Heterogeneous Encoder to achieve high performance with only ~9k parameters.

```bash
uv run python -m src.benchmark --encoder_type he --use_edge_features --emb_dim 32 --dropout 0.5
```

### 3. GIN Baseline (Reproduction)

Reproduces the GIN results by disabling edge features.

```bash
uv run python -m src.benchmark --encoder_type default --emb_dim 64 --dropout 0.5
```

## 📚 References

[1] Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2019). **Strategies for Pre-training Graph Neural Networks**. [_arXiv:1905.12265_](https://arxiv.org/abs/1905.12265).

[2] Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). **Open Graph Benchmark: Datasets for Real-World Graph Machine Learning**. [_Advances in Neural Information Processing Systems (NeurIPS)_](https://arxiv.org/abs/2005.00687).

[3] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). **How Powerful are Graph Neural Networks?**. [_arXiv:1810.00826_](https://arxiv.org/abs/1810.00826).

[4] Bruns, W. **tiny-GIN-for-ogbg-molhiv**. [GitHub Repo](https://github.com/willy-b/tiny-GIN-for-ogbg-molhiv).
