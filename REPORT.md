# Technical Report: Efficient Graph Isomorphism Networks with Heterogeneous Encoding for OGBG-MolHIV

## 1. Introduction & Contribution

This report details the submission of three model variants based on the Graph Isomorphism Network (GINE) architecture for the `ogbg-molhiv` dataset. The study explores the trade-offs between architectural complexity (edge features), encoding strategies (heterogeneous vs. uniform), and parameter efficiency.

Our experiments reveal a clear hierarchy:

1.  **Performance:** The standard **GINE** (with edge features and default encoders) achieves the highest Test AUC (**0.7921**), demonstrating that edge information contributes positively to predictive power.
2.  **Efficiency:** The proposed **Heterogeneous Encoder (HE)** reduces the model's parameter count by **~71%** (from 33k to 9k). Crucially, it achieves a Test AUC of **0.7903**, which is statistically comparable to the heavy model, validating that molecular graphs can be encoded extremely compactly without meaningful performance loss.

## 2. Submissions & Experimental Details

### Submission 1: GINE + Heterogeneous Encoder (GINE-HE)

**Mechanism:**
This model introduces a novel **Heterogeneous Encoder** designed to respect the semantic type of molecular features. Unlike standard OGB encoders which treat all features as categorical integers:

- **Categorical Features** (e.g., Atom Type) are mapped via `nn.Embedding` layers with dimensions chosen according to feature cardinality and complexity.
- **Numerical Features** (e.g., Degree, Charge) scaled through a learnable weight parameter to preserve ordinal relationships.
- **Boolean Features** (e.g., Aromaticity) are processed as raw `0.0/1.0` float scalars.

These type-specific outputs are concatenated, projected to `emb_dim` and normalized via **Batch Normalization** before entering the GINE layers. This allows for a significantly smaller embedding dimension ($d=32$) to carry the same information density as larger standard embeddings.

**OGB Hyperparameter Tuning:**
We tuned the embedding dimension specifically to demonstrate parameter efficiency. A smaller dimension ($32$) was sufficient for GINE-HE, whereas the default encoder required $64$ to maintain performance. We observed that slightly extending the training duration (60 epochs) allowed this compact model to converge to a better optima.

- **Format:** `lr: [0.001*], num_layers: [2*], mlp_layers: [1, 2*, 3], emb_dim: [32*, 64], dropout: [0.475*, 0.5], lr_gamma: [0.5, 0.707*], max_epochs: [50, 60*]`

**Reproducibility:**

```bash
uv run python -m src.benchmark --encoder_type he --use_edge_features --emb_dim 32 --dropout 0.5 --lr_gamma 0.707 --max_epochs 60
```

**Results:**

- **Test AUC:** 0.7903 ± 0.0079
- **Val AUC:** 0.8099 ± 0.0080
- **Params:** 9,393

**Conclusion:** GINE-HE serves as the **efficiency champion** of this study. It achieves competitive performance (within 0.2% of the best heavy model) while using **3.5x fewer parameters**.

---

### Submission 2: Standard GINE (Default Encoder)

**Mechanism:**
This model implements the standard **GINE** architecture described by Hu et al. [1]. It incorporates edge features $e_{uv}$ into the aggregation step: $h_v^{(k)} = \text{MLP}^{(k)} \left( (1 + \epsilon^{(k)}) \cdot h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \text{ReLU}(h_u^{(k-1)} + e_{uv}) \right)$. It utilizes the default `ogb.graphproppred.mol_encoder` atom and bond encoders, which projects all node and edge features (regardless of type) to a fixed embedding dimension and sums them.

**OGB Hyperparameter Tuning:**
We identified that adding edge features increased the model's capacity and risk of overfitting. To counter this, we conducted a fine-grained sweep of dropout rates. While standard values like `0.6` degraded performance, a precise tune to `0.525` provided the necessary regularization. Additionally, extending the training duration slightly allowed the model to converge more robustly across multiple runs.

- **Format:** `lr: [0.001*], num_layers: [2*], mlp_layers: [2*], emb_dim: [32, 64*], dropout: [0.5, 0.51, 0.525*, 0.6], lr_gamma: [0.5, 0.707*], max_epochs: [50, 60*]`

**Reproducibility:**

```bash
uv run python -m src.benchmark --encoder_type default --use_edge_features --emb_dim 64 --dropout 0.525 --lr_gamma 0.707 --max_epochs 60
```

**Results:**

- **Test AUC:** 0.7921 ± 0.0128
- **Val AUC:** 0.7987 ± 0.0075
- **Params:** 33,217

---

### Submission 3: GIN (Baseline Reproduction)

**Mechanism:**
This is an improvement of the current lightweight leaderboard entry (Tiny-GIN) by Bruns [4]. It is a standard GIN [3] without edge features. Our implementation builds upon the original by injecting **Batch Normalization** _inside_ the MLPs of the GINConv layers and using a `StepLR` scheduler instead of a single decay step.

**OGB Hyperparameter Tuning:**
To ensure that the gains observed in Submission 2 (Standard GINE) were due to architectural improvements (Edge Features) and not simply random hyperparameter luck, we subjected this baseline GIN to an extensive hyperparameter search:

- We tested **Dropout** at [0.475, 0.5, 0.525].
- We tested **Max Epochs** at [50, 60].
- We tested **Gamma** at [0.5, 0.707].

**Observation:** We found that a slightly lower dropout (`0.475`) was optimal for the simpler GIN architecture. The resulting model achieves **0.7908**, significantly outperforming the original leaderboard baseline (0.7835), validating the impact of our Batch Norm and Scheduler improvements.

- **Format:** `lr: [0.001*], num_layers: [2*], emb_dim: [32, 64*], dropout: [0.475*, 0.5, 0.525], lr_gamma: [0.5*, 0.707], max_epochs: [50, 60*]`

**Reproducibility:**

```bash
uv run python -m src.benchmark --encoder_type default --emb_dim 64 --dropout 0.475 --lr_gamma 0.5 --max_epochs 60
```

**Results:**

- **Test AUC:** 0.7908 ± 0.0102
- **Val AUC:** 0.7944 ± 0.0140
- **Params:** 32,385

---

## 3. Ablation Studies & Negative Results

- **GraphNorm vs. BatchNorm:** We experimented with `GraphNorm` to handle graph-size variability. While Train/Val AUC increased (>0.85), Test AUC degraded significantly (~0.75). We hypothesize this is due to `ogbg-molhiv`'s scaffold splitting; GraphNorm removes global statistics (like total molecule size/charge) that vary between scaffolds, causing overfitting to local training structures. **BatchNorm** was retained as it enforces global regularization.
- **Jumping Knowledge (JK):** Concatenating all layers degraded performance, suggesting that the initial raw embeddings (Layer 0) allowed the model to overfit on simple atom counts rather than learning structural isomorphism.
- **Depth:** 2 GNN layers consistently outperformed 3 or 4 layers, likely due to over-smoothing on the small average graph size (~26 nodes) of the dataset.

## 📚 References

[1] Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2019). **Strategies for Pre-training Graph Neural Networks**. [_arXiv:1905.12265_](https://arxiv.org/abs/1905.12265).

[2] Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). **Open Graph Benchmark: Datasets for Real-World Graph Machine Learning**. [_Advances in Neural Information Processing Systems (NeurIPS)_](https://arxiv.org/abs/2005.00687).

[3] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). **How Powerful are Graph Neural Networks?**. [_arXiv:1810.00826_](https://arxiv.org/abs/1810.00826).

[4] Bruns, W. **tiny-GIN-for-ogbg-molhiv**. [GitHub Repo](https://github.com/willy-b/tiny-GIN-for-ogbg-molhiv).
