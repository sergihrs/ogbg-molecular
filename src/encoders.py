import torch
import torch.nn as nn

# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


class CustomAtomEncoder(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        atom_dim: int = 16,
        chiral_dim: int = 4,
        hybrid_dim: int = 8,
        numerical_dim: int = 1,
    ):
        super(CustomAtomEncoder, self).__init__()

        # OGB atom feature dimensions
        atom_dims = get_atom_feature_dims()

        # Embeddings for categorical features
        self.atom_embedding = nn.Embedding(atom_dims[0], atom_dim)
        self.chirality_embedding = nn.Embedding(atom_dims[1], chiral_dim)
        self.hybrid_embedding = nn.Embedding(atom_dims[6], hybrid_dim)

        # Linear projections for Numerical (bias=False to avoid redundancy)
        self.degree_proj = nn.Linear(1, numerical_dim, bias=False)
        self.charge_proj = nn.Linear(1, numerical_dim, bias=False)
        self.num_h_proj = nn.Linear(1, numerical_dim, bias=False)
        self.radical_proj = nn.Linear(1, numerical_dim, bias=False)

        # Is Aromatic and Is In Ring are used as-is (0/1)

        # Calculate input dimension for the projection layer
        self.input_dim = (
            atom_dim + chiral_dim + hybrid_dim + (numerical_dim * 4) + 2
        )  # 2 for aromatic and ring features

        # Final projection to the GNN's expected emb_dim
        self.lin = nn.Linear(self.input_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)  # Normalize before GINE's ReLU

        # Kaiming Initialization (Best for ReLU networks)
        nn.init.kaiming_uniform_(self.atom_embedding.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.chirality_embedding.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.hybrid_embedding.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.lin.weight, nonlinearity="relu")

        # Initialize scalar weights to 1.0 so they start at a reasonable scale
        nn.init.ones_(self.degree_proj.weight)
        nn.init.ones_(self.charge_proj.weight)
        nn.init.ones_(self.num_h_proj.weight)
        nn.init.ones_(self.radical_proj.weight)

    def forward(self, x):
        # x shape: [num_nodes, 9]

        # Embed categorical features
        atom_emb = self.atom_embedding(x[:, 0])
        chiral_emb = self.chirality_embedding(x[:, 1])
        hybrid_emb = self.hybrid_embedding(x[:, 6])

        # Project numerical features
        degree_feat = self.degree_proj(x[:, 2].float().view(-1, 1))
        charge_feat = self.charge_proj(x[:, 3].float().view(-1, 1))
        num_h_feat = self.num_h_proj(x[:, 4].float().view(-1, 1))
        radical_feat = self.radical_proj(x[:, 5].float().view(-1, 1))

        # Raw booleans
        aromatic_feat = x[:, 7].float().view(-1, 1)
        ring_feat = x[:, 8].float().view(-1, 1)

        # Concatenate all features
        features = torch.cat(
            [
                atom_emb,
                chiral_emb,
                hybrid_emb,
                degree_feat,
                charge_feat,
                num_h_feat,
                radical_feat,
                aromatic_feat,
                ring_feat,
            ],
            dim=1,
        )

        # Linear Pojection -> Batch Norm -> (GINE will apply ReLU in aggregation)
        return self.bn(self.lin(features))


class CustomBondEncoder(torch.nn.Module):
    def __init__(self, emb_dim: int, bond_dim: int = 8, stereo_dim: int = 8):
        super(CustomBondEncoder, self).__init__()

        bond_dims = get_bond_feature_dims()

        # Categorical Embeddings
        self.bond_type_emb = nn.Embedding(bond_dims[0], bond_dim)
        self.bond_stereo_emb = nn.Embedding(bond_dims[1], stereo_dim)

        # Input dimension: embeddings + 1 for 'Is Conjugated' boolean
        self.input_dim = bond_dim + stereo_dim + 1

        # Final projection to GNN's expected emb_dim
        self.fusion = nn.Linear(self.input_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)

        # Kaiming Initialization for ReLU-based GINE
        nn.init.kaiming_uniform_(self.bond_type_emb.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.bond_stereo_emb.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fusion.weight, nonlinearity="relu")

    def forward(self, edge_attr):
        # edge_attr shape: [num_edges, 3]

        type_emb = self.bond_type_emb(edge_attr[:, 0])
        stereo_emb = self.bond_stereo_emb(edge_attr[:, 1])
        conj_feat = edge_attr[:, 2].float().view(-1, 1)

        # Concatenate categorical and boolean signals
        features = torch.cat([type_emb, stereo_emb, conj_feat], dim=1)

        # Project -> Batch Norm -> (GINE ReLU)
        return self.bn(self.fusion(features))
