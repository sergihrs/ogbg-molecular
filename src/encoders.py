import torch
import torch.nn as nn
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


class CustomAtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(CustomAtomEncoder, self).__init__()

        # OGB atom feature dimensions
        atom_dims = get_atom_feature_dims()

        # Atomic Number -> Medium Embedding
        self.atom_embedding = nn.Embedding(atom_dims[0], 8)

        # Chirality -> Small Embedding
        self.chirality_embedding = nn.Embedding(atom_dims[1], 3)

        # Degree -> Numerical

        # Formal Charge -> Small Embedding
        self.charge_embedding = nn.Embedding(atom_dims[3], 3)

        # Num Explicit H -> Numerical

        # Radical Electrons -> Numerical

        # Hybridization -> Small Embedding
        self.hybrid_embedding = nn.Embedding(atom_dims[6], 3)

        # Is Aromatic -> Boolean (Float)
        # Is In Ring -> Boolean (Float)

        # Calculate input dimension for the projection layer
        self.input_dim = 8 + 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1

        # Final projection to the GNN's expected emb_dim
        self.project = nn.Sequential(nn.Linear(self.input_dim, emb_dim), nn.ReLU())

        # Initialize weights specifically for better convergence
        nn.init.xavier_uniform_(self.atom_embedding.weight)

    def forward(self, x):
        # x shape: [num_nodes, 9]

        # --- Categorical Features ---
        atom_emb = self.atom_embedding(x[:, 0])  # [N, 8]
        chiral_emb = self.chirality_embedding(x[:, 1])  # [N, 3]
        charge_emb = self.charge_embedding(x[:, 3])  # [N, 3]
        hybrid_emb = self.hybrid_embedding(x[:, 6])  # [N, 3]

        # --- Numerical Features (Scaled) ---
        degree_feat = x[:, 2].float().view(-1, 1) * 0.1  # Scale max degree ~10 to 1.0
        num_h_feat = x[:, 4].float().view(-1, 1) * 0.2  # Scale max H ~5 to 1.0
        radical_feat = x[:, 5].float().view(-1, 1)  # Usually 0, 1, or 2

        # --- Boolean Features ---
        aromatic_feat = x[:, 7].float().view(-1, 1)
        ring_feat = x[:, 8].float().view(-1, 1)

        # --- Concatenate All Features ---
        features = torch.cat(
            [
                atom_emb,
                chiral_emb,
                charge_emb,
                hybrid_emb,
                degree_feat,
                num_h_feat,
                radical_feat,
                aromatic_feat,
                ring_feat,
            ],
            dim=1,
        )

        return self.project(features)


class CustomBondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(CustomBondEncoder, self).__init__()

        bond_dims = get_bond_feature_dims()

        # Bond Type  -> Embedding
        self.bond_type_emb = nn.Embedding(
            bond_dims[0], 3
        )  # Single, Double, Triple, Aromatic

        # Bond Stereo -> Embedding
        self.bond_stereo_emb = nn.Embedding(bond_dims[1], 3)

        # Is Conjugated -> Boolean

        # Calculate input dimension for the projection layer
        self.input_dim = 3 + 3 + 1

        self.project = nn.Sequential(nn.Linear(self.input_dim, emb_dim), nn.ReLU())

    def forward(self, edge_attr):
        # edge_attr shape: [num_edges, 3]

        # --- Categorical ---
        type_emb = self.bond_type_emb(edge_attr[:, 0])  # [E, 8]
        stereo_emb = self.bond_stereo_emb(edge_attr[:, 1])  # [E, 3]

        # --- Boolean ---
        conj_feat = edge_attr[:, 2].float().view(-1, 1)  # [E, 1]

        # --- Concatenate ---
        features = torch.cat([type_emb, stereo_emb, conj_feat], dim=1)

        # --- Project ---
        return self.project(features)
