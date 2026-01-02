import torch
import torch.nn as nn

# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


class CustomAtomEncoder(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        atom_dim: int = 8,
        chirality_dim: int = 3,
        hybrid_dim: int = 3,
    ):
        super(CustomAtomEncoder, self).__init__()

        # OGB atom feature dimensions
        atom_dims = get_atom_feature_dims()

        # Atomic Number -> Medium Embedding
        self.atom_embedding = nn.Embedding(atom_dims[0], atom_dim)

        # Chirality -> Small Embedding
        # self.chirality_embedding = nn.Embedding(atom_dims[1], chirality_dim)
        # Chirality -> Dummy ([1,0] if CHI_TETRAHEDRAL_CW, [0,1] if CHI_TETRAHEDRAL_CCW, [1,1])

        # Degree -> Numerical
        # Formal Charge -> Numerical
        # Num Explicit H -> Numerical
        # Radical Electrons -> Numerical

        # Hybridization -> Small Embedding
        self.hybrid_embedding = nn.Embedding(atom_dims[6], hybrid_dim)

        # Is Aromatic -> Boolean (Float)
        # Is In Ring -> Boolean (Float)

        # Calculate input dimension for the projection layer
        self.input_dim = atom_dim + 2 + hybrid_dim + 6

        # Final projection to the GNN's expected emb_dim
        self.lin = nn.Linear(self.input_dim, emb_dim)

        # Initialize weights for better convergence
        nn.init.xavier_uniform_(self.atom_embedding.weight)
        # nn.init.xavier_uniform_(self.chirality_embedding.weight)
        nn.init.xavier_uniform_(self.hybrid_embedding.weight)

    def forward(self, x):
        # x shape: [num_nodes, 9]

        atom_emb = self.atom_embedding(x[:, 0])
        chiral_emb = torch.zeros((x.size(0), 2))
        chiral_emb[x[:, 1] == 1, 0] = 1
        chiral_emb[x[:, 1] == 2, 1] = 1
        degree_feat = x[:, 2].float().view(-1, 1) / 10  # Scale degree 0,10 -> 0,1
        charge_emb = x[:, 3].float().view(-1, 1) / 5  # Scale charge -5,5 -> -1,1
        num_h_feat = x[:, 4].float().view(-1, 1) / 8  # Scale numH 0,8 -> 0,1
        radical_feat = x[:, 5].float().view(-1, 1)  # Scale radical electrons 0,4 -> 0,1
        hybrid_emb = self.hybrid_embedding(x[:, 6])
        aromatic_feat = x[:, 7].float().view(-1, 1)  # False, True -> 0,1
        ring_feat = x[:, 8].float().view(-1, 1)  # False, True -> 0,1

        # Concatenate all features
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

        # Project to emb_dim
        return self.lin(features)


class CustomBondEncoder(torch.nn.Module):
    def __init__(self, emb_dim: int, bond_dim: int = 3, stereo_dim: int = 3):
        super(CustomBondEncoder, self).__init__()

        bond_dims = get_bond_feature_dims()

        # Bond Type  -> Small Embedding
        self.bond_type_emb = nn.Embedding(bond_dims[0], bond_dim)

        # Bond Stereo -> Small Embedding
        self.bond_stereo_emb = nn.Embedding(bond_dims[1], stereo_dim)

        # Is Conjugated -> Boolean

        # Calculate input dimension for the projection layer
        self.input_dim = bond_dim + stereo_dim + 1

        self.lin = nn.Linear(self.input_dim, emb_dim)

        # Initialize weights for better convergence
        nn.init.xavier_uniform_(self.bond_type_emb.weight)
        nn.init.xavier_uniform_(self.bond_stereo_emb.weight)

    def forward(self, edge_attr):
        # edge_attr shape: [num_edges, 3]

        type_emb = self.bond_type_emb(edge_attr[:, 0])
        stereo_emb = self.bond_stereo_emb(edge_attr[:, 1])
        conj_feat = edge_attr[:, 2].float().view(-1, 1)  # False, True -> 0,1

        # Concatenate all features
        features = torch.cat([type_emb, stereo_emb, conj_feat], dim=1)

        # Project to emb_dim
        return self.lin(features)
