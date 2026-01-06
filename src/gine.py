import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.models import MLP

from .encoders import (
    HeterogeneousAtomEncoder,
    HeterogeneousBondEncoder,
    ZeroBondEncoder,
)


class MolecularGINE(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        mlp_num_layers: int = 2,
        jumping_knowledge: bool = False,
        use_edge_features: bool = True,
        encoder_type: str = "he",
    ):
        super().__init__()
        self.dropout_ratio = dropout
        self.jumping_knowledge = jumping_knowledge

        # Encoders
        atom_encoder_cls = (
            HeterogeneousAtomEncoder if encoder_type == "he" else AtomEncoder
        )
        bond_encoder_cls = (
            HeterogeneousBondEncoder
            if encoder_type == "he"
            else BondEncoder
            if use_edge_features
            else ZeroBondEncoder
        )
        self.atom_encoder = atom_encoder_cls(emb_dim=emb_dim)
        self.bond_encoder = bond_encoder_cls(emb_dim=emb_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        for layer_idx in range(num_layers):
            mlp = MLP(
                in_channels=emb_dim,
                hidden_channels=emb_dim,
                out_channels=emb_dim,
                num_layers=mlp_num_layers,
                batch_norm=True,
            )
            self.convs.append(GINEConv(mlp))
            # Skip last batch norm after final layer
            if layer_idx < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(emb_dim))
                self.activations.append(nn.ReLU())

        # The final predictor needs to process this large concatenated vector
        if jumping_knowledge:
            full_dim = emb_dim * (num_layers + 1)
        else:
            full_dim = emb_dim
        self.lin_pred = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(full_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, out_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Node embeddings  (Layer 0 representation)
        h = self.atom_encoder(x)

        # Edge embeddings
        edge_emb = self.bond_encoder(edge_attr)

        # List to store graph-level representations from every layer
        hidden_reps = [global_add_pool(h, batch)]  # Include initial representation

        for layer_idx in range(len(self.convs)):
            h = self.convs[layer_idx](h, edge_index, edge_attr=edge_emb)
            # Pool this layer's representation and save it
            hidden_reps.append(global_add_pool(h, batch))

            # Apply BatchNorm, ReLU, Dropout except after last layer
            if layer_idx < len(self.batch_norms):
                h = self.batch_norms[layer_idx](h)
                h = self.activations[layer_idx](h)
                h = F.dropout(h, p=self.dropout_ratio, training=self.training)

        # Concatenate all layers (Jumping Knowledge)
        if self.jumping_knowledge:
            h_graph = torch.cat(
                hidden_reps, dim=1
            )  # Shape: [batch_size, emb_dim * (num_layers+1)]
        else:
            h_graph = hidden_reps[-1]  # Shape: [batch_size, emb_dim]

        # Final prediction
        return self.lin_pred(h_graph)
