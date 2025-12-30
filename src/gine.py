import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.models import MLP

from .encoders import CustomAtomEncoder, CustomBondEncoder


class MolecularGINE(nn.Module):
    def __init__(
        self,
        emb_dim,
        out_dim,
        num_layers=1,
        dropout=0.0,
        mlp_hidden_dim=None,
        mlp_num_layers=1,
        mlp_dropout=0.0,
    ):
        super().__init__()
        self.dropout_ratio = dropout

        # Encoders
        self.atom_encoder = CustomAtomEncoder(emb_dim=emb_dim)
        self.bond_encoder = CustomBondEncoder(emb_dim=emb_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        for layer_idx in range(num_layers):
            mlp = MLP(
                in_channels=emb_dim,
                hidden_channels=mlp_hidden_dim or emb_dim,
                out_channels=emb_dim,
                num_layers=mlp_num_layers,
                dropout=mlp_dropout,
                activation=nn.PReLU(),
            )
            self.convs.append(GINEConv(mlp))
            # Skip last batch norm after final layer
            if layer_idx < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(emb_dim))
                self.activations.append(nn.PReLU())

        # The final predictor needs to process this large concatenated vector
        full_dim = emb_dim * (num_layers + 1)
        self.lin_pred = nn.Sequential(
            nn.Linear(full_dim, emb_dim),
            nn.PReLU(),
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

        # for conv, bn in zip(self.convs, self.batch_norms):
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
        h_graph = torch.cat(
            hidden_reps, dim=1
        )  # Shape: [batch_size, emb_dim * (num_layers+1)]

        # Final prediction
        return self.lin_pred(h_graph)
