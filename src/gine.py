import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.models import MLP

from .encoders import CustomAtomEncoder, CustomBondEncoder


class MolecularGINE(nn.Module):
    def __init__(self, emb_dim, out_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout_ratio = dropout

        # Encoders
        self.atom_encoder = CustomAtomEncoder(emb_dim=emb_dim)
        self.bond_encoder = CustomBondEncoder(emb_dim=emb_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            mlp = MLP([emb_dim, emb_dim, emb_dim])
            self.convs.append(GINEConv(mlp))
            # Skip last batch norm after final layer
            if layer_idx < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        # The final predictor needs to process this large concatenated vector
        full_dim = emb_dim * num_layers
        self.lin_pred = nn.Sequential(
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
        hidden_reps = []  # Do NOT include initial features

        # for conv, bn in zip(self.convs, self.batch_norms):
        for layer_idx in range(len(self.convs)):
            conv = self.convs[layer_idx]
            # Pool this layer's representation and save it
            h = conv(h, edge_index, edge_attr=edge_emb)
            hidden_reps.append(global_add_pool(h, batch))

            # Apply BatchNorm, ReLU, Dropout except after last layer
            if layer_idx < len(self.batch_norms):
                h = self.batch_norms[layer_idx](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout_ratio, training=self.training)

        # Concatenate all layers (Jumping Knowledge)
        h_graph = torch.cat(
            hidden_reps, dim=1
        )  # Shape: [batch_size, emb_dim * num_layers]

        # Final prediction
        return self.lin_pred(h_graph)
