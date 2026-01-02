import warnings

import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset  # noqa: E402


def load_data(dataset_name: str = "ogbg-molhiv", batch_size: int = 32):
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
    dataset = PygGraphPropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]], batch_size=batch_size, shuffle=False
    )
    evaluator = Evaluator(name=dataset_name)
    return dataset, train_loader, valid_loader, test_loader, evaluator
