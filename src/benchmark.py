import sys
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

with warnings.catch_warnings():
    from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

from src.gine import MolecularGINE

warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


def load_data():
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
    return dataset, train_loader, valid_loader, test_loader


def train_step(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for _, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass  # Skip batches with single molecule/node if they cause BatchNorm issues (optional)

        pred = model(batch)
        y = batch.y.to(torch.float32)

        # --- Handling Missing Labels (NaNs) ---
        # Whether a label is valid (not NaN)
        is_labeled = y == y

        # Loss is calculated only on labeled tasks
        # We perform logical indexing to flatten the valid predictions and targets
        loss = criterion(pred[is_labeled], y[is_labeled])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_step(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    # Concatenate all batches
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # OGB Evaluator expects a dictionary
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    # Returns a dict like {'rocauc': 0.78}
    return evaluator.eval(input_dict)


def train_loop(
    model,
    device,
    train_loader,
    valid_loader,
    test_loader,
    optimizer,
    criterion,
    evaluator,
    num_epochs,
) -> tuple[float, float]:
    tqdm_range = tqdm(range(num_epochs), unit="epoch")
    best_val_auc = 0
    final_test_auc = 0

    for epoch in tqdm_range:
        # Train
        loss = train_step(model, device, train_loader, optimizer, criterion)

        # Evaluate
        train_result = eval_step(model, device, train_loader, evaluator)
        val_result = eval_step(model, device, valid_loader, evaluator)
        test_result = eval_step(model, device, test_loader, evaluator)

        train_auc = train_result["rocauc"]
        val_auc = val_result["rocauc"]
        test_auc = test_result["rocauc"]

        # Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc

        tqdm_range.set_postfix(
            {
                "loss": f"{loss:.4f}",
                "train": f"{train_auc:.4f}",
                "val": f"{val_auc:.4f}",
                "test": f"{test_auc:.4f}",
            }
        )

    return best_val_auc, final_test_auc


def get_gine_parameters(hyperparams, dataset):
    model = MolecularGINE(
        emb_dim=hyperparams["emb_dim"],
        out_dim=dataset.num_tasks,
        num_layers=hyperparams["num_layers"],
        dropout=hyperparams["dropout"],
    )

    # Print model architecture and number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def main_gine(dataset, train_loader, valid_loader, test_loader):
    print("\n--- Benchmarking GINE Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hyperparams = {
        "emb_dim": 32,
        "out_dim": dataset.num_tasks,
        "num_layers": 2,
        "dropout": 0.5,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "num_epochs": 2,
    }
    print("Hyperparameters:", hyperparams)

    evaluator = Evaluator(name=dataset.name)
    criterion = nn.BCEWithLogitsLoss()

    test_aucs = []  # Report mean and std over multiple runs
    val_aucs = []  # Report mean and std over multiple runs
    for i in range(10):  # 10 runs
        print("\n--- New Run (Seed", i, ") ---")
        model = MolecularGINE(
            emb_dim=hyperparams["emb_dim"],
            out_dim=hyperparams["out_dim"],
            num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"],
        ).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )
        val_auc, test_auc = train_loop(
            model,
            device,
            train_loader,
            valid_loader,
            test_loader,
            optimizer,
            criterion,
            evaluator,
            hyperparams["num_epochs"],
        )
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)

        print(
            f"Run {i} - Best Val ROC-AUC: {val_auc:.4f}, Test ROC-AUC: {test_auc:.4f}"
        )

    print("\n--- BENCHMARK REPORT ---")
    print(
        f"\nFinal Results over 10 runs: \n"
        f"Validation ROC-AUC: {torch.tensor(val_aucs).mean():.4f} ± {torch.tensor(val_aucs).std():.4f}\n"
        f"Test ROC-AUC: {torch.tensor(test_aucs).mean():.4f} ± {torch.tensor(test_aucs).std():.4f}"
    )
    print("\nNumber of parameters:", get_gine_parameters(hyperparams, dataset))


if __name__ == "__main__":
    dataset, train_loader, valid_loader, test_loader = load_data()

    if sys.argv[1] == "gine":
        main_gine(dataset, train_loader, valid_loader, test_loader)
    elif sys.argv[1] == "csnn":
        print("CSNN main not implemented yet.")
    else:
        print("Unknown model type. Use 'gine' or 'csnn'.")
