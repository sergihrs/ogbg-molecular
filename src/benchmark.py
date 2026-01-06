import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import load_data
from src.gine import MolecularGINE


def set_seed(seed: int):
    """
    Set torch's manual seed and deterministic behaviour for reproducibility.
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_step(model, device, loader, optimizer, lr_scheduler, criterion):
    model.train()
    total_loss = 0

    for _, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.to(torch.float32)

        # Handle missing Nans in labels
        is_labeled = y == y
        loss = criterion(pred[is_labeled], y[is_labeled])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Step the learning rate scheduler after each epoch
    lr_scheduler.step()

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
    lr_scheduler,
    criterion,
    evaluator,
    max_epochs: int,
    patience: int,
    plot_path: str,
) -> tuple[float, float]:
    tqdm_range = tqdm(range(max_epochs), unit="epoch")
    best_model_epoch = 0
    best_train_auc = 0
    best_val_auc = 0
    final_test_auc = 0
    patience_counter = 0
    best_model_state = None

    # Track metrics for plotting
    train_aucs = []
    val_aucs = []
    test_aucs = []

    for epoch in tqdm_range:
        # Train
        loss = train_step(
            model, device, train_loader, optimizer, lr_scheduler, criterion
        )

        # Evaluate
        train_result = eval_step(model, device, train_loader, evaluator)
        val_result = eval_step(model, device, valid_loader, evaluator)
        test_result = eval_step(model, device, test_loader, evaluator)

        train_auc = train_result["rocauc"]
        val_auc = val_result["rocauc"]
        test_auc = test_result["rocauc"]

        # Store metrics
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)

        # Checkpoint
        if val_auc > best_val_auc and train_auc >= best_train_auc:
            best_train_auc = train_auc
            best_val_auc = val_auc
            final_test_auc = test_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            best_model_epoch = epoch
        else:
            patience_counter += 1

        tqdm_range.set_postfix(
            {
                "loss": f"{loss:.4f}",
                "train": f"{train_auc:.4f}",
                "val": f"{val_auc:.4f}",
                "test": f"{test_auc:.4f}",
                "patience": f"{patience_counter}/{patience}",
            }
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # Restore best model
    if best_model_state is not None:
        print(f"Restoring best model from epoch {best_model_epoch}.")
        model.load_state_dict(best_model_state)

    # Save plot
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_aucs) + 1)
    plt.plot(epochs_range, train_aucs, label="Train", marker="o", markersize=3)
    plt.plot(epochs_range, val_aucs, label="Validation", marker="s", markersize=3)
    plt.plot(epochs_range, test_aucs, label="Test", marker="^", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("Train/Validation/Test ROC-AUC vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return best_val_auc, final_test_auc


def get_num_parameters(hyperparams: dict, num_tasks: int):
    "Return the number of trainable parameters of MolecularGINE for the given hyperparameters"
    model = MolecularGINE(
        emb_dim=hyperparams["emb_dim"],
        out_dim=num_tasks,
        num_layers=hyperparams["num_layers"],
        dropout=hyperparams["dropout"],
        mlp_num_layers=hyperparams["mlp_num_layers"],
        jumping_knowledge=hyperparams["jumping_knowledge"],
        use_edge_features=hyperparams["use_edge_features"],
        encoder_type=hyperparams["encoder_type"],
    )

    # Print model architecture and number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def main(hyperparams: dict):
    print("\n--- Benchmarking GINE Model ---")
    # CPU is faster for sequential processing of graphs when deterministic
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Print hyperparameters
    print("Hyperparameters:", hyperparams)

    dataset, *loaders, evaluator = load_data(
        dataset_name=hyperparams["dataset_name"], batch_size=hyperparams["batch_size"]
    )
    train_loader, valid_loader, test_loader = loaders

    # Print number of trainable parameters
    trainable_params = get_num_parameters(hyperparams, dataset.num_tasks)
    print(f"Number of trainable parameters: {trainable_params}")

    test_aucs, val_aucs = [], []  # Report mean and std over multiple runs

    # Create directories for saving models, plots, and reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"models/{timestamp}", exist_ok=True)
    os.makedirs(f"plots/{timestamp}", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    for i in range(hyperparams["runs"]):
        print("\n--- New Run (Seed", i, ") ---")
        # Different seed each run for variability
        set_seed(i)
        # Create new model each run
        model = MolecularGINE(
            emb_dim=hyperparams["emb_dim"],
            out_dim=dataset.num_tasks,
            num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"],
            mlp_num_layers=hyperparams["mlp_num_layers"],
            jumping_knowledge=hyperparams["jumping_knowledge"],
            use_edge_features=hyperparams["use_edge_features"],
            encoder_type=hyperparams["encoder_type"],
        ).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )
        # Use MultiStepLR to allow for more flexible LR scheduling
        num_lr_updates = (
            1
            if hyperparams["lr_update_once"]
            else hyperparams["max_epochs"] // hyperparams["lr_step_size"]
        )
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                hyperparams["lr_step_size"] * i for i in range(1, num_lr_updates + 1)
            ],
            gamma=hyperparams["lr_gamma"],
        )
        criterion = nn.BCEWithLogitsLoss()
        plot_path = f"plots/{timestamp}/run_{i}.png"
        val_auc, test_auc = train_loop(
            model,
            device,
            train_loader,
            valid_loader,
            test_loader,
            optimizer,
            lr_scheduler,
            criterion,
            evaluator,
            hyperparams["max_epochs"],
            hyperparams["patience"],
            plot_path,
        )
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)

        # Save the best model of the run
        model_path = f"models/{timestamp}/run_{i}_best_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Best model of Run {i} saved to {model_path}")

        print(
            f"Run {i} - Best Val ROC-AUC: {val_auc:.4f}, Test ROC-AUC: {test_auc:.4f}"
        )

    # Generate final report
    report_path = f"reports/report_{timestamp}.txt"
    with open(report_path, "w") as report_file:
        report_file.write("--- BENCHMARK REPORT ---\n")
        report_file.write(
            f"\nFinal Results over {hyperparams['runs']} runs:\n"
            f"Validation ROC-AUC: {torch.tensor(val_aucs).mean():.6f} ± {torch.tensor(val_aucs).std():.6f}\n"
            f"Test ROC-AUC: {torch.tensor(test_aucs).mean():.6f} ± {torch.tensor(test_aucs).std():.6f}\n"
        )
        report_file.write(f"\nNumber of parameters: {trainable_params}\n")
        report_file.write(f"\nHyperparameters: {hyperparams}\n")

        # Individual run results
        report_file.write("\nIndividual Run Results:\n")
        for i, (val_auc, test_auc) in enumerate(zip(val_aucs, test_aucs)):
            report_file.write(
                f"Run {i} - Best Val ROC-AUC: {val_auc:.6f}, Test ROC-AUC: {test_auc:.6f}\n"
            )
    print(f"\nFinal report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking GINE Model")

    # Dataset and Infrastructure
    parser.add_argument("--dataset_name", type=str, default="ogbg-molhiv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of runs with different seeds"
    )

    # Model Architecture
    parser.add_argument(
        "--encoder_type", type=str, default="he", choices=["he", "default"]
    )
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--mlp_num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--jumping_knowledge", action="store_true", default=False)
    parser.add_argument("--use_edge_features", action="store_true", default=False)

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--lr_update_once", action="store_true", default=False)

    args = parser.parse_args()
    hyperparams = vars(args)

    main(hyperparams)
