"""Train a Stack cell-set response classifier."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from stack.classifier.dataset import CellSetClassificationDataset, DEFAULT_LABELS, DEFAULT_TEST_SAMPLE_IDS
from stack.classifier.model import StackCellSetClassifier

log = logging.getLogger("stack.classifier.train")


def parse_csv_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_dataloaders(
    *,
    adata_path: str,
    genelist_path: str,
    stack_model,
    gene_name_col: Optional[str],
    batch_size: int,
    num_workers: int,
    test_sample_ids: Sequence[str],
    labels: Sequence[str],
    random_seed: Optional[int],
    filter_organism: bool,
):
    dataset_kwargs = {
        "genelist_path": genelist_path,
        "sample_size": stack_model.n_cells,
        "test_sample_ids": test_sample_ids,
        "labels": labels,
        "gene_name_col": gene_name_col,
        "filter_organism": filter_organism,
        "random_state": random_seed,
    }
    train_dataset = CellSetClassificationDataset(
        adata_path,
        split="train",
        **dataset_kwargs,
    )
    test_dataset = CellSetClassificationDataset(
        adata_path,
        split="test",
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: StackCellSetClassifier,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for features, labels, _metadata in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)["logits"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        loss_sum += loss.item() * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_size

    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "n_samples": float(total),
    }


@torch.no_grad()
def evaluate(
    model: StackCellSetClassifier,
    dataloader: DataLoader,
    *,
    criterion: torch.nn.Module,
    device: torch.device,
    labels: Sequence[str],
) -> tuple[Dict[str, float], list[Dict[str, object]]]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    rows: list[Dict[str, object]] = []

    for features, target, metadata in dataloader:
        features = features.to(device)
        target = target.to(device)
        logits = model(features)["logits"]
        loss = criterion(logits, target)
        probabilities = logits.softmax(dim=1)
        predictions = logits.argmax(dim=1)

        batch_size = target.numel()
        loss_sum += loss.item() * batch_size
        correct += int((predictions == target).sum().item())
        total += batch_size

        sample_ids = metadata["sample_id"]
        if not isinstance(sample_ids, list):
            sample_ids = list(sample_ids)
        for idx in range(batch_size):
            rows.append(
                {
                    "sample_id": sample_ids[idx],
                    "true_label": labels[int(target[idx].item())],
                    "pred_label": labels[int(predictions[idx].item())],
                    "prob_NMPR": float(probabilities[idx, 0].item()),
                    "prob_MPR": float(probabilities[idx, 1].item()),
                    "prob_pCR": float(probabilities[idx, 2].item()),
                }
            )

    metrics = {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "n_samples": float(total),
    }
    return metrics, rows


def save_prediction_rows(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "true_label", "pred_label", "prob_NMPR", "prob_MPR", "prob_pCR"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Stack for cell-set response classification")
    parser.add_argument("--checkpoint", required=True, help="Path to Stack checkpoint")
    parser.add_argument("--adata", required=True, help="Pre-biopsy AnnData .h5ad path")
    parser.add_argument("--genelist", required=True, help="Pickled Stack gene list")
    parser.add_argument("--gene-name-col", default="gene_symbols", help="Optional adata.var gene symbol column")
    parser.add_argument("--output-dir", default="outputs/classifier", help="Directory for classifier outputs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Cell-set batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-5, help="AdamW learning rate for full fine-tuning")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument(
        "--test-sample-ids",
        default=",".join(DEFAULT_TEST_SAMPLE_IDS),
        help="Comma-separated sample_id values reserved for test",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated labels in class-index order",
    )
    parser.add_argument(
        "--no-filter-organism",
        action="store_true",
        help="Disable Homo sapiens filtering in the dataset",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    torch.manual_seed(args.random_seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_sample_ids = parse_csv_values(args.test_sample_ids)
    labels = parse_csv_values(args.labels)
    from ..model_loading import load_model_from_checkpoint

    stack_model = load_model_from_checkpoint(args.checkpoint, device=device)
    model = StackCellSetClassifier(stack_model, n_classes=len(labels)).to(device)

    train_loader, test_loader = build_dataloaders(
        adata_path=args.adata,
        genelist_path=args.genelist,
        stack_model=stack_model,
        gene_name_col=args.gene_name_col,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_sample_ids=test_sample_ids,
        labels=labels,
        random_seed=args.random_seed,
        filter_organism=not args.no_filter_organism,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        test_metrics, prediction_rows = evaluate(
            model,
            test_loader,
            criterion=criterion,
            device=device,
            labels=labels,
        )
        record = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        history.append(record)
        log.info(
            "epoch=%s train_loss=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            test_metrics["loss"],
            test_metrics["accuracy"],
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": labels,
            "test_sample_ids": test_sample_ids,
            "args": vars(args),
        },
        output_dir / "stack_cellset_classifier.pt",
    )
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(history, handle, indent=2)
    save_prediction_rows(prediction_rows, output_dir / "test_predictions.csv")


if __name__ == "__main__":
    main()
