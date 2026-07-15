"""Train a frozen-encoder Stack cell-set response classifier."""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from stack.classifier.dataset import (
    ClassifierFold,
    CellSetClassificationDataset,
    DEFAULT_LABELS,
    make_stratified_holdout_folds,
    read_sample_label_map,
)
from stack.classifier.model import HEAD_TYPES, StackCellSetClassifier

log = logging.getLogger("stack.classifier.train")


@dataclass(frozen=True)
class TrialConfig:
    trial_id: int
    head_type: str
    lr: float
    weight_decay: float
    sample_size: int
    max_samples: Optional[int]
    middle_dim: Optional[int]
    dropout: float


def parse_csv_values(value: Optional[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_float_values(value: Optional[str], fallback: Sequence[float]) -> tuple[float, ...]:
    parts = parse_csv_values(value)
    if not parts:
        return tuple(float(item) for item in fallback)
    return tuple(float(part) for part in parts)


def parse_optional_int_values(
    value: Optional[str],
    fallback: Sequence[Optional[int]],
) -> tuple[Optional[int], ...]:
    parts = parse_csv_values(value)
    if not parts:
        return tuple(fallback)

    parsed: list[Optional[int]] = []
    for part in parts:
        if part.lower() in {"none", "null", "all"}:
            parsed.append(None)
        else:
            parsed.append(int(part))
    return tuple(parsed)


def parse_sample_sizes(value: Optional[str], default_sample_size: int) -> tuple[int, ...]:
    parts = parse_csv_values(value)
    if not parts or (len(parts) == 1 and parts[0].lower() == "auto"):
        return (int(default_sample_size),)
    return tuple(int(part) for part in parts)


def parse_folds(value: str, n_folds: int) -> tuple[int, ...]:
    if value.lower() == "all":
        return tuple(range(n_folds))
    folds = tuple(int(part) for part in parse_csv_values(value))
    invalid = [fold for fold in folds if fold < 0 or fold >= n_folds]
    if invalid:
        raise ValueError(f"Fold indices must be in [0, {n_folds - 1}], got {invalid}")
    return folds


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _metadata_list(value, batch_size: int) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted] * batch_size
    return [value] * batch_size


def build_dataloader(
    *,
    adata_path: str,
    genelist_path: str,
    stack_model,
    gene_name_col: Optional[str],
    batch_size: int,
    num_workers: int,
    sample_ids: Sequence[str],
    split: str,
    label_col: str,
    groupby_col: str,
    labels: Sequence[str],
    sample_size: int,
    max_samples: Optional[int],
    random_seed: Optional[int],
    filter_organism: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = CellSetClassificationDataset(
        adata_path,
        genelist_path=genelist_path,
        split=split,
        sample_ids=sample_ids,
        label_col=label_col,
        groupby_col=groupby_col,
        labels=labels,
        sample_size=sample_size,
        max_samples=max_samples,
        gene_name_col=gene_name_col,
        filter_organism=filter_organism,
        random_state=random_seed,
    )
    generator = None
    if shuffle and random_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(random_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


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
        "n_cell_sets": total,
    }


@torch.no_grad()
def evaluate_cellsets(
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

        sample_ids = _metadata_list(metadata["sample_id"], batch_size)
        for idx in range(batch_size):
            target_idx = int(target[idx].item())
            pred_idx = int(predictions[idx].item())
            row: Dict[str, object] = {
                "sample_id": str(sample_ids[idx]),
                "true_label": labels[target_idx],
                "pred_label": labels[pred_idx],
            }
            for label_idx, label in enumerate(labels):
                row[f"prob_{label}"] = float(probabilities[idx, label_idx].item())
            rows.append(row)

    metrics = {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "n_cell_sets": total,
    }
    return metrics, rows


def build_patient_prediction_rows(
    cellset_rows: Iterable[Dict[str, object]],
    labels: Sequence[str],
) -> list[Dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in cellset_rows:
        sample_id = str(row["sample_id"])
        true_label = str(row["true_label"])
        probabilities = np.array([float(row[f"prob_{label}"]) for label in labels])
        if sample_id not in grouped:
            grouped[sample_id] = {
                "true_label": true_label,
                "probabilities": [],
            }
        elif grouped[sample_id]["true_label"] != true_label:
            raise ValueError(
                f"Conflicting labels for sample_id {sample_id}: "
                f"{grouped[sample_id]['true_label']} and {true_label}"
            )
        grouped[sample_id]["probabilities"].append(probabilities)

    patient_rows: list[Dict[str, object]] = []
    for sample_id, entry in grouped.items():
        stacked = np.stack(entry["probabilities"], axis=0)
        mean_probabilities = stacked.mean(axis=0)
        pred_idx = int(mean_probabilities.argmax())
        patient_row: Dict[str, object] = {
            "sample_id": sample_id,
            "true_label": entry["true_label"],
            "pred_label": labels[pred_idx],
            "n_cell_sets": int(stacked.shape[0]),
        }
        for label_idx, label in enumerate(labels):
            patient_row[f"prob_{label}"] = float(mean_probabilities[label_idx])
        patient_rows.append(patient_row)
    return patient_rows


def compute_classification_metrics(
    rows: Sequence[Dict[str, object]],
    labels: Sequence[str],
) -> Dict[str, object]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n_classes = len(labels)
    confusion = [[0 for _ in labels] for _ in labels]

    for row in rows:
        true_idx = label_to_idx[str(row["true_label"])]
        pred_idx = label_to_idx[str(row["pred_label"])]
        confusion[true_idx][pred_idx] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[idx][idx] for idx in range(n_classes))
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    for idx, label in enumerate(labels):
        tp = confusion[idx][idx]
        fn = sum(confusion[idx]) - tp
        fp = sum(confusion[row_idx][idx] for row_idx in range(n_classes)) - tp
        recall_denom = tp + fn
        f1_denom = 2 * tp + fp + fn
        per_class_recall[label] = tp / recall_denom if recall_denom else 0.0
        per_class_f1[label] = 2 * tp / f1_denom if f1_denom else 0.0

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean(list(per_class_f1.values()))) if labels else 0.0,
        "balanced_accuracy": (
            float(np.mean(list(per_class_recall.values()))) if labels else 0.0
        ),
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion,
        "n_patients": total,
    }


def evaluate(
    model: StackCellSetClassifier,
    dataloader: DataLoader,
    *,
    criterion: torch.nn.Module,
    device: torch.device,
    labels: Sequence[str],
) -> tuple[Dict[str, object], list[Dict[str, object]], list[Dict[str, object]]]:
    cellset_metrics, cellset_rows = evaluate_cellsets(
        model,
        dataloader,
        criterion=criterion,
        device=device,
        labels=labels,
    )
    patient_rows = build_patient_prediction_rows(cellset_rows, labels)
    patient_metrics = compute_classification_metrics(patient_rows, labels)
    metrics: Dict[str, object] = {
        "cellset": cellset_metrics,
        "patient": patient_metrics,
    }
    return metrics, cellset_rows, patient_rows


def save_prediction_rows(
    rows: Iterable[Dict[str, object]],
    output_path: Path,
    *,
    labels: Sequence[str],
) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "true_label", "pred_label"]
    if any("n_cell_sets" in row for row in rows):
        fieldnames.append("n_cell_sets")
    fieldnames.extend(f"prob_{label}" for label in labels)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(data, handle, indent=2)


def create_summary_writer(log_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, ModuleNotFoundError) as exc:
        log.warning("TensorBoard is unavailable: %s", exc)
        return None
    return SummaryWriter(log_dir=str(log_dir))


def log_eval_metrics(writer, prefix: str, metrics: Dict[str, object], step: int) -> None:
    if writer is None:
        return
    cellset = metrics["cellset"]
    patient = metrics["patient"]
    writer.add_scalar(f"{prefix}/cellset_loss", cellset["loss"], step)
    writer.add_scalar(f"{prefix}/cellset_accuracy", cellset["accuracy"], step)
    writer.add_scalar(f"{prefix}/patient_accuracy", patient["accuracy"], step)
    writer.add_scalar(f"{prefix}/patient_macro_f1", patient["macro_f1"], step)
    writer.add_scalar(
        f"{prefix}/patient_balanced_accuracy",
        patient["balanced_accuracy"],
        step,
    )
    for label, recall in patient["per_class_recall"].items():
        writer.add_scalar(f"{prefix}/recall/{label}", recall, step)


def iter_trial_configs(
    *,
    head_types: Sequence[str],
    lrs: Sequence[float],
    weight_decays: Sequence[float],
    sample_sizes: Sequence[int],
    max_samples_values: Sequence[Optional[int]],
    middle_dims: Sequence[Optional[int]],
    dropouts: Sequence[float],
) -> tuple[TrialConfig, ...]:
    configs: list[TrialConfig] = []
    trial_id = 0
    for head_type, lr, weight_decay, sample_size, max_samples, middle_dim, dropout in itertools.product(
        head_types,
        lrs,
        weight_decays,
        sample_sizes,
        max_samples_values,
        middle_dims,
        dropouts,
    ):
        if head_type not in HEAD_TYPES:
            raise ValueError(f"Unknown head_type '{head_type}'. Expected one of {HEAD_TYPES}")
        configs.append(
            TrialConfig(
                trial_id=trial_id,
                head_type=head_type,
                lr=float(lr),
                weight_decay=float(weight_decay),
                sample_size=int(sample_size),
                max_samples=max_samples,
                middle_dim=middle_dim,
                dropout=float(dropout),
            )
        )
        trial_id += 1
    return tuple(configs)


def build_split_summary(
    folds: Sequence[ClassifierFold],
    sample_labels: Dict[str, str],
    labels: Sequence[str],
) -> Dict[str, object]:
    test_ids = folds[0].test_sample_ids if folds else ()
    val_counts = {sample_id: 0 for sample_id in sample_labels}
    for fold in folds:
        for sample_id in fold.val_sample_ids:
            val_counts[sample_id] += 1

    label_counts = {
        label: sum(1 for value in sample_labels.values() if value == label)
        for label in labels
    }
    test_label_counts = {
        label: sum(1 for sample_id in test_ids if sample_labels[sample_id] == label)
        for label in labels
    }
    repeated_val_samples = {
        sample_id: count
        for sample_id, count in val_counts.items()
        if count > 1 and sample_id not in test_ids
    }
    never_val_samples = [
        sample_id
        for sample_id, count in val_counts.items()
        if count == 0 and sample_id not in test_ids
    ]

    return {
        "labels": list(labels),
        "label_counts": label_counts,
        "test_sample_ids": list(test_ids),
        "test_label_counts": test_label_counts,
        "folds": [asdict(fold) for fold in folds],
        "val_use_counts": val_counts,
        "repeated_val_samples": repeated_val_samples,
        "never_val_samples": never_val_samples,
    }


def run_training_fold(
    *,
    stack_model,
    config: TrialConfig,
    fold: ClassifierFold,
    args: argparse.Namespace,
    labels: Sequence[str],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    seed = args.random_seed + fold.fold * 1000 + config.trial_id
    torch.manual_seed(seed)

    run_dir = output_dir / f"fold_{fold.fold}" / f"trial_{config.trial_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = create_summary_writer(run_dir / "tensorboard", enabled=not args.no_tensorboard)

    common_loader_kwargs = {
        "adata_path": args.adata,
        "genelist_path": args.genelist,
        "stack_model": stack_model,
        "gene_name_col": args.gene_name_col,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "label_col": args.label_col,
        "groupby_col": args.groupby_col,
        "labels": labels,
        "sample_size": config.sample_size,
        "max_samples": config.max_samples,
        "random_seed": seed,
        "filter_organism": not args.no_filter_organism,
    }
    train_loader = build_dataloader(
        split="train",
        sample_ids=fold.train_sample_ids,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = build_dataloader(
        split="val",
        sample_ids=fold.val_sample_ids,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = build_dataloader(
        split="test",
        sample_ids=fold.test_sample_ids,
        shuffle=False,
        **common_loader_kwargs,
    )

    model = StackCellSetClassifier(
        stack_model,
        n_classes=len(labels),
        head_type=config.head_type,
        middle_dim=config.middle_dim,
        dropout=config.dropout,
        freeze_stack=True,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history: list[Dict[str, object]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.classifier.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_metrics, _val_cellset_rows, _val_patient_rows = evaluate(
            model,
            val_loader,
            criterion=criterion,
            device=device,
            labels=labels,
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/cellset_accuracy", train_metrics["accuracy"], epoch)
            log_eval_metrics(writer, "val", val_metrics, epoch)

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        val_loss = float(val_metrics["cellset"]["loss"])
        improved = val_loss < best_val_loss - args.min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.classifier.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log.info(
            "fold=%s trial=%s epoch=%s train_loss=%.4f val_loss=%.4f val_patient_acc=%.4f",
            fold.fold,
            config.trial_id,
            epoch,
            train_metrics["loss"],
            val_metrics["cellset"]["loss"],
            val_metrics["patient"]["accuracy"],
        )
        if epochs_without_improvement >= args.patience:
            break

    model.classifier.load_state_dict(best_state)
    val_metrics, val_cellset_rows, val_patient_rows = evaluate(
        model,
        val_loader,
        criterion=criterion,
        device=device,
        labels=labels,
    )
    test_metrics, test_cellset_rows, test_patient_rows = evaluate(
        model,
        test_loader,
        criterion=criterion,
        device=device,
        labels=labels,
    )
    log_eval_metrics(writer, "best_val", val_metrics, best_epoch)
    log_eval_metrics(writer, "test", test_metrics, best_epoch)
    if writer is not None:
        writer.close()

    checkpoint = {
        "classifier_state_dict": model.classifier.state_dict(),
        "labels": tuple(labels),
        "fold": asdict(fold),
        "config": asdict(config),
        "best_epoch": best_epoch,
        "stack_checkpoint": args.checkpoint,
    }
    torch.save(checkpoint, run_dir / "stack_cellset_classifier_head.pt")
    save_prediction_rows(val_cellset_rows, run_dir / "val_cellset_predictions.csv", labels=labels)
    save_prediction_rows(val_patient_rows, run_dir / "val_patient_predictions.csv", labels=labels)
    save_prediction_rows(test_cellset_rows, run_dir / "test_cellset_predictions.csv", labels=labels)
    save_prediction_rows(test_patient_rows, run_dir / "test_patient_predictions.csv", labels=labels)

    run_record: Dict[str, object] = {
        "fold": fold.fold,
        "trial_id": config.trial_id,
        "config": asdict(config),
        "split": asdict(fold),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_val": val_metrics,
        "test": test_metrics,
        "run_dir": str(run_dir),
    }
    write_json(history, run_dir / "history.json")
    write_json(run_record, run_dir / "metrics.json")
    return run_record


def aggregate_runs(run_records: Sequence[Dict[str, object]]) -> list[Dict[str, object]]:
    grouped: dict[int, list[Dict[str, object]]] = {}
    for record in run_records:
        grouped.setdefault(int(record["trial_id"]), []).append(record)

    aggregate: list[Dict[str, object]] = []
    for trial_id, records in sorted(grouped.items()):
        aggregate.append(
            {
                "trial_id": trial_id,
                "config": records[0]["config"],
                "n_folds": len(records),
                "mean_val_loss": float(
                    np.mean([record["best_val"]["cellset"]["loss"] for record in records])
                ),
                "mean_val_patient_accuracy": float(
                    np.mean([record["best_val"]["patient"]["accuracy"] for record in records])
                ),
                "mean_val_patient_macro_f1": float(
                    np.mean([record["best_val"]["patient"]["macro_f1"] for record in records])
                ),
                "mean_test_patient_accuracy": float(
                    np.mean([record["test"]["patient"]["accuracy"] for record in records])
                ),
                "mean_test_patient_macro_f1": float(
                    np.mean([record["test"]["patient"]["macro_f1"] for record in records])
                ),
                "mean_test_patient_balanced_accuracy": float(
                    np.mean(
                        [
                            record["test"]["patient"]["balanced_accuracy"]
                            for record in records
                        ]
                    )
                ),
            }
        )
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Stack for patient response classification")
    parser.add_argument("--checkpoint", required=True, help="Path to Stack checkpoint")
    parser.add_argument("--adata", required=True, help="Pre-biopsy AnnData .h5ad path")
    parser.add_argument("--genelist", required=True, help="Pickled Stack gene list")
    parser.add_argument("--gene-name-col", default="gene_name", help="Optional adata.var gene symbol column")
    parser.add_argument("--label-col", default="response", help="adata.obs response label column")
    parser.add_argument("--groupby-col", default="sample_id", help="adata.obs patient/sample column")
    parser.add_argument("--output-dir", default="outputs/classifier", help="Directory for classifier outputs")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience on val loss")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum val loss improvement")
    parser.add_argument("--batch-size", type=int, default=8, help="Cell-set batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fallback AdamW learning rate")
    parser.add_argument("--lrs", default="", help="Comma-separated learning-rate grid")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Fallback AdamW weight decay")
    parser.add_argument("--weight-decays", default="", help="Comma-separated weight-decay grid")
    parser.add_argument("--head-type", default="linear", choices=HEAD_TYPES, help="Fallback classifier head")
    parser.add_argument("--head-types", default="", help="Comma-separated classifier head grid")
    parser.add_argument("--middle-dims", default="", help="Comma-separated hidden dims; empty uses hidden_dim // 2")
    parser.add_argument("--dropouts", default="0.2", help="Comma-separated dropout grid")
    parser.add_argument("--sample-size", type=int, default=None, help="Fallback cells per cell set")
    parser.add_argument("--sample-sizes", default="", help="Comma-separated sample-size grid; empty uses model.n_cells")
    parser.add_argument("--max-samples", type=int, default=None, help="Fallback max cell sets per sample")
    parser.add_argument("--max-samples-list", default="", help="Comma-separated max cell sets per sample; use none for all")
    parser.add_argument("--n-folds", type=int, default=7, help="Number of validation folds")
    parser.add_argument("--folds", default="all", help="'all' or comma-separated fold indices")
    parser.add_argument("--test-per-class", type=int, default=2, help="Held-out test samples per class")
    parser.add_argument("--val-per-class", type=int, default=1, help="Validation samples per class per fold")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument(
        "--test-sample-ids",
        default="",
        help="Comma-separated fixed test sample_id values; empty means stratified auto-select",
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
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
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

    labels = parse_csv_values(args.labels)
    sample_labels = read_sample_label_map(
        args.adata,
        label_col=args.label_col,
        groupby_col=args.groupby_col,
    )
    unknown_labels = sorted({label for label in sample_labels.values() if label not in labels})
    if unknown_labels:
        raise ValueError(f"Found labels outside --labels {labels}: {unknown_labels}")

    test_sample_ids = parse_csv_values(args.test_sample_ids)
    all_folds = make_stratified_holdout_folds(
        sample_labels,
        labels=labels,
        test_per_class=args.test_per_class,
        n_folds=args.n_folds,
        val_per_class=args.val_per_class,
        random_state=args.random_seed,
        test_sample_ids=test_sample_ids or None,
    )
    selected_fold_indices = set(parse_folds(args.folds, args.n_folds))
    folds = tuple(fold for fold in all_folds if fold.fold in selected_fold_indices)
    split_summary = build_split_summary(all_folds, sample_labels, labels)
    write_json(split_summary, output_dir / "split_summary.json")
    if split_summary["repeated_val_samples"] or split_summary["never_val_samples"]:
        log.warning(
            "Validation folds are not a strict non-overlapping 7-fold CV. "
            "Repeated val samples: %s; never-val samples: %s",
            split_summary["repeated_val_samples"],
            split_summary["never_val_samples"],
        )

    from ..model_loading import load_model_from_checkpoint

    stack_model = load_model_from_checkpoint(args.checkpoint, device=device)
    default_sample_size = args.sample_size or int(stack_model.n_cells)
    sample_sizes = parse_sample_sizes(args.sample_sizes, default_sample_size)
    max_samples_fallback = (args.max_samples,) if args.max_samples is not None else (None,)
    configs = iter_trial_configs(
        head_types=parse_csv_values(args.head_types) or (args.head_type,),
        lrs=parse_float_values(args.lrs, (args.lr,)),
        weight_decays=parse_float_values(args.weight_decays, (args.weight_decay,)),
        sample_sizes=sample_sizes,
        max_samples_values=parse_optional_int_values(args.max_samples_list, max_samples_fallback),
        middle_dims=parse_optional_int_values(args.middle_dims, (None,)),
        dropouts=parse_float_values(args.dropouts, (0.2,)),
    )

    run_records: list[Dict[str, object]] = []
    for config in configs:
        for fold in folds:
            run_records.append(
                run_training_fold(
                    stack_model=stack_model,
                    config=config,
                    fold=fold,
                    args=args,
                    labels=labels,
                    device=device,
                    output_dir=output_dir,
                )
            )

    aggregate = aggregate_runs(run_records)
    best_by_val_loss = min(aggregate, key=lambda item: item["mean_val_loss"]) if aggregate else None
    best_by_val_macro_f1 = (
        max(aggregate, key=lambda item: item["mean_val_patient_macro_f1"])
        if aggregate
        else None
    )
    summary = {
        "runs": run_records,
        "aggregate_by_trial": aggregate,
        "best_by_mean_val_loss": best_by_val_loss,
        "best_by_mean_val_patient_macro_f1": best_by_val_macro_f1,
    }
    write_json(summary, output_dir / "runs_summary.json")


if __name__ == "__main__":
    main()
