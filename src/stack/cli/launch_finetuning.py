#!/usr/bin/env python3
"""CLI script for fine-tuning the StateICL model with a frozen teacher."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict
from pathlib import Path

print(sys.executable)
print(sys.version)

if __package__ in {None, ""}:  # Support running this file directly from an IDE debugger.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:  # pragma: no cover - runtime import resolution
    from stack.cli_utils import apply_config_from_file, filter_unused_arguments
except ImportError:  # pragma: no cover - script execution fallback
    from cli_utils import apply_config_from_file, filter_unused_arguments  # type: ignore

# Backwards compatibility alias --------------------------------------------------------------------
def _override_model_config_n_cells(*args, **kwargs):
    """Lazily dispatch to override_model_config_n_cells for callers importing this alias."""

    deps = _import_training_modules()
    return deps["override_model_config_n_cells"](*args, **kwargs)


def _import_training_modules():
    """Lazy import heavy dependencies so ``--help`` stays fast."""

    try:
        import torch  # type: ignore
        import pytorch_lightning as pl  # type: ignore
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
        from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger
        from pytorch_lightning.strategies import DDPStrategy

        from stack.finetune.datamodule import FinetuneDataModule, MultiDatasetDataModule
        from stack.finetune.lightning import LightningFinetunedModel
        from stack.finetune.utils import (
            build_model_config,
            build_scheduler_config,
            configure_logger,
            override_model_config_n_cells,
            parse_dataset_configs,
        )
    except ImportError:  # pragma: no cover - script execution fallback
        import torch  # type: ignore
        import pytorch_lightning as pl  # type: ignore
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint  # type: ignore
        from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger  # type: ignore
        from pytorch_lightning.strategies import DDPStrategy  # type: ignore

        from finetune.datamodule import FinetuneDataModule, MultiDatasetDataModule  # type: ignore
        from finetune.lightning import LightningFinetunedModel  # type: ignore
        from finetune.utils import (  # type: ignore
            build_model_config,
            build_scheduler_config,
            configure_logger,
            override_model_config_n_cells,
            parse_dataset_configs,
        )

    torch.set_float32_matmul_precision("high")
    return {
        "torch": torch,
        "pl": pl,
        "EarlyStopping": EarlyStopping,
        "LearningRateMonitor": LearningRateMonitor,
        "ModelCheckpoint": ModelCheckpoint,
        "Logger": Logger,
        "TensorBoardLogger": TensorBoardLogger,
        "WandbLogger": WandbLogger,
        "DDPStrategy": DDPStrategy,
        "FinetuneDataModule": FinetuneDataModule,
        "MultiDatasetDataModule": MultiDatasetDataModule,
        "LightningFinetunedModel": LightningFinetunedModel,
        "build_model_config": build_model_config,
        "build_scheduler_config": build_scheduler_config,
        "configure_logger": configure_logger,
        "override_model_config_n_cells": override_model_config_n_cells,
        "parse_dataset_configs": parse_dataset_configs,
    }


def configure_callbacks(args: argparse.Namespace,
                        model_checkpoint_cls,
                        early_stopping_cls,
                        lr_monitor_cls,
                        ):
    """Return the default callback suite used for fine-tuning."""
    return [
        model_checkpoint_cls(
            dirpath=args.save_dir,
            filename="finetuned-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        early_stopping_cls(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode="min",
            verbose=True,
        ),
        lr_monitor_cls(logging_interval="epoch"),
    ]


def build_parser(parents=None) -> argparse.ArgumentParser:
    parent_parsers = list(parents) if parents else []
    parser = argparse.ArgumentParser(
        description="Fine-tune the StateICL model with a frozen teacher",
        parents=parent_parsers,
    )

    # Checkpoint arguments
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to pre-trained model checkpoint")

    # Data arguments
    parser.add_argument("--dataset_configs", type=str, nargs="+", default=None, help="Dataset configuration strings")
    parser.add_argument("--genelist_path", type=str, default=None, help="Path to saved gene list file (.pkl)")
    parser.add_argument("--sample_size", type=int, default=128, help="Number of cells per sample")
    parser.add_argument(
        "--replacement_ratio",
        type=float,
        default=0.25,
        help="Fraction of cells to replace (should match 1 - n_kept_cell/sample_size)",
    )
    parser.add_argument(
        "--intra_file_replacement_prob",
        type=float,
        default=1.0,
        help="Probability of using file-level replacement first",
    )
    parser.add_argument("--min_cells_per_group", type=int, default=128, help="Minimum cells per group")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument(
        "--split_strategy",
        type=str,
        choices=["random", "lopo"],
        default="random",
        help="Group split strategy. Use 'lopo' for leave-one-patient-out paired datasets.",
    )
    parser.add_argument("--fold_index", type=int, default=0, help="LOPO fold index used when split_strategy='lopo'")
    parser.add_argument(
        "--val_fold_offset",
        type=int,
        default=1,
        help="Validation patient offset from the held-out test patient for LOPO splits",
    )
    parser.add_argument(
        "--paired_sampling_method",
        type=str,
        choices=[
            "method1_capacity_strict",
            "method2_capacity_repeat_pre",
            "method3_post_only_strict",
            "method4_post_only_repeat_pre",
        ],
        default="method1_capacity_strict",
        help="Paired pre/post sampling method for NSCLC fine-tuning datasets",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cache_file", type=str, default=None, help="Path to metadata cache file")
    parser.add_argument("--max_memory_gb", type=float, default=None, help="Maximum memory usage in GB")
    parser.add_argument("--cache_ratio", type=float, default=0.1, help="Fraction of memory to use for caching")
    parser.add_argument("--block_size", type=int, default=1000, help="Number of rows per cache block")

    # Model arguments
    parser.add_argument("--n_hidden", type=int, default=100, help="Hidden dimension after gene reduction")
    parser.add_argument("--token_dim", type=int, default=16, help="Token dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of attention layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--mask_rate_min", type=float, default=0.1, help="Minimum mask rate")
    parser.add_argument("--mask_rate_max", type=float, default=0.8, help="Maximum mask rate")
    parser.add_argument("--sw_weight", type=float, default=0.1, help="Sliced Wasserstein weight")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--finetune_strategy",
        type=str,
        choices=["stage1", "full"],
        default="stage1",
        help="Student parameter update strategy. stage1 trains only query/cls/decoder-tail parameters.",
    )
    parser.add_argument("--head_lr", type=float, default=1e-4, help="Learning rate for stage1 query and cls parameters")
    parser.add_argument("--decoder_lr", type=float, default=1e-5, help="Learning rate for stage1 decoder-tail parameters")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--resample_each_epoch", action="store_true", help="Resample training data each epoch")

    # Scheduler arguments
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "reduce_on_plateau"],
        default="cosine",
        help="Learning rate scheduler",
    )
    parser.add_argument("--scheduler_T_max", type=int, default=20, help="T_max for cosine scheduler")
    parser.add_argument(
        "--scheduler_warmup_epochs",
        type=int,
        default=0,
        help="Warmup epochs for cosine scheduler",
    )
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6, help="Minimum LR for cosine scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=10, help="Patience for reduce_on_plateau")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Factor for reduce_on_plateau")

    # System arguments
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs (-1 for all available)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["ddp", "ddp_find_unused_parameters_false"],
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")

    # Logging and checkpointing
    parser.add_argument("--project_name", type=str, default="SHIFTAB Finetuning with Frozen Teacher")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for logging")
    parser.add_argument("--save_dir", type=str, default="../shiftab_finetuned_checkpoints")
    parser.add_argument("--logger", type=str, choices=["wandb", "tensorboard"], default="tensorboard")
    parser.add_argument("--log_every_n_steps", type=int, default=50)

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)

    return parser


def log_configuration(args: argparse.Namespace, model_config: Dict[str, any], data_module: FinetuneDataModule, logger: Logger) -> None:
    logging.info("=" * 80)
    logging.info("FINE-TUNING CONFIGURATION WITH FROZEN TEACHER")
    logging.info("=" * 80)
    logging.info("Checkpoint: %s", args.checkpoint_path)
    logging.info("Model configuration:")
    for key, value in model_config.items():
        logging.info("  %s: %s", key, value)
    logging.info("Training setup:")
    logging.info("  Using Frozen Teacher architecture for stable targets")
    logging.info("Dataset configurations:")
    for idx, config in enumerate(data_module.dataset_configs):
        logging.info("  Dataset %s: %s - %s", idx + 1, config.type, config.path)
        if config.type == "human":
            logging.info("    Donor col: %s, Cell type col: %s", config.donor_col, config.cell_type_col)
        elif config.type == "drug":
            logging.info(
                "    Condition col: %s, Cell line col: %s, Control condition: %s",
                config.condition_col,
                config.cell_line_col,
                config.control_condition,
            )
        elif config.type == "paired":
            logging.info(
                "    Patient col: %s, Timepoint col: %s, Cell type col: %s, Pre/Post: %s/%s",
                config.patient_col,
                config.timepoint_col,
                config.cell_type_col,
                config.pre_condition,
                config.post_condition,
            )
        logging.info("    Filter organism: %s", config.filter_organism)
    logging.info("Gene list: %s (%s genes)", args.genelist_path, data_module.n_genes)
    logging.info("Sample size: %s", args.sample_size)
    logging.info("Split strategy: %s (fold=%s, val_offset=%s)", args.split_strategy, args.fold_index, args.val_fold_offset)
    logging.info("Paired sampling method: %s", args.paired_sampling_method)
    logging.info("N kept cells (Student): %s", int((1.0 - args.replacement_ratio) * args.sample_size))
    logging.info("Batch size: %s", args.batch_size)
    logging.info("Fine-tune strategy: %s", args.finetune_strategy)
    logging.info("Learning rate: %s", args.learning_rate)
    logging.info("Head learning rate: %s", args.head_lr)
    logging.info("Decoder learning rate: %s", args.decoder_lr)
    logging.info("Max epochs: %s", args.max_epochs)
    logging.info("Save dir: %s", args.save_dir)
    logging.info("Run name: %s", args.run_name)
    logging.info("Logger: %s", type(logger).__name__)
    logging.info("=" * 80)


def validate_model_gene_dimension(model_config: Dict[str, any], data_n_genes: int) -> None:
    """Ensure checkpoint gene dimension matches the prepared dataset."""
    model_n_genes = model_config.get("n_genes")
    if model_n_genes is None:
        raise ValueError("Checkpoint model_config is missing required n_genes")
    if int(model_n_genes) != int(data_n_genes):
        raise ValueError(
            "Checkpoint n_genes does not match fine-tuning data: "
            f"checkpoint={model_n_genes}, data={data_n_genes}. "
            "Use the checkpoint's gene list or rebuild a compatible checkpoint."
        )


def prepare_lopo_output_paths(args: argparse.Namespace) -> None:
    """Keep LOPO fold artifacts separated and named by fold/seed."""
    if args.split_strategy != "lopo":
        return

    fold_component = f"fold_{args.fold_index}"
    normalized_save_dir = os.path.normpath(args.save_dir)
    if os.path.basename(normalized_save_dir) == fold_component:
        args.save_dir = normalized_save_dir
    else:
        args.save_dir = os.path.join(args.save_dir, fold_component)

    run_component = f"{fold_component}_seed_{args.random_seed}"
    if args.run_name:
        if run_component not in args.run_name:
            args.run_name = f"{args.run_name}_{run_component}"
    else:
        args.run_name = run_component


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    # config_parser.add_argument("--config", type=str, default=None, help="Path to a YAML/JSON config file")
    config_parser.add_argument("--config", type=str, default="/data/home/zhangyaojie/Lung_stack/configs/finetuning/ft_parsecg.yaml")

    parser = build_parser(parents=[config_parser])

    config_args, remaining_argv = config_parser.parse_known_args()
    apply_config_from_file(parser, config_args.config)
    args = parser.parse_args(remaining_argv)
    filter_unused_arguments(args, ("dataset_configs", "genelist_path"), parser)
    prepare_lopo_output_paths(args)

    deps = _import_training_modules()
    torch = deps["torch"]
    pl = deps["pl"]
    EarlyStopping = deps["EarlyStopping"]
    LearningRateMonitor = deps["LearningRateMonitor"]
    ModelCheckpoint = deps["ModelCheckpoint"]
    Logger = deps["Logger"]
    TensorBoardLogger = deps["TensorBoardLogger"]
    WandbLogger = deps["WandbLogger"]
    DDPStrategy = deps["DDPStrategy"]
    FinetuneDataModule = deps["FinetuneDataModule"]
    MultiDatasetDataModule = deps["MultiDatasetDataModule"]
    LightningFinetunedModel = deps["LightningFinetunedModel"]
    build_model_config = deps["build_model_config"]
    build_scheduler_config = deps["build_scheduler_config"]
    configure_logger = deps["configure_logger"]
    override_model_config_n_cells = deps["override_model_config_n_cells"]
    parse_dataset_configs = deps["parse_dataset_configs"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if config_args.config:
        logging.info("Loaded fine-tuning defaults from %s", config_args.config)

    dataset_configs = parse_dataset_configs(args.dataset_configs)
    scheduler_cfg = build_scheduler_config(vars(args))

    data_module = MultiDatasetDataModule(
        dataset_configs=dataset_configs,
        genelist_path=args.genelist_path,
        sample_size=args.sample_size,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        split_strategy=args.split_strategy,
        fold_index=args.fold_index,
        val_fold_offset=args.val_fold_offset,
        paired_sampling_method=args.paired_sampling_method,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.random_seed,
        cache_file=args.cache_file,
        replacement_ratio=args.replacement_ratio,
        intra_file_replacement_prob=args.intra_file_replacement_prob,
        min_cells_per_group=args.min_cells_per_group,
        max_memory_gb=args.max_memory_gb,
        cache_ratio=args.cache_ratio,
        block_size=args.block_size,
        resample_each_epoch=args.resample_each_epoch,
    )

    # Setup data module to get dataset info
    data_module.setup()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Save dataset split information
    split_info = data_module.get_split_info()
    split_info_path = os.path.join(args.save_dir, "dataset_splits.json")
    try:
        with open(split_info_path, "w") as f:
            json.dump(split_info, f, indent=4)
        logging.info("Saved dataset split information to %s", split_info_path)
    except Exception as exc:
        logging.error("Failed to save dataset split information: %s", exc)

    # Calculate n_kept_cell from replacement_ratio
    n_kept_cell = int((1.0 - args.replacement_ratio) * args.sample_size)

    # Initialize model - either from checkpoint or from scratch
    if args.checkpoint_path:
        logging.info("Loading model from checkpoint: %s", args.checkpoint_path)
        model_config = override_model_config_n_cells(args.checkpoint_path, args.sample_size)
        validate_model_gene_dimension(model_config, data_module.n_genes)

        student_model = LightningFinetunedModel.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            strict=False,
            model_config=model_config,
            learning_rate=args.learning_rate,
            head_lr=args.head_lr,
            decoder_lr=args.decoder_lr,
            weight_decay=args.weight_decay,
            scheduler_config=scheduler_cfg,
            n_kept_cell=n_kept_cell,
            finetune_strategy=args.finetune_strategy,
        )
    else:
        logging.info("Creating model from scratch (no checkpoint provided)")
        model_config = build_model_config(vars(args), data_module.n_genes)

        student_model = LightningFinetunedModel(
            model_config=model_config,
            checkpoint_path=None,
            learning_rate=args.learning_rate,
            head_lr=args.head_lr,
            decoder_lr=args.decoder_lr,
            weight_decay=args.weight_decay,
            scheduler_config=scheduler_cfg,
            n_kept_cell=n_kept_cell,
            finetune_strategy=args.finetune_strategy,
        )

    logger = configure_logger(
        logger_type=args.logger,
        project_name=args.project_name,
        run_name=args.run_name,
        save_dir=args.save_dir,
    )
    callbacks = configure_callbacks(args, ModelCheckpoint, EarlyStopping, LearningRateMonitor)

    strategy = DDPStrategy(find_unused_parameters=False) if args.strategy == "ddp" else args.strategy
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=strategy,
        precision=args.precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
        deterministic=False,
        benchmark=True,
    )

    log_configuration(args, model_config, data_module, logger)

    trainer.fit(student_model, datamodule=data_module)

    if data_module.test_dataset is not None and len(data_module.test_dataset) > 0:
        logging.info("Running test evaluation with frozen teacher...")
        test_results = trainer.test(student_model, datamodule=data_module, ckpt_path="best")
        logging.info("Test results: %s", test_results)

    logging.info("Fine-tuning pipeline completed successfully!")


if __name__ == "__main__":
    main()
