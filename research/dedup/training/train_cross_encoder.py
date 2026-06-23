from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from research.dedup.model_registry import ModelManager, SKU_RERANKER_INSTRUCTION

from .common import (
    DEFAULT_MODEL_OUTPUT_ROOT,
    DEFAULT_SPLIT_DATA_PATH,
    binary_metrics_from_predictions,
    frame_to_pair_dataset,
    git_commit,
    package_versions,
    read_split_pairs,
    safe_slug,
    split_frame,
    split_label_counts,
    supported_kwargs,
)
from .data import write_json


DEFAULT_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformers CrossEncoder reranker.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_SPLIT_DATA_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--default-prompt-name")
    parser.add_argument("--sku-match-prompt", default=SKU_RERANKER_INSTRUCTION)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-peft-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="query,value",
        help="Comma-separated module names for PEFT LoRA. Use q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj for Qwen.",
    )
    parser.add_argument(
        "--no-merge-peft-lora",
        dest="merge_peft_lora",
        action="store_false",
        help="Save adapter-only PEFT weights instead of merging LoRA into the final CrossEncoder.",
    )
    parser.set_defaults(merge_peft_lora=True)
    parser.add_argument(
        "--pos-weight",
        type=float,
        help="BCE positive-class weight. Default: negatives / positives from train split.",
    )
    return parser


def _args_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _default_output_dir(model_name: str) -> Path:
    return DEFAULT_MODEL_OUTPUT_ROOT / f"{safe_slug(model_name)}_cross_encoder"


def _cross_encoder_imports() -> tuple[Any, Any, Any]:
    from sentence_transformers import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    try:
        from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
    except ImportError:
        from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments
    return CrossEncoderTrainer, CrossEncoderTrainingArguments, BinaryCrossEntropyLoss


def _prompt_options(args: argparse.Namespace) -> tuple[dict[str, str] | None, str | None]:
    if args.default_prompt_name:
        return {args.default_prompt_name: args.sku_match_prompt}, args.default_prompt_name
    return None, None


def _load_cross_encoder(args: argparse.Namespace) -> Any:
    prompts, default_prompt_name = _prompt_options(args)
    manager = ModelManager(
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        use_pool=False,
    )
    try:
        return manager.load_cross_encoder(
            args.model_name,
            device=args.device,
            trust_remote_code=args.trust_remote_code or None,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
            num_labels=1,
            max_length=args.max_length,
        )
    except KeyError:
        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {"num_labels": 1, "max_length": args.max_length}
        if args.device:
            kwargs["device"] = args.device
        if args.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if prompts is not None:
            kwargs["prompts"] = prompts
        if default_prompt_name is not None:
            kwargs["default_prompt_name"] = default_prompt_name
        if args.cache_dir is not None:
            kwargs["cache_folder"] = str(args.cache_dir)
        if args.local_files_only:
            kwargs["local_files_only"] = True
        return CrossEncoder(args.model_name, **kwargs)


def _build_dataset(frame: Any) -> Any:
    from datasets import Dataset

    data = frame_to_pair_dataset(frame)
    data["labels"] = data["labels"].astype(float)
    return Dataset.from_pandas(data, preserve_index=False)


def _training_arguments(args: argparse.Namespace, output_dir: Path, cross_encoder_training_args: Any) -> Any:
    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "dataloader_num_workers": args.dataloader_num_workers,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": 25,
        "logging_first_step": True,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
        "seed": args.seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    return cross_encoder_training_args(**supported_kwargs(cross_encoder_training_args.__init__, kwargs))


def _auto_pos_weight(train_frame: Any, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    positives = int(train_frame["same_base_product"].eq(1).sum())
    negatives = int(train_frame["same_base_product"].eq(0).sum())
    if positives <= 0:
        return 1.0
    return max(1.0, negatives / positives)


def _parse_lora_target_modules(value: str) -> list[str] | None:
    targets = [item.strip() for item in value.split(",") if item.strip()]
    return targets or None


def _transformer_module(cross_encoder: Any) -> Any:
    for module in cross_encoder:
        if hasattr(module, "model") and hasattr(module, "modality_config"):
            return module
    raise TypeError("CrossEncoder object does not expose a Transformer module; cannot apply PEFT LoRA")


def _base_model(cross_encoder: Any) -> Any:
    base = getattr(_transformer_module(cross_encoder), "model", None)
    if base is None:
        raise TypeError("CrossEncoder Transformer module does not expose .model; cannot apply PEFT LoRA")
    return base


def _set_base_model(cross_encoder: Any, base_model: Any) -> None:
    _transformer_module(cross_encoder).model = base_model


def _maybe_enable_gradient_checkpointing(cross_encoder: Any, args: argparse.Namespace) -> None:
    if not args.gradient_checkpointing:
        return
    base = _base_model(cross_encoder)
    config = getattr(base, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")


def _maybe_apply_lora(cross_encoder: Any, args: argparse.Namespace) -> Any:
    if not args.use_peft_lora:
        return cross_encoder
    from peft import LoraConfig, TaskType, get_peft_model

    base = _base_model(cross_encoder)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_parse_lora_target_modules(args.lora_target_modules),
        bias="none",
    )
    peft_model = get_peft_model(base, lora_config)
    _set_base_model(cross_encoder, peft_model)
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()
    return cross_encoder


def _maybe_merge_lora_for_export(cross_encoder: Any, args: argparse.Namespace) -> Any:
    if not args.use_peft_lora or not args.merge_peft_lora:
        return cross_encoder
    base = _base_model(cross_encoder)
    if not hasattr(base, "merge_and_unload"):
        print("LoRA merge skipped: PEFT model does not expose merge_and_unload()")
        return cross_encoder
    _set_base_model(cross_encoder, base.merge_and_unload())
    print("Merged LoRA adapter into final CrossEncoder model")
    return cross_encoder


def _predict_probabilities(model: Any, frame: Any, batch_size: int) -> Any:
    import numpy as np
    import torch

    pairs = list(zip(frame["sentence_A"].tolist(), frame["sentence_B"].tolist(), strict=False))
    try:
        raw_scores = model.predict(
            pairs,
            batch_size=batch_size,
            activation_fct=torch.nn.Sigmoid(),
            convert_to_numpy=True,
        )
    except TypeError:
        raw_scores = model.predict(pairs, batch_size=batch_size)
    return np.asarray(raw_scores, dtype=float).reshape(-1)


def _evaluate_split(model: Any, frame: Any, *, batch_size: int, threshold: float) -> dict[str, float]:
    if frame.empty:
        return {}
    scores = _predict_probabilities(model, frame, batch_size)
    predictions = (scores >= threshold).astype(int)
    metrics = binary_metrics_from_predictions(frame["same_base_product"].to_numpy(), predictions)
    metrics["score_min"] = float(scores.min()) if len(scores) else 0.0
    metrics["score_max"] = float(scores.max()) if len(scores) else 0.0
    metrics["score_mean"] = float(scores.mean()) if len(scores) else 0.0
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or _default_output_dir(args.model_name)
    frame = read_split_pairs(args.data_path, smoke_limit=args.smoke_limit)
    train_frame = split_frame(frame, "train")
    dev_frame = split_frame(frame, "dev")
    test_frame = split_frame(frame, "test")
    if train_frame.empty or dev_frame.empty:
        raise ValueError("split dataset must contain non-empty train and dev splits")

    pos_weight = _auto_pos_weight(train_frame, args.pos_weight)
    manifest_base = {
        "trainer": "sentence_transformers.CrossEncoderTrainer",
        "model_name": args.model_name,
        "data_path": str(args.data_path),
        "output_dir": str(output_dir),
        "score_type": "sigmoid_probability",
        "prediction_threshold": args.prediction_threshold,
        "pos_weight": pos_weight,
        "split_label_counts": split_label_counts(frame),
        "rows": {
            "total": int(len(frame)),
            "train": int(len(train_frame)),
            "dev": int(len(dev_frame)),
            "test": int(len(test_frame)),
        },
        "args": _args_payload(args),
        "git_commit": git_commit(),
        "dependencies": package_versions(["sentence-transformers", "transformers", "datasets", "accelerate", "torch"]),
    }

    if args.dry_run:
        write_json(output_dir / "dry_run_manifest.json", manifest_base)
        print(f"Dry run ok. Manifest: {output_dir / 'dry_run_manifest.json'}")
        return 0

    import torch

    cross_encoder_trainer, training_args_cls, bce_loss_cls = _cross_encoder_imports()
    model = _load_cross_encoder(args)
    _maybe_enable_gradient_checkpointing(model, args)
    model = _maybe_apply_lora(model, args)
    train_dataset = _build_dataset(train_frame)
    dev_dataset = _build_dataset(dev_frame)

    loss = bce_loss_cls(model=model, pos_weight=torch.tensor(pos_weight))
    trainer = cross_encoder_trainer(
        model=model,
        args=_training_arguments(args, output_dir, training_args_cls),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        loss=loss,
    )
    train_result = trainer.train()

    final_output_dir = output_dir / "final"
    model = _maybe_merge_lora_for_export(model, args)
    model.save_pretrained(str(final_output_dir))

    metrics = {
        "train_result": train_result.metrics,
        "train": _evaluate_split(
            model,
            train_frame,
            batch_size=args.per_device_eval_batch_size,
            threshold=args.prediction_threshold,
        ),
        "dev": _evaluate_split(
            model,
            dev_frame,
            batch_size=args.per_device_eval_batch_size,
            threshold=args.prediction_threshold,
        ),
    }
    if not test_frame.empty:
        metrics["test"] = _evaluate_split(
            model,
            test_frame,
            batch_size=args.per_device_eval_batch_size,
            threshold=args.prediction_threshold,
        )

    manifest = {
        **manifest_base,
        "final_output_dir": str(final_output_dir),
        "metrics": metrics,
    }
    write_json(output_dir / "training_manifest.json", manifest)
    write_json(final_output_dir / "training_manifest.json", manifest)
    print(f"Saved model: {final_output_dir}")
    print(f"Saved manifest: {output_dir / 'training_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
