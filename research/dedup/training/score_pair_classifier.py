from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import DEFAULT_SPLIT_DATA_PATH, package_versions, read_split_pairs, safe_slug
from .data import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a frozen dedup split with a sequence-classification pair model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_SPLIT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--score-column")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke-limit", type=int)
    return parser


def _load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    kwargs: dict[str, Any] = {}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.local_files_only:
        kwargs["local_files_only"] = True
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), **kwargs)
    if (args.model_path / "adapter_config.json").exists():
        from peft import AutoPeftModelForSequenceClassification

        model = AutoPeftModelForSequenceClassification.from_pretrained(str(args.model_path), **kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(args.model_path), **kwargs)
    return model, tokenizer


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _score_batches(frame: Any, model: Any, tokenizer: Any, *, device: str, max_length: int, batch_size: int) -> list[float]:
    import torch

    model.to(device)
    model.eval()
    scores: list[float] = []
    for offset in range(0, len(frame), batch_size):
        chunk = frame.iloc[offset : offset + batch_size]
        encoded = tokenizer(
            chunk["sentence_A"].tolist(),
            chunk["sentence_B"].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        if logits.shape[-1] == 1:
            probabilities = torch.sigmoid(logits.reshape(-1))
        else:
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
        scores.extend(float(value) for value in probabilities.detach().cpu().tolist())
    return scores


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = read_split_pairs(args.data_path, smoke_limit=args.smoke_limit)
    model, tokenizer = _load_model_and_tokenizer(args)
    device = _resolve_device(args.device)
    scores = _score_batches(
        frame,
        model,
        tokenizer,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    model_label = args.model_path.parent.name if args.model_path.name == "final" else args.model_path.name
    score_column = args.score_column or f"score_{safe_slug(model_label)}"
    output = frame.copy()
    output[score_column] = scores
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)

    manifest = {
        "scorer": "transformers.AutoModelForSequenceClassification",
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "output_path": str(args.output_path),
        "score_column": score_column,
        "score_type": "softmax_same_probability",
        "rows": int(len(output)),
        "device": device,
        "dependencies": package_versions(["transformers", "peft", "torch"]),
    }
    write_json(args.output_path.with_suffix(args.output_path.suffix + ".manifest.json"), manifest)
    print(f"Saved scores: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
