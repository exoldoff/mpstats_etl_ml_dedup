from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from research.dedup.model_registry import SKU_RERANKER_INSTRUCTION

from .common import DEFAULT_SPLIT_DATA_PATH, package_versions, read_split_pairs, safe_slug
from .data import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a frozen dedup split with a SentenceTransformers CrossEncoder.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_SPLIT_DATA_PATH)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--score-column")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--default-prompt-name")
    parser.add_argument("--sku-match-prompt", default=SKU_RERANKER_INSTRUCTION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--activation", choices=["sigmoid", "identity"], default="sigmoid")
    return parser


def _prompt_options(args: argparse.Namespace) -> tuple[dict[str, str] | None, str | None]:
    if args.default_prompt_name:
        return {args.default_prompt_name: args.sku_match_prompt}, args.default_prompt_name
    return None, None


def _load_model(args: argparse.Namespace) -> Any:
    from sentence_transformers import CrossEncoder

    prompts, default_prompt_name = _prompt_options(args)
    kwargs: dict[str, Any] = {}
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
    return CrossEncoder(str(args.model_path), **kwargs)


def _score(model: Any, frame: Any, *, batch_size: int, activation: str) -> list[float]:
    import numpy as np
    import torch

    activation_fct = torch.nn.Sigmoid() if activation == "sigmoid" else torch.nn.Identity()
    pairs = list(zip(frame["sentence_A"].tolist(), frame["sentence_B"].tolist(), strict=False))
    try:
        raw_scores = model.predict(
            pairs,
            batch_size=batch_size,
            activation_fct=activation_fct,
            convert_to_numpy=True,
        )
    except TypeError:
        raw_scores = model.predict(pairs, batch_size=batch_size)
    return [float(value) for value in np.asarray(raw_scores, dtype=float).reshape(-1)]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = read_split_pairs(args.data_path, smoke_limit=args.smoke_limit)
    model = _load_model(args)
    scores = _score(model, frame, batch_size=args.batch_size, activation=args.activation)

    model_path = Path(str(args.model_path))
    model_name = model_path.parent.name if model_path.name == "final" else model_path.name
    score_column = args.score_column or f"score_{safe_slug(model_name)}"
    output = frame.copy()
    output[score_column] = scores
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)

    manifest = {
        "scorer": "sentence_transformers.CrossEncoder",
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "output_path": str(args.output_path),
        "score_column": score_column,
        "score_type": f"{args.activation}_cross_encoder_score",
        "rows": int(len(output)),
        "dependencies": package_versions(["sentence-transformers", "transformers", "torch"]),
    }
    write_json(args.output_path.with_suffix(args.output_path.suffix + ".manifest.json"), manifest)
    print(f"Saved scores: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
