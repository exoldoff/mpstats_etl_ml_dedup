#!/usr/bin/env bash
set -euo pipefail

cd /workspace

command_name="${1:-doctor}"
if [[ $# -gt 0 ]]; then
  shift
fi

precision_args=()
case "${DEDUP_TRAINING_PRECISION:-fp16}" in
  bf16)
    precision_args=(--bf16)
    ;;
  fp16)
    precision_args=(--fp16)
    ;;
  fp32|none)
    precision_args=()
    ;;
  *)
    echo "Unknown DEDUP_TRAINING_PRECISION=${DEDUP_TRAINING_PRECISION}. Use bf16, fp16, fp32, or none." >&2
    exit 2
    ;;
esac

case "${command_name}" in
  doctor)
    exec python3 -m research.dedup.training.runtime_doctor "$@"
    ;;
  prepare)
    exec python3 -m research.dedup.training.prepare_dataset "$@"
    ;;
  smoke-rubert)
    exec python3 -m research.dedup.training.train_pair_classifier \
      --model-name cointegrated/rubert-tiny2 \
      --output-dir artifacts/models/dedup/rubert_tiny2_smoke \
      --smoke-limit 96 \
      --num-train-epochs 1 \
      --per-device-train-batch-size 16 \
      --per-device-eval-batch-size 32 \
      "${precision_args[@]}" \
      "$@"
    ;;
  train-rubert)
    exec python3 -m research.dedup.training.train_pair_classifier \
      --model-name cointegrated/rubert-tiny2 \
      --output-dir artifacts/models/dedup/rubert_tiny2_v1 \
      --num-train-epochs 5 \
      --learning-rate 3e-5 \
      --per-device-train-batch-size 32 \
      --per-device-eval-batch-size 64 \
      "${precision_args[@]}" \
      "$@"
    ;;
  train-mmarco)
    exec python3 -m research.dedup.training.train_cross_encoder \
      --model-name cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
      --output-dir artifacts/models/dedup/mmarco_v1 \
      --num-train-epochs 3 \
      --learning-rate 2e-5 \
      --per-device-train-batch-size 16 \
      --per-device-eval-batch-size 32 \
      --gradient-accumulation-steps 2 \
      "${precision_args[@]}" \
      "$@"
    ;;
  train-bge)
    exec python3 -m research.dedup.training.train_cross_encoder \
      --model-name BAAI/bge-reranker-v2-m3 \
      --output-dir artifacts/models/dedup/bge_reranker_v2_m3_v1 \
      --num-train-epochs 3 \
      --learning-rate 2e-5 \
      --per-device-train-batch-size 8 \
      --per-device-eval-batch-size 16 \
      --gradient-accumulation-steps 4 \
      "${precision_args[@]}" \
      "$@"
    ;;
  train-qwen)
    exec python3 -m research.dedup.training.train_cross_encoder \
      --model-name Qwen/Qwen3-Reranker-0.6B \
      --output-dir artifacts/models/dedup/qwen3_reranker_0_6b_v1 \
      --num-train-epochs 2 \
      --learning-rate 1e-5 \
      --per-device-train-batch-size 1 \
      --per-device-eval-batch-size 2 \
      --gradient-accumulation-steps 16 \
      --default-prompt-name sku_match \
      --trust-remote-code \
      "${precision_args[@]}" \
      "$@"
    ;;
  score-pair)
    exec python3 -m research.dedup.training.score_pair_classifier "$@"
    ;;
  score-cross)
    exec python3 -m research.dedup.training.score_cross_encoder "$@"
    ;;
  calibrate)
    exec python3 -m research.dedup.training.calibrate_scores "$@"
    ;;
  bash|sh)
    exec "${command_name}" "$@"
    ;;
  *)
    exec "${command_name}" "$@"
    ;;
esac
