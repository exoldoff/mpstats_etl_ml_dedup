# Dedup Fine-Tuning Runbook

Дата: 2026-06-29.

Этот runbook реализует архитектуру из
`docs/DEDUP_FINE_TUNING_EXPERIMENT_PLAN.md`. Production `pipeline/` и web-app
не меняются: весь код живёт в `research/dedup/training/`, а локальные
датасеты, модели и score-cache пишутся в ignored-папки.

## Решение: scripts first

Обучение делаем отдельными Python-модулями, не notebook-ячейками:

1. Notebook удобен для графиков и выводов, но опасен для дорогой H200-сессии:
   легко выполнить ячейки не в том порядке.
2. Training script даёт воспроизводимую команду, manifest, commit hash,
   версии библиотек и один формат артефактов.
3. Notebook остаётся на следующем шаге: прочитать готовые score CSV и показать
   сравнение fine-tuned vs old zero-shot benchmark.

## Бэкап старого benchmark

Старые результаты прогона моделей сохранены до новых экспериментов:

```text
artifacts/backups/dedup_model_runs/20260629_204244/
```

Внутри:

- `MANIFEST.txt` — список сохранённых файлов;
- `SHA256SUMS.txt` — контрольные суммы исходных файлов;
- `dedup_model_run_results_20260629_204244.tar.gz`;
- `dedup_model_run_results_20260629_204244.tar.gz.sha256`.

Проверка tarball:

```bash
cd /Users/exoldoff/.codex/worktrees/13ef/mpstats
shasum -a 256 -c artifacts/backups/dedup_model_runs/20260629_204244/dedup_model_run_results_20260629_204244.tar.gz.sha256
```

## Локальная подготовка датасета

Сначала freeze. Эта команда пишет пары, excluded rows, conflicts и manifest,
но останавливается с кодом `2`, если есть конфликты меток:

```bash
python3 -m research.dedup.training.prepare_dataset --fail-on-conflicts
```

Ожидаемо сейчас есть несколько конфликтных строк между CSV и Telegram SQLite.
Их надо посмотреть в:

```text
research/dedup/data/training/dedup_pairs_v1_conflicts.csv
```

Если конфликты понятны и мы осознанно исключаем их из train loss, создать
финальный split:

```bash
python3 -m research.dedup.training.prepare_dataset
```

Главный файл для обучения:

```text
research/dedup/data/training/dedup_pairs_v1_split.csv
```

Текущий фактический freeze:

- `2465` binary pairs;
- `1686` negative / `779` positive;
- `3` conflict rows;
- `33` excluded `uncertain`;
- `2465` unique pair keys.

Splitter по умолчанию сначала пробует strict all-edge components. На текущих
данных он находит giant component на `717` строк, почти целиком
`coconut_oil`. Поэтому автоматически включается fallback
`positive_record_holdout`: positive-дубли держатся вместе, raw records
назначаются в split один раз, crossing negative pairs выносятся в dropped.
Фактический training-ready split:

- train/dev/test: `1404 / 167 / 165`;
- dropped crossing pairs: `729`, все negative;
- raw-id leakage: `0`;
- dropped audit:
  `research/dedup/data/training/dedup_pairs_v1_split_dropped.csv`.

Для аудита strict-режима:

```bash
python3 -m research.dedup.training.prepare_dataset --large-component-strategy strict
```

## Что копировать на H200

На удалённой машине нужен не весь локальный мусор, а минимальный bundle:

```text
research/dedup/training/
research/dedup/model_registry.py
research/dedup/threshold_calibration.py
research/dedup/data/training/dedup_pairs_v1_split.csv
research/dedup/data/training/dedup_pairs_v1_split_manifest.json
requirements-research.txt
docs/DEDUP_FINE_TUNING_EXPERIMENT_PLAN.md
docs/DEDUP_TRAINING_RUNBOOK.md
```

Если запускаем прямо из repo checkout, отдельно копировать не надо.

Важно: vLLM нужен для инференса, не для обучения. Для training-команд нужен
PyTorch/Transformers runtime с CUDA под H200. Если H200-образ уже содержит
рабочий `torch`, не переустанавливать его без причины; доставить только
research-зависимости:

```bash
pip install -r requirements-research.txt
```

## Обязательный smoke перед дорогим запуском

```bash
python3 -m research.dedup.training.train_pair_classifier \
  --model-name cointegrated/rubert-tiny2 \
  --output-dir artifacts/models/dedup/rubert_tiny2_smoke \
  --smoke-limit 96 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --bf16
```

Если smoke не проходит, большой запуск не начинать.

## Модели первого цикла

### RuBERT tiny2

Это не embedding-модель в нашем эксперименте, а cross-encoder style
pair-classifier: на вход идут `sentence_A` и `sentence_B`, на выходе
вероятность `same_base_product`.

```bash
python3 -m research.dedup.training.train_pair_classifier \
  --model-name cointegrated/rubert-tiny2 \
  --output-dir artifacts/models/dedup/rubert_tiny2_v1 \
  --num-train-epochs 5 \
  --learning-rate 3e-5 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 64 \
  --bf16
```

PEFT/LoRA для tiny2 не обязателен. Если нужно проверить adapter-only режим:

```bash
python3 -m research.dedup.training.train_pair_classifier \
  --model-name cointegrated/rubert-tiny2 \
  --output-dir artifacts/models/dedup/rubert_tiny2_lora_v1 \
  --use-peft-lora \
  --lora-target-modules query,value \
  --num-train-epochs 5 \
  --bf16
```

### mMARCO MiniLM

Самый дешёвый CrossEncoder sanity baseline.

```bash
python3 -m research.dedup.training.train_cross_encoder \
  --model-name cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --output-dir artifacts/models/dedup/mmarco_v1 \
  --num-train-epochs 3 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 64 \
  --bf16
```

### BGE reranker v2 m3

Стабильный multilingual reranker baseline.

```bash
python3 -m research.dedup.training.train_cross_encoder \
  --model-name BAAI/bge-reranker-v2-m3 \
  --output-dir artifacts/models/dedup/bge_reranker_v2_m3_v1 \
  --num-train-epochs 3 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --gradient-accumulation-steps 2 \
  --bf16
```

### Qwen3 Reranker 0.6B

Главный практичный кандидат. `Qwen3-Reranker-4B` в первом цикле не тюним,
оставляем только benchmark-only.

```bash
python3 -m research.dedup.training.train_cross_encoder \
  --model-name Qwen/Qwen3-Reranker-0.6B \
  --output-dir artifacts/models/dedup/qwen3_reranker_0_6b_v1 \
  --num-train-epochs 2 \
  --learning-rate 1e-5 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --default-prompt-name sku_match \
  --trust-remote-code \
  --bf16
```

### Jina reranker v3

Jina оставляем в плане, но не запускаем первой H200-сессией. У неё listwise /
remote-code path, поэтому сначала стабилизируем frozen split, score-cache и
threshold loop на tiny2, mMARCO, BGE и Qwen 0.6B.

## Скоринг после обучения

RuBERT tiny2 / pair-classifier:

```bash
python3 -m research.dedup.training.score_pair_classifier \
  --model-path artifacts/models/dedup/rubert_tiny2_v1/final \
  --output-path artifacts/reports/fine_tuning/rubert_tiny2_scores.csv \
  --score-column ft_rubert_tiny2 \
  --batch-size 128
```

CrossEncoder:

```bash
python3 -m research.dedup.training.score_cross_encoder \
  --model-path artifacts/models/dedup/bge_reranker_v2_m3_v1/final \
  --output-path artifacts/reports/fine_tuning/bge_reranker_v2_m3_scores.csv \
  --score-column ft_bge_reranker_v2_m3 \
  --activation sigmoid \
  --batch-size 32
```

## Threshold calibration

Каждый fine-tuned score прогоняем через ту же dev/test threshold-логику, что и
старый benchmark:

```bash
python3 -m research.dedup.training.calibrate_scores \
  --score-path artifacts/reports/fine_tuning/rubert_tiny2_scores.csv \
  --score-column ft_rubert_tiny2 \
  --method ft_rubert_tiny2 \
  --reports-dir artifacts/reports/fine_tuning
```

Итоги появятся в:

```text
artifacts/reports/fine_tuning/<method>_binary_threshold_summary.csv
artifacts/reports/fine_tuning/<method>_binary_threshold_predictions.csv
```

Сравнивать прирост надо с сохранённым старым benchmark:

```text
artifacts/reports/binary_threshold_summary.csv
artifacts/reports/score_cache/model_scores_cache_latest.csv
```

Главные критерии те же:

- false merge должен оставаться низким;
- recall должен вырасти относительно zero-shot;
- финальный threshold выбирается только на `dev`;
- `test` смотрим один раз как честную проверку.
