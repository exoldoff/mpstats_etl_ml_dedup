# Dedup Fine-Tuning Runbook

Дата: 2026-06-29.

Этот runbook реализует архитектуру из
`docs/DEDUP_FINE_TUNING_EXPERIMENT_PLAN.md`. Production `pipeline/` и web-app
не меняются: весь код живёт в `research/dedup/training/`, а локальные
датасеты, модели и score-cache пишутся в ignored-папки.

## Решение: scripts first

Обучение делаем отдельными Python-модулями, не notebook-ячейками:

1. Notebook удобен для графиков и выводов, но опасен для дорогой GPU-сессии:
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

Текущая разметка зафиксирована как финальный training snapshot. Новую разметку
больше не добираем; обучение и сравнение идут от clean CSV:

```text
research/dedup/data/training/dedup_pairs_final_split.csv
```

Локальный backup финального freeze:

```text
artifacts/backups/dedup_training_freeze/20260629_final_a5000/
```

Сначала freeze. Эта команда пишет пары, excluded rows, conflicts и manifest,
но останавливается с кодом `2`, если есть конфликты меток:

```bash
python3 -m research.dedup.training.prepare_dataset --fail-on-conflicts
```

Ожидаемо сейчас есть несколько конфликтных строк между CSV и Telegram SQLite.
Их надо посмотреть в:

```text
research/dedup/data/training/dedup_pairs_final_conflicts.csv
```

Если конфликты понятны и мы осознанно исключаем их из train loss, создать
финальный split:

```bash
python3 -m research.dedup.training.prepare_dataset
```

Главный файл для обучения:

```text
research/dedup/data/training/dedup_pairs_final_split.csv
```

CSV-схема основного split намеренно чистая: `27` колонок, без служебных
`csv_label`, `sqlite_label`, `status`, `source_priority`, component internals
и notebook-only полей. Конфликты/excluded/dropped лежат отдельными audit CSV и
в train loss не попадают.

Продажи для бизнес-калибровки не лежат в split CSV. Перед threshold
calibration нужен отдельный lookup:

```bash
python3 -m research.dedup.training.export_sales_lookup \
  --duckdb-path mpstats.duckdb \
  --output-path research/dedup/data/training/sales_volume_lookup.csv
```

Файл `sales_volume_lookup.csv` содержит только `raw_record_id,sales_volume` и
нужен, чтобы после обучения появились стратегии `threshold_max_weighted_f1` и
`threshold_weighted_cost`. Если lookup не доступен, calibration с
`--require-weighted` должна упасть, а не молча перейти в unweighted режим.

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
  `research/dedup/data/training/dedup_pairs_final_split_dropped.csv`.

Для аудита strict-режима:

```bash
python3 -m research.dedup.training.prepare_dataset --large-component-strategy strict
```

Для отдельной benchmark/evaluation проверки false merge можно создать
pair-stratified split без выкидывания negative:

```bash
python3 -m research.dedup.training.prepare_dataset \
  --large-component-strategy pair_stratified \
  --prefix dedup_pairs_final_pair_stratified
```

Он сохраняет все `2465` binary pairs и даёт примерно `253` negative / `117`
positive в test при split `70/15/15`. Это лучше для стабильной оценки false
merge, но такой split может иметь raw-id leakage между train/dev/test; поэтому
он benchmark/stress split, а не строгий no-leak split.

## Что копировать на GPU-сервер

### Вариант A: Docker, предпочтительно

Образ содержит код и зависимости. Локальные данные, модели, score-cache и
Hugging Face cache монтируются volume-ами, чтобы не запекать CSV и веса модели
в image.

Сборка образа:

```bash
docker build \
  -f docker/dedup-training/Dockerfile \
  -t mpstats-dedup-training:cu128 \
  .
```

Проверка GPU/runtime внутри контейнера:

```bash
docker run --rm --gpus all \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  mpstats-dedup-training:cu128 doctor
```

Precision по умолчанию — `fp16`, это профиль под маленькие encoder-модели на
A5000 24GB. Для H200 можно включить `bf16`:

```bash
docker run --rm --gpus all \
  -e DEDUP_TRAINING_PRECISION=bf16 \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  mpstats-dedup-training:cu128 smoke-rubert
```

Подготовка split внутри контейнера:

```bash
docker run --rm --gpus all \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  mpstats-dedup-training:cu128 prepare
```

Если на сервер скопирован `mpstats.duckdb`, lookup продаж можно сделать прямо
в контейнере:

```bash
docker run --rm \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/mpstats.duckdb:/workspace/mpstats.duckdb:ro" \
  mpstats-dedup-training:cu128 export-sales-lookup \
    --duckdb-path /workspace/mpstats.duckdb \
    --output-path /workspace/research/dedup/data/training/sales_volume_lookup.csv
```

Если `sales_volume_lookup.csv` уже экспортирован локально и скопирован на
сервер, DuckDB для calibration больше не нужен.

Обязательный smoke:

```bash
docker run --rm --gpus all \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  mpstats-dedup-training:cu128 smoke-rubert
```

Первый полный cheap baseline:

```bash
docker run --rm --gpus all \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  mpstats-dedup-training:cu128 train-rubert
```

Дальше по готовности:

```bash
docker run --rm --gpus all \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  mpstats-dedup-training:cu128 train-mmarco

docker run --rm --gpus all \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  mpstats-dedup-training:cu128 train-bge

docker run --rm --gpus all \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  mpstats-dedup-training:cu128 train-qwen
```

`train-qwen` использует `bf16` по умолчанию, даже если общий default контейнера
`fp16`: Qwen reranker поднимает BF16 weights, а fp16 GradScaler падает на
BF16 gradients. Если precision задаётся явно, использовать:

```bash
docker run --rm --gpus all \
  -e DEDUP_TRAINING_PRECISION=bf16 \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  mpstats-dedup-training:cu128 train-qwen
```

Если Docker daemon на сервере не настроен под GPU, сначала проверить host:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Вариант B: без Docker

На удалённой машине нужен не весь локальный мусор, а минимальный bundle:

```text
research/dedup/training/
research/dedup/model_registry.py
research/dedup/threshold_calibration.py
research/dedup/data/training/dedup_pairs_final_split.csv
research/dedup/data/training/dedup_pairs_final_split_manifest.json
research/dedup/data/training/sales_volume_lookup.csv
docker/dedup-training/
requirements-research.txt
docs/DEDUP_FINE_TUNING_EXPERIMENT_PLAN.md
docs/DEDUP_TRAINING_RUNBOOK.md
```

Если запускаем прямо из repo checkout, отдельно копировать не надо.

Важно: vLLM нужен для инференса, не для обучения. Для training-команд нужен
PyTorch/Transformers runtime с CUDA на GPU-сервере. Если GPU-образ уже содержит
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
  --fp16
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
  --fp16
```

PEFT/LoRA для tiny2 не обязателен. Если нужно проверить adapter-only режим:

```bash
python3 -m research.dedup.training.train_pair_classifier \
  --model-name cointegrated/rubert-tiny2 \
  --output-dir artifacts/models/dedup/rubert_tiny2_lora_v1 \
  --use-peft-lora \
  --lora-target-modules query,value \
  --num-train-epochs 5 \
  --fp16
```

### mMARCO MiniLM

Самый дешёвый CrossEncoder sanity baseline.

```bash
python3 -m research.dedup.training.train_cross_encoder \
  --model-name cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --output-dir artifacts/models/dedup/mmarco_v1 \
  --num-train-epochs 3 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --gradient-accumulation-steps 2 \
  --fp16
```

### BGE reranker v2 m3

Стабильный multilingual reranker baseline.

```bash
python3 -m research.dedup.training.train_cross_encoder \
  --model-name BAAI/bge-reranker-v2-m3 \
  --output-dir artifacts/models/dedup/bge_reranker_v2_m3_v1 \
  --num-train-epochs 3 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 16 \
  --gradient-accumulation-steps 4 \
  --fp16
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
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --default-prompt-name sku_match \
  --trust-remote-code \
  --bf16
```

### Jina reranker v3

Jina оставляем в плане, но не запускаем первой GPU-сессией. У неё listwise /
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
старый benchmark. Для финального сравнения обязателен weighted режим по
продажам:

```bash
python3 -m research.dedup.training.calibrate_scores \
  --score-path artifacts/reports/fine_tuning/rubert_tiny2_scores.csv \
  --score-column ft_rubert_tiny2 \
  --method ft_rubert_tiny2 \
  --reports-dir artifacts/reports/fine_tuning \
  --sales-lookup-path research/dedup/data/training/sales_volume_lookup.csv \
  --require-weighted
```

В Docker-команде `calibrate` флаг `--require-weighted` включён по умолчанию
через `DEDUP_REQUIRE_WEIGHTED_CALIBRATION=1`. Отключать это можно только для
диагностики:

```bash
docker run --rm --gpus all \
  -e DEDUP_REQUIRE_WEIGHTED_CALIBRATION=0 \
  -v "$PWD/research/dedup/data:/workspace/research/dedup/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  mpstats-dedup-training:cu128 calibrate \
    --score-path artifacts/reports/fine_tuning/rubert_tiny2_scores.csv \
    --score-column ft_rubert_tiny2 \
    --method ft_rubert_tiny2 \
    --reports-dir artifacts/reports/fine_tuning
```

Нормальный вывод calibration должен содержать:

```text
sales_volume_status: {'status': 'joined_sales_volume', ...}
weights_available: True
```

В summary должны быть строки `threshold_max_weighted_f1` и
`threshold_weighted_cost`. Если видны только `threshold_max_f1` и
`threshold_cost_sensitive`, это не финальный продуктовый режим.

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
- `test` смотрим один раз как честную проверку;
- главный режим выбора — `threshold_weighted_cost`, обычные `max_f1` и
  `cost_sensitive` остаются диагностикой.
