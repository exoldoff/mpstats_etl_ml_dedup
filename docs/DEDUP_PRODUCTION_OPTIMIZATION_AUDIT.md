# Dedup Production Optimization Audit

Дата: 2026-06-27.

Scope проверки:

- `notebooks/00_eda.ipynb` - `notebooks/06_grouped_sku_demo.ipynb`;
- `research/dedup/*.py`, `research/dedup/matchers/*.py`;
- `scripts/benchmark_classifier.py`, `scripts/profile_classifier.py`,
  `scripts/classifier_perf_utils.py`, `scripts/duckdb_benchmark.py`,
  `scripts/duckdb_mock_data.py`, `scripts/launch_local_app.sh`;
- актуальная FAISS-документация по on-disk / mmap index loading.

Главный вывод: текущий dedup flow уже хороший research benchmark, но до
production ему не хватает не "ещё одной модели", а воспроизводимого runtime
слоя: кэшируемых embeddings/indexes/scores, typed artifacts, run manifest,
жёстких safety gates против false merge и понятного места в будущей identity
mapping схеме.

## P0. Сделать до production

### 1. Кэш embeddings и FAISS index

Сейчас `notebooks/01_candidate_generation.ipynb` каждый раз:

1. читает срез из DuckDB;
2. строит `product_records`;
3. заново вызывает `model.encode(...)`;
4. строит `faiss.IndexFlatIP`;
5. сохраняет только `candidates_*.csv`.

Это нормально для research, но в production дорого, нестабильно и плохо
масштабируется.

Что улучшить:

- Ввести content-addressed кэш по ключу:
  `category_run + project_name + input_row_hash + model_name + text_builder_version + normalize_vectors + dimension`.
- Сохранять:
  - нормализованный node catalog;
  - embedding texts;
  - embeddings как `float32` `.npy`/`np.memmap`;
  - FAISS index через `faiss.write_index`;
  - `run_manifest.json`.
- При повторном запуске сначала читать готовые embeddings/index, а не
  повторять model/API calls.
- Для текущего `IndexFlatIP` самый быстрый практический выигрыш сначала даст
  именно кэш embeddings: flat index rebuild дешевле, чем повторный embedding.
- Для больших категорий перейти на persisted IVF/on-disk index. FAISS
  поддерживает on-disk IVF и `IO_FLAG_MMAP`; важно делать это как отдельный
  индексный режим, а не просто механически добавить флаг к текущему
  `IndexFlatIP`.

FAISS caveats для будущей реализации:

- on-disk storage в FAISS в первую очередь описан для IVF inverted lists;
- `IO_FLAG_MMAP` пытается memory-map IVF data, но search всё равно зависит от
  OS page cache и SSD;
- GPU indexes надо сохранять как CPU index;
- `read_index` не должен читать непроверенные/чужие index files: нужен
  manifest, checksum и контроль того, что файл создан нашим pipeline.

Решение production v1 от 2026-06-30:

- сначала кэшировать embeddings, потому что текущий bottleneck —
  повторный `model.encode(...)`;
- сохранять `embeddings.npy` и читать его через `np.load(..., mmap_mode="r")`;
- `IndexFlatIP` пересобирать на каждом run из cached embeddings;
- не добавлять `IO_FLAG_MMAP` к текущему flat-index flow;
- не использовать `faiss.read_index` для production v1;
- on-disk IVF / mmap FAISS index рассматривать позже отдельным режимом,
  только после recall@k benchmark.

### 2. Вынести runtime из notebook cells в сервисный слой

Notebook 03 уже содержит несколько больших code cells: загрузка labels,
baseline/bi-encoder/cross-encoder/reranker scoring, sales-volume join,
threshold calibration, export и визуализации. Notebook 04/06 тоже держат
существенную business logic внутри ячеек.

Перед production лучше сделать такие чистые модули:

- `research/dedup/retrieval_artifacts.py` или будущий production service:
  построение/чтение embeddings + FAISS index + candidate pairs;
- `research/dedup/scoring.py`: единый batch scoring API, score cache и
  статусы skipped/failed/ready;
- `research/dedup/artifact_schema.py`: typed artifact columns, required
  columns, schema validation;
- `research/dedup/run_manifest.py`: manifest каждого запуска;
- позже, при интеграции, перенести выбранный вариант в `pipeline/services`
  и repository layer, а notebooks оставить как thin presentation.

Смысл: notebooks должны показывать и объяснять, а не быть единственным местом,
где живёт production логика.

### 3. Typed artifacts вместо CSV-only контракта

Сейчас основные связи между шагами:

- `candidates_*.csv`;
- `labeling_*.csv`;
- `binary_threshold_summary.csv`;
- `binary_threshold_predictions.csv`;
- `fusion_components_*.csv`;
- `fusion_pair_eval_*.csv`.

CSV удобен для ручного просмотра, но для production это слабый контракт:
типы теряются, bool/NaN читаются неоднозначно, нет версии схемы и нет связи с
конкретным cube snapshot.

Что добавить:

- machine-readable artifact format: DuckDB tables или Parquet;
- CSV оставить как export/debug;
- schema version для каждого artifact;
- `run_manifest.json` рядом с artifact:
  - git commit;
  - category_run/project/category aliases;
  - input table/query;
  - input row count и node count;
  - model alias/id/backend;
  - FAISS index type/top_k/min_similarity;
  - thresholds/cost config;
  - package versions;
  - created_at;
  - checksums для embeddings/index/scores.

### 4. Production safety gate против false merge и graph chaining

В dedup false merge опаснее false split. Текущий threshold layer уже учитывает
cost, но graph step всё равно строит connected components: один плохой edge
может склеить большую family.

Что добавить до production:

- hard gate на выбор fusion-run:
  - `max_false_merge_count` на dev/test;
  - min precision для same-base links;
  - max component size или max brand diversity без ручной проверки;
  - запрет auto-merge для компонент с конфликтующими pack/brand/weight
    сигналами.
- quarantine статус для рискованных компонент: не склеивать автоматически,
  а отдавать в review/diagnostics.
- component-level validation поверх pairwise metrics:
  - component size distribution;
  - false-link examples;
  - "dangerous bridge edge" diagnostics;
  - доля продаж внутри рискованных компонент.

### 5. Будущий identity mapping слой отдельно от monthly cube facts

Production dedup не должен физически схлопывать monthly facts. Нужен
отдельный identity layer:

- `dedup_sku_node`: marketplace/article/title snapshot или стабильный
  product node;
- `dedup_sku_edge`: scored pair edges, method, score, threshold, decision;
- `dedup_sku_group`: текущие family/pack ids;
- `dedup_run`: manifest/config/status;
- фактовая таблица куба остаётся помесячной и join-ится к identity layer.

DB impact будущей интеграции: это новая schema/migrations/repository layer,
а не изменение существующих monthly fact rows.

## P1. Следующий слой оптимизаций

### 6. Убрать `DataFrame.apply` и `iterrows` с горячего пути

Нормально для 60k пар, но плохо для больших категорий:

- `add_cross_marketplace_flags`;
- `add_hard_negative_flags`;
- `add_pack_variant_flags`;
- `same_pack_signature_mask`;
- `_score_matcher` в notebook 03 делает `row for _, row in pairs.iterrows()`;
- demo/grouping notebooks собирают dict rows в Python loops.

Что делать:

- flags по marketplace/weights считать vectorized pandas/numpy;
- scoring передавать списки dict/dataclass или сразу text-pairs, без Series;
- pair keys и same-pack signatures делать как pure vectorized helpers;
- оставить row-wise код только для маленьких review tables.

### 7. Score cache для pairwise/reranker models

Сейчас scores пересчитываются при каждом запуске notebook 03. Для production
нужен cache:

`method + model_id + model_version + pair_key + text_a_hash + text_b_hash -> score`.

Это особенно важно для:

- Polza/API embeddings;
- Qwen/Jina/BGE rerankers;
- повторного threshold tuning на тех же scores;
- ручного расширения gold-set, когда старые пары не должны пересчитываться.

### 8. Bounded retries/rate limits для online embeddings

`PolzaEmbeddingModel` уже имеет timeout, но нет retry/backoff и явного
rate-limit/cost accounting.

Перед production добавить:

- bounded retries на 429/5xx/network reset;
- batch failure reporting;
- cache by text hash;
- dry-run estimate: сколько texts/batches будет отправлено;
- запрет silent partial results.

### 9. DuckDB pushdown вместо `SELECT *` там, где это runtime

В notebooks 00/01/06 есть чтение широких срезов из DuckDB. Для EDA это
допустимо, но production runtime должен:

- выбирать только нужные колонки;
- агрегировать monthly rows до node catalog в DuckDB;
- фильтровать project/category на SQL side;
- не материализовать весь cube slice в pandas без необходимости.

`scripts/duckdb_benchmark.py` уже содержит полезные идеи: explicit schema,
staging table, anti-join/hash подходы. Их стоит использовать как ориентир,
но не тащить benchmark script напрямую в runtime.

### 10. Output hygiene для notebooks

Сейчас часть `.ipynb` хранит outputs с абсолютными локальными путями и
кусочками данных. Перед production/review лучше:

- хранить notebooks без outputs или чистить outputs перед commit;
- воспроизводимость держать через tracked docs + ignored artifacts;
- не считать сохранённые output cells источником правды.

## P2. Методологические улучшения

### 11. Category-run config из данных/настроек, не только hardcoded registry

`research/dedup/category_runs.py` сейчас удобен для трёх research категорий.
Для production список runs должен приходить из web settings/project config или
из repository layer, а не из Python constants.

### 12. Улучшить labeling как active learning loop

Notebook 02 делает стартовую стратифицированную выборку. Следующий шаг:

- добавлять пары из model disagreement;
- добавлять near-threshold пары;
- добавлять graph bridge-risk edges;
- отдельно добирать high-sales false-merge risk.

Так gold-set будет расти в тех местах, где production риск самый большой.

### 13. Добавить cluster-level metrics

Notebook 05 сейчас полезно показывает link-level family/pack report и
примеры ошибок. Для production decision нужны ещё:

- B-cubed precision/recall/F1;
- pairwise cluster precision/recall на компонентах;
- component purity по brand/pack;
- sales-weighted component risk.

### 14. `trust_remote_code=True` только через allowlist

Jina reranker использует `trust_remote_code=True`. В research это нормально,
но production loader должен иметь allowlist моделей и явный security review.

## По файлам

### `notebooks/00_eda.ipynb`

Оставить как research/diagnostic notebook. Если какие-то проверки станут
production checks, переносить их в service/data-quality layer, а не выполнять
EDA notebook в runtime.

### `notebooks/01_candidate_generation.ipynb`

Главный кандидат на оптимизацию:

- embeddings/index cache;
- persisted FAISS;
- SQL column pushdown;
- manifest;
- vectorized embedding text builder;
- explicit smoke mode, который не бьёт по API/model limits.

### `notebooks/02_labeling_dataset.ipynb`

Хороший research старт. Улучшать не скорость, а sampling strategy:
near-threshold, disagreement, graph-bridge и sales-weighted risk strata.

### `notebooks/03_matching_comparison.ipynb`

Перед production распилить на сервисные функции. Важные улучшения:

- score cache;
- retry/backoff для API;
- batch/scoring telemetry;
- threshold selection с hard constraints;
- единый typed output contract.

### `notebooks/04_fusion_pack_grouping.ipynb`

Сейчас есть ручной дефолт `MY_FUSION_METHOD = "reranker_qwen3_4b"`. Для
production это должно стать config/run selection policy с safety constraints,
а не notebook edit.

Graph-quality metrics и risk summaries теперь должны жить в
`notebooks/04_fusion_pack_grouping.ipynb`, чтобы production-кандидат не
зависел от отдельного read-only report notebook.

### `notebooks/06_grouped_sku_demo.ipynb`

Оставить как human preview. Перед production убрать fallback на локальный
desktop path из runtime-логики и перенести DuckDB aggregation в repository
helper.

### `research/dedup/embedding_candidates.py`

Добавить persisted-index режим рядом с текущим `IndexFlatIP`:

- `build_flat_index(...)` для малого среза;
- `write/read index` helpers;
- `build_ivf_on_disk(...)` для больших категорий;
- guarded `mmap` loading, только для trusted artifacts.

### `research/dedup/model_registry.py`

Уже хорошая основа. Добавить:

- model artifact/version metadata;
- explicit allowlist для `trust_remote_code`;
- pool memory diagnostics;
- cache key helpers для score/embedding artifacts.

### `research/dedup/threshold_calibration.py`

Сильный слой, но selection policy стоит усилить hard constraints для
false-merge и component-risk. Cost sorting не должен быть единственной
защитой.

### `research/dedup/fusion.py` и `research/dedup/clustering.py`

Сделать production bridge checks:

- max component size;
- brand/pack conflict flags;
- bridge edge diagnostics;
- quarantine output.

### `scripts/*`

Это полезные benchmark/profiling tools, но не production runtime.

- `duckdb_benchmark.py`: брать идеи pushdown/staging/hash, но переносить в
  repositories/services.
- `classifier_perf_utils.py`: monkeypatch instrumentation оставить в scripts;
  для CI можно добавить отдельный lightweight performance smoke.
- `launch_local_app.sh`: нормальный local-dev wrapper, к dedup production
  напрямую не относится.

## Рекомендуемый порядок работ

1. Добавить manifest + typed schema validation для существующих CSV artifacts.
2. Добавить embeddings cache и `run_manifest.json` в candidate generation
   / production retrieval. Production v1 закрыт через
   `pipeline/services/dedup/retrieval_cache.py`; research notebooks могут
   переиспользовать идею позже.
3. Добавить persisted FAISS index для small/medium runs; отдельно проверить
   mmap/IVF режим на большом synthetic slice.
4. Добавить score cache для notebook 03.
5. Усилить fusion selection hard gates и quarantine risky components.
6. Только после этого проектировать production DB schema/migrations для
   identity layer.
