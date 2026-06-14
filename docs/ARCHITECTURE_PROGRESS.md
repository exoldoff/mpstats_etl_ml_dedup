# SKU Deduplication Research Progress

Этот файл ведётся как практический журнал к `docs/ARCHITECTURE.md`.
`ARCHITECTURE.md` фиксирует целевую методологию, а здесь записывается, что
уже сделано на research-этапе, какие артефакты получились и какие проверки
пройдены.

Правило ведения: после каждого завершённого research-этапа добавлять новую
запись сверху или обновлять текущий этап, если работа ещё относится к нему.
Не переносить сюда production-решения до выбора технологии; `pipeline/` и
`mpstats_app/` остаются вне scope research-журнала.

## Текущий статус

- Scope: категория `Соусы`, research-only.
- Главная ветка работ: кандидаты -> ручная разметка gold-set -> сравнение
  matching engines -> кластеризация.
- Primary blocking для `01_candidate_generation.ipynb` теперь соответствует
  целевой архитектуре: dense embeddings -> FAISS top-k.
- Текущий блокер для метрик: нужна ручная разметка
  `research/dedup/data/labeling_sauces.csv`.

## 2026-06-23 — FAISS embedding candidate generation

### Зачем

Первичный blocking должен соответствовать архитектурной схеме: не lexical
эвристики как основной поиск, а top-k ближайших соседей по dense embeddings.
Rule/fuzzy logic остаётся для baseline matching и вспомогательных feature-флагов,
но не формирует основной список candidate pairs.

### Что сделано

- Добавлен `research/dedup/embedding_candidates.py`:
  - принимает `product_records` и aligned embeddings;
  - нормализует vectors;
  - строит FAISS `IndexFlatIP`;
  - ищет top-k соседей;
  - сохраняет совместимый pairwise contract для `02_labeling_dataset.ipynb`
    и `03_matching_comparison.ipynb`.
- `notebooks/01_candidate_generation.ipynb` переписан на flow:
  `SentenceTransformer` -> embeddings -> FAISS top-k -> `candidates_sauces.csv`.
- Default local model для notebook: `intfloat/multilingual-e5-small`; модель
  можно заменить через `DEDUP_EMBEDDING_MODEL`.
- Для smoke/debug добавлены `DEDUP_FAISS_RECORD_LIMIT` и
  `DEDUP_CANDIDATES_PATH`, чтобы проверять flow без перезаписи основного CSV.
- `baseline_similarity_score` в CSV оставлен для совместимости, но теперь равен
  `embedding_similarity_score`.
- Добавлен `requirements-research.txt` для notebook-зависимостей:
  `faiss-cpu`, `sentence-transformers`.
- Добавлен unit-test FAISS top-k логики через fake FAISS backend, чтобы тесты
  не скачивали модель.

### Проверки

- `python3 -m pytest research/dedup/tests` — 16 passed.
- `python3 -m compileall research/dedup` — ok.
- `git diff --check` — ok.
- `nbclient` на `notebooks/01_candidate_generation.ipynb` с
  `DEDUP_FAISS_RECORD_LIMIT=300`, `DEDUP_FAISS_TOP_K=10`,
  `DEDUP_FAISS_MAX_CANDIDATES=1000`,
  `DEDUP_CANDIDATES_PATH=research/dedup/data/candidates_sauces_smoke.csv` —
  ok, 1000 candidate pairs, `candidate_source=faiss_embedding_topk`.
- `nbclient` на полном `notebooks/01_candidate_generation.ipynb` —
  ok, локально перегенерировано 60 000 candidate pairs, 25 093
  cross-marketplace, `candidate_source=faiss_embedding_topk`.
- `nbclient` на `notebooks/02_labeling_dataset.ipynb` — ok, локально
  перегенерировано 400 labeling pairs, 216 cross-marketplace,
  `candidate_source=faiss_embedding_topk`, `label` пустой для ручной разметки.
- На локальном Python 3.13/macOS важно импортировать `faiss` до
  `sentence-transformers`/`torch`; обратный порядок давал native crash
  (`exit 139`). Notebook фиксирует этот порядок.

### Следующий шаг

Разметить `research/dedup/data/labeling_sauces.csv`, затем запускать
`notebooks/03_matching_comparison.ipynb` для сравнения pairwise methods на
FAISS-based gold-set.

## 2026-06-23 — Cross-marketplace candidate generation

### Зачем

Перед ручной разметкой нужно было исправить потерю межмаркетплейсных дублей:
старый candidate generation агрегировал записи по одному `Артикул`, поэтому
один и тот же артикул из разных marketplace схлопывался до генерации пар.
Такой gold-set не покрывал бы важный класс дублей.

### Что сделано

- В `research/dedup/candidates.py` ключ product-record изменён с одного
  `Артикул` на `marketplace + Артикул` через `raw_record_id`.
- Повторы одного `Артикул` по месяцам внутри одного marketplace по-прежнему
  агрегируются в одну запись.
- В candidates добавлены поля:
  `raw_record_id_a`, `raw_record_id_b`, `marketplace_a`, `marketplace_b`,
  `marketplaces_a`, `marketplaces_b`, `is_cross_marketplace_pair`.
- В `research/dedup/labeling.py` добавлена отдельная страта
  `cross_marketplace_candidate`, чтобы cross-marketplace пары осознанно
  попадали в gold-set.
- В `notebooks/01_candidate_generation.ipynb` добавлен sanity-check
  `same_marketplace` vs `cross_marketplace`.
- В `notebooks/02_labeling_dataset.ipynb` добавлена разбивка разметочного
  набора по стратам и cross-marketplace count.

### Получившиеся артефакты

CSV-файлы перегенерированы локально, но они игнорируются через `.gitignore`
как рабочие данные:

- `research/dedup/data/candidates_sauces.csv`
- `research/dedup/data/labeling_sauces.csv`

Фактическое распределение после регенерации:

| Артефакт | Всего пар | Cross-marketplace | Same-marketplace |
| --- | ---: | ---: | ---: |
| `candidates_sauces.csv` | 60 381 | 28 885 | 31 496 |
| `labeling_sauces.csv` | 400 | 205 | 195 |

Разбивка `labeling_sauces.csv`:

| Страта | Пар | Cross-marketplace |
| --- | ---: | ---: |
| `cross_marketplace_candidate` | 80 | 80 |
| `hard_negative_candidate` | 80 | 34 |
| `high_similarity` | 88 | 30 |
| `medium_similarity` | 60 | 23 |
| `pack_variant_candidate` | 72 | 31 |
| `random_easy_negative` | 20 | 7 |

### Проверки

- `python3 -m pytest research/dedup/tests` — 15 passed.
- `python3 -m compileall research/dedup` — ok.
- `git diff --check` — ok.
- `nbclient` на `notebooks/01_candidate_generation.ipynb` — ok.
- `nbclient` на `notebooks/02_labeling_dataset.ipynb` — ok.

### Следующий шаг

Заполнить `label` и при необходимости `notes` в
`research/dedup/data/labeling_sauces.csv`, затем запускать
`notebooks/03_matching_comparison.ipynb`.

## 2026-06-23 — Baseline matching scaffold

### Зачем

Пока идёт ручная разметка, нужен runnable-каркас для сравнения первых
matching engines из `ARCHITECTURE.md`: A и B.

### Что сделано

- Добавлен общий интерфейс `PairMatcher`.
- Добавлен baseline A: `RuleBasedMatcher` на fuzzy title similarity +
  brand/weight/pack fusion.
- Добавлен baseline B: `BiEncoderMatcher` с ленивой загрузкой
  `sentence-transformers`; если зависимости нет, notebook не падает, а метод
  получает статус skipped.
- Добавлен `research/dedup/fusion.py` с `decide_label()` по разделу 5
  архитектуры.
- Добавлен `notebooks/03_matching_comparison.ipynb`, который:
  - graceful обрабатывает пустой или отсутствующий `labeling_sauces.csv`;
  - считает per-class precision/recall/F1;
  - строит confusion matrix;
  - отдельно выводит `false_merge` как самую дорогую ошибку.

### Проверки

- `python3 -m pytest research/dedup/tests` — passed на момент этапа.
- `python3 -m compileall research/dedup` — ok.
- `nbclient` на `notebooks/03_matching_comparison.ipynb` с пустой разметкой —
  ok.

### Следующий шаг

После ручной разметки перезапустить notebook 03 и откалибровать thresholds на
dev-set, не смешивая dev и held-out test.
