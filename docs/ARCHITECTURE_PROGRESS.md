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

- Scope: category-runs `sauces`, `coconut_oil`, `soap`, research-only.
- Главная ветка работ: кандидаты -> ручная разметка gold-set -> сравнение
  matching engines -> кластеризация.
- `sauces` сохраняет legacy-пути `research/dedup/data/*_sauces.csv` и
  `artifacts/reports/binary_threshold_*.csv`.
- `coconut_oil` и `soap` пишут в отдельные ignored-папки:
  `research/dedup/data/<slug>/` и `artifacts/reports/<slug>/`, поэтому новые
  прогоны не перетирают старые sauce-артефакты.
- Все notebook-загрузки для category-run фильтруют DuckDB не только по
  `Категория`, но и по `__project_name`.
- Входные строки для production pipeline и research-ноутбуков дополнительно
  фильтруются по продажам: нули отбрасываются, а внутри каждого среза
  остаются верхние 80% положительных строк (`Продажи` не ниже нижнего
  20%-квантиля).
- Primary blocking для `01_candidate_generation.ipynb` теперь соответствует
  целевой архитектуре: dense embeddings -> FAISS top-k.
- Research-модели управляются через `research/dedup/model_registry.py`:
  alias -> backend/model id, общий cache dir `research/dedup/models/`,
  in-process pool и offline-режим `DEDUP_MODEL_LOCAL_ONLY=1`.
- Online embedding-модели идут через Polza.ai-compatible backend
  `polza_embedding`, не через абстрактные provider env-переменные.
- Gold-set переводится на binary-контракт: для модели остаются
  `exact_duplicate` и `different_product`; legacy
  `same_product_different_pack` пользователь заменяет на `exact_duplicate`.
  В threshold calibration legacy label временно мапится в positive
  `same_base_product`, чтобы старые размеченные строки не ломали dev/test
  отчёты.
- Benchmark SKU matching перешёл на forced binary evaluation:
  для каждого method выбирается один `threshold_same` только на dev, а test
  используется только для финальной проверки. Главные отчёты теперь лежат в
  reports-папке текущего category-run:
  `binary_threshold_summary.csv`,
  `binary_threshold_predictions.csv` и, если доступны объёмы продаж,
  `binary_threshold_by_volume_bucket.csv`.
  Manual review, LLM-review и triage на текущем benchmark-этапе не нужны.
- Downstream теперь идёт через `notebooks/04_fusion_pack_grouping.ipynb`:
  notebook выбирает fusion-run только по `dev`, строит `family` и `pack`
  группы и сохраняет `fusion_components_<suffix>.csv` /
  `fusion_pair_eval_<suffix>.csv` в data-папку текущего category-run.
- Финальный downstream-отчёт теперь один:
  `notebooks/05_evaluation_report.ipynb` объединяет прежнюю проверку
  family/pack graph resolution и итоговый report по matching + fusion.
- Для просмотра результата на реальных строках DuckDB добавлен
  `notebooks/06_grouped_sku_demo.ipynb`: он накладывает `fusion_*` на
  `mpstats_products` и показывает склеенные SKU-группы.

## 2026-06-27 — Pre-embedding collapse для пустого бренда

### Зачем

В категориях с пустым `Бренд` одинаковые title могут попадать в FAISS как
отдельные записи и занимать ближайшие embedding-соседи друг друга. Для
research blocking это шум: такие SKU уже очевидно один и тот же record до
модельного сравнения.

### Что сделано

- `prepare_product_records()` теперь по умолчанию схлопывает строки с пустым
  брендом и одинаковым title после лёгкой нормализации регистра/пробелов.
- Схлопывание происходит до построения embedding-текстов в
  `01_candidate_generation.ipynb`, потому что notebook использует
  `prepare_product_records()`.
- Для диагностики в record остаются `source_raw_record_ids`, `source_skus` и
  `collapsed_record_count`; поведение можно отключить через
  `CandidateGenerationConfig(collapse_empty_brand_exact_titles=False)`.

### Проверки

- `python3 -m pytest research/dedup/tests/test_candidates.py research/dedup/tests/test_embedding_candidates.py`
- `python3 -m compileall research/dedup`

## 2026-06-27 — Category-runs для `Кокосовое масло` и `Мыло`

### Зачем

Нужно расширить research benchmark за пределы `Соусы`, но не перетирать
готовые sauce CSV/отчёты. Особенно важно для `Мыло`: в локальном dev-кубе
есть другой проект с той же категорией, поэтому фильтр только по `Категория`
смешивает срезы.

### Что сделано

- Добавлен `research/dedup/category_runs.py`:
  - `sauces` -> проект `Соусы_тест`, категории `Соусы` / `Соус`;
  - `coconut_oil` -> проект `кокос_тест`, категория `Кокосовое масло`;
  - `soap` -> проект `мыло_тест`, категория `Мыло`.
- Ноутбуки `00`-`06` теперь читают `DEDUP_CATEGORY_RUN`.
- `research.dedup` автоматически загружает локальный `.env` при импорте;
  shell-переменные остаются приоритетнее `.env`.
- Для `sauces` сохранены legacy-пути:
  `research/dedup/data/*_sauces.csv`, `artifacts/reports/binary_threshold_*`.
- Для новых category-runs пути изолированы:
  `research/dedup/data/coconut_oil/`, `artifacts/reports/coconut_oil/`,
  `research/dedup/data/soap/`, `artifacts/reports/soap/`.
- Загрузки из `mpstats_products` и sales-volume join дополнительно фильтруют
  `__project_name`, чтобы `soap` не подхватил чужой проект `тест`.
- Загрузки из `mpstats_products` в `00`, `01`, `03` и `06` применяют
  sales-фильтр нижнего 20%-квантиля, чтобы старые кубы без нового
  processing-фильтра давали тот же research-вход.
- Аннотатор без аргументов тоже понимает `DEDUP_CATEGORY_RUN`; старый запуск
  без env по-прежнему открывает sauce gold-set.

### Проверки

- `ast.parse` code cells в `notebooks/00_eda.ipynb` -
  `notebooks/06_grouped_sku_demo.ipynb` — ok.
- Узкие pytest/compile проверки см. в commit/финальном ответе текущего
  изменения.

## 2026-06-27 — Demo notebook для сгруппированных SKU из DuckDB

### Зачем

После quality-отчёта нужен отдельный человеческий просмотр результата:
не метрики, а таблица реальных SKU из DuckDB, которые текущий research-run
склеил в `family` и `pack` группы.

### Что сделано

- Добавлен `notebooks/06_grouped_sku_demo.ipynb`.
- Notebook автоматически находит `mpstats.duckdb` через
  `MPSTATS_DUCKDB_PATH`, локальный `mpstats.duckdb` или desktop-путь.
- Читает `mpstats_products` по категории `Соус` / `Соусы`, агрегирует строки
  по `marketplace + Артикул`, затем join-ит с
  `research/dedup/data/fusion_components_sauces.csv`.
- Показывает coverage, склеенные family-группы, pack-группы, подробную
  карточку выбранной family и pair-edges, которые склеили группу.
- Сохраняет просмотренную витрину в ignored CSV:
  `artifacts/reports/dedup_grouped_sku_demo.csv`.

### Проверки

- `ast.parse` code cells в `notebooks/06_grouped_sku_demo.ipynb` — ok.
- `nbclient` на `notebooks/06_grouped_sku_demo.ipynb` — ok.

## 2026-06-27 — Единый downstream evaluation notebook

### Зачем

`05_clustering_resolution.ipynb` и `06_evaluation_report.ipynb` читали одни и
те же `binary_threshold_*` / `fusion_*` артефакты и дублировали смысловой
слой. Для чтения результата удобнее один финальный notebook после `04`.

### Что сделано

- Вместо двух downstream notebooks добавлен
  `notebooks/05_evaluation_report.ipynb`.
- В один notebook объединены:
  - состояние gold-set;
  - ranking methods на held-out `test`;
  - выбранный fusion-run;
  - family/pack link metrics;
  - размеры компонентов;
  - dangerous false merges и graph false/missed links.
- `notebooks/05_clustering_resolution.ipynb` и
  `notebooks/06_evaluation_report.ipynb` удалены из текущего workflow.
- `docs/AI_INDEX.md`, `docs/ARCHITECTURE.md` и `docs/USER_GUIDE.md`
  обновлены под flow `03 -> 04 -> 05`.

### Проверки

- `ast.parse` code cells в `notebooks/05_evaluation_report.ipynb` — ok.
- `nbclient` на `notebooks/05_evaluation_report.ipynb` — ok.

## 2026-06-26 — Fusion/pack grouping после binary benchmark

### Зачем

После `03_matching_comparison.ipynb` нужен отдельный слой, который не
пересчитывает model scores, а превращает выбранные binary pair predictions в
понятные группы товаров. Важно не смешивать два уровня: базовый товар
(`family`) и конкретную фасовку (`pack`).

### Что сделано

- В `research/dedup/fusion.py` добавлены helper-ы:
  - `select_fusion_run(...)` выбирает `method + threshold_strategy` только по
    `dev`; по умолчанию сначала берёт `threshold_weighted_cost` с весом
    продаж, а если weighted-строк нет — `threshold_cost_sensitive`;
  - `prepare_fusion_pair_edges(...)` добавляет `fusion_family_edge`,
    `same_pack_signature`, `fusion_pack_edge` и true/pred labels для графа.
- Добавлен `notebooks/04_fusion_pack_grouping.ipynb`:
  - читает `artifacts/reports/binary_threshold_summary.csv` и
    `artifacts/reports/binary_threshold_predictions.csv`;
  - использует видимые параметры `MY_FUSION_METHOD`,
    `MY_FUSION_THRESHOLD_STRATEGY`, `MY_FUSION_EVAL_SPLIT` прямо в notebook;
  - сохраняет `fusion_components_sauces.csv` и
    `fusion_pair_eval_sauces.csv` в ignored `research/dedup/data/`.
- Старые downstream notebooks сдвинуты:
  - `05_clustering_resolution.ipynb` теперь читает `fusion_*` и показывает
    family/pack link metrics;
  - `06_evaluation_report.ipynb` собирает итог по binary benchmark и fusion.
  Позже они объединены в `05_evaluation_report.ipynb`.
- `docs/AI_INDEX.md`, `docs/ARCHITECTURE.md`,
  `docs/THRESHOLD_CALIBRATION_REPORT.md` и `docs/USER_GUIDE.md` обновлены под
  flow `03 -> 04 -> 05 -> 06`; актуальный flow после объединения —
  `03 -> 04 -> 05`.

### Проверки

- `python3 -m pytest research/dedup/tests/test_fusion.py research/dedup/tests/test_clustering.py research/dedup/tests/test_threshold_calibration.py` — 14 passed.
- `python3 -m compileall research/dedup` — ok.
- `ast.parse` code cells в `notebooks/04_fusion_pack_grouping.ipynb`,
  `notebooks/05_clustering_resolution.ipynb`,
  `notebooks/06_evaluation_report.ipynb` — ok.
- `nbclient` на `notebooks/04_fusion_pack_grouping.ipynb`,
  `notebooks/05_clustering_resolution.ipynb`,
  `notebooks/06_evaluation_report.ipynb` — ok.

## 2026-06-26 — Forced binary threshold benchmark для SKU matching

### Зачем

Двухпороговая схема `threshold_auto_same` / `threshold_auto_diff` отправляла
слишком много пар в ручную проверку и усложняла текущий research benchmark.
На этом этапе нужна простая binary evaluation: выше порога — тот же базовый
товар, ниже порога — другой товар.

### Что сделано

- `research/dedup/threshold_calibration.py` переписан на forced binary
  контракт:
  - `same_base_product=1` для `exact_duplicate` и legacy
    `same_product_different_pack`;
  - `same_base_product=0` для `different_product`;
  - `predicted_binary = 1 if score >= threshold_same else 0`.
- Threshold выбирается только на dev для стратегий:
  - `threshold_max_f1`;
  - `threshold_cost_sensitive` с `FP_COST=5`, `FN_COST=1`;
  - `threshold_max_weighted_f1`, если есть объёмы продаж;
  - `threshold_weighted_cost`, если есть объёмы продаж.
- Weighted-метрики строятся по объёму продаж, не по выручке:
  `pair_weight = log1p(max(sales_volume_a, sales_volume_b))`.
  Если объём нельзя надёжно подтянуть, используется `pair_weight=1`,
  `weight_source=unit_weight_fallback`, weighted-метрики остаются пустыми.
- `notebooks/03_matching_comparison.ipynb` больше не пишет старые
  `matching_*` / workbook / pair-review artifacts и сохраняет только:
  - `artifacts/reports/binary_threshold_summary.csv`;
  - `artifacts/reports/binary_threshold_predictions.csv`;
  - `artifacts/reports/binary_threshold_by_volume_bucket.csv`, если sales
    volume доступен.
- `docs/ARCHITECTURE.md`, `docs/AI_INDEX.md`,
  `docs/THRESHOLD_CALIBRATION_REPORT.md` и `docs/USER_GUIDE.md` обновлены
  под новый benchmark.

### Проверки

- `python3 -m pytest research/dedup/tests/test_threshold_calibration.py` —
  7 passed.

## 2026-06-25 — Cost-sensitive threshold calibration для SKU matching

### Зачем

Один threshold по `macro_f1` плохо соответствует цене ошибок в SKU
deduplication: false merge разных товаров опаснее, чем пропуск дубля или
ручная проверка. Поэтому benchmark теперь оптимизирует безопасный auto-merge,
а не общую симметричную метрику.

### Что сделано

- Добавлен `research/dedup/threshold_calibration.py`:
  - binary target `same_base_product`;
  - `exact_duplicate` и legacy `same_product_different_pack` мапятся в
    positive, `different_product` — в negative;
  - если во входных данных уже есть `same_base_product`, используется она;
  - для каждого `method` на dev калибруются `threshold_auto_same` и
    `threshold_auto_diff`;
  - `threshold_auto_same` выбирается по максимальному recall среди порогов,
    где `auto_same_precision >= 0.97` и `false_merge_count <= 0`;
  - `threshold_auto_diff` выбирается по максимальному coverage среди порогов,
    где `auto_diff_precision >= 0.95`.
- `notebooks/03_matching_comparison.ipynb` теперь применяет dev-пороги к test
  без переобучения и сохраняет компактный отчёт:
  - `artifacts/reports/threshold_summary.xlsx`;
  - `artifacts/reports/threshold_pair_review.csv`;
  - внутри workbook остаются dev/test calibration, forced/triage confusion
    matrices и pair review.
- `notebooks/04_clustering_resolution.ipynb` выбирает метод только по dev
  calibration и строит graph edge только из `auto_same`; `manual_review` не
  превращается в автосклейку.
- `notebooks/05_evaluation_report.ipynb` показывает dev calibration как
  главный выборочный отчёт, а test — только как held-out проверку.
- Добавлен `docs/THRESHOLD_CALIBRATION_REPORT.md` с критерием auto-merge,
  объяснением, почему `macro_f1` не production-критерий, и списком артефактов.

### Проверки

- `python3 -m pytest research/dedup/tests/test_threshold_calibration.py` — 5 passed.
- `python3 -m pytest research/dedup/tests` — 42 passed.
- `python3 -m compileall research/dedup` — ok.
- `ast.parse` code cells в `notebooks/03_matching_comparison.ipynb`,
  `notebooks/04_clustering_resolution.ipynb`,
  `notebooks/05_evaluation_report.ipynb` — ok.

## 2026-06-25 — Binary pair labels без different-pack класса

### Зачем

Pack/multipack не должен быть отдельным ML-классом: модель должна отвечать,
тот же это базовый товар или другой товар. Конкретная фасовка выделяется
после модели deterministic правилами, чтобы не путать classifier и не
занижать pairwise-метрики.

### Что сделано

- Pairwise label contract сужен до двух классов:
  `exact_duplicate`, `different_product`; `uncertain` остаётся только как
  ручная служебная метка вне метрик.
- Legacy `same_product_different_pack` больше не генерируется кодом и не
  входит в evaluation; такие старые строки нужно заменить на
  `exact_duplicate` перед прогоном метрик.
- Fusion больше не использует pack/multipack для отдельного класса:
  brand mismatch остаётся жёстким negative signal, дальше решение идёт по
  rerank score и порогам.
- Pack-граф в `04_clustering_resolution.ipynb` отделён от ML-label:
  family edge = binary `exact_duplicate`, pack edge = `exact_duplicate` +
  deterministic `same_pack_signature`.
- Аннотатор теперь сохраняет только `w -> exact_duplicate`,
  `s -> different_product`, `d -> uncertain`.

## 2026-06-25 — Qwen reranker safe mode для Mac MPS

### Зачем

`Qwen/Qwen3-Reranker-4B` на Mac может падать при загрузке с
`MPS backend out of memory`: модель почти целиком забивает доступный MPS
лимит, поэтому уменьшение scoring batch само по себе не помогает.

### Что сделано

- В `ModelSpec` добавлен параметр `device`, который прокидывается в
  `sentence_transformers.CrossEncoder`.
- Alias `reranker_qwen3_4b` / `qwen3_4b` теперь по умолчанию грузится на
  `cpu`, а `batch_size` остаётся `1`.
- Добавлен лёгкий alias `reranker_qwen3_0_6b` / `qwen3_0_6b` для быстрого
  Qwen-smoke, если 4B на CPU слишком медленный.
- `03_matching_comparison.ipynb` показывает `device` в таблице benchmark и
  прямо подсказывает ставить `MY_RERANKER_MAX_PAIRS = 12-30` для первого
  CPU-прогона Qwen-4B.

### Как пользоваться

- Для проверки 4B без MPS OOM:
  `MY_RERANKER_MODELS = ["qwen3_4b"]`, `MY_RERANKER_MAX_PAIRS = 12`.
- Если нужно быстрее проверить именно Qwen-семейство:
  `MY_RERANKER_MODELS = ["qwen3_0_6b"]`.
- Перед повторным запуском после MPS OOM лучше перезапустить Jupyter kernel,
  чтобы освободить уже занятые моделью/torch ресурсы.

## 2026-06-25 — Polza.ai backend для online embeddings

### Зачем

Online-модели в research-слое должны подключаться через Polza.ai, а не через
произвольные provider-переменные. Это сохраняет один понятный контракт для
API-ключей, model ids и endpoint-ов.

### Что сделано

- В `research/dedup/model_registry.py` добавлен backend `polza_embedding`.
- Добавлены aliases:
  - `polza_embedding_3_small` -> `openai/text-embedding-3-small`;
  - `polza_embedding_3_large` -> `openai/text-embedding-3-large`;
  - `polza_qwen3_embedding_4b` -> `qwen/qwen3-embedding-4b`.
- Оригинальные Polza model ids тоже принимаются напрямую:
  `openai/text-embedding-3-small`, `openai/text-embedding-3-large`,
  `qwen/qwen3-embedding-4b`; backend определяется registry автоматически.
- Добавлен `PolzaEmbeddingModel` с интерфейсом `encode(...)`, совместимым с
текущим notebook/matcher-кодом:
  - POST `{POLZA_BASE_URL}/embeddings`;
  - `Authorization: Bearer ...`;
  - request body `model`, `input`, `encoding_format`;
  - `dimensions` отправляется только если явно задан.
- Добавлен helper `fetch_polza_models(...)` для публичного каталога:
  `/models?type=embedding&include_providers=true`.
- `BiEncoderMatcher` теперь умеет backend `polza_embedding` и не требует
  установленный `sentence-transformers` для Polza aliases.
- `notebooks/01_candidate_generation.ipynb` и
  `notebooks/03_matching_comparison.ipynb` получили env-переменные:
  `DEDUP_EMBEDDING_BACKEND` и `DEDUP_BI_ENCODER_BACKEND`.

### Как пользоваться

- Ключ: `POLZA_API_KEY` или `POLZA_AI_API_KEY`.
- Base URL по умолчанию: `https://polza.ai/api/v1`; override:
  `POLZA_BASE_URL`.
- Candidate generation через Polza:
  `DEDUP_EMBEDDING_MODEL=openai/text-embedding-3-small`.
- Старый alias тоже работает:
  `DEDUP_EMBEDDING_MODEL=polza_embedding_3_small`.
- `DEDUP_EMBEDDING_BACKEND=polza_embedding` нужен только для нового прямого
  Polza model id, которого ещё нет в registry.
- Bi-encoder comparison через Polza:
  `DEDUP_BI_ENCODER_MODEL=openai/text-embedding-3-small`.
- `DEDUP_BI_ENCODER_BACKEND=polza_embedding` нужен только для нового прямого
  Polza model id, которого ещё нет в registry.

### Проверки

- `python3 -m pytest research/dedup/tests/test_model_registry.py
  research/dedup/tests/test_matchers.py` — 20 passed.
- `python3 -m compileall research/dedup` — ok.
- `ast.parse` всех code cells в `01_candidate_generation.ipynb` и
  `03_matching_comparison.ipynb` — ok.
- `nbclient` на `notebooks/03_matching_comparison.ipynb` в лёгком режиме
  (`DEDUP_RUN_BI_ENCODER=0`, `DEDUP_RUN_CROSS_ENCODER=0`,
  `DEDUP_RERANKER_BENCHMARK_MODELS=`) — ok.

## 2026-06-25 — Model registry и локальный cache/pool

### Зачем

Модели в research notebooks больше не должны выглядеть как магическая загрузка
прямо из ячейки: нужно явно видеть, какой alias выбран, какой реальный model id
будет загружен и куда он попадёт на диске.

### Что сделано

- Добавлен `research/dedup/model_registry.py`:
  - registry alias-ов `embedding_e5_small`, `bi_encoder_e5_small`,
    `cross_encoder_mmarco`, `reranker_qwen3_4b`, `reranker_bge_v2_m3`,
    `reranker_jina_v3`;
  - `ModelManager` с единым cache dir `research/dedup/models/`;
  - override cache dir через `DEDUP_MODEL_CACHE_DIR`;
  - offline-режим `DEDUP_MODEL_LOCAL_ONLY=1`;
  - in-process pool, чтобы повторный запуск ячеек не пересоздавал уже
    загруженную модель в рамках одного Python kernel.
- `BiEncoderMatcher`, `CrossEncoderMatcher`, `JinaRerankerMatcher` теперь
  грузят реальные модели через `ModelManager`, но сохраняют `model_factory`
  для unit tests и fake-моделей.
- `notebooks/01_candidate_generation.ipynb` больше не импортирует
  `SentenceTransformer` напрямую: embedding модель берётся через
  `MODEL_MANAGER.load_sentence_transformer(DEDUP_EMBEDDING_MODEL)`.
- `notebooks/03_matching_comparison.ipynb` показывает alias/input,
  реальный model id, cache dir и offline-флаг для bi-encoder/cross-encoder и
  reranker benchmark.
- `research/dedup/models/` добавлен в `.gitignore`.

### Как пользоваться

- Обычный запуск: ничего не менять, defaults используют локальный кэш
  `research/dedup/models/`.
- Поменять место кэша:
  `DEDUP_MODEL_CACHE_DIR=/path/to/models`.
- Не ходить в сеть и брать только уже скачанное:
  `DEDUP_MODEL_LOCAL_ONLY=1`.
- Выбрать модель в notebook:
  `DEDUP_EMBEDDING_MODEL=embedding_e5_small`,
  `DEDUP_BI_ENCODER_MODEL=bi_encoder_e5_small`,
  `DEDUP_CROSS_ENCODER_MODEL=cross_encoder_mmarco`,
  `RERANKER_BENCHMARK_MODELS = ["reranker_qwen3_4b", "reranker_bge_v2_m3", "reranker_jina_v3"]`.
- Лёгкий smoke-запуск notebook-3 без тяжёлых моделей:
  `DEDUP_RUN_BI_ENCODER=0 DEDUP_RUN_CROSS_ENCODER=0 DEDUP_RERANKER_BENCHMARK_MODELS=`.

### Проверки

- `python3 -m pytest research/dedup/tests/test_model_registry.py
  research/dedup/tests/test_matchers.py` — 15 passed.
- `python3 -m compileall research/dedup` — ok.
- `ast.parse` всех code cells в `01_candidate_generation.ipynb` и
  `03_matching_comparison.ipynb` — ok.
- `nbclient` на `notebooks/03_matching_comparison.ipynb` в лёгком режиме
  (`DEDUP_RUN_BI_ENCODER=0`, `DEDUP_RUN_CROSS_ENCODER=0`,
  `DEDUP_RERANKER_BENCHMARK_MODELS=`) — ok.
- `git diff --check` — ok.

## 2026-06-25 — BGE reranker в benchmark list

### Зачем

В all-model benchmark должен быть не только Qwen/Jina, но и BGE reranker как
сильный multilingual cross-encoder вариант.

### Что сделано

- Добавлен alias `reranker_bge_v2_m3`:
  - model id: `BAAI/bge-reranker-v2-m3`;
  - backend: `sentence_transformers.CrossEncoder`;
  - короткие aliases: `bge_v2_m3`, `bge_m3`.
- `notebooks/03_matching_comparison.ipynb` теперь по умолчанию запускает
  `["reranker_qwen3_4b", "reranker_bge_v2_m3", "reranker_jina_v3"]`.

### Проверки

- `python3 -m pytest research/dedup/tests/test_model_registry.py
  research/dedup/tests/test_matchers.py` — 20 passed.
- `python3 -m compileall research/dedup` — ok.
- `ast.parse` code cells в `notebooks/03_matching_comparison.ipynb` — ok.
- `nbclient` на `notebooks/03_matching_comparison.ipynb` в лёгком режиме
  (`DEDUP_RUN_BI_ENCODER=0`, `DEDUP_RUN_CROSS_ENCODER=0`,
  `DEDUP_RERANKER_BENCHMARK_MODELS=`) — ok.

## 2026-06-25 — All-model benchmark в notebook-3

### Зачем

После первых ошибок cross-encoder стало видно, что часть пар сложна даже для
человека: товары очень близки по тексту, но отличаются назначением, вкусом или
pack. Перед обучением своей fusion-модели полезно быстро проверить более
сильные готовые reranker-модели на том же gold-set.

### Что сделано

- `CrossEncoderMatcher` получил:
  - отдельное `method_name`, чтобы в одном notebook можно было сравнивать
    несколько cross-encoder моделей;
  - `prompts` / `default_prompt_name` для instruction-aware rerankers вроде
    Qwen3;
  - `trust_remote_code` на случай моделей с кастомным кодом.
- Добавлен `research/dedup/matchers/jina_reranker.py`:
  - lazy load через `transformers.AutoModel`;
  - вызов `model.rerank(query, documents)`;
  - тот же pairwise score -> fusion -> label контракт, что у остальных
    matchers.
- В конец `notebooks/03_matching_comparison.ipynb` добавлен общий
  benchmark всех текущих matching-моделей:
  - `rule_based_fuzzy`;
  - `bi_encoder_zero_shot`;
  - `cross_encoder_zero_shot`;
  - `Qwen/Qwen3-Reranker-4B`;
  - `BAAI/bge-reranker-v2-m3`;
  - `jinaai/jina-reranker-v3`;
  - после добавления model registry запуск управляется alias-ами в
    `RERANKER_BENCHMARK_MODELS`, а model id/cache/backend берутся из
    `research/dedup/model_registry.py`;
  - итоговая общая таблица сохраняется в
    `all_model_benchmark_summary_sauces.csv` и
    `all_model_benchmark_predictions_sauces.csv`;
  - отдельные результаты новых reranker-моделей дополнительно сохраняются в
    `reranker_benchmark_summary_sauces.csv` и
    `reranker_benchmark_predictions_sauces.csv`;
  - основные `matching_*` CSV не перетираются.
- `requirements-research.txt` обновлён под актуальные reranker dependencies:
  `sentence-transformers>=5.0`, `transformers>=4.51`.

### Проверки

- `python3 -m pytest research/dedup/tests` — 27 passed.
- `python3 -m compileall research/dedup` — ok.
- `nbclient` на `notebooks/03_matching_comparison.ipynb` в лёгком режиме
  до снятия env-gate (`DEDUP_RUN_BI_ENCODER=0`,
  `DEDUP_RUN_CROSS_ENCODER=0`) — ok.

### Следующий шаг

Запустить общий benchmark сначала с
`RERANKER_BENCHMARK_MAX_PAIRS = 120`, затем поставить `0` для всего gold-set,
если машина вывозит. Сравнивать строки
`all_model_benchmark_calibrated / test`.

## 2026-06-25 — Ready-made cross-encoder rerank в notebook-3

### Зачем

Архитектура предусматривает cross-encoder как более внимательную проверку
пары после FAISS/bi-encoder retrieval. После первых baseline-метрик стало
видно, что главная проблема — false merges на похожих вкусах/типах одного
бренда. Поэтому следующий шаг — добавить готовый cross-encoder без
дообучения и проверить, снижает ли он опасные ошибки.

### Что сделано

- Добавлен `research/dedup/matchers/cross_encoder.py`:
  - lazy load через `sentence_transformers.CrossEncoder`;
  - default model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`;
  - используется raw rerank score модели; порог в notebook-3 подбирается
    по фактическому диапазону score на dev;
  - graceful skip, если `sentence-transformers` или модель недоступны.
- В конец `notebooks/03_matching_comparison.ipynb` дописана новая секция
  `cross_encoder_zero_shot`:
  - использует те же `dev`/`test`;
  - калибрует `threshold_high` на dev;
  - сравнивает cross-encoder с уже посчитанными baselines;
  - дописывает cross-encoder results в `matching_summary_sauces.csv`,
    `matching_predictions_sauces.csv`, `matching_false_merges_sauces.csv`.

### Следующий шаг

### Первые цифры

Готовый cross-encoder успешно запустился, но пока не обогнал
`rule_based_fuzzy` на held-out test:

| Метод | macro-F1 | `exact_duplicate` precision | false-merge rate |
| --- | ---: | ---: | ---: |
| `rule_based_fuzzy` | 0.606 | 0.579 | 18.5% |
| `cross_encoder_zero_shot` | 0.572 | 0.500 | 21.9% |
| `bi_encoder_zero_shot` | 0.470 | 1.000 | 15.2% |

Вывод: готовый reranker сам по себе не решает товарный matching. Это
полезный результат для защиты: следующая итерация должна быть fine-tuning
cross-encoder на наших парах или LLM-judge для спорных случаев.

### Проверки

- `python3 -m pytest research/dedup/tests` — 24 passed.
- `python3 -m compileall research/dedup` — ok.
- `nbclient` на `notebooks/03_matching_comparison.ipynb` с cross-encoder —
  ok, CSV matching-артефакты обновлены.

### Следующий шаг

Перезапустить `04` и `05`, чтобы graph/report подхватили новый метод, затем
решать: fine-tuning cross-encoder или LLM-judge на low-confidence/false-merge
парах.

## 2026-06-24 — Gold-set validation, calibrated matching, clustering/report

### Зачем

После ручной разметки нужно было перейти от scaffolding к честным метрикам:
проверить CSV, не тюнить пороги на том же наборе, на котором репортятся
результаты, и собрать первый graph-resolution/report слой.

### Что сделано

- Проверен `research/dedup/data/labeling_sauces.csv`:
  - 400 строк, пустых labels нет;
  - invalid labels нет;
  - 379 пар входят в 3-class evaluation;
  - 21 пара оставлена как `uncertain`;
  - дубликатов unordered pair и self-pairs нет.
- `notebooks/03_matching_comparison.ipynb` обновлён:
  - default bi-encoder model: `intfloat/multilingual-e5-small`;
  - добавлен stratified dev/test split;
  - `threshold_high` калибруется на dev;
  - held-out test используется для отчётных метрик;
  - сохраняются локальные CSV-артефакты:
    `matching_summary_sauces.csv`, `matching_predictions_sauces.csv`,
    `matching_false_merges_sauces.csv`.
- Добавлен `research/dedup/clustering.py`:
  - connected components для family graph (`exact_duplicate` +
    `same_product_different_pack`);
  - pack graph только по `exact_duplicate`;
  - helper для pair-level component flags и component-size summary.
- Добавлен `notebooks/04_clustering_resolution.ipynb`:
  - читает calibrated predictions из notebook-3;
  - выбирает лучший calibrated method по held-out `test`;
  - строит partial true/predicted family и pack components;
  - сохраняет `clustering_components_sauces.csv` и
    `clustering_pair_eval_sauces.csv`.
- Добавлен `notebooks/05_evaluation_report.ipynb`:
  - собирает label QA, matching summary, false-merge examples и partial
    clustering metrics в один отчётный notebook.

### Получившиеся baseline-цифры

Текущий лучший calibrated baseline по held-out macro-F1 — `rule_based_fuzzy`:

| Метрика | Значение |
| --- | ---: |
| Held-out pairs | 151 |
| `threshold_high` | 0.86 |
| macro-F1 | 0.606 |
| `exact_duplicate` precision | 0.579 |
| false-merge rate | 18.5% |

Zero-shot `intfloat/multilingual-e5-small` без reranker слишком агрессивен
на default threshold: много false merges. При `threshold_high=1.0` достигает
высокой precision для `exact_duplicate`, но почти теряет recall; это полезный
аргумент в пользу cross-encoder/reranker, а не финальное решение.

Partial graph metrics на held-out test для выбранного `rule_based_fuzzy`:

| Graph | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| family | 0.594 | 0.695 | 0.641 |
| pack | 0.579 | 0.379 | 0.458 |

Важно: это partial sanity-check по sampled gold-set pairs, не полная
cluster-quality метрика по всей категории.

### Проверки

- `python3 -m pytest research/dedup/tests/test_clustering.py` — 3 passed.
- `python3 -m compileall research/dedup` — ok.
- `nbclient` на `notebooks/03_matching_comparison.ipynb` — ok.
- `nbclient` на `notebooks/04_clustering_resolution.ipynb` — ok.
- `nbclient` на `notebooks/05_evaluation_report.ipynb` — ok.

### Следующий шаг

Перед финальной технологией нужно добавить более сильный reranker/fusion:
cross-encoder или LLM-judge для low-confidence/false-merge-prone пар.
Также стоит вручную пересмотреть несколько sanity-suspect labels, если
нужен максимально чистый held-out report.

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
  embeddings -> FAISS top-k -> `candidates_sauces.csv`.
- После добавления model registry default alias для notebook:
  `embedding_e5_small` (`intfloat/multilingual-e5-small`); модель можно
  заменить через `DEDUP_EMBEDDING_MODEL`.
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
