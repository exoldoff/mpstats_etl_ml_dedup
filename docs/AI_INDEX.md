# AI Index

Короткий индекс проекта для быстрого входа без полного сканирования
репозитория. Это роутер для агента: что читать, где искать код и какие
границы не пересекать.

## Обязательный порядок чтения

1. `AGENTS.md` — правила работы, коммиты, проверки, ограничения.
2. `docs/AI_INDEX.md` — этот индекс.
3. `docs/AGENT_NAVIGATION.md` — операционный навигатор для нетривиальных
   задач, документации и выбора между треками.
4. `docs/ARCHITECTURE.md` — архитектурный контекст и research-направление,
   особенно для задач по SKU deduplication.
5. `docs/ARCHITECTURE_PROGRESS.md` — что уже сделано на текущем research-этапе.
6. `README.md` — только если нужен человеческий quick start или общая картина.
7. Релевантные файлы по задаче из разделов ниже.

Не сканируй весь проект без причины: сначала определи трек задачи.

## Система навигации для агентов

- `AGENTS.md` задаёт обязательные правила: границы, коммиты, проверки и
  финальный формат.
- `docs/AI_INDEX.md` остаётся коротким роутером по проекту.
- `docs/AGENT_NAVIGATION.md` описывает рабочий алгоритм: как выбрать маршрут,
  какие источники читать, какие проверки запускать и как не смешивать
  research, production web-app и docs-only задачи.
- `docs/ARCHITECTURE.md` фиксирует стабильную research-методологию, а
  `docs/ARCHITECTURE_PROGRESS.md` хранит mutable статус.

Если задача выглядит как "разобраться в проекте", "обновить документацию",
"куда смотреть" или может затронуть больше одного трека, сначала используй
`docs/AGENT_NAVIGATION.md`, затем читай только выбранные источники.

## Два рабочих трека

| Трек | Основные пути | Правило |
| --- | --- | --- |
| Локальная production web-app | `mpstats_app/`, `web/`, `pipeline/`, `classifiers/` | Единственный пользовательский интерфейс проекта. Здесь работают реальные workflow, настройки, CSV-справочники, DuckDB и API. |
| Research SKU deduplication | `research/dedup/`, `notebooks/`, `docs/ARCHITECTURE.md`, `docs/ARCHITECTURE_PROGRESS.md` | Песочница для сравнения методов дедупликации. До выбора технологии не переносить код в `pipeline/` и не трогать `mpstats_app/`. |

Если пользователь явно говорит `research-only`, работай только в
`research/dedup/`, `notebooks/` и research-документации.

## Markdown-документы

| Файл | Статус | Когда читать |
| --- | --- | --- |
| `AGENTS.md` | источник правил | всегда перед работой |
| `docs/AI_INDEX.md` | индекс для агентов | всегда после `AGENTS.md` |
| `docs/AGENT_NAVIGATION.md` | навигационная система для агентов | docs/onboarding задачи, выбор между треками, нетривиальные изменения |
| `docs/ARCHITECTURE.md` | источник research-архитектуры | задачи по dedup, matching, evaluation, notebooks, ML-методологии |
| `docs/ARCHITECTURE_PROGRESS.md` | журнал research-этапов | понять, что уже сделано, какие CSV/ноутбуки/проверки актуальны |
| `docs/THRESHOLD_CALIBRATION_REPORT.md` | критерии SKU matching benchmark | cost-sensitive пороги, manual review, почему macro-F1 не главный критерий |
| `docs/DEDUP_MODEL_HF_AUDIT.md` | HF-аудит моделей dedup research | prefixes, pooling, prompts, reranker settings, known mismatches |
| `docs/DEDUP_FINE_TUNING_EXPERIMENT_PLAN.md` | план первого supervised fine-tuning эксперимента SKU dedup | после завершения ручной разметки: состав gold/training set, выбор reranker-моделей, split без leakage, метрики и порядок обучения |
| `docs/DEDUP_TRAINING_RUNBOOK.md` | команды запуска первого fine-tuning цикла | подготовка frozen split, запуск smoke/full training на H200, scoring fine-tuned моделей и threshold calibration |
| `docs/DEDUP_PRODUCTION_OPTIMIZATION_AUDIT.md` | аудит прод-готовности dedup flow | что оптимизировать перед переносом из notebooks/research в production: FAISS/mmap, artifacts, manifests, safety gates |
| `README.md` | краткий обзор | запуск, структура, состояние проекта |
| `docs/USER_GUIDE.md` | пользовательская инструкция web-app | изменения UI, workflow, расчётов, статусов, справочника, классификатора |
| `docs/PIPELINE_OVERVIEW.md` | краткое объяснение pipeline | вопросы про текущие шаги pipeline и pandas/SQL |
| `filter.md` | справочник MPStats-фильтров | задачи с `filterModel` и CSV-колонкой `Фильтр` |
| `.cursor/agents/mpstats-tasks-handbook.md` | узкий handbook | обновление `TASKS` из CSV-справочника |
| `справочник tasks архив.md` | append-only архив | после обновления `TASKS`, только дописывать недостающие задачи |

## Research Dedup: текущая архитектура

Источник: `docs/ARCHITECTURE.md`.

- Scope: SKU deduplication для category-run `sauces`, `coconut_oil`,
  `soap`. Базовый legacy-run `sauces` остаётся совместимым со старыми
  `*_sauces.csv`; новые категории пишут в отдельные папки и не перетирают
  sauce-артефакты.
- Research env: локальный `.env` в корне репозитория загружается автоматически
  при импорте `research.dedup`. Там можно держать `DEDUP_CATEGORY_RUN`,
  `MPSTATS_DUCKDB_PATH`, `POLZA_API_KEY`/`POLZA_AI_API_KEY`, `POLZA_BASE_URL`
  и остальные `DEDUP_*`; переменные, явно заданные в shell, приоритетнее `.env`.
- Переключатель research-ноутбуков: `DEDUP_CATEGORY_RUN=sauces|coconut_oil|soap`.
  Встроенные project filters:
  - `sauces` -> `Соусы_тест` / `Соус`, `Соусы`;
  - `coconut_oil` -> `кокос_тест` / `Кокосовое масло`;
  - `soap` -> `мыло_тест` / `Мыло`.
- Входной контракт: `mpstats_products` после classification, где
  `Артикул` — id карточки, `SKU` — title товара.
- Готовые структурные поля:
  - `unit_amount` -> `Вес, кг (ед.)`
  - `total_amount` -> `Вес, кг`
  - `multipack_count` выводится как `round(total/unit)`
  - `brand` -> `Бренд`
- Выходные классы пары:
  - `exact_duplicate`
  - `different_product`
- Бизнес-правило для модели: `200 г` vs `3x200 г` — положительная пара
  `exact_duplicate`, если это тот же базовый товар; конкретная фасовка
  выделяется после модели deterministic pack-правилами.
- Brand — полезный сигнал, но не абсолютный stop-gate при missing brand.
- Fusion: brand signal + rerank score; deterministic pack rules живут
  отдельным post-processing слоем.
- Метрики: binary per-class precision/recall/F1, confusion matrix, false-merge
  как дорогая ошибка, retrieval recall@k отдельно от pairwise classification,
  clustering metrics отдельно от pairwise metrics.

Research-код остаётся независимым: `research/dedup/` не должен импортировать
`pipeline/` или `mpstats_app/`.

## Research Dedup: карта файлов

| Файл/папка | Роль |
| --- | --- |
| `research/dedup/candidates.py` | product records и вспомогательные feature-флаги для candidate pairs |
| `research/dedup/embedding_candidates.py` | FAISS top-k candidate generation по dense embeddings |
| `research/dedup/category_runs.py` | единый конфиг category-run: project/category aliases и изолированные пути артефактов |
| `research/dedup/labeling.py` | stratified sampling для ручной разметки gold-set |
| `research/dedup/metrics.py` | dependency-light classification report и confusion matrix |
| `research/dedup/fusion.py` | выбор fusion-run по dev-summary и подготовка family/pack pair edges |
| `research/dedup/model_registry.py` | единый registry/manager/pool для local-моделей и Polza.ai online embeddings |
| `research/dedup/matchers/` | baseline matching engines A/B и общий интерфейс |
| `research/dedup/training/` | research-only freeze/split/train/score/calibrate scripts для supervised reranker fine-tuning |
| `research/dedup/threshold_calibration.py` | forced binary dev calibration для `threshold_same`, cost/weighted strategies и compact CSV |
| `research/dedup/annotator.py` | helper для ручной разметки |
| `research/dedup/tests/` | узкие тесты research-модулей |
| `tools/dedup_labeling_bot/` | отдельный Telegram-бот для многопользовательской разметки `labeling_*.csv` |
| `docker/dedup-training/` | Docker image и entrypoint для H200/A5000 fine-tuning runtime |
| `research/dedup/data/` | локальные CSV-артефакты, игнорируются `.gitignore` |
| `notebooks/00_eda.ipynb` | EDA по выбранному `DEDUP_CATEGORY_RUN` |
| `notebooks/01_candidate_generation.ipynb` | FAISS embedding blocking + supplemental training coverage pairs, генерация `candidates_<suffix>.csv` |
| `notebooks/02_labeling_dataset.ipynb` | генерация `labeling_<suffix>.csv` для одного run или multi-category labeling CSV для fine-tuning reranker |
| `notebooks/03_matching_comparison.ipynb` | сравнение baseline/reranker methods на размеченном gold-set выбранного run |
| `notebooks/04_fusion_pack_grouping.ipynb` | post-03 fusion для одного или нескольких category-runs, family/pack grouping, graph-quality диагностика, 3D components и CSV `fusion_*_<suffix>.csv` |
| `notebooks/05_grouped_sku_demo.ipynb` | demo-flow дедупликатора: DuckDB-кандидаты -> bi-encoder/FAISS -> cross-encoder -> группы SKU |

Локальные CSV/backup/model artifacts — рабочие данные, они не коммитятся и
могут отличаться между машинами. Текущий research-state, frozen split,
fine-tuning outputs и report paths смотри в `docs/ARCHITECTURE_PROGRESS.md`
и узких `docs/DEDUP_*.md`; если пользователь просит выводы по CSV/DB,
сначала инспектируй конкретные указанные artifacts.

Для новых category-runs пути изолированы:

- `research/dedup/data/coconut_oil/` и `artifacts/reports/coconut_oil/`;
- `research/dedup/data/soap/` и `artifacts/reports/soap/`.

## Research Dedup: текущий статус

Смотри подробности в `docs/ARCHITECTURE_PROGRESS.md`.

Сделано:

- Ноутбуки `00`-`06` принимают `DEDUP_CATEGORY_RUN`; `sauces` сохраняет
  legacy-пути, `coconut_oil` и `soap` пишут в изолированные папки.
- `00_eda.ipynb` уже был выполнен по категории `Соусы`; для новых категорий
  его нужно запускать отдельно с нужным `DEDUP_CATEGORY_RUN`.
- Candidate generation по `marketplace + Артикул`, без потери
  cross-marketplace дублей: primary blocking теперь идёт через dense
  embeddings + FAISS top-k. Если `Подкатегория` заполнена, основной FAISS
  top-k считается внутри одной подкатегории; пустые подкатегории и маленький
  `global_safety` ищут full-global, а полностью пустой срез автоматически
  откатывается в старый global FAISS. Default notebook alias:
  `embedding_e5_small` (`intfloat/multilingual-e5-small`); заменить можно
  через `DEDUP_EMBEDDING_MODEL`. Для обучающей разметки candidate CSV также
  добавляет supplemental пары вне FAISS top-k:
  `supplemental_lexical_overlap`, `supplemental_same_brand_pack`,
  `supplemental_cross_marketplace_random`,
  `supplemental_random_control`; финальный cap сохраняет часть таких строк.
- Stratified labeling dataset с отдельной стратой
  `cross_marketplace_candidate`; `02_labeling_dataset.ipynb` также умеет
  собрать multi-category датасет, например 3000 строк по
  `sauces,coconut_oil,soap` через `DEDUP_LABELING_CATEGORY_RUNS`.
  При перегенерации `02` сохраняет уже заполненные `label`-строки из
  существующего `LABELING_PATH`, пишет backup `*_preserved_labels.csv`,
  добирает незаполненный остаток, ограничивает пары с двумя известными
  разными брендами через `DEDUP_LABELING_MAX_DIFFERENT_BRAND_SHARE`
  (по умолчанию 20%) и поддерживает разные random-срезы через
  `DEDUP_LABELING_BATCH_ID`.
- Telegram-бот `tools/dedup_labeling_bot/` в unique-режиме назначает новый
  `DEDUP_TELEGRAM_BATCH_SIZE` не последовательным блоком CSV, а spread-random
  по разным частям ещё доступной выборки; навигация внутри уже выданного
  батча остаётся стабильной.
- Baseline matching scaffold:
  - A: rule-based fuzzy + structural fusion.
  - B: zero-shot bi-encoder с graceful skip без `sentence-transformers`.
  - D0: готовый cross-encoder rerank без дообучения, дописан новой секцией
    в конец `03_matching_comparison.ipynb`.
- Research-модели больше не загружаются напрямую из notebook cells:
  `research/dedup/model_registry.py` задаёт alias -> backend/model id,
  `ModelManager` управляет локальным кэшем `research/dedup/models/`,
  in-process pool и offline-флагом `DEDUP_MODEL_LOCAL_ONLY=1`.
  Cache dir можно заменить через `DEDUP_MODEL_CACHE_DIR`.
- Перед добавлением/сменой embedding/reranker модели смотри
  `docs/DEDUP_MODEL_HF_AUDIT.md`: там зафиксированы HF prefixes, pooling,
  prompts, max length и known mismatches для E5, BERTA, RuModernBERT, Qwen,
  BGE, Jina и cross-encoders.
- Online embedding-модели подключаются только через Polza.ai backend:
  известные Polza model ids `openai/text-embedding-3-small`,
  `openai/text-embedding-3-large`, `qwen/qwen3-embedding-4b` можно писать
  напрямую без backend override; старые aliases `polza_embedding_3_small`,
  `polza_embedding_3_large`, `polza_qwen3_embedding_4b` остаются рабочими.
  Ключ берётся из `POLZA_API_KEY` или `POLZA_AI_API_KEY`, base URL по
  умолчанию `https://polza.ai/api/v1`. `DEDUP_EMBEDDING_BACKEND=polza_embedding`
  или `DEDUP_BI_ENCODER_BACKEND=polza_embedding` нужны только для нового
  прямого Polza model id, которого ещё нет в registry.
- В конце `03_matching_comparison.ipynb` есть общий forced binary benchmark
  всех matching-моделей на frozen split:
  `research/dedup/data/training/dedup_pairs_final_pair_stratified_split.csv`
  (`2465` размеченных пар) по умолчанию. Сравниваются `rule_based_fuzzy`,
  `bi_encoder_zero_shot`, `cross_encoder_zero_shot`,
  `reranker_qwen3_4b`, `reranker_bge_v2_m3`, `reranker_jina_v3` и
  fine-tuned модели вроде `ft_bge_reranker_v2_m3`. Новые zero-shot
  reranker-модели выбираются alias-ами registry в
  `RERANKER_BENCHMARK_MODELS` или через env
  `DEDUP_RERANKER_BENCHMARK_MODELS`; fine-tuned модели добавляются в
  `MY_FINE_TUNED_MODELS` через `model_path` или готовый `score_path`. Главный
  отчёт frozen/fine-tuning comparison по умолчанию лежит в
  `artifacts/reports/fine_tuning/binary_threshold_summary.csv`.
  Test split используется только для финальной проверки выбранных на dev
  `threshold_same`.
  `reranker_qwen3_4b` по умолчанию грузится на CPU, потому что на MPS с
  лимитом около 9GB падает по памяти; для быстрого Qwen-smoke есть alias
  `qwen3_0_6b`.
- `03_matching_comparison.ipynb` запускается без разметки и показывает
  заглушки вместо падения.
- Production processing и research-ноутбуки фильтруют хвост продаж:
  строки с `Продажи < 15` отбрасываются. В ноутбуках это дополнительно
  применяется при чтении старых `mpstats_products` из DuckDB.

Актуальный downstream:

- Gold-set для threshold calibration использует binary target
  `same_base_product`: `exact_duplicate` и legacy
  `same_product_different_pack` считаются positive, `different_product` —
  negative. Если во входных данных уже есть колонка `same_base_product`, она
  считается источником правды.
- `notebooks/03_matching_comparison.ipynb` теперь сохраняет готовый `split`
  из frozen CSV; stratified dev/test split остаётся fallback только для
  старых labeling CSV без split. На dev подбирается один `threshold_same` для
  `threshold_max_f1`, `threshold_cost_sensitive` и weighted-стратегий, если
  доступны объёмы продаж. Raw predictions и старые `matching_*` CSV не
  перезаписываются; compact outputs пишутся в reports-папку notebook
  comparison: `binary_threshold_summary.csv`,
  `binary_threshold_predictions.csv`,
  `binary_threshold_by_volume_bucket.csv` при available sales volume.
- `notebooks/04_fusion_pack_grouping.ipynb` читает compact outputs из `03`,
  выбирает method/strategy только по `dev`; дефолт — `threshold_weighted_cost`
  с весом продаж, fallback — `threshold_cost_sensitive`. Для нового
  frozen/fine-tuning benchmark он умеет один раз читать общий
  `artifacts/reports/fine_tuning/`, восстановить `category_run` через frozen
  split, разложить пары по `sauces` / `coconut_oil` / `soap`, сохранить
  `fusion_components_<suffix>.csv` плюс `fusion_pair_eval_<suffix>.csv` в
  data-папку каждого category-run и показать graph-quality diagnostics,
  false/missed links и 3D component visualization.
- `notebooks/05_grouped_sku_demo.ipynb` показывает интерактивную demo-линию
  дедупликатора: берёт небольшой срез похожих SKU из `mpstats_products`,
  строит bi-encoder embeddings, ищет соседей через FAISS, rerank-ит пары
  cross-encoder и показывает дерево `каноничный SKU -> входящие SKU` с
  агрегатами куба. Production v1 перенесён в основной pipeline отдельным
  identity layer, а notebook остаётся demo/research-песочницей для будущих
  проверок и hard-negative улучшений.

## Production Web-App: карта проекта

| Зона | Файлы | Когда читать |
| --- | --- | --- |
| Web-app backend | `mpstats_app/` | API, настройки, DuckDB repository, сервисы локального приложения |
| Web frontend | `web/` | React UI локальной web-app |
| Pipeline services | `pipeline/services/` | шаги 1-6 основного workflow |
| Production ML-дедуп | `pipeline/services/dedup/`, `pipeline/services/dedup/model_profile.json`, `mpstats_app/api/dedup.py`, `pipeline/migrations/012_dedup_identity.sql`, `tests/test_dedup_service.py`, `tests/test_dedup_api.py` | запуск dedup после куба, FAISS K=30, fine-tuned BGE scoring, identity tables |
| Pipeline repositories | `pipeline/repositories/` | CSV/JSON/DuckDB data layer |
| DuckDB migrations | `pipeline/migrations/` | schema changes |
| Классификатор | `classifiers/rules.csv`, `classifiers/engine.py`, `mpstats_app/services/classifier_rules_service.py` | правила web-редактора и движок классификации |
| Качество данных | `pipeline/data_quality/`, `pipeline/services/data_quality_service.py`, `mpstats_app/api/quality.py` | проверки продаж, цен, дублей, периодов и согласованности |
| Web API tests | `tests/test_web_api.py` | backend route regressions |
| User docs | `docs/USER_GUIDE.md` | пользовательские сценарии web-app |

## Production Web-App: поток данных

```text
MPStats API
  -> data/projects/{project}/raw
  -> data/projects/{project}/processed
  -> mpstats.duckdb / mpstats_products
  -> optional dedup_* identity tables
  -> data/projects/{project}/reports|exports
Справочник категорий MP STATS.csv
  -> план задач marketplace + category + year + month
  -> mpstats.duckdb / cube_registry / mpstats_products
```

Настройки web workflow:

- `mpstats_app/api/settings.py`
- `mpstats_app/services/project_service.py`
- `pipeline/step1_config.py`

Локальный `pipeline/step1_export_config.json` игнорируется, потому что там
может быть cookie.

## Быстрые команды

Установка Python-зависимостей:

```bash
python3 -m pip install -r requirements.txt
```

Локальная web-app:

```bash
open "MPStats Local App.command"
```

Windows без Node.js:

```bat
"MPStats Local App.bat"
```

Research tests:

```bash
python3 -m pytest research/dedup/tests
```

Research compile check:

```bash
python3 -m compileall research/dedup
```

Notebook smoke через `nbclient`:

Для `notebooks/01_candidate_generation.ipynb` предварительно нужны research
dependencies:

```bash
python3 -m pip install -r requirements-research.txt
```

```bash
python3 - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

for notebook in [
    "notebooks/01_candidate_generation.ipynb",
    "notebooks/02_labeling_dataset.ipynb",
    "notebooks/03_matching_comparison.ipynb",
    "notebooks/04_fusion_pack_grouping.ipynb",
    "notebooks/05_grouped_sku_demo.ipynb",
]:
    path = Path(notebook)
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()
    print(f"ok: {notebook}")
PY
```

Production compile check:

```bash
python3 -m compileall pipeline classifiers mpstats_app
```

Узкая проверка web API:

```bash
python3 -m pytest tests/test_web_api.py
```

## Правила изменений

- Research-only задачи: не трогать `pipeline/` и `mpstats_app/`, если
  пользователь явно не попросил интеграцию.
- Production web-app задачи: держать routes тонкими, бизнес-логику в services,
  работу с файлами/SQL в repositories/data layer.
- Если меняется schema/SQL/repository/model — описать DB impact, добавить или
  обновить миграцию и тест.
- Если меняется пользовательский сценарий web-app, интерфейс, расчёты,
  статусы, справочник, классификатор или workflow — обновить
  `docs/USER_GUIDE.md`.
- Если задача зависит от актуального API библиотек/сервисов — использовать
  Context7 MCP, если доступен; иначе официальную документацию.
- Рабочие CSV/DB/outputs не считать доказательством пользовательского
  production workflow без явного подтверждения.
- В грязном worktree не трогать чужие изменения и не стадить их.

## Что уже есть

- Локальная FastAPI + React web-app для основного MPStats workflow.
- Сервисы шагов 1-6 в `pipeline/services/`.
- Data layer и DuckDB-миграции для production workflow.
- Web UI для справочника категорий и правил классификатора.
- MVP проверок качества данных.
- Production ML-дедуп v1 для `Соус/Соусы`, `Кокосовое масло`, `Мыло`:
  FAISS `K=30`, fine-tuned BGE `ft_bge_reranker_v2_m3`,
  `threshold_weighted_cost=0.872321`, identity tables без мутации фактов.
- Research-песочница `research/dedup/` и notebooks для SKU deduplication.
- Regression tests для web API, pipeline services, parser/quality и
  research/dedup.

## Чего не хватает

- CI для автоматического запуска тестов.
- Browser/e2e-регрессий frontend.
- Полной настройки порогов качества данных через web UI.
- Полной уборки исторических локальных DB/CSV-артефактов из tracked-части.
- Дополнительных hard-negative/quality проходов по fine-tuned reranker перед
  расширением production ML-дедупа на новые категории.

## Риски

- `docs/ARCHITECTURE.md` является обязательным контекстом для research-задач;
  перед staging проверяй `git status`, чтобы не захватить чужие/untracked
  документы случайно.
- Рабочие данные лежат рядом с кодом; не коммитить локальные `mpstats.duckdb`,
  `data/`, CSV/XLSX outputs и notebook outputs.
- В research важно не смешать уровни оценки: retrieval recall@k, pairwise
  classification и clustering metrics считаются отдельно.
- False merge дороже false negative: для `exact_duplicate` важнее precision,
  чем recall.
