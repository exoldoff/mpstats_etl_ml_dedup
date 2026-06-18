# AI Index

Короткий индекс проекта для быстрого входа без полного сканирования
репозитория. Это роутер для агента: что читать, где искать код и какие
границы не пересекать.

## Обязательный порядок чтения

1. `AGENTS.md` — правила работы, коммиты, проверки, ограничения.
2. `docs/AI_INDEX.md` — этот индекс.
3. `docs/ARCHITECTURE.md` — архитектурный контекст и research-направление,
   особенно для задач по SKU deduplication.
4. `docs/ARCHITECTURE_PROGRESS.md` — что уже сделано на текущем research-этапе.
5. `README.md` — только если нужен человеческий quick start или общая картина.
6. Релевантные файлы по задаче из разделов ниже.

Не сканируй весь проект без причины: сначала определи трек задачи.

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
| `docs/ARCHITECTURE.md` | источник research-архитектуры | задачи по dedup, matching, evaluation, notebooks, ML-методологии |
| `docs/ARCHITECTURE_PROGRESS.md` | журнал research-этапов | понять, что уже сделано, какие CSV/ноутбуки/проверки актуальны |
| `docs/THRESHOLD_CALIBRATION_REPORT.md` | критерии SKU matching benchmark | cost-sensitive пороги, manual review, почему macro-F1 не главный критерий |
| `docs/DEDUP_MODEL_HF_AUDIT.md` | HF-аудит моделей dedup research | prefixes, pooling, prompts, reranker settings, known mismatches |
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
| `research/dedup/threshold_calibration.py` | forced binary dev calibration для `threshold_same`, cost/weighted strategies и compact CSV |
| `research/dedup/annotator.py` | helper для ручной разметки |
| `research/dedup/tests/` | узкие тесты research-модулей |
| `research/dedup/data/` | локальные CSV-артефакты, игнорируются `.gitignore` |
| `notebooks/00_eda.ipynb` | EDA по выбранному `DEDUP_CATEGORY_RUN` |
| `notebooks/01_candidate_generation.ipynb` | FAISS embedding blocking, генерация `candidates_<suffix>.csv` |
| `notebooks/02_labeling_dataset.ipynb` | генерация `labeling_<suffix>.csv` |
| `notebooks/03_matching_comparison.ipynb` | сравнение baseline/reranker methods на размеченном gold-set выбранного run |
| `notebooks/04_fusion_pack_grouping.ipynb` | выбор fusion-run, family/pack grouping и CSV `fusion_*_<suffix>.csv` |
| `notebooks/05_evaluation_report.ipynb` | финальный research-отчёт по matching + fusion + graph quality выбранного run |
| `notebooks/06_grouped_sku_demo.ipynb` | демонстрация склеенных SKU из DuckDB + `fusion_*` выбранного run |

Текущие локальные CSV после последнего research-этапа:

- `research/dedup/data/candidates_sauces.csv` — 60 000 пар из
  `faiss_embedding_topk`, 25 093 cross-marketplace.
- `research/dedup/data/labeling_sauces.csv` — 400 пар из
  `faiss_embedding_topk`, 216 cross-marketplace.

Эти CSV — рабочие данные, они не коммитятся.

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
  embeddings + FAISS top-k. Default notebook alias:
  `embedding_e5_small` (`intfloat/multilingual-e5-small`); заменить можно
  через `DEDUP_EMBEDDING_MODEL`.
- Stratified labeling dataset с отдельной стратой
  `cross_marketplace_candidate`.
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
  всех matching-моделей на одном срезе: `rule_based_fuzzy`,
  `bi_encoder_zero_shot`, `cross_encoder_zero_shot`,
  `reranker_qwen3_4b`, `reranker_bge_v2_m3`, `reranker_jina_v3`. Новые
  reranker-модели выбираются alias-ами registry в
  `RERANKER_BENCHMARK_MODELS` или через env
  `DEDUP_RERANKER_BENCHMARK_MODELS`; главный отчёт лежит в reports-папке
  текущего `DEDUP_CATEGORY_RUN` как `binary_threshold_summary.csv`.
  Test split используется
  только для финальной проверки выбранных на dev `threshold_same`.
  `reranker_qwen3_4b` по умолчанию грузится на CPU, потому что на MPS с
  лимитом около 9GB падает по памяти; для быстрого Qwen-smoke есть alias
  `qwen3_0_6b`.
- `03_matching_comparison.ipynb` запускается без разметки и показывает
  заглушки вместо падения.

Актуальный downstream:

- Gold-set для threshold calibration использует binary target
  `same_base_product`: `exact_duplicate` и legacy
  `same_product_different_pack` считаются positive, `different_product` —
  negative. Если во входных данных уже есть колонка `same_base_product`, она
  считается источником правды.
- `notebooks/03_matching_comparison.ipynb` теперь делает stratified dev/test
  split и подбирает один `threshold_same` на dev для стратегий
  `threshold_max_f1`, `threshold_cost_sensitive` и weighted-стратегий, если
  доступны объёмы продаж. Raw predictions и старые `matching_*` CSV не
  перезаписываются; compact outputs пишутся в reports-папку текущего
  category-run: `binary_threshold_summary.csv`,
  `binary_threshold_predictions.csv`,
  `binary_threshold_by_volume_bucket.csv` при available sales volume.
- `notebooks/04_fusion_pack_grouping.ipynb` читает compact outputs из `03`,
  выбирает method/strategy только по `dev`; дефолт — `threshold_weighted_cost`
  с весом продаж, fallback — `threshold_cost_sensitive`; сохраняет
  `fusion_components_<suffix>.csv` плюс `fusion_pair_eval_<suffix>.csv` в
  data-папку текущего category-run.
- `notebooks/05_evaluation_report.ipynb` читает `binary_threshold_*` и
  `fusion_*`, без старых `matching_*` / `auto_same` / `manual_review`
  артефактов, и собирает текущий research-отчёт вместе с family/pack graph
  diagnostics.
- `notebooks/06_grouped_sku_demo.ipynb` показывает результат как витрину:
  реальные SKU из `mpstats_products` в DuckDB + текущие
  `fusion_family_id` / `fusion_pack_id`; это demo текущего research-среза,
  а не production identity mapping всего куба.
  Следующий ML-шаг — улучшать scorer/rerank/fusion на hard negatives без
  manual review / LLM-review в текущем benchmark-этапе, не переносить код в
  production pipeline.

## Production Web-App: карта проекта

| Зона | Файлы | Когда читать |
| --- | --- | --- |
| Web-app backend | `mpstats_app/` | API, настройки, DuckDB repository, сервисы локального приложения |
| Web frontend | `web/` | React UI локальной web-app |
| Pipeline services | `pipeline/services/` | шаги 1-6 основного workflow |
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
  -> data/projects/{project}/merged
  -> data/projects/{project}/exports
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
    "notebooks/05_evaluation_report.ipynb",
    "notebooks/06_grouped_sku_demo.ipynb",
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
- Research-песочница `research/dedup/` и notebooks для SKU deduplication.
- Regression tests для web API, pipeline services, parser/quality и
  research/dedup.

## Чего не хватает

- CI для автоматического запуска тестов.
- Browser/e2e-регрессий frontend.
- Полной настройки порогов качества данных через web UI.
- Полной уборки исторических локальных DB/CSV-артефактов из tracked-части.
- Размеченного gold-set для `notebooks/03_matching_comparison.ipynb`.
- Реализаций research engines C/D/E, clustering notebook и финального отчёта.

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
