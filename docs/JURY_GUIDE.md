# Карта проекта для жюри

Этот файл нужен как быстрый маршрут по репозиторию перед защитой. Он не
заменяет подробную документацию, а показывает, где лежат ключевые части
проекта и в каком порядке их смотреть.

## Что решает проект

MPstats-FMCG-ETL собирает данные MPStats по маркетплейсам, приводит их к
единому аналитическому виду и помогает убрать дубли SKU. Для конкурса важны
две линии:

1. Production web-app: локальный интерфейс для загрузки, классификации,
   обработки и ML-дедупликации данных.
2. Research notebooks: прозрачная исследовательская история, как мы пришли к
   embeddings + FAISS + reranker + graph grouping.

## Быстрый маршрут

1. `README.md` — что это за приложение и как его запустить.
2. `docs/JURY_GUIDE.md` — эта карта.
3. `notebooks/README.md` — порядок research-notebooks.
4. `notebooks/00_eda.ipynb` — входной EDA выбранной категории.
5. `notebooks/01_candidate_generation.ipynb` ->
   `notebooks/05_grouped_sku_demo.ipynb` — основная ML-история.
6. `docs/ARCHITECTURE.md` — целевая методология SKU-дедупликации.
7. `docs/ARCHITECTURE_PROGRESS.md` — что уже реализовано и проверено.
8. `docs/USER_GUIDE.md` — пользовательская инструкция для локальной web-app.
9. `Dockerfile` и `compose.yaml` — воспроизводимый контейнерный запуск web-app.

## Как устроен репозиторий

| Путь | Что внутри |
| --- | --- |
| `mpstats_app/`, `web/` | Локальная web-app: пользовательский интерфейс, API и сервисы. |
| `pipeline/` | Основной data pipeline: загрузка, нормализация, DuckDB, exports, production ML-dedup layer. |
| `classifiers/` | Правила и движок классификации товаров. |
| `research/dedup/` | Исследовательские модули дедупликации: candidates, labeling, matchers, calibration, graph grouping. |
| `notebooks/` | Конкурсный research-рассказ: от EDA до demo дерева SKU. |
| `docs/` | Архитектура, прогресс, runbooks, пользовательская инструкция и эта карта. |
| `Dockerfile`, `compose.yaml` | Docker-контур: сборка frontend/backend, healthcheck и volume для локальных данных. |
| `docs/archive/` | Старые аудиты и исторические материалы, которые полезны как контекст, но не являются главным входом. |
| `docs/assets/` | Вспомогательные картинки/таблицы для документации. |
| `tools/dedup_labeling_bot/` | Отдельный Telegram-инструмент для ручной разметки пар. |

## Главные артефакты исследования

- Candidate CSV: `research/dedup/data/*/candidates_<suffix>.csv` или legacy
  `research/dedup/data/candidates_sauces.csv`.
- Labeling CSV: `research/dedup/data/*/labeling_<suffix>.csv`.
- Threshold reports: `artifacts/reports/fine_tuning/binary_threshold_*.csv`.
- Fusion outputs: `fusion_components_<suffix>.csv`,
  `fusion_pair_eval_<suffix>.csv`.
- Demo export: `grouped_sku_demo_<suffix>.csv`.

Эти файлы обычно игнорируются git, потому что зависят от локальной базы,
секретов, модели и конкретного запуска. В репозитории хранится код,
notebooks и документация, а не пользовательские данные.

## Что показывать на защите

1. Web-app как продуктовый контур: загрузка, классификация, справочники,
   ML-дедуп.
2. Notebook `01`: как мы строим пары-кандидаты через embeddings + FAISS.
3. Notebook `02`: как собираем gold-set для ручной разметки.
4. Notebook `03`: как сравниваем модели и выбираем threshold только на dev.
5. Notebook `04`: как из pairwise решений получаются family/pack группы.
6. Notebook `05`: короткое end-to-end demo на реальных строках DuckDB.
7. Engineering-проверки: `git status --short`, узкий pytest и
   `docker compose config` / `docker compose build`.

## Инженерные решения

- Git хранит код, миграции, тесты, frontend source/build и документацию; `.env`,
  `mpstats.duckdb`, локальные выгрузки, cache моделей и пользовательские CSV
  игнорируются.
- macOS/Windows запускатели создают project-local `.venv` и не требуют
  установки зависимостей в системный Python.
- Docker-контур собирает frontend через `npm ci`, ставит Python-зависимости из
  `requirements.txt`, поднимает FastAPI на `127.0.0.1:8055` и проверяет
  `/api/health`.
- В Docker все mutable данные вынесены в volume `mpstats-runtime`: база,
  pipeline-файлы, справочник, правила, ручные override и cache моделей.

## Что не является мусором

- `docs/archive/` — архивные отчёты и аудиты. Они не главный маршрут, но
  показывают историю инженерных решений.
- `artifacts/`, `research/dedup/data/`, `mpstats.duckdb`, `.env` — локальные
  рабочие данные и секреты. Они нужны для запуска, но не должны попадать в git.
