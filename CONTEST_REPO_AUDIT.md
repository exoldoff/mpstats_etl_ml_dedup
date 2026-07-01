# Предконкурсный аудит репозитория

Дата аудита: 2026-07-02.
Цель: проверить, что репозиторий понятно выглядит для жюри, запускается с нуля и не тащит локальные секреты/данные.

## Краткий вывод

Статус после минимальных правок: `YES_WITH_NOTES`.

Проект имеет понятную web-app точку входа, Docker Compose конфигурацию, единый `requirements.txt`, тесты и research-ноутбуки `00`-`05`. Основные конкурсные риски были в упаковке: README недопродавал ML-дедуп, в notebook defaults были личные абсолютные пути, не было `.env.example`, training Dockerfile не копировал `requirements.txt`, временные `outputs/` могли попасть в Docker build context, а импорт `mpstats_app.main` мог трогать локальную DuckDB через dedup cleanup.

Безопасно исправлено: README, `.env.example`, переносимые настройки notebooks `03`/`05`, training Dockerfile, ignore rules, tracked CSV с headline ML-метриками, инструкции и startup-only dedup maintenance.

Осталось проверить перед загрузкой: Docker build/run на машине с запущенным Docker daemon, чистый архив без ignored локальных данных, а также текущее незакоммиченное состояние `notebooks/00_eda.ipynb`, которое было изменено до аудита и не включается в этот cleanup автоматически.

## Этап 1. Аудит без правок

### 1. README

| Статус | Приоритет | Что найдено | Почему важно | Что исправить |
| --- | --- | --- | --- | --- |
| WARNING | P1 | README объяснял локальный ETL, но слабо показывал бизнес-задачу, pipeline, ML-дедуп, входы/выходы, структуру repo и метрики. | Для конкурса проект выглядел как обычная обработка таблиц, а не как ML-продукт. | Добавить business context, pipeline-схему, clean-start, I/O, структуру и ML headline metrics. |
| WARNING | P1 | README ссылался на runtime CSV `Справочник категорий MP STATS.csv` и `classifiers/rules.csv`, которые игнорируются и могут отсутствовать в clean checkout. | Проверяющий мог решить, что repo неполный. | Явно описать, что это локальные runtime-настройки, создаваемые через UI или импортируемые из приватного набора. |
| OK | P2 | macOS/Windows/Docker quick start уже был описан. | Базовая точка входа есть. | Оставить и расширить clean-start оговорками. |

### 2. Docker / запуск

| Статус | Приоритет | Что найдено | Почему важно | Что исправить |
| --- | --- | --- | --- | --- |
| OK | P2 | `compose.yaml` валиден, runtime вынесен в volume `mpstats-runtime`, healthcheck есть. | Основной Docker route выглядит воспроизводимо. | Запустить build/run на машине с Docker daemon. |
| WARNING | P1 | `docker/dedup-training/Dockerfile` копировал `requirements-research.txt`, но не `requirements.txt`, хотя research-файл делает `-r requirements.txt`. | Training image мог падать на `pip install -r requirements-research.txt`. | Копировать оба requirements файла. Исправлено. |
| WARNING | P1 | `outputs/` не был в `.dockerignore`/`.gitignore`, рядом был временный Office-файл. | В build context или zip мог попасть локальный мусор. | Добавить `outputs/` в ignore rules. Исправлено для новых файлов. |
| OK | P1 | High-confidence secret grep по tracked-файлам не нашёл API keys/tokens. `.env` игнорируется. | Публичный repo не должен раскрывать доступы. | Перед upload повторить grep и не добавлять `.env`. |

### 3. Python-зависимости

| Статус | Приоритет | Что найдено | Почему важно | Что исправить |
| --- | --- | --- | --- | --- |
| OK | P1 | Единый основной источник зависимостей — `requirements.txt`; `pyproject.toml` синхронизирован. | Окружение можно воспроизводить одним install. | Поддерживать parity. |
| WARNING | P2 | `requirements-research.txt` существовал как compatibility alias, но notebooks README вёл через него. | Создавалось ощущение двух разных dependency surfaces. | В notebooks README указать `requirements.txt`, alias оставить совместимым. Исправлено. |
| WARNING | P2 | Зависимости тяжёлые: `torch`, `sentence-transformers`, `faiss-cpu`, `datasets`, `peft`. | Это ожидаемо для ML, но установка может быть долгой. | В README оставить Docker/launcher путь и не дробить requirements без необходимости. |

### 4. Ноутбуки

| Статус | Приоритет | Что найдено | Почему важно | Что исправить |
| --- | --- | --- | --- | --- |
| OK | P2 | `notebooks/01`-`05` открываются как JSON, имеют русские введения и пустые outputs. | Research story читается последовательно. | Сохранять outputs вне git. |
| WARNING | P1 | `03_matching_comparison.ipynb` и `05_grouped_sku_demo.ipynb` содержали личные абсолютные пути к backup/model/DuckDB. | На машине жюри такие пути ломают воспроизводимость и выглядят непрофессионально. | Заменить на `None`, HF model id, `MPSTATS_DUCKDB_PATH` и repo-relative path. Исправлено. |
| WARNING | P1 | `03_matching_comparison.ipynb` по умолчанию разрешал full live scoring (`MY_ALLOW_LIVE_MODEL_SCORING=True`, `MY_MAX_EVAL_PAIRS=0`). | Clean-start мог неожиданно загрузить модели и долго считать. | Сделать score-CSV-first safe default, live scoring включать явно. Исправлено. |
| WARNING | P1 | `notebooks/00_eda.ipynb` уже был изменён до аудита: выполненные outputs, локальный PST path и включённый optional mail archive EDA. | Это может помешать clean upload, но файл уже был dirty и содержит чужой/предыдущий diff. | Не включать его в мой коммит; перед upload отдельно решить, сохранять ли этот notebook snapshot или откатить/очистить outputs. |

### 5. Данные

| Статус | Приоритет | Что найдено | Почему важно | Что исправить |
| --- | --- | --- | --- | --- |
| OK | P1 | `.gitignore` исключает `.env`, DuckDB, `data/`, `artifacts/`, model cache, локальные CSV/SQLite/Parquet. | Приватные рабочие данные не должны попасть в git. | Перед upload проверять `git status --short --ignored`. |
| WARNING | P1 | В clean repo нет пользовательских данных, справочника и правил. Это правильно для приватности, но требует явного bootstrap-сценария. | Жюри должно понимать, что создать через UI, а что не входит в архив. | README и USER_GUIDE уточнены; для демо можно отдельно подготовить sanitized sample, если правила конкурса разрешают. |
| WARNING | P2 | `outputs/mpstats_fmcg_etl_pitch.pptx` уже tracked, а `outputs/` теперь игнорируется только для новых файлов. | Нужно осознанно решить, входит ли презентация в конкурсную поставку. | Перед upload оставить tracked presentation только если это нужный артефакт. |

### 6. Код

| Статус | Приоритет | Что найдено | Почему важно | Что исправить |
| --- | --- | --- | --- | --- |
| OK | P1 | Явная точка входа web-app: `MPStats Local App.command`, `MPStats Local App.bat`, `mpstats_app.main:create_app`, Docker entrypoint. | Проверяющий видит, как запустить продукт. | Оставить web-app единственным пользовательским UI. |
| OK | P1 | Слои разделены: routes в `mpstats_app/api`, бизнес-логика в services, DuckDB/file work в repositories, ML-dedup в `pipeline/services/dedup`. | Архитектура выглядит поддерживаемой. | Не смешивать UI/domain/repository слои. |
| WARNING | P2 | Полный поиск мёртвого кода не выполнялся, чтобы не сканировать проект без причины. | Аудит предпродажный, не полноценный refactor. | После конкурса можно сделать отдельный cleanup. |

### 7. Тесты и проверки

| Статус | Приоритет | Что найдено | Почему важно | Что исправить |
| --- | --- | --- | --- | --- |
| OK | P1 | Есть тесты `tests/test_app_config.py`, `tests/test_web_api.py`, `tests/test_dedup_service.py`, research/dedup tests. | Есть минимальная regression surface. | Перед upload запускать узкий набор. |
| WARNING | P1 | Системный Python 3.13 подцепил внешний pytest plugin `langsmith`, из-за чего обычный pytest завис на импорте. | Проверки могут казаться сломанными из-за чужого окружения. | Запускать с `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` в `.venv` или в чистом Docker. |
| WARNING | P1 | Импорт `mpstats_app.main` создавал `app = create_app()`, а `DedupService.__init__` сразу запускал cleanup dedup runs. | Test collection/import могли писать в локальную DuckDB и ловить lock. | Перенести cleanup в FastAPI lifespan после `repository.ensure_ready()`. Исправлено. |
| WARNING | P1 | Полный Docker build/run пока не подтверждён текущей машиной. | Это ключевая проверка переносимости. | Запустить `docker compose build && docker compose up` при доступном Docker daemon. |

### 8. Конкурсная упаковка

| Статус | Приоритет | Что найдено | Почему важно | Что исправить |
| --- | --- | --- | --- | --- |
| WARNING | P1 | До правок README не показывал ML-вклад, метрики и data contract на первом экране. | Жюри могло недооценить проект. | README расширен. |
| OK | P1 | Tracked summary `docs/assets/contest_ml_metrics_summary.csv` добавляет переносимую короткую метрику. | Headline ML evidence теперь не зависит только от ignored artifacts. | При новых benchmark обновлять CSV и README вместе. |
| WARNING | P1 | Локальные model cache и data artifacts занимают много места в worktree, но ignored. | Нельзя отправлять весь каталог как zip без фильтрации. | Делать upload из git archive / clean clone, не из полной рабочей папки. |

## Этап 2. Минимальные безопасные правки

Исправлено:

- README: добавлены business context, pipeline-схема, clean-start, I/O, структура проекта, ML-дедуп и headline metrics.
- `.env.example`: добавлен безопасный шаблон без секретов.
- `docs/assets/contest_ml_metrics_summary.csv`: добавлена короткая tracked-сводка ML benchmark.
- `notebooks/03_matching_comparison.ipynb`: убраны локальные backup paths; live scoring выключен по умолчанию.
- `notebooks/05_grouped_sku_demo.ipynb`: убраны локальные DuckDB/model fallback paths; оставлен HF model id и явный `MY_DUCKDB_PATH`/`MPSTATS_DUCKDB_PATH`.
- `notebooks/README.md`: основной install путь синхронизирован на `requirements.txt`.
- `docs/USER_GUIDE.md`: добавлены `.env.example` и переносимые model path/HF id инструкции.
- `docs/ARCHITECTURE_PROGRESS.md`: обновлено правило ведения progress после production transfer.
- `docker/dedup-training/Dockerfile`: добавлен `requirements.txt` в Docker context.
- `.dockerignore` и `.gitignore`: добавлен `outputs/` для новых локальных артефактов.
- `DedupService`: cleanup stale/failed dedup runs перенесён из конструктора в startup-maintenance, который вызывается из FastAPI lifespan.

Сознательно не исправлено автоматически:

- `notebooks/00_eda.ipynb`: уже был dirty до аудита и содержит большой diff с выполненными outputs. Автоматическая правка могла бы перезаписать чужие изменения.
- Реальный справочник категорий и `classifiers/rules.csv`: локальные runtime/business files могут содержать приватные правила. Вместо коммита приватных CSV описан UI/bootstrap сценарий.
- Полный Docker build/run: требует доступного Docker daemon и может быть долгим из-за ML-зависимостей.

## Финальный статус

### Что было исправлено

См. раздел "Этап 2". Главный эффект: clean README, переносимые notebook defaults, безопасный env-шаблон, исправленный training Dockerfile, tracked ML metrics summary и import-safe app startup.

### Что осталось нерешённым

- Проверить `notebooks/00_eda.ipynb`: очистить outputs/локальный PST path или осознанно оставить текущий snapshot.
- Решить, нужен ли tracked `outputs/mpstats_fmcg_etl_pitch.pptx` в конкурсной загрузке.
- Подготовить sanitized sample справочника/правил, если жюри должно увидеть наполненный UI без приватных данных.
- Запустить Docker build/run на машине с доступным Docker daemon.

### Команды, которые были запущены

```bash
git status -sb
docker compose config
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_app_config.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_dedup_service.py::test_service_marks_stale_dedup_runs_failed_on_start tests/test_dedup_service.py::test_service_prunes_repeated_failed_dedup_runs_on_startup_maintenance
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --collect-only -q
python3 -m compileall mpstats_app pipeline/services pipeline/repositories research/dedup
.venv/bin/python -m pip install -r requirements.txt --dry-run
docker compose build
git grep -n -I -E 'sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{20,}' -- . ':!package-lock.json' ':!web/package-lock.json'
git grep -n -I -E '<local absolute path patterns>' -- README.md docs/USER_GUIDE.md docs/ARCHITECTURE_PROGRESS.md notebooks/README.md notebooks/03_matching_comparison.ipynb notebooks/05_grouped_sku_demo.ipynb .env.example docs/assets/contest_ml_metrics_summary.csv
```

### Команды, которые прошли успешно

- `docker compose config`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_app_config.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_dedup_service.py::test_service_marks_stale_dedup_runs_failed_on_start tests/test_dedup_service.py::test_service_prunes_repeated_failed_dedup_runs_on_startup_maintenance`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --collect-only -q`
- `python3 -m compileall mpstats_app pipeline/services pipeline/repositories research/dedup`
- `.venv/bin/python -m pip install -r requirements.txt --dry-run`
- JSON smoke для `notebooks/03_matching_comparison.ipynb` и `notebooks/05_grouped_sku_demo.ipynb`
- Secret grep по tracked-файлам

### Команды, которые упали или не завершились

- Обычный `python3 -m pytest tests/test_app_config.py` завис на внешнем pytest plugin autoload (`langsmith`) в системном Python 3.13; с `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` тот же тест прошёл.
- До правки `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --collect-only -q` падал на DuckDB lock при импорте `mpstats_app.main`; после переноса dedup cleanup в lifespan collect проходит.
- `docker compose build` не стартовал из-за недоступного Docker daemon: Docker CLI не смог подключиться к локальному Docker socket.
- `docker compose up` не запускался, потому что build не мог начаться без Docker daemon.

### Готов ли репозиторий к загрузке на конкурс

`YES_WITH_NOTES`

Репозиторий стал заметно лучше упакован для конкурса, но перед финальной отправкой нужно не включать локальные ignored данные, проверить/решить состояние `00_eda.ipynb` и прогнать Docker build/run.
