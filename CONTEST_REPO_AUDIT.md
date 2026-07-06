# Contest Repository Audit

Дата проверки: 2026-07-06

Ветка: `ml-dedup-contest-ver`

Цель аудита: проверить, что репозиторий выглядит профессионально, запускается
с нуля классическим способом и через Docker, не содержит явных секретов и
понятен проверяющему конкурса.

## Краткий итог

| Зона | Статус | Приоритет | Что найдено | Что исправлено / что делать |
| --- | --- | --- | --- | --- |
| README | OK | P1 | Есть бизнес-задача, pipeline, quick start, Docker, структура, данные и ML-результат. Не хватало видимой схемы и пояснения уровней ML-оценки. | Добавлена SVG-схема pipeline и короткое объяснение retrieval / pairwise / graph metrics. |
| Docker / запуск | OK | P1 | `Dockerfile` и `compose.yaml` рабочие; контейнер собрался, поднялся и ответил на `/api/health`. В чистом контейнере runtime CSV могут быть пустыми до настройки через UI. | Docker подтверждён. Перед демонстрацией вручную импортировать/создать справочник и правила. |
| Python-зависимости | OK | P2 | Есть единый `requirements.txt`; `requirements-research.txt` является alias. `pyproject.toml` синхронизирован. | Установка зависимостей прошла. В `.env.example` добавлен alias `POLZA_AI_API_KEY`. |
| Ноутбуки | WARNING | P0/P1 | `00_eda.ipynb` содержал включённый mail-аудит, локальный PST-путь и сохранённые outputs; `04_fusion_pack_grouping.ipynb` уже был грязным и содержит outputs с локальными путями. | `00_eda.ipynb` очищен и mail-аудит выключен по умолчанию. `04` не трогался, чтобы не удалить чужие незакоммиченные результаты. |
| Данные | OK | P1 | Пользовательские DB/CSV/artifacts игнорируются. Локально есть ignored `mpstats.duckdb` и `artifacts/`, но они не tracked. | Дополнительных приватных данных в tracked-части не найдено. |
| Код | WARNING | P1 | Backend/tests проходят. TypeScript strict check падал из-за типа `loadProjects`; также есть крупные файлы-комбайны и архитектурный долг в dedup-service. | `loadProjects` исправлен, добавлен `npm run typecheck`. Крупные файлы оставлены как refactor debt. |
| Тесты и проверки | OK | P1 | Есть backend/research tests, compile check, frontend build. Не было явного frontend typecheck script. | `npm run typecheck` добавлен; итоговые проверки перечислены ниже. |
| Конкурсная упаковка | OK | P1 | Проект уже объясняет бизнес-проблему и ML-вклад. Notebook README был слишком коротким для быстрого маршрута жюри. | В `notebooks/README.md` добавлен маршрут для жюри по notebook `00`-`05`. |

## 1. README

Статус: OK  
Приоритет: P1

Что найдено:

- README объясняет, что проект делает: локальная web-app для MPStats ETL,
  классификации, DuckDB-куба и ML-дедупликации SKU.
- Бизнес-задача сформулирована: заменить ручную склейку месячных выгрузок
  воспроизводимым pipeline.
- Есть инструкция запуска через `.command`, `.bat` и Docker.
- Описаны входы, выходы, структура репозитория и основной ML-result.
- Не хватало встроенной визуальной схемы и короткого объяснения трёх уровней
  ML-оценки.

Почему важно:

- Проверяющий должен понять ценность проекта с первой страницы, без поиска по
  внутренним docs.

Что исправлено:

- В README добавлена `docs/assets/ml_dedup_pipeline_flow.svg`.
- Добавлено короткое разделение метрик на retrieval, pairwise matching и graph
  grouping.

## 2. Docker / Запуск

Статус: OK  
Приоритет: P1

Что найдено:

- Основной Docker-контур: `Dockerfile` + `compose.yaml`.
- Runtime state вынесен в volume `mpstats-runtime`.
- `.dockerignore` исключает `.env*`, локальные данные, notebooks, DB и model
  cache.
- `scripts/docker_entrypoint.sh` создаёт минимальные runtime-директории и
  header-only CSV, если правил ещё нет.
- Классический launcher `scripts/launch_local_app.sh` создаёт `.venv`,
  проверяет зависимости, собирает frontend и запускает backend.

Почему важно:

- Репозиторий должен стартовать на другой машине без локальных путей,
  пользовательских cookies и рабочей базы.

Что исправить / заметка:

- Для реальной демонстрации в чистом Docker нужно через UI создать или
  импортировать справочник категорий и правила классификатора. Это не блокирует
  старт приложения, но важно для ручного checklist.

## 3. Python-зависимости

Статус: OK  
Приоритет: P2

Что найдено:

- Основной dependency surface: `requirements.txt`.
- `requirements-research.txt` оставлен как compatibility alias.
- `pyproject.toml` содержит тот же набор основных зависимостей.
- Установка зависимостей в локальную `.venv` прошла.

Почему важно:

- Проверяющий должен воспроизвести окружение одной командой, без угадывания
  feature-specific requirements.

Что исправлено:

- В `.env.example` добавлен `POLZA_AI_API_KEY` как поддерживаемый alias для
  research/provider-настроек.

## 4. Ноутбуки

Статус: WARNING  
Приоритет: P0/P1

Что найдено:

- `01_candidate_generation.ipynb`, `02_labeling_dataset.ipynb`,
  `03_matching_comparison.ipynb`, `05_grouped_sku_demo.ipynb` проходят
  `nbformat.validate`, имеют русские intro/выводы, outputs пустые.
- `00_eda.ipynb` содержал включённый `MY_RUN_MAIL_ARCHIVE_EDA = True`,
  локальный `/Users/exoldoff/Desktop/backup.pst` и сохранённые outputs.
- `04_fusion_pack_grouping.ipynb` уже был изменён до аудита и содержит
  большой сохранённый output с локальными путями `/Users/exoldoff/...`.

Почему важно:

- Notebook-артефакты открываются проверяющим как часть конкурсной истории.
  Локальные пути и приватный mail-контекст создают ощущение "работает только у
  автора" и могут раскрывать лишний контекст.

Что исправлено:

- В `00_eda.ipynb` mail-блок выключен по умолчанию.
- `MY_MAIL_ARCHIVE_PATH` заменён на `None` с нейтральным примером.
- Outputs `00_eda.ipynb` очищены.

Что осталось:

- Перед финальной публикацией решить по `04_fusion_pack_grouping.ipynb`:
  оставить текущий визуальный output как конкурсный артефакт или очистить /
  перегенерировать в нейтральной среде. Автоматически файл не чистился, потому
  что он уже был грязным до аудита.

## 5. Данные

Статус: OK  
Приоритет: P1

Что найдено:

- `.gitignore` исключает `.env*`, DB, DuckDB, data/artifacts/output,
  локальные справочники и model cache.
- В tracked-файлах не найдено пользовательских `.duckdb`, `.sqlite`,
  `.parquet`, `.xlsx` или `.env` кроме `.env.example`.
- Локальные `mpstats.duckdb`, `.env` и `artifacts/` есть, но ignored.

Почему важно:

- Конкурсный архив не должен тащить коммерческие данные, cookies, токены,
  локальные базы или тяжёлые артефакты.

Что исправить / заметка:

- Перед отправкой проверить `git status --ignored -s` и не добавлять ignored
  локальные данные вручную.

## 6. Код

Статус: WARNING  
Приоритет: P1/P2

Что найдено:

- Production/research граница в целом выдержана: `research/dedup` не импортирует
  `pipeline`/`mpstats_app`, а production-слой не импортирует `research`.
- Точка входа backend: `mpstats_app.main:create_app`.
- Routes в основном тонкие, бизнес-логика вынесена в services/repositories.
- `npx tsc --noEmit` падал из-за несовместимого callback-типа `loadProjects`.
- Есть архитектурный долг: крупные файлы `web/src/App.tsx`,
  `mpstats_app/repositories/duckdb_repository.py`,
  `pipeline/services/dedup/service.py`.

Почему важно:

- TypeScript-ошибка снижает доверие к frontend-контрактам. Крупные файлы не
  блокируют запуск, но усложняют поддержку.

Что исправлено:

- `loadProjects` теперь явно возвращает `Promise<void>`.
- Добавлен script `npm run typecheck`.
- `npm audit fix` обновил frontend lockfile в рамках существующих semver-
  диапазонов; `npm audit` и `npm audit --omit=dev` теперь показывают 0
  vulnerabilities.

Что осталось:

- Дробление крупных файлов и вынос shared helpers лучше делать отдельным
  refactor-этапом, не перед конкурсной загрузкой.

## 7. Тесты и проверки

Статус: OK  
Приоритет: P1

Что найдено:

- Есть backend, pipeline, dedup и research tests.
- Есть Docker healthcheck.
- Есть frontend build.
- До аудита не было явного frontend `typecheck` script.

Что исправлено:

- Добавлен `web/package.json` script `typecheck`.

## 8. Конкурсная упаковка

Статус: OK  
Приоритет: P1

Что найдено:

- Жюри видит бизнес-проблему, локальный product workflow и ML-вклад.
- ML-результат вынесен в README, краткая CSV-сводка лежит в
  `docs/assets/contest_ml_metrics_summary.csv`.
- `notebooks/README.md` был корректным, но слишком коротким как маршрут
  конкурсной истории.

Что исправлено:

- В `notebooks/README.md` добавлен "Маршрут для жюри" по notebook `00`-`05`.
- В `docs/AI_INDEX.md` удалена stale-ссылка на несуществующие
  `docs/DEDUP_*.md`.

## Финальный статус

Готовность к загрузке на конкурс: `YES_WITH_NOTES`

Что было исправлено:

- Выключен и обезличен необязательный mail-аудит в `00_eda.ipynb`.
- Очищены outputs `00_eda.ipynb`.
- README усилен SVG-схемой и объяснением уровней ML-оценки.
- `notebooks/README.md` дополнен маршрутом для жюри.
- `.env.example` дополнен `POLZA_AI_API_KEY`.
- Исправлена TypeScript-ошибка `loadProjects`.
- Добавлен `npm run typecheck`.
- Обновлён `web/package-lock.json`, чтобы закрыть dev/build-tooling npm audit
  warnings (`vite`, `esbuild`, `@babel/core`).
- Удалена stale-ссылка `docs/DEDUP_*.md` из `docs/AI_INDEX.md`.

Что осталось нерешённым:

- `notebooks/04_fusion_pack_grouping.ipynb` уже был изменён до аудита и
  содержит локальные outputs. Перед финальной публикацией нужно решить:
  коммитить его как защитный output или очистить/перегенерировать отдельно.
- В чистом Docker runtime нет пользовательского справочника и правил, поэтому
  после старта нужно создать/импортировать их через UI.
- Нет CI. Это не блокирует локальный запуск, но желательно добавить отдельным
  этапом.
- Есть крупные файлы-комбайны; это refactor debt, не P0 перед конкурсом.

Команды, которые прошли успешно:

- `.venv/bin/python -m pip install -r requirements.txt`
- `docker compose config`
- `python3 -m compileall pipeline classifiers mpstats_app research/dedup`
- `npm --prefix web run typecheck`
- `npm --prefix web run build`
- `npm --prefix web audit --omit=dev`
- `npm --prefix web audit`
- `.venv/bin/python -m pytest tests research/dedup/tests`
- `docker build --check -f docker/dedup-training/Dockerfile .`
- `docker compose build`
- `docker compose up -d`
- HTTP smoke: `GET http://127.0.0.1:8055/api/health`
- `docker compose down`
- `MPSTATS_APP_NO_BROWSER=1 MPSTATS_APP_PORT=8060 MPSTATS_APP_PORT_END=8060 scripts/launch_local_app.sh`
- Notebook static check: `nbformat.validate` + `ast.parse` для `notebooks/00`-`05`

Команды, которые падали и почему:

- Первый `npx tsc --noEmit` падал из-за `loadProjects(): Promise<{projects}>`
  там, где `changeTab` ожидал `Promise<void>`. Исправлено.

Что проверить вручную перед отправкой:

- Открыть README на GitHub и убедиться, что SVG-схема отображается.
- В UI Docker-запуска создать/импортировать справочник категорий и правила.
- Открыть `notebooks/04_fusion_pack_grouping.ipynb` и принять решение по
  сохранённым outputs.
- На машине проверяющего первый ML-дедуп может скачать Hugging Face модели в
  Docker volume; это нормально, но требует сети и времени.

DB impact: нет. Schema, SQL, migrations, repositories и model profile не
менялись.
