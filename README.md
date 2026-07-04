# MPstats-FMCG-ETL

MPstats-FMCG-ETL — локальное web-приложение для выгрузки и обработки данных MPStats в удобный аналитический датасет.

Проект помогает B2B-командам, FMCG-брендам, категорийным менеджерам, аналитикам продаж и trade marketing-командам регулярно собирать данные по категориям и маркетплейсам, приводить их к единой структуре, считать весовые и ценовые метрики, классифицировать товары по бизнес-правилам и готовить данные для отчётов, BI и ассортиментного анализа.

Бизнес-задача: заменить ручную склейку месячных MPStats-выгрузок воспроизводимым локальным pipeline, который сохраняет факты в DuckDB и отдельно строит ML-слой identity-групп SKU. Для поддержанных FMCG-категорий приложение показывает, какие товары являются одним базовым SKU, даже если названия отличаются между маркетплейсами или фасовками.

## Pipeline

```text
MPStats API / CSV
  -> raw CSV
  -> standardize + weight parsing
  -> classifier rules + manual overrides
  -> DuckDB: mpstats_products + cube_registry
  -> ML-dedup: FAISS candidates + fine-tuned BGE cross-encoder
  -> dedup_* identity tables + mpstats_products_dedup
  -> reports / exports / ML-dedup CSV
```

Для ML-дедупликации используется отдельный identity layer: исходная таблица фактов `mpstats_products` не мутируется, а канонические группы пишутся в `dedup_*` и `mpstats_products_dedup`.

## Как запустить web-приложение

**Windows**

Запустите файл:

```text
MPStats Local App.bat
```

**macOS**

Откройте терминал в папке проекта и выполните:

```bash
open "MPStats Local App.command"
```

Запускатель создаст локальное Python-окружение `.venv`, проверит зависимости
и при необходимости установит их из `requirements.txt`, включая пакеты для
`ML-дедуп`.

**Docker**

Для проверки переносимости на другой машине можно поднять тот же web-app в
контейнере:

```bash
docker compose config
docker compose build
docker compose up
```

После старта откройте `http://127.0.0.1:8055`. Compose хранит локальную
DuckDB, настройки, справочник, правила классификатора и cache моделей в
именованном volume `mpstats-runtime`; образ остаётся переносимым и не зависит
от локальных рабочих файлов. Остановить контейнер можно через
`Ctrl+C`, затем `docker compose down`.

После запуска откроется локальное приложение в браузере. Дальше всё делается через интерфейс: вставьте MPStats cookie, выберите проект, период, категории, создайте план и нажмите запуск.

Другие пользовательские способы запуска не поддерживаются: работа с проектом сосредоточена в web-интерфейсе.

### Быстрый старт с чистого окружения

1. Установите Python 3.10+ и, для macOS-режима, Node.js 20+.
2. Скопируйте `.env.example` в `.env`, только если нужны API/model/cache-переменные.
3. Запустите web-app через `.command`, `.bat` или Docker.
4. В интерфейсе задайте MPStats cookie, при необходимости MPStats API token, проект, категории и период.
5. Если локальный справочник категорий или правила классификатора отсутствуют, создайте их через вкладки `Категории` и `Классификатор`.

В чистом git-архиве нет пользовательской DuckDB, MPStats cookie, рабочих CSV/XLSX и model cache. Приложение стартует без них, но реальные выгрузки и ML-дедуп требуют доступа к MPStats и локальных данных.

## Что умеет приложение

- скачивает отчёты MPStats по маркетплейсам, категориям, периодам и фильтрам;
- приводит разные выгрузки к единой структуре;
- парсит вес и объём из названий товаров;
- считает продажи в кг/тоннах и цену за кг;
- классифицирует товары по настраиваемым правилам;
- сохраняет результат в локальные файлы и локальную DuckDB-базу;
- запускает production ML-дедуп для категорий `Соус/Соусы`, `Кокосовое масло` и `Мыло`;
- строит candidate pairs через FAISS и оценивает пары fine-tuned BGE cross-encoder моделью;
- даёт web-интерфейс для справочника категорий, умного плана, классификатора, куба и ML-дедупа.

## Входы и выходы

Входы:

- MPStats cookie для категорийных выгрузок;
- MPStats API token для строк справочника, которые грузятся `По предмету`;
- справочник категорий, который можно создать/редактировать через UI;
- правила классификатора и ручные override, которые также редактируются через UI;
- локальная DuckDB/CSV-артефакты, если нужно открыть уже сохранённый проект.

Выходы:

- raw/processed/classified CSV внутри `data/projects/<project>/`;
- локальная DuckDB `mpstats.duckdb`;
- отчёты и выгрузки в `data/projects/<project>/reports|exports`;
- ML-дедуп identity tables `dedup_*`;
- материализованная таблица `mpstats_products_dedup`;
- CSV `ML-дедуп` из вкладки `Данные` -> `Выгрузка`.

## ML-результаты

Текущий production-профиль: `ft_bge_reranker_v2_m3`, `threshold_weighted_cost`, FAISS `K=30`, графовая группировка Leiden.

В отчётной safe-конфигурации fine-tuned BGE на server `positive_record_holdout`
benchmark (`test=165`: 48 negative + 117 positive) даёт 0 ложных склеек:

| Метод | Precision | Recall | F1 | False merges | False splits | Weighted cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ft_bge_reranker_v2_m3` | 100.0% | 79.5% | 88.6% | 0 | 24 | 131.3 |

Короткая сводка лежит в `docs/assets/contest_ml_metrics_summary.csv`.

## Структура проекта

- `mpstats_app/` — FastAPI backend, API routes, сервисы web-app.
- `web/` — React/Vite frontend локального приложения.
- `pipeline/` — ETL-сервисы, repositories, DuckDB migrations и production ML-дедуп.
- `classifiers/` — локальные CSV правил и движок классификации.
- `research/dedup/` — research-only код для candidate generation, scoring, clustering и training helpers.
- `notebooks/` — конкурсная research-история `00`-`05` по SKU-дедупликации.
- `tests/` — regression/smoke tests для app, pipeline и dedup.
- `docs/` — пользовательская инструкция, архитектура и описание ML-методологии.

## Данные и локальное состояние

Репозиторий поставляется без пользовательской DuckDB, MPStats cookie, рабочих
CSV/XLSX, model cache и локальных `.env`-настроек. Эти файлы создаются при
запуске приложения или подключаются из рабочего окружения.

Файлы `Справочник категорий MP STATS.csv`, `classifiers/rules.csv` и
`classifiers/manual_overrides.csv` являются runtime-настройками: их можно
создать через web UI или импортировать из рабочего набора.

## Документация

- `docs/ARCHITECTURE.md` — архитектура research-направления SKU deduplication.
- `docs/ARCHITECTURE_PROGRESS.md` — текущий журнал research/production-dedup этапов и технических заметок.
- `docs/USER_GUIDE.md` — подробная пользовательская инструкция.
- `notebooks/README.md` — порядок research-ноутбуков по SKU-дедупликации.

## License

This project is licensed under the GNU Affero General Public License v3.0.
See the LICENSE file for details.
