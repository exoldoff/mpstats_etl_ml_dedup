# Флоу презентации проекта MPstats-FMCG-ETL

Документ нужен как опорный сценарий для описания проекта, презентации и защиты.
Логика ниже идет не по структуре папок, а по истории продукта: от проблемы и
сырого MPStats до локальной web-app, исследовательского ML-дедупа и переноса
выбранного решения в основной pipeline.

## 1. Главная мысль

MPstats-FMCG-ETL - локальная платформа для подготовки MPStats-данных и
ML-нормализации SKU в FMCG. Проект делает полный путь:

```text
MPStats API
  -> raw/processed/classified файлы
  -> DuckDB-куб
  -> отчеты и выгрузки
  -> ML-дедуп базовых товаров и фасовок
```

Что важно подчеркнуть сразу:

- это не разовый скрипт и не отдельный notebook;
- основной пользовательский сценарий живет в web-app;
- research-слой нужен не для красоты, а чтобы доказать, какой ML-подход
  безопасно переносить в production;
- дедуп не ломает исходные факты: фактовые строки остаются, а ML-группы
  добавляются отдельным identity-слоем.

## 2. Бизнес-проблема

В MPStats один и тот же товар часто выглядит как несколько разных SKU:
разные маркетплейсы, артикулы, хвосты в названии, бренд, вес, наборы,
описания акций и фасовок.

Из-за этого:

- продажи одного товара дробятся между карточками;
- ассортимент кажется шире реального;
- цена за кг и доли товаров считаются хуже;
- сравнение Ozon / WB / Яндекс.Маркет становится ручной работой;
- бизнес-правки и классификация быстро превращаются в набор разрозненных CSV.

Хорошая формулировка для слайда:

> Проект превращает разрозненные выгрузки MPStats в локальный аналитический
> куб и добавляет ML-слой, который связывает разные карточки одного товара
> без потери исходных фактов.

## 3. Хронология работ

### Этап 1. Сначала сделали рабочий локальный ETL

Показать, что база проекта не ML, а надежный pipeline:

1. Пользователь создает проект и задает MPStats cookie.
2. Web-app строит план по категориям, маркетплейсам, годам и месяцам.
3. Для каждого среза скачивается raw-файл.
4. Processed-слой приводит разные выгрузки к общей схеме.
5. Вес и объем парсятся в отдельные поля.
6. Классификатор применяет бизнес-правила.
7. Classified-файл сохраняется в DuckDB.
8. `cube_registry` хранит, какие срезы уже загружены.

Что подсветить:

- единица работы: проект + маркетплейс + категория + год + месяц;
- промежуточные файлы остаются локально в `data/projects/<project>/...`;
- итоговая аналитика строится из DuckDB, а не из набора открытых Excel-файлов;
- повторная загрузка того же среза защищена через registry и overwrite-логику.

Источники: `docs/PIPELINE_OVERVIEW.md`, `docs/USER_GUIDE.md`, `mpstats_app/`,
`pipeline/`.

### Этап 2. Добавили управляемую бизнес-классификацию

Классификатор нужен, потому что часть разметки FMCG остается бизнесовой:
подкатегория, тип, вид мяса, ручные исправления SKU.

Текущий локальный файл правил:

| Метрика | Значение |
| --- | ---: |
| Всего правил в `classifiers/rules.csv` | 226 |
| Активных правил | 222 |
| Ручных override-правок в `classifiers/manual_overrides.csv` | 3 |
| Основная целевая колонка | `Подкатегория` |

Что показать:

- вкладку `Классификатор`;
- приоритеты правил;
- условия `contains`, `regex`, `equals`, `otherwise`, числовые сравнения;
- режимы `fill_empty` и `overwrite`;
- ручные правки SKU без редактирования сырых CSV.

Важно сказать:

> ML не заменяет весь бизнес-словарь. В проекте есть управляемый слой правил,
> а ML-дедуп решает другую задачу: связывает карточки одного товара.

### Этап 3. Ускорили и укрепили DuckDB-контур

Этот блок нужен, чтобы показать инженерную зрелость, не только notebooks.

Что было сделано:

- импорт classified CSV через DuckDB staging SQL;
- замена среза в одной транзакции вместо опасного частичного обновления;
- прямой CSV export через DuckDB `COPY`;
- плоские CSV/XLSX выгрузки строятся из одного SQL-запроса;
- управляемые DuckDB-подключения, read-only режим для отчетов и экспортов;
- защита от слишком больших XLSX: если превышен лимит Excel, пользователь
  переключается на CSV.

Локальный dev-срез в текущей DuckDB:

| Метрика | Значение |
| --- | ---: |
| Строк в `mpstats_products` | 913 592 |
| Записей в `cube_registry` | 137 |
| Проектов в локальном кубе | 6 |
| Крупнейшая категория | `Мыло`, 773 314 строк |
| Категория `Соус` | 9 953 строк |
| Категория `Кокосовое масло` | 1 388 строк |

Примечание для защиты: это локальный dev evidence-срез из текущего worktree, а
не гарантия состояния пользовательской машины.

Источники: `docs/archive/DUCKDB_OPTIMIZATION_REPORT.md`, локальная
`mpstats.duckdb`.

### Этап 4. Перешли к задаче SKU-дедупликации

Здесь объяснить, почему это отдельная ML-задача.

Что такое результат дедупа:

- `family` - базовый товар;
- `pack` - конкретная фасовка внутри базового товара;
- `200 г` и `3 x 200 г` могут быть одним базовым товаром, но разными фасовками;
- false merge опаснее false split: ошибочная склейка разных товаров портит
  справочник и отчеты, а пропущенный дубль можно добрать позже.

Начальный research scope:

- `sauces` - `Соусы_тест` / `Соус`, `Соусы`;
- `coconut_oil` - `кокос_тест` / `Кокосовое масло`;
- `soap` - `мыло_тест` / `Мыло`.

Исторический EDA по соусам:

| Метрика | Значение |
| --- | ---: |
| Строк категории `Соус` в EDA | 19 473 |
| Уникальных title | 11 890 |
| Заполненность бренда | 97.1% |
| Пропуски/ошибки веса и веса единицы | 0.0% |
| Оценочный multipack > 1 | 20.6% |
| Весовые аномалии | < 0.3% |

Зачем это важно: бренд и вес полезны, но не должны быть жесткими стоп-факторами.
Финальное решение принимает модельный score плюс post-processing.

Источники: `docs/ARCHITECTURE.md`, `docs/ARCHITECTURE_PROGRESS.md`.

### Этап 5. Candidate generation: не полный перебор, а близкие пары

Полный перебор SKU слишком дорогой и дает слишком много очевидных негативов.
Поэтому сначала строятся векторные представления названий, затем FAISS ищет
ближайших соседей.

Что сделали:

- dense embeddings + FAISS top-k;
- subcategory-aware blocking, если `Подкатегория` заполнена;
- full-global fallback для пустой подкатегории;
- pre-embedding collapse для exact-title дублей с одинаковым брендом;
- supplemental pairs вне FAISS top-k, чтобы разметка не была заложником
  текущего top-k.

Текущие candidate CSV:

| Category-run | Candidate pairs | Основной источник |
| --- | ---: | --- |
| `sauces` | 107 539 | FAISS + supplemental |
| `coconut_oil` | 21 640 | FAISS + supplemental |
| `soap` | 479 288 | FAISS + supplemental |

Mini-test моделей для retrieval:

| Модель | recall@20 | cross-marketplace recall@20 | Комментарий |
| --- | ---: | ---: | --- |
| `intfloat/multilingual-e5-small` | 100.00% | 100.00% | текущий default, но gold-set был частично собран из этого pipeline |
| `ai-forever/FRIDA` | 87.16% | 84.15% | сильный независимый локальный кандидат, тяжелее |
| `cointegrated/rubert-tiny2` | 76.35% | 68.29% | быстрый, но пропускает много positive pairs |
| `deepvk/RuModernBERT-small` | 66.22% | 57.32% | слабее для этой задачи без дообучения |

Что сказать:

> Мы отдельно проверяли первый этап. Если настоящий дубль не попал в кандидаты,
> никакой reranker потом его уже не спасет.

Источники: `docs/assets/embedding_recall_mini_tests.csv`,
`research/dedup/data/*/candidates_*.csv`.

### Этап 6. Разметка: gold-set и Telegram-инструмент

Для оценки и обучения нужен размеченный набор пар. Случайные пары почти все
будут `different_product`, поэтому выборка делалась стратифицированно:
cross-marketplace, hard negatives, pack variants, high/medium/low similarity.

Фактические данные разметки:

| Артефакт | Метрика | Значение |
| --- | --- | ---: |
| `labeling_sauces.csv` | строк | 400 |
| `labeling_sauces.csv` | `exact_duplicate` | 148 |
| `labeling_sauces.csv` | `different_product` | 231 |
| `labeling_sauces.csv` | `uncertain` | 21 |
| multi-category labeling CSV | строк | 3 000 |
| финальный clean snapshot | training pairs | 2 465 |
| финальный clean snapshot | negative / positive | 1 686 / 779 |
| финальный clean snapshot | category counts | sauces 1 046, coconut 725, soap 694 |

Telegram labeling bot:

| Метрика локального state | Значение |
| --- | ---: |
| Пользователей | 5 |
| Команд | 2 |
| Assignments всего | 2 641 |
| Labeled assignments | 2 021 |
| Player answers | 1 625 |
| Discussion posts | 18 |

Что подсветить:

- бот вынесен в `tools/dedup_labeling_bot/`, отдельно от research-пакета;
- есть password auth, SQLite sidecar, уникальная выдача строк, overlap-режим,
  команды, очки и milestone-сообщения;
- это не просто "мы руками разметили CSV", а отдельный операторский инструмент.

Источники: `research/dedup/data/training/*manifest.json`,
`tools/dedup_labeling_bot/`, SQLite sidecar разметки.

### Этап 7. Обучение и сравнение моделей

Обучение вынесено в scripts-first подход:

- `research/dedup/training/prepare_dataset.py`;
- `train_cross_encoder.py`;
- `score_cross_encoder.py`;
- `calibrate_scores.py`;
- Docker runtime для GPU-сервера.

Почему так:

- GPU-сессию опасно вести через notebook-ячейки;
- скрипт дает воспроизводимую команду, manifest, split и score CSV;
- notebook остается витриной сравнения.

Split-ы:

| Split | Назначение | Размер |
| --- | --- | ---: |
| strict no-leak fallback `positive_record_holdout` | обучение | train/dev/test = 1 404 / 167 / 165 |
| dropped crossing negatives | защита от leakage | 729 |
| pair-stratified benchmark | финальный stress-test | 2 465 пар, train/dev/test = 1 726 / 370 / 369 |

Главный benchmark: threshold выбирается только на dev, test используется
только для финальной проверки.

Итоговая таблица test, стратегия `threshold_weighted_cost`:

| Модель | Precision | Recall | F1 | False merge | False split | Cost | Weighted F1 | Weighted cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fine-tuned BGE | 95.4% | 84.6% | 89.7% | 5 | 19 | 44 | 87.0% | 261.2 |
| BGE zero-shot reranker | 93.1% | 22.0% | 35.5% | 2 | 96 | 106 | 35.2% | 499.1 |
| E5 bi-encoder zero-shot | 93.8% | 12.2% | 21.6% | 1 | 108 | 113 | 21.2% | 517.7 |
| Rule-based fuzzy | 90.3% | 22.8% | 36.4% | 3 | 95 | 110 | 33.2% | 537.1 |
| mMARCO cross-encoder zero-shot | 0.0% | 0.0% | 0.0% | 2 | 123 | 133 | 0.0% | 635.3 |

Вывод для слайда:

> Дообученный BGE стал первым вариантом, который одновременно держит высокую
> точность и нормальное покрытие. Поэтому именно он перенесен в production v1.

Разбор по объему продаж для fine-tuned BGE:

| Bucket | Pairs | Precision | Recall | F1 | False merge | False split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high | 113 | 90.5% | 73.1% | 80.9% | 2 | 7 |
| medium | 130 | 91.4% | 82.1% | 86.5% | 3 | 7 |
| low | 107 | 100.0% | 90.4% | 94.9% | 0 | 5 |
| zero | 19 | 100.0% | 100.0% | 100.0% | 0 | 0 |

Хорошая фраза:

> Самые дорогие ошибки концентрируются на high/medium продажах, поэтому мы
> выбирали не максимальный F1, а weighted-cost стратегию.

Источники: `artifacts/reports/fine_tuning/all_models_terminal_summary.csv`,
`artifacts/reports/fine_tuning/binary_threshold_summary.csv`,
`artifacts/reports/fine_tuning/binary_threshold_by_volume_bucket.csv`.

### Этап 8. Из пар в группы товаров

Pairwise score сам по себе не является финальным продуктом. Пользователю нужна
группа SKU: базовый товар и фасовки.

Что сделали:

- positive-pairs превращаются в граф;
- `family` строится через Leiden community detection, чтобы слабое bridge-edge
  не обязательно склеивало огромную компоненту;
- `pack` строится внутри family по deterministic signature веса/набора;
- connected components сохранены как audit-baseline.

Fusion quality на test:

| Category-run | Family precision | Family recall | Family F1 | Pack precision | Pack recall | Pack F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sauces` | 94.1% | 80.0% | 86.5% | 92.9% | 76.5% | 83.9% |
| `coconut_oil` | 100.0% | 91.4% | 95.5% | 100.0% | 88.9% | 94.1% |
| `soap` | 86.4% | 76.0% | 80.9% | 66.7% | 66.7% | 66.7% |

Компоненты research outputs:

| Category-run | Nodes | Families | Pack groups | Grouped nodes | Largest family |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sauces` | 585 | 514 | 556 | 139 | 3 |
| `coconut_oil` | 250 | 157 | 233 | 119 | 18 |
| `soap` | 382 | 339 | 370 | 85 | 3 |

Что сказать:

> Мы не остановились на "модель сказала score". Финальный слой переводит score
> в структуру, которую можно показать пользователю и использовать в отчете.

Источники: `research/dedup/data/*/fusion_pair_eval_*.csv`,
`research/dedup/data/*/fusion_components_*.csv`.

### Этап 9. Перенос ML-дедупа в production pipeline

Это главный поворот проекта: dedup больше не только notebook.

Production v1:

- запускается после сохранения classified-файлов в DuckDB;
- поддерживает `Соус/Соусы`, `Кокосовое масло`, `Мыло`;
- строит SKU-node catalog из `mpstats_products`;
- FAISS retrieval: `top_k=30`;
- scorer: fine-tuned BGE cross-encoder;
- activation: `sigmoid`;
- стратегия: `threshold_weighted_cost`;
- thresholds по категориям:
  - `sauces = 0.917444`;
  - `coconut_oil = 0.872321`;
  - `soap = 0.930329`;
  - общий fallback `0.872321`.

Production model profile:

```text
method = ft_bge_reranker_v2_m3
hf_model_id = exoldoff/bge-reranker-v2-m3-cross-encoder-marketplaces-rus
embedding_model = intfloat/multilingual-e5-small
faiss_top_k = 30
```

DB impact production v1:

| Таблица | Роль |
| --- | --- |
| `dedup_runs` | запуск, модель, threshold, статус, manifest |
| `dedup_sku_nodes` | SKU-node catalog |
| `dedup_sku_edges` | пары, score, решение |
| `dedup_sku_groups` | family/pack assignment |
| `mpstats_products_dedup` | materialized browser/export table |
| `dedup_node_embeddings` | cache embeddings отдельных node |
| `dedup_pair_score_cache` | cache score пары |
| `dedup_identity_assignments` | стабильное node -> group назначение |

Важно:

- `mpstats_products` не схлопывается и не мутируется;
- отчеты и выгрузки получают ML-поля через join;
- если fine-tuned модель недоступна, run падает, а не делает тихий fallback на
  rule-based/zero-shot.

Источники: `pipeline/services/dedup/`, `pipeline/migrations/012-015_*.sql`,
`mpstats_app/api/dedup.py`, `web/src/App.tsx`, `docs/USER_GUIDE.md`.

### Этап 10. Инкрементальность и cache

Что было сделано после первого production-встраивания:

- identity-cache: уже известные SKU-node сохраняют группу;
- node embedding-cache в DuckDB;
- pair score-cache для cross-encoder;
- snapshot-cache в `data/projects/<project>/dedup_cache/<category>/<cache_key>/`;
- `embeddings.npy` читается через `np.load(..., mmap_mode="r")`;
- `IndexFlatIP` пересобирается в памяти, FAISS index-файлы в v1 не сохраняются.

Почему это важно:

- повторный запуск не обязан заново считать все embeddings;
- если появились новые SKU, сравниваются новые node со старыми группами и
  между собой;
- corrupt/mismatched cache не ломает run, а приводит к rebuild;
- пользователь видит cache status в UI.

Источники: `pipeline/services/dedup/retrieval_cache.py`,
`pipeline/services/dedup/service.py`, `docs/ARCHITECTURE_PROGRESS.md`.

### Этап 11. Пользовательская поверхность ML-дедупа

Что есть в web-app:

- вкладка `Данные` -> `ML-дедуп`;
- список eligible-категорий;
- карточка текущего run: статус, процент, этап, время, node, FAISS-кандидаты,
  проскоренные пары, группы;
- таблица запусков с cache status;
- browser результата `mpstats_products_dedup`;
- режимы browser:
  - `Основная`: базовый товар -> фасовки;
  - `Дубли`: канон фасовки -> входящие SKU-node;
  - `Каноны`: только верхний уровень;
- экспорт `CSV 2 уровня` и `CSV каноны`;
- переключатель `Дедупликация ON/OFF` в обычной выгрузке.

Что удобно показать в демо:

1. Открыть `Данные` -> `Куб`: показать, что данные уже лежат в DuckDB.
2. Открыть `Данные` -> `ML-дедуп`: показать run и cache status.
3. В browser выбрать `Основная`: показать базовый товар и фасовки.
4. В `Выгрузка` включить `Дедупликация ON`: показать, что строки фактов
   остаются, но SKU нормализуется.

### Этап 12. Демонстрационная цифра по кокосовому маслу

Это хороший финальный proof, потому что уже идет из production/web-app слоя, а
не только из notebook.

Локальный dev export `Кокосовое масло`, run `2026-07-01`:

| Метрика | Значение |
| --- | ---: |
| Строк фактов в выгрузке | 1 388 |
| Уникальных артикулов | 581 |
| Уникальных исходных SKU-title | 555 |
| Уникальных SKU после `Дедупликация ON` | 65 |
| Схлопнулось уникальных SKU-title | 490 |
| Снижение уникальности | 88.29% |
| Compression ratio | 8.54x |
| Продажи, шт | 208 343 |
| Выручка, руб | 178 572 138 |

По маркетплейсам:

| Маркетплейс | Строк | Артикулов | Было SKU | Стало SKU | Снижение | Compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ozon | 977 | 425 | 400 | 47 | 88.25% | 8.51x |
| WB | 346 | 127 | 129 | 29 | 77.52% | 4.45x |
| Яндекс.Маркет | 65 | 29 | 33 | 11 | 66.67% | 3.00x |

Top-группы для слайда:

| Нормализованный SKU | Строк | Было SKU | Артикулов | Продажи | Выручка |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Кокосовое масло` | 728 | 305 | 339 | 75 023 | 71 250 194 |
| `Масло кокосовое Roi Thai рафинированное` | 159 | 74 | 81 | 59 802 | 44 637 677 |
| `Масло МСТ, кокосовое (MCT Oil, кето диета)` | 59 | 43 | 43 | 9 337 | 7 855 749 |

Формулировка:

> На локальном срезе кокосового масла ML-дедуп уменьшил 555 вариантов названий
> до 65 нормализованных SKU, при этом 1 388 фактовых строк остались на месте.

Источник: `artifacts/reports/coconut_oil/dedup_export_stats_2026-07-01.md`.
Это локальный evidence-срез.

### Этап 13. Что сейчас в worktree как WIP

В текущем worktree есть незакоммиченные изменения по ручному исправлению
ML-группы:

- новый API route `/api/dedup/products/split`;
- schema `DedupProductSplitPayload`;
- таблица `dedup_manual_overrides`;
- repository/service методы для `split_singleton`;
- применение override к `dedup_sku_groups` и materialized table.

Как подавать:

- если успеешь довести до тестов и коммита, это хороший слайд про "человек
  может исправить редкую ошибку модели";
- пока это WIP, не продавать как финально готовую фичу.

## 4. Рекомендуемый порядок слайдов

### Слайд 1. Название и one-liner

Заголовок:

```text
MPstats-FMCG-ETL: локальный куб MPStats и ML-дедуп SKU
```

Подзаголовок:

```text
От выгрузок маркетплейсов до нормализованных товарных групп без потери фактовых строк.
```

### Слайд 2. Проблема данных

Показать 2-3 примера разных названий одного товара.

Главные тезисы:

- один товар = много карточек;
- веса, наборы, бренд и маркетинговые хвосты ломают аналитику;
- ручная чистка не масштабируется.

### Слайд 3. Что делает продукт

Схема:

```text
MPStats -> raw -> processed -> classified -> DuckDB -> отчеты/выгрузки -> ML-дедуп
```

Подсветить: локально, воспроизводимо, через web-app.

### Слайд 4. Web-app как основной пользовательский сценарий

Показать разделы:

- `Проекты`;
- `Умный план`;
- `Категории`;
- `Данные`;
- `Классификатор`.

Фраза:

> Пользователь не редактирует JSON руками. Настройки, классификатор,
> куб, отчеты и ML-дедуп доступны через интерфейс.

### Слайд 5. Data pipeline

Раскрыть путь raw/processed/classified/DuckDB.

Цифры:

- локальный dev cube: 913 592 строк;
- `cube_registry`: 137 срезов;
- крупнейший локальный срез по категории: `Мыло`, 773 314 строк.

### Слайд 6. Бизнес-правила и классификатор

Цифры:

- 226 правил;
- 222 активных;
- 3 ручные override-правки.

Показать, что это управляемый слой, а не hardcode в notebook.

### Слайд 7. Почему SKU-дедуп сложнее обычной классификации

Разделить:

- базовый товар;
- фасовка;
- разные маркетплейсы;
- false merge дороже false split.

### Слайд 8. Research-путь дедупликации

Схема:

```text
EDA -> кандидаты -> разметка -> сравнение моделей -> threshold -> graph grouping -> demo
```

Показать notebooks `00`-`05`.

### Слайд 9. Candidate generation

Показать таблицу recall@20 и candidate counts.

Главная мысль:

> Сначала проверяется retrieval. Это отдельный этап качества, не спрятанный
> внутри итоговой метрики.

### Слайд 10. Разметка

Показать:

- 3 000 строк multi-category labeling CSV;
- 2 465 clean training pairs;
- Telegram bot как инструмент командной разметки.

### Слайд 11. Сравнение моделей

Поставить таблицу fine-tuned BGE vs zero-shot/rule-based.

Главная цифра:

```text
Fine-tuned BGE: precision 95.4%, recall 84.6%, F1 89.7%
```

Сразу объяснить:

- threshold выбран на dev;
- test только финальный readout;
- weighted-cost выбран из-за цены false merge.

### Слайд 12. Из model score в товарные группы

Показать:

- family/pack;
- Leiden graph grouping;
- fusion metrics по категориям.

Особенно хороший блок:

```text
Кокосовое масло: family F1 95.5%, pack F1 94.1% на test.
```

### Слайд 13. Production-встраивание

Показать, что выбранный метод перенесен в `pipeline/services/dedup/`.

Тезисы:

- запускается после сохранения куба;
- отдельные identity tables;
- `mpstats_products` не мутируется;
- fine-tuned model required;
- category-specific thresholds.

### Слайд 14. Cache и инкрементальность

Коротко:

- identity-cache;
- embedding-cache;
- pair score-cache;
- snapshot-cache;
- повторный запуск не пересчитывает все с нуля.

Это хороший инженерный слайд, но не растягивать.

### Слайд 15. UI результата

Показать вкладку `Данные` -> `ML-дедуп`:

- карточка run;
- cache status;
- browser `Основная` / `Дубли` / `Каноны`;
- экспорт CSV.

### Слайд 16. Демонстрация эффекта на кокосовом масле

Главная таблица:

```text
555 исходных SKU-title -> 65 нормализованных SKU
88.29% reduction
8.54x compression
1 388 фактовых строк сохранены
```

Это самый понятный бизнес-результат.

### Слайд 17. Надежность и ограничения

Что сказать честно:

- локальные CSV/DB evidence не являются состоянием пользовательской машины;
- текущий production v1 поддерживает только три категории;
- false merge полностью не исчезает, поэтому нужен ручной review/override;
- WIP manual split можно показать только после завершения и тестов.

### Слайд 18. Итог

Финальный тезис:

> Мы построили не только модель, а полный контур: загрузка данных, куб,
> классификация, разметка, обучение, оценка, graph grouping и production
> ML-дедуп внутри локальной web-app.

## 5. Что точно стоит подсветить

- Локальная web-app как главный продукт, не notebook-first подход.
- DuckDB как локальный аналитический слой.
- Управляемый классификатор правил.
- Research и production разделены до момента выбора подхода.
- Candidate generation отдельно оценивается recall@k.
- Разметка сделана не только вручную в CSV, а через отдельный Telegram-tool.
- Fine-tuned BGE сильно выигрывает у zero-shot и rule-based по балансу
  precision/recall.
- Threshold выбирается на dev, test не используется для подбора.
- Ошибка false merge явно дороже false split.
- Grouping решает два уровня: базовый товар и фасовка.
- Production dedup не мутирует исходные факты.
- Cache и инкрементальность превращают research-идею в рабочий runtime.
- Кокосовый export дает понятную бизнес-цифру: 555 -> 65 SKU.

## 6. Что лучше не раздувать

- Не делать длинный рассказ про все модели Hugging Face. Достаточно:
  baseline, bi-encoder, reranker, fine-tuned BGE.
- Не продавать old sauce-only цифры как текущий финальный результат: сейчас
  главный статус multi-category и production v1.
- Не говорить, что дедуп физически схлопывает продажи. Это неверно: строки
  фактов остаются.
- Не показывать `macro-F1` как главный критерий. Для проекта важнее
  false merge, weighted cost и test readout.
- Не обещать поддержку всех категорий: production v1 ограничен `Соусы`,
  `Кокосовое масло`, `Мыло`.
- Не делать вид, что локальная dev DB равна данным пользователя.

## 7. Где брать таблицы и скрины

| Что нужно | Файл / место |
| --- | --- |
| Общая карта проекта | `docs/JURY_GUIDE.md` |
| Pipeline overview | `docs/PIPELINE_OVERVIEW.md` |
| Пользовательский сценарий | `docs/USER_GUIDE.md` |
| Research methodology | `docs/ARCHITECTURE.md` |
| Текущий progress | `docs/ARCHITECTURE_PROGRESS.md` |
| Порядок notebooks | `notebooks/README.md` |
| Retrieval recall table | `docs/assets/embedding_recall_mini_tests.csv` |
| Training dataset manifest | `research/dedup/data/training/dedup_pairs_final_manifest.json` |
| Fine-tuned model ranking | `artifacts/reports/fine_tuning/all_models_terminal_summary.csv` |
| Full threshold benchmark | `artifacts/reports/fine_tuning/binary_threshold_summary.csv` |
| Volume bucket breakdown | `artifacts/reports/fine_tuning/binary_threshold_by_volume_bucket.csv` |
| Fusion outputs | `research/dedup/data/*/fusion_pair_eval_*.csv` |
| Coconut oil export stats | `artifacts/reports/coconut_oil/dedup_export_stats_2026-07-01.md` |
| Production profile | `pipeline/services/dedup/model_profile.json` |
| Dedup service | `pipeline/services/dedup/service.py` |
| Dedup API | `mpstats_app/api/dedup.py` |
| Dedup UI | `web/src/App.tsx`, `web/src/styles.css` |
| DB schema | `pipeline/migrations/012-015_*.sql` |

## 8. Короткий вариант устного рассказа

Мы начали с практической проблемы MPStats: данные маркетплейсов нужны для
аналитики, но приходят разрозненными срезами и разными названиями товаров.
Сначала был собран локальный web workflow: проект, план загрузок, raw-файлы,
processed-слой, классификация, DuckDB-куб, отчеты и выгрузки. Это уже дает
пользователю рабочий инструмент без ручного редактирования файлов.

Дальше появилась главная ML-задача: один и тот же FMCG-товар может иметь много
SKU. Мы отдельно исследовали этот слой в notebooks: EDA, поиск пар-кандидатов
через embeddings и FAISS, стратифицированная разметка, Telegram-инструмент для
разметчиков, обучение reranker-моделей и cost-sensitive threshold calibration.
Главный результат benchmark: fine-tuned BGE на test дал precision 95.4%,
recall 84.6% и F1 89.7% в weighted-cost режиме.

После этого выбранный вариант был перенесен в основной pipeline. Теперь
ML-дедуп запускается после сохранения куба, пишет отдельные identity tables,
не меняет исходные фактовые строки и дает browser/CSV-выгрузки в web-app.
На локальном срезе кокосового масла 555 вариантов SKU-title были сведены к 65
нормализованным SKU, при этом 1 388 фактовых строк остались в данных.

Итог: это не только модель, а полный локальный продуктовый контур от MPStats
до аналитического куба и ML-нормализации товарных карточек.
