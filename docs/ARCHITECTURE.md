# SKU Deduplication Pipeline — Architecture (v3, зафиксировано)

## 0. Контекст и цель
Дедупликация и нормализация SKU из MPStats для аналитического куба.
Стартовый research scope был одной категорией — **«Соусы»**. Текущий
research-run расширяется до трёх изолированных category-runs:
`sauces`, `coconut_oil`, `soap`. Модельная задача остаётся общей, но
gold-set, threshold calibration и отчёты должны считаться отдельно по run,
чтобы `Мыло` или `Кокосовое масло` не перетирали и не маскировали результаты
`Соусы`.
Проект конкурсный (JMLC) → пайплайн должен явно демонстрировать ML-глубину
(embeddings, fine-tuning, cross-encoder), а не только rule-based эвристики.

Модельный выход для пары SKU:
- `exact_duplicate`
- `different_product`

`exact_duplicate` здесь означает "тот же базовый товар" для ML-слоя. Если
это тот же продукт, но другая фасовка (`200 г` vs `3×200 г`), пара всё равно
считается positive для модели. Конкретная фасовка выделяется после модели
deterministic pack-правилами и не является отдельным ML-классом.

Для threshold calibration используется бинарный target `same_base_product`:

- `1`: `exact_duplicate` и legacy `same_product_different_pack`;
- `0`: `different_product`.

Если во входных данных уже есть колонка `same_base_product`, она считается
источником правды.

## 1. Входные данные (уже готовы, вне scope этого пайплайна)
- `category` — дана из источника, не требует классификации.
- `unit_amount`, `unit`, `multipack_count`, `total_amount` (или аналог) —
  уже распаршены детерминированным скриптом из сырого title в отдельные колонки.
- `brand` (опционально) — присутствует в источнике для большинства/топовых
  позиций. **Подтверждено EDA на реальных данных категории «Соус»**:
  заполненность `Бренд` — 97.1% (19,473 строк, «Соус»). Решение не меняется,
  но уточняется: это полезный сигнал, но НЕ жёсткий стоп-гейт — даже среди
  заполненных значений есть неоднозначные/грязные написания.
  **Решение: не строим отдельную ML-инфраструктуру под извлечение бренда.**
  Подробности — раздел 2.

### 1.1 Контракт данных (важно, открыт до подтверждения)
В репозитории есть два разных контракта по результатам аудита:
- `processed` — после `standardize_dataframe()` + `parse_weights_dataframe()`.
  Колонки: `Дата, Маркетплейс, Категория, SKU, Бренд, Название, Продажи шт,
  Продавец, Средняя цена руб, Выручка руб, Вес кг сырой, Вес кг (ед.),
  Вес кг, Вес аномалия, Вес причина, Объем кг, Объем т, Год, Цена за кг`.
- `classified` — после `classification_service.py`: удаляет `Вес кг сырой`,
  `Вес аномалия`, `Вес причина`, `Объем кг`; добавляет `месяц`;
  переименовывает `SKU → Артикул`, `Название → SKU`.

**Рабочая гипотеза**: дедупликация работает на `classified`-контракте,
потому что именно там `SKU` = строка названия товара (то, что мы реально
дедуплицируем). **Подтверждено практикой**: `00_eda.ipynb` успешно поднял
19 473 строки / 11 890 уникальных title по категории «Соус» из
`mpstats_products` с разумными числами — грануляция верна, дальше не
перепроверяем.

Маппинг абстрактных полей из этого документа на реальные колонки:
- `unit_amount` (вес одной единицы) → `Вес, кг (ед.)`
- `total_amount` (общий вес позиции) → `Вес, кг`
- `multipack_count` — явной колонки нет, выводится как
  `round(Вес, кг / Вес, кг (ед.))`
- `brand` → `Бренд` (после `.strip().lower()`, без дальнейшей обработки)
- `subcategory` → `Подкатегория` (если заполнена; используется как
  retrieval-blocking scope, не как финальный ML-класс)

**EDA-подтверждение (категория «Соус», 19 473 строки)**: общий вес и вес
единицы пропущены/некорректны в 0.0% строк; оценочный multipack > 1 у 20.6%
строк; аномалии (`pack_count_estimate_too_large` и т.п.) встречаются у <0.3%
строк. Формула `multipack_count = round(total/unit)` надёжна на этих данных,
доп. чистка не требуется.

## 2. Brand handling (упрощено, не приоритет)
- Если `brand` есть в источнике:
  лёгкая нормализация surface form (lowercase, trim, fuzzy-dedup через
  rapidfuzz/простую кластеризацию написаний) — чтобы `Heinz` / `HEINZ` /
  `Хайнц`, если они оба встречаются как значения поля brand, схлопнулись
  в один `canonical_brand`. Это дешёвая операция над словарём уникальных
  значений поля, не над парами товаров.
- Если `brand` отсутствует:
  никакого NER / weak-supervision / LLM-extraction. Просто `brand = None`.
  В fusion-слое (раздел 5) бренд в этом случае не используется как жёсткий
  гейт — решение принимается по семантическому сходству текста
  (embeddings / cross-encoder), которые сами по себе чувствительны к бренду,
  если он упомянут в названии.
- Если title совпадает после лёгкой нормализации регистра/пробелов и
  нормализованный `brand` тоже совпадает, такие строки схлопываются в один
  research-record до embedding blocking. Пустой `brand` считается отдельным
  значением, поэтому одинаковые безбрендовые SKU тоже схлопываются. Это
  детерминированный pre-processing, чтобы очевидные exact-title дубли не
  занимали соседние места в FAISS и не путали downstream-разметку.
- Если в ходе работы окажется, что доля missing-brand и качество матчинга
  на этих строках — проблема, эскалируем точечно (не сейчас).

## 3. Общая схема пайплайна

```
[1] Raw SKU (category дана, weight/pack уже в колонках)
        │
        ▼
[2] Light brand normalization (см. раздел 2)
        │
        ▼
[3] Blocking / retrieval
    bi-encoder kNN внутри category/subcategory (FAISS/hnswlib)
    + маленький global safety-net; если subcategory нет, full-global внутри category
        │
        ▼
[4] Pairwise matching / rerank — ML-ядро, технологии для сравнения:
    A) Rule-based fuzzy baseline
    B) Bi-encoder zero-shot (готовая мультиязычная модель)
    C) Bi-encoder fine-tuned (контрастивно, на своих данных)
    D) Cross-encoder fine-tuned (парный классификатор)
    E) LLM-judge — каскадный хвост, только low-confidence пары
        │
        ▼
[5] Fusion / calibration:
    brand signal + rerank_score → same_base_product score
    + dev-only threshold_same → forced binary same/different
        │
        ▼
[6] Графовая кластеризация, 2 уровня:
    Level 1 — Product Family по positive pairwise edges
    Level 2 — pack-группа внутри семьи по deterministic pack signature
        │
        ▼
[7] Чистый справочник SKU + отчёт сравнения технологий
```

## 4. Matching engines — детали по каждому (A–E)

| # | Метод | Где крутится | Роль |
|---|---|---|---|
| A | Rule-based fuzzy (rapidfuzz) | локально | baseline, без модели |
| B | Bi-encoder zero-shot (multilingual-e5 / LaBSE / mpnet) | локально или API embeddings | retrieval + baseline rerank |
| C | Bi-encoder fine-tuned контрастивно | train в облаке, inference локально | основной retrieval после baselines |
| D | Cross-encoder fine-tuned / reranker scorer | train в облаке, inference локально/cloud endpoint | финальный rerank, точность |
| E | LLM-judge | API (Claude) | только спорные пары с низкой confidence от B/C/D |

Позитивные/негативные пары для C и D майнятся без ручной разметки:
позитив — общий `canonical_brand` (где есть) + общий `total_amount`/pack-сигнатура;
негатив — hard negatives внутри той же категории (близкие по эмбеддингу,
но разный продукт). Небольшой gold-set вручную всё равно нужен — для
финальной оценки и для калибровки порогов.

## 5. Fusion и binary threshold evaluation — псевдокод

```python
def pair_score(a, b, rerank_score):
    if a.canonical_brand and b.canonical_brand:
        if a.canonical_brand != b.canonical_brand:
            return low_score
    # если brand отсутствует у одной/обоих сторон — не блокируем,
    # решение остаётся за rerank_score
    return rerank_score

def predict_binary(score, threshold_same):
    return 1 if score >= threshold_same else 0
```

`threshold_same` выбирается только на dev split. Test split не используется
ни для выбора порога, ни для выбора модели.

Для каждого method считаются несколько стратегий:

- `threshold_max_f1` — максимальный обычный F1 на dev;
- `threshold_cost_sensitive` — минимум
  `FP_COST * false_merge_count + FN_COST * false_split_count`;
- `threshold_max_weighted_f1` и `threshold_weighted_cost` — только если есть
  надёжный объём продаж по SKU A/B.

Стартовая цена ошибок: `FP_COST = 5`, `FN_COST = 1`. Если объём продаж
доступен, вес пары считается как `log1p(max(sales_volume_a, sales_volume_b))`.
Revenue / turnover / GMV не используются как основной бизнес-вес. Если
объёмы нельзя подтянуть надёжно, benchmark работает как обычный unweighted
binary benchmark с `pair_weight = 1`.

## 6. Кластеризация
- Level 1: graph grouping по positive binary-рёбрам выбранной стратегии
  → товарная семья / базовый продукт. Research default теперь —
  Louvain community detection по weighted-рёбрам (`score`) с фиксированным
  seed, чтобы слабый bridge-edge не обязательно склеивал всю connected
  component. Старый режим connected components остаётся доступен как baseline
  и сохраняется в audit-колонках.
- Level 2: внутри выбранной family группировка по deterministic pack signature из
  готовых weight/multipack колонок → финальная pack-группа.
- Research v1 делает это отдельным шагом после `03_matching_comparison.ipynb`:
  `04_fusion_pack_grouping.ipynb` выбирает `method + threshold_strategy`
  только по `dev`, затем сохраняет `fusion_components_<suffix>.csv` и
  `fusion_pair_eval_<suffix>.csv` для downstream-отчётов. Для общего
  frozen/fine-tuning benchmark notebook может запускаться один раз на
  `sauces,coconut_oil,soap`: он восстанавливает `category_run` из frozen
  split и строит отдельные family/pack графы внутри каждой категории.
  `fusion_family_id` / `fusion_pack_id` остаются финальным contract для
  downstream, а `connected_family_id` / `connected_pack_id` нужны для сравнения
  с прежним connected-components поведением.
- Текущий threshold benchmark не создаёт `manual_review` / triage-зону.

## 6.1 Организация кода на research-этапе (пересмотрено по запросу)
Предыдущая версия предлагала сразу встраивать matching-код в
`pipeline/services/dedup/`. **Пересмотрено**: сейчас research-этап —
сравнение технологий (rule-based / embeddings zero-shot / fine-tuned /
cross-encoder / LLM-judge), а не финальная интеграция. `pipeline/` и
`mpstats_app/` не трогаем вообще до тех пор, пока технология не выбрана.

Текущая организация:
- `notebooks/` — сами ноутбуки-деливераблы конкурса (план — раздел 7).
- `research/dedup/` — отдельная песочница, не импортируется из `pipeline/`
  и не импортирует `pipeline/`. Туда выносятся только чисто переиспользуемые
  функции, чтобы не копипастить код между ноутбуками (генерация кандидатов,
  метрики, обёртки над моделями). Импортируется из ноутбуков как обычный
  локальный пакет (`from research.dedup import ...`).

Когда технология выбрана по результатам сравнения (раздел 8) — тогда, и
только тогда, финальный выбранный вариант переносится в
`pipeline/services/...` под реальную структуру репозитория. До этого
момента это намеренно не «production-код».

## 7. Деливераблы конкурса — план ноутбуков (обновлено по итогам EDA)
- `00_eda.ipynb` — готово (раздел 1.2).
- `01_candidate_generation.ipynb` — candidate generation внутри выбранного
  `DEDUP_CATEGORY_RUN`: dense embeddings + FAISS top-k. Если `Подкатегория`
  заполнена, основной поиск идёт внутри неё; пустая/отсутствующая
  `Подкатегория` не блокирует строки и откатывает их в full-global fallback.
  Brand/weight остаются вспомогательными признаками (не жёсткими гейтами).
  Чтобы обучающий датасет не был заложником текущего `FAISS_TOP_K`, candidate
  CSV дополнительно включает supplemental training coverage pairs вне FAISS
  top-k: lexical overlap, same-brand/same-pack controls,
  cross-marketplace random и random controls. Явный блок hard-negative mining:
  пары с похожим brand+weight, но разными flavor-токенами (EDA уже нашла 10
  таких примеров вручную — нужно находить их систематически, не вручную).
  Выход — ранжированный список пар-кандидатов с baseline similarity score и
  `candidate_source`.
- `02_labeling_dataset.ipynb` — стратифицированная выборка пар из 01 для
  ручной разметки: cross-marketplace пары, hard negatives, pack variants,
  высокое/среднее/низкое similarity и немного случайных лёгких негативов для
  контроля. Экспорт в CSV для ручной разметки:
  `exact_duplicate`, `different_product`, `uncertain`. Legacy label
  `same_product_different_pack` при чтении старых данных временно мапится в
  `same_base_product=1`, но новый gold-set должен использовать 2 модельных
  класса. Для быстрого benchmark можно собирать 300-500 строк на один
  category-run; для fine-tuning reranker целевой режим — multi-category
  датасет около 3000 строк, примерно поровну по `sauces`, `coconut_oil`,
  `soap`, с теми же стратами внутри каждой категории.
- `03_matching_comparison.ipynb` — A vs B vs C vs D vs E на gold-set из 02,
  метрики — см. раздел 8.
- `04_fusion_pack_grouping.ipynb` — выбор fusion-run по `dev`, multi-category
  family/pack grouping, compact CSV `fusion_*`, graph-quality diagnostics,
  примеры ошибок и 3D-визуализация components.
- `06_grouped_sku_demo.ipynb` — демонстрационная витрина: реальные SKU из
  DuckDB, на которые наложены текущие `fusion_family_id` / `fusion_pack_id`.

## 8. Оценка и тестирование (методология)

### 8.1 Три уровня оценки — не путать друг с другом
1. **Candidate generation (retrieval)** — recall@k: для каждой истинной пары
   `same_base_product=1` из gold-set проверяем, попал ли партнёр в top-k
   кандидатов на этапе blocking/retrieval. Если recall@k низкий —
   никакой matching engine дальше это не спасёт, нужно ловить отдельно от
   точности классификации.
2. **Pairwise classification** — точность по самой паре, то, что выдают
   matching engines A-E после fusion.
3. **Clustering (финальный результат)** — точность итоговых групп после
   graph resolution. Меряем отдельно от (2): один ложноположительный
   pairwise-edge может «склеить» в один кластер несколько разных продуктов
   (classic chaining problem в entity resolution) — pairwise-метрики это
   не покажут, а кластерные метрики покажут.

### 8.2 Gold-set — как собирать, чтобы он был информативным
Случайная выборка пар не работает: подавляющее большинство случайных пар —
`different_product`, на таком наборе любая модель будет выглядеть отлично.
Берём стратифицированную выборку из кандидатов `01_candidate_generation`
(детали состава — раздел 7, `02_labeling_dataset.ipynb`).
Делим на dev (тюнинг порогов) и held-out test (финальные цифры для
сравнения технологий) — пороги не тюнятся на том же сете, на котором потом
репортятся итоговые метрики. Ориентир по объёму для стартового benchmark:
300-500 размеченных пар. Для дообучения reranker нужен отдельный более
широкий manual dataset: около 3000 пар по всем трём category-runs, чтобы
модель видела разные типы товаров и не переучивалась на одну категорию.

### 8.3 Метрики
- Forced binary metrics по `same_base_product`: `precision`, `recall`, `F1`,
  `accuracy`, `false_merge_count`, `false_split_count`, rates и cost.
- Cost-sensitive thresholding на dev:
  - `threshold_same`;
  - `threshold_max_f1`;
  - `threshold_cost_sensitive`;
  - weighted strategies, если есть объём продаж.
- Weighted metrics, если доступны продажи в штуках: `weighted_precision`,
  `weighted_recall`, `weighted_f1`, `weighted_false_merge_cost`,
  `weighted_false_split_cost`, `weighted_total_cost`.
- Downstream fusion по умолчанию использует `threshold_weighted_cost`, чтобы
  ошибки на SKU с большим объёмом продаж весили сильнее. Для финального
  сравнения reranker / fine-tuned моделей weighted-строки обязательны:
  fallback `threshold_cost_sensitive` допустим только как диагностика.
- Breakdown по bucket объёма продаж: `zero / low / medium / high`. Cutoffs
  считаются только на dev и применяются к test без пересчёта.
- **Асимметрия цены ошибки**: false merge (`same_base_product=0`, но
  предсказано `1`) — самая дорогая ошибка, потому что портит справочник и
  downstream-отчёты. False split дешевле: дубль можно обработать позже.
- Recall@k для retrieval (раздел 8.1, п.1).
- B-cubed precision/recall/F1 (или ARI) — для финальных кластеров.

### 8.4 Сравнение технологий (A–E)
На одном held-out test-set, для каждой технологии: dev-пороги применяются без
изменений, дальше считаются forced binary metrics, latency на 1000 пар и
стоимость на 1000 пар для API/cloud-вариантов. Текущий benchmark не делает
manual review, LLM-review или triage. Test нельзя использовать для выбора
threshold или модели.

### 8.5 Качественный анализ
Отдельно показать 5-10 примеров ошибок каждого типа в
`04_fusion_pack_grouping.ipynb` (особенно hard negatives, смерженные
неправильно) — не метрика, но полезно и для дебага, и для защиты проекта.

## 9. Открытые вопросы / следующие шаги (обновлено)
- ~~Контракт `mpstats_products` vs `merge_service.py`~~ — снято с приоритета
  пользователем: «обрабатывается → классифицируется по словарю, это не
  важно». Не блокирует research-этап.
- ~~`classifiers/engine.py` + `rules.csv`~~ — снято с приоритета, не
  пересекается с текущей задачей по словам пользователя.
- ~~Бренд — валидация на реальных данных~~ — закрыто: EDA подтвердил
  97.1% заполненности (раздел 1.2).
- Собрать multi-category labeling set для fine-tuning reranker:
  `DEDUP_LABELING_CATEGORY_RUNS=sauces,coconut_oil,soap`,
  `DEDUP_LABELING_TARGET_SIZE=3000`, `02_labeling_dataset.ipynb`.
- Определить конкретные модели B/C для эмбеддингов (мультиязычные, RU/EN).
- Решить, переносить ли что-то из `research/dedup/` в `pipeline/` — только
  после выбора технологии по итогам сравнения (раздел 8.4).
