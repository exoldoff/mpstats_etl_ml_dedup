# Dedup Fine-Tuning Experiment Plan

Дата: 2026-06-29.

Документ фиксирует первый план обучения моделей для research-only SKU dedup.
Production `pipeline/` и web-app не меняются: здесь только подготовка
эксперимента, обучение scorer/reranker и честное сравнение с текущими
zero-shot / threshold benchmark результатами.

## 1. Что уже собрано

### Новый multi-category set

Файл:
`research/dedup/data/labeling_sauces_coconut_oil_soap.csv`

Размер:

- 3000 строк всего;
- 2000 строк с метками прямо в CSV;
- 2000 финализированных строк в Telegram sidecar
  `labeling_sauces_coconut_oil_soap.csv.telegram_state.sqlite`;
- CSV и SQLite совпадают не идеально: 1898 строк размечены в обоих источниках,
  102 строки есть только в CSV, 102 строки есть только в SQLite, 3 строки имеют
  конфликт метки между CSV и SQLite;
- если брать union CSV+SQLite с приоритетом SQLite, получается 2102 строки с
  какой-либо меткой, из них 2090 бинарно пригодны для обучения.

Распределение union с приоритетом SQLite:

| category_run | different_product | exact_duplicate | uncertain |
| --- | ---: | ---: | ---: |
| coconut_oil | 371 | 357 | 4 |
| sauces | 554 | 114 | 4 |
| soap | 531 | 163 | 4 |
| **Итого** | **1456** | **634** | **12** |

Важно: перед обучением нужно создать один подготовленный training artifact,
который явно мержит CSV + SQLite sidecar и выносит 3 конфликтные строки в
отдельный ручной review. Не тренироваться напрямую на текущем CSV, иначе часть
ответов из Telegram потеряется.

### Старый sauce set

Файл:
`research/dedup/data/labeling_sauces.csv`

Размер:

- 400 строк;
- 400 строк с метками;
- 400 уникальных пар;
- пересечение с новым set: 1 пара, метка совпадает.

Распределение:

| category_run | different_product | exact_duplicate | uncertain |
| --- | ---: | ---: | ---: |
| old_sauces / unknown | 231 | 148 | 21 |

### Общий объем для первого обучения

Фактический freeze, который пишет `research.dedup.training.prepare_dataset`
после дедупликации пар, исключения конфликтов CSV-vs-Telegram и приоритета
нового multi-category set:

- 2465 уникальных бинарно пригодных пар;
- 779 positive (`exact_duplicate`, включая legacy positive-логику);
- 1686 negative (`different_product`);
- 33 `uncertain` не идут в train loss, но остаются для ручного анализа;
- 3 conflict rows между CSV и Telegram SQLite пишутся отдельно и не идут в
  train loss.

Текущий split использует fallback `positive_record_holdout`, потому что strict
all-edge component split создаёт giant component на 717 строк почти целиком из
`coconut_oil`. Training-ready split после удаления crossing negative pairs:
`1404 / 167 / 165` для train/dev/test; dropped audit сохраняется как
`research/dedup/data/training/dedup_pairs_final_split_dropped.csv`.
Основной training CSV зафиксирован в clean-схеме
`dedup_training_clean_v1`: без служебных колонок разметчика и notebook-only
диагностик.

Это уже достаточно для первого supervised fine-tune reranker. Для большой 4B
модели это все еще маленький датасет, поэтому первый прогон должен быть
осторожным: короткие эпохи, dev-only early stopping, отдельный untouched test.

## 2. Что уже есть по benchmark

Главный актуальный отчет:
`artifacts/reports/binary_threshold_summary.csv`

Он сравнивает методы в forced-binary режиме:

- threshold выбирается только на `dev`;
- `test` используется только для финальной проверки;
- false merge дороже false split;
- weighted-стратегии используют `log1p(max(sales_volume_a, sales_volume_b))`.

Лучшие test-ориентиры из текущего отчета:

| Режим сравнения | Лучший метод | Test-результат |
| --- | --- | --- |
| Минимальная weighted cost | `reranker_qwen3_0_6b + threshold_weighted_cost` | precision 1.000, recall 0.254, F1 0.405, false_merge 0, false_split 44, weighted_total_cost 172.9 |
| Баланс F1 | `reranker_qwen3_4b + threshold_max_f1` | precision 0.705, recall 0.932, F1 0.803, false_merge 23, false_split 4 |
| Практичный компромисс для fusion | `reranker_qwen3_4b + threshold_weighted_cost` | precision 0.879, recall 0.492, F1 0.630, false_merge 4, false_split 30, weighted_total_cost 188.4 |

Вывод: zero-shot rerankers уже дают сильный сигнал, но trade-off тяжелый:
строгий Qwen 0.6B почти не склеивает лишнего, но теряет много дублей; Qwen 4B
лучше ловит positive, но при F1-пороге дает слишком много false merge. Цель
fine-tune: поднять recall при сохранении низкого false merge, особенно на
hard negatives и cross-marketplace парах.

Старые recall@20 embedding mini-tests по sauces:

| embedding | recall@20 | cross_recall@20 | Комментарий |
| --- | ---: | ---: | --- |
| `intfloat/multilingual-e5-small` | 100.0% | 100.0% | biased baseline, потому что gold-set был собран вокруг E5-кандидатов |
| `ai-forever/FRIDA` | 87.2% | 84.1% | сильный альтернативный retrieval-кандидат |
| `cointegrated/rubert-tiny2` | 76.4% | 68.3% | легкий baseline, слабее |
| `deepvk/RuModernBERT-small` | 66.2% | 57.3% | raw encoder, не dedicated sentence embedder |

Embedding retriever пока не главный bottleneck для fine-tune: первый фокус -
pairwise scorer/reranker. `cointegrated/rubert-tiny2` в таблице выше не надо
читать как готовую embedding-модель: для нашего эксперимента он интересен как
маленький encoder, который можно специально заточить под бинарную pairwise
задачу.

## 3. Какие модели тюнить

### Первый эшелон

1. `Qwen/Qwen3-Reranker-0.6B`
   - Почему: лучший текущий weighted-cost baseline, 0 false merge на test в
     строгом режиме, доступный размер.
   - Как тюнить: supervised reranker на бинарных парах, с category-neutral
     instruction.
   - Роль: основной practical-кандидат.

2. `BAAI/bge-reranker-v2-m3`
   - Почему: легкий multilingual reranker, fast inference, нормальный
     CrossEncoder/FlagEmbedding path.
   - Как тюнить: тем же CSV и тем же split, чтобы понять, дает ли более простая
     архитектура лучший прирост от наших hard negatives.
   - Роль: стабильный open baseline, проще Qwen.

3. `cointegrated/rubert-tiny2`
   - Почему: маленькая русскоязычная BERT-like модель; как retrieval embedding
     она была слабее E5/FRIDA, но как supervised pair classifier может хорошо
     выучить наши SKU-признаки.
   - Как тюнить: не как bi-encoder, а как cross-encoder / sequence
     classification: на вход подается пара текстов SKU, на выходе один binary
     score.
   - Роль: обязательный cheap supervised baseline под русские товарные
     названия.

4. `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
   - Почему: уже есть как `cross_encoder_zero_shot`, маленький и быстрый.
   - Как тюнить: binary CrossEncoder baseline.
   - Роль: cheap sanity baseline. Если даже он сильно прибавит, значит датасет
     действительно учит SKU-специфичные признаки.

### Второй эшелон

5. `jinaai/jina-reranker-v3`
   - Почему: сильный 0.6B multilingual/listwise reranker; в публичной карточке
     сравнивается конкурентно с Qwen/BGE.
   - Ограничение: `trust_remote_code=True`, listwise API, лицензия
     `cc-by-nc-4.0`; fine-tune path менее прямой, чем CrossEncoderTrainer.
   - Роль: оставить в плане fine-tune, но запускать после простого
     CrossEncoder/Qwen/BGE/RuBERT loop, когда уже понятен frozen split,
     score-cache и threshold protocol.

### Benchmark-only, без fine-tune в первом цикле

`Qwen/Qwen3-Reranker-4B` остается в сравнении как сильный zero-shot scorer и
quality ceiling, но не входит в fine-tune plan. Причина простая: локально он
тяжелый, а первый supervised цикл должен отладить данные, split, loss,
threshold calibration и error analysis на более дешевых моделях.

### Отдельно: bi-encoder

`intfloat/multilingual-e5-small` fine-tune имеет смысл как второй этап, когда
reranker уже улучшен:

- цель bi-encoder fine-tune - улучшить candidate retrieval, а не финальный
  scorer;
- нужен другой loss: contrastive / multiple negatives, а не binary
  CrossEncoder loss;
- на текущем E5-biased gold-set нельзя честно объявлять победу retriever, надо
  оценивать на multi-category positives и на hard missed pairs.

## 4. Как делаем эксперимент

### 4.0. Архитектура обучения

Текущий benchmark-код умеет считать scores и калибровать thresholds, но
обучение должно быть отдельным research-слоем, чтобы не смешать data freeze,
training и evaluation. Целевая структура:

```text
labeling CSV + Telegram SQLite + old sauce labels
        │
        ▼
[A] frozen training dataset
    dedup_pairs_final.csv
    dedup_pairs_final_conflicts.csv
    dedup_pairs_final_manifest.json
        │
        ▼
[B] component-aware split
    dedup_pairs_final_split.csv
    dedup_pairs_final_split_manifest.json
        │
        ▼
[C] model-specific training adapters
    CrossEncoderTrainer branch: Qwen 0.6B, BGE, mMARCO
    Transformers Trainer branch: RuBERT tiny2 pair classifier
    Custom/listwise branch: Jina v3
        │
        ▼
[D] unified score cache
    one score column per model on the same frozen split
        │
        ▼
[E] existing threshold calibration + fusion notebooks
    binary_threshold_* -> fusion_* -> evaluation report
```

Research-модули для этого стоит держать под `research/dedup/training/`:

| Компонент | Ответственность |
| --- | --- |
| `dataset_freeze` | Смержить CSV + SQLite + old sauces, удалить/вынести конфликты, построить `same_base_product`, сохранить manifest. |
| `splitter` | Построить graph components по `raw_record_id_*`, разложить components в train/dev/test без leakage, сохранить split manifest. |
| `text_builder` | Единообразно собрать `sentence_A` / `sentence_B` из brand, title, subcategory, unit/total/multipack. |
| `train_cross_encoder` | Общий supervised branch для моделей, которые грузятся как `sentence_transformers.CrossEncoder`: Qwen 0.6B, BGE, mMARCO. |
| `train_pair_classifier` | Transformers branch для `cointegrated/rubert-tiny2`: `AutoModelForSequenceClassification`, tokenizer pair input, `Trainer`. |
| `train_jina_adapter` | Отдельный experimental branch для Jina: custom/listwise scoring, `trust_remote_code=True`, adapter/PEFT только после smoke. |
| `score_models` | Прогнать zero-shot и fine-tuned модели на frozen split, записать scores в общий cache. |
| `training_report` | Собрать metrics, model manifests, error breakdown и ссылки на артефакты. |

Implementation note: первый воспроизводимый слой реализован в
`research/dedup/training/`, а команды запуска зафиксированы в
`docs/DEDUP_TRAINING_RUNBOOK.md`.

Model-specific training paths:

- **Qwen 0.6B / BGE / mMARCO**: основной путь —
  `sentence-transformers` v4-style `CrossEncoderTrainer` +
  `BinaryCrossEntropyLoss`. Для Qwen предпочтительна sequence-classification
  совместимая версия/обертка, если она стабильнее стандартного reranker
  template. Scores могут быть raw logits или sigmoid probabilities, но
  threshold calibration всегда проводится заново на dev именно для выбранного
  score type.
- **RuBERT tiny2**: не тренировать как bi-encoder. Это отдельный binary
  pair-classifier: `AutoTokenizer` получает пару `(sentence_A, sentence_B)`,
  `AutoModelForSequenceClassification(..., num_labels=2)` учится через
  `transformers.Trainer`. Финальный score для calibration — вероятность или
  logit класса `same_base_product=1`; формат фиксируется в manifest.
- **Jina v3**: не притворяется обычным CrossEncoderTrainer. Public inference
  API — `AutoModel(..., trust_remote_code=True).rerank(query, documents)`,
  listwise и асимметричный. Поэтому первый шаг для Jina — zero-shot /
  adapter-smoke на frozen split; fine-tune/adapter-tune идёт отдельным
  experimental branch только после того, как базовый loop стабилен.
- **Qwen 4B**: full fine-tune не запускаем на A5000, но допускаем
  LoRA-only эксперимент: base model заморожена, обучаются adapters,
  `gradient_checkpointing` включён, финальный adapter merge-ится перед
  обычным CrossEncoder scoring.

Каждый training run обязан сохранить:

- `model_id`, `model_revision` или exact local path;
- commit hash репозитория;
- train/dev/test counts и label distribution;
- hyperparameters (`lr`, batch size, epochs, max length, loss, pos_weight);
- score type (`raw_logit`, `sigmoid_probability`, `softmax_same_probability`);
- path к frozen dataset, split manifest, model artifact и score cache;
- environment summary: `sentence-transformers`, `transformers`, `torch`,
  `datasets`, `peft`/`accelerate`, если использовались.

Smoke-gates перед дорогим прогоном:

1. `dataset_freeze` находит 3 текущих label conflicts и не пускает их в train.
2. `splitter` проверяет, что ни один `raw_record_id` component не попал сразу
   в train и test.
3. Training smoke на 16-32 парах проходит forward/backward для каждого trainer
   branch.
4. Scoring smoke на dev пишет score cache с тем же pair count, что входной
   split.
5. Threshold calibration читает этот cache без специальных notebook-hacks.

Dependency impact перед реализацией:

- текущий `requirements-research.txt` уже содержит `sentence-transformers>=5.0`
  и `transformers>=4.51`;
- для training implementation нужно будет добавить минимум `datasets` и
  `accelerate`, потому что `CrossEncoderTrainer` / `transformers.Trainer`
  ожидают dataset/trainer runtime;
- `peft` нужен только если запускаем LoRA/adapters для Qwen/Jina, поэтому это
  optional dependency, не обязательная для первого smoke на RuBERT/BGE/mMARCO.

### Шаг 0. Freeze исходных данных

Собрать один immutable artifact:

- вход: новый CSV + Telegram SQLite + старый sauce CSV;
- выход:
  `research/dedup/data/training/dedup_pairs_final.csv`;
- отдельный файл конфликтов:
  `research/dedup/data/training/dedup_pairs_final_conflicts.csv`;
- отдельный manifest:
  `research/dedup/data/training/dedup_pairs_final_manifest.json`.

Правила:

- `uncertain` не идет в train loss;
- legacy `same_product_different_pack`, если встретится, мапится в positive;
- дубли пар схлопываются по unordered pair key;
- при конфликте меток пара исключается до ручного решения;
- в clean training artifact сохраняются только поля, нужные для обучения и
  анализа ошибок: `split`, `pair_id`, `pair_key`, `category_run`, label/target,
  `sentence_A/B`, raw ids, marketplace, sku/title/brand и pack/weight fields.
  Служебные поля разметчика и notebook-only диагностики остаются вне основного
  train CSV.

### Шаг 1. Split без leakage

Нельзя просто random split по строкам: один и тот же товар может попасть в
train и test через разные пары.

План split:

1. Построить граф по `raw_record_id_a/raw_record_id_b`.
2. Разбить connected components на `train/dev/test`, чтобы связанные товары не
   протекали между split.
3. Сохранить стратификацию по:
   - `category_run`;
   - `same_base_product`;
   - marketplace/pack context в clean CSV для error analysis.

Первый размер:

- train: 70%;
- dev: 15%;
- test: 15%;
- test не трогать до финального сравнения.

### Шаг 2. Zero-shot baseline на том же frozen split

Перед обучением нужно пересчитать текущие модели на новом frozen split:

- `rule_based_fuzzy`;
- `bi_encoder_zero_shot`;
- `cross_encoder_zero_shot`;
- `reranker_qwen3_0_6b`;
- `reranker_qwen3_4b`, если доступен runtime;
- `reranker_bge_v2_m3`;
- `reranker_jina_v3`.

Это станет честной точкой "до fine-tune", потому что старый benchmark был
только на sauces.

### Шаг 3. Fine-tune scorer/reranker

Основной training API: `sentence-transformers` v4-style
`CrossEncoderTrainer` + `BinaryCrossEntropyLoss`.

Dataset columns:

- `sentence_A`: нормализованный текст SKU A;
- `sentence_B`: нормализованный текст SKU B;
- `labels`: `1.0` для `same_base_product`, `0.0` для `different_product`;
- metadata columns сохраняются рядом, но не подаются в модель напрямую.

Training details:

- loss: `BinaryCrossEntropyLoss`;
- `pos_weight`: ratio negatives/positives на train, потому что negative больше;
- 2-4 эпохи максимум для первого прогона;
- learning rate: стартовать с `2e-5` для малых CrossEncoder/BGE/RuBERT; для
  Qwen 0.6B использовать более осторожный LR/LoRA, если full fine-tune
  нестабилен;
- early stopping по dev weighted cost или dev false-merge-constrained F1;
- сохранять model artifact и score cache с manifest.

Для Qwen instruction:

```text
Decide whether two ecommerce products are the same base SKU. Pay attention to
brand, product line, flavor or scent or purpose, unit size, total size, and pack
count. Treat different pack counts as the same base product when the underlying
product is the same.
```

### Шаг 4. Threshold calibration после обучения

После каждой обученной модели не сравниваем raw logits напрямую.
Повторяем тот же binary threshold protocol:

- dev выбирает `threshold_same`;
- test только проверяет;
- считать все стратегии:
  `threshold_max_f1`, `threshold_cost_sensitive`,
  `threshold_max_weighted_f1`, `threshold_weighted_cost`;
- главный продуктовый критерий: `threshold_weighted_cost`;
- вторичный критерий: F1 при controlled false merge.

Calibration после fine-tune запускается с `--require-weighted`: если
`sales_volume_a/sales_volume_b` или `sales_volume_lookup.csv` не доступны,
команда должна завершиться ошибкой. Таблица только с `threshold_max_f1` и
`threshold_cost_sensitive` считается диагностической, но не финальным
продуктовым сравнением.

Для строгого train/dev/test используется no-leak split по raw records. Если
нужна более устойчивая оценка false merge, дополнительно считать
pair-stratified benchmark split: он сохраняет все размеченные negative, но
может иметь raw-id leakage, поэтому не заменяет строгую проверку.

### Шаг 5. Error analysis

Для каждой модели сохранить:

- false merges на test;
- false splits на test;
- breakdown по category;
- breakdown по marketplace/pack/weight context;
- score distributions до/после fine-tune.

Отдельно проверить 3 текущих conflict rows из CSV/SQLite merge: они не должны
попасть в train/test без ручного решения.

### Шаг 6. Downstream graph check

Лучшие 1-2 модели прогнать через текущий `04_fusion_pack_grouping.ipynb`:

- family false links;
- family size distribution;
- pack-level correctness;
- ручная витрина через `06_grouped_sku_demo.ipynb`.

Выбирать модель только по pairwise F1 нельзя: нам важнее отсутствие опасных
false merge в итоговых компонентах.

## 5. Что считать успехом

Минимальный success criterion для первого fine-tune:

- test weighted_total_cost ниже текущего zero-shot baseline на том же frozen
  split;
- false_merge_count не растет относительно выбранного product-safe baseline;
- recall по positive растет хотя бы на hard-negative/pack/cross-marketplace
  сегментах;
- model artifact воспроизводим: есть training data manifest, split manifest,
  model manifest и score cache.

Желаемый success criterion:

- `Qwen 0.6B fine-tuned` сохраняет 0-2 false merge на test, но заметно
  увеличивает recall относительно текущего строгого режима;
- `BGE v2-m3 fine-tuned` становится близким к Qwen 0.6B по weighted cost и
  быстрее/дешевле на inference;
- `RuBERT tiny2 fine-tuned` дает понятный cheap baseline и показывает, сколько
  качества можно получить из маленькой модели, заточенной именно под наши пары;
- `Jina v3` остается в плане как более сложный fine-tune candidate;
- `Qwen 4B` остается benchmark-only quality ceiling, но не блокирует локальный
  research loop.

## 6. Короткий порядок запуска

1. Export/merge labels: CSV + Telegram SQLite + old sauces.
2. Resolve 3 conflicts вручную.
3. Create frozen component-aware split.
4. Re-run zero-shot benchmark на frozen split.
5. Fine-tune `rubert_tiny2`, `cross_encoder_mmarco` и `bge_v2_m3` как быстрый
   smoke.
6. Fine-tune `qwen3_0_6b`.
7. Fine-tune / adapter-tune `jina_v3`, если первые прогоны показали прирост и
   training path не раздувает эксперимент.
8. Re-run threshold calibration and fusion/grouping.
9. Обновить итоговый report/visuals.
