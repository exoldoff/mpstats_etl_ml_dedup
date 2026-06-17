# SKU Matching Binary Threshold Benchmark

Этот отчёт описывает текущий критерий для benchmark SKU matching.
Актуальные числа появляются после запуска `notebooks/03_matching_comparison.ipynb`
и сохраняются в `artifacts/reports/`.

## Правило предсказания

Benchmark больше не делает triage и не отправляет пары в ручную проверку.
Для каждой пары есть один `score` и один порог:

```python
predicted_binary = 1 if score >= threshold_same else 0
```

`same_base_product` строится так:

- `1`: `exact_duplicate` и legacy `same_product_different_pack`;
- `0`: `different_product`.

`threshold_same` выбирается только на `dev`. `test` используется только для
финальной оценки выбранного на `dev` порога.

## Threshold Strategies

Для каждого `method` считаются:

- `threshold_max_f1` — максимальный обычный F1 на `dev`;
- `threshold_cost_sensitive` — минимальная цена ошибки:
  `FP_COST * false_merge_count + FN_COST * false_split_count`;
- `threshold_max_weighted_f1` — только если доступны объёмы продаж;
- `threshold_weighted_cost` — только если доступны объёмы продаж.

Стартовые параметры:

```python
FP_COST = 5
FN_COST = 1
```

False merge дороже false split, потому что ложная склейка разных товаров
портит справочник и downstream-отчёты.

## Вес по объёму продаж

Основной бизнес-вес ошибки строится по объёму продаж, не по выручке.
Revenue / turnover / GMV не используются как основной вес.

Если есть `sales_volume_a` и `sales_volume_b` или notebook может надёжно
подтянуть `Продажи, шт` из `mpstats_products` по `raw_record_id`, вес пары:

```python
pair_importance = max(sales_volume_a, sales_volume_b)
pair_weight = log1p(pair_importance)
```

Если объёмы продаж нельзя подтянуть надёжно, benchmark не пытается матчить по
названию. Он использует `pair_weight = 1`, оставляет weighted-метрики пустыми
и пишет `weight_source = unit_weight_fallback`.

## Основные артефакты

- `artifacts/reports/binary_threshold_summary.csv` — одна строка на
  `method + split + threshold_strategy`.
- `artifacts/reports/binary_threshold_predictions.csv` — предсказания по
  парам: score, true label, predicted label, `false_merge`, `false_split`,
  `pair_weight` и контекст пары.
- `artifacts/reports/binary_threshold_by_volume_bucket.csv` — появляется
  только при доступных weighted-метриках.

Bucket-и объёма продаж: `zero / low / medium / high`. Cutoffs считаются только
на `dev` и применяются к `test` без пересчёта.

## Как читать итог

В конце notebook выводит компактную таблицу только по `test`:

- `method`;
- `threshold_strategy`;
- `threshold_same`;
- `precision`, `recall`, `f1`;
- `false_merge_count`, `false_split_count`;
- `cost`;
- `weighted_f1`, `weighted_total_cost`;
- `weight_source`.

Если weighted-метрики недоступны, обычный unweighted binary benchmark всё
равно считается полностью.

## Downstream Fusion

Следующий notebook — `notebooks/04_fusion_pack_grouping.ipynb`. Он читает
`binary_threshold_summary.csv` и `binary_threshold_predictions.csv`, выбирает
`method + threshold_strategy` только по `dev`, а затем строит два уровня
групп. Дефолт для downstream — `threshold_weighted_cost`, то есть минимальная
цена ошибок с весом продаж; если weighted-строк нет, fallback —
`threshold_cost_sensitive`.

- `fusion_family_id` — базовый товар;
- `fusion_pack_id` — конкретная фасовка внутри family по deterministic
  `unit/total/multipack` полям.

Результаты сохраняются в `research/dedup/data/fusion_components_sauces.csv`
и `research/dedup/data/fusion_pair_eval_sauces.csv`.
