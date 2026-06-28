# Dedup Family Size Diagnostic

Дата проверки: 2026-06-27.

## Вывод

Маленькие `fusion_family_id` по 2-3 SKU в текущей витрине не доказывают,
что в категории больше трёх дублей не бывает.

Текущий `04_fusion_pack_grouping.ipynb` строит graph components не по всему
каталогу и не по `candidates_sauces.csv` на 60 000 пар, а по compact
benchmark predictions из `artifacts/reports/binary_threshold_predictions.csv`.
В проверенном артефакте это 379 уникальных размеченных пар и 716 узлов.
На таком sparse gold-set graph большие связные компоненты почти не могут
появиться.

## Проверенные числа

Текущий saved run:

- `fusion_method`: `reranker_qwen3_4b`;
- `fusion_threshold_strategy`: `threshold_weighted_cost`;
- `fusion_threshold_same`: `7.25`;
- rows in `fusion_pair_eval_sauces.csv`: 379;
- unique nodes in selected prediction graph: 716;
- predicted family positive edges: 78;
- predicted pack positive edges: 53;
- true family positive edges from labels: 148.

Распределение predicted family sizes:

| SKU in family | Family count |
| --- | ---: |
| 1 | 564 |
| 2 | 70 |
| 3 | 4 |

Распределение true family sizes по ручным labels на том же graph:

| SKU in true family | Family count |
| --- | ---: |
| 1 | 431 |
| 2 | 126 |
| 3 | 11 |

То есть потолок `3` виден даже в `true_family`, не только в model output.
Это признак sparse pair sample, а не доказательство реального максимального
числа дублей.

## Sanity-check против full candidate slice

В `research/dedup/data/candidates_sauces.csv` найдено 11 085 уникальных
candidate nodes. Даже по простому exact normalized title есть группы больше
трёх SKU:

- `same normalized title`: max 8, groups with more than 3 nodes = 50;
- `same brand + title`: max 8, groups with more than 3 nodes = 35;
- `same brand + title + unit + total + pack`: max 8, groups with more than
  3 nodes = 35.

Эти проверки не заменяют модельный dedup, но показывают, что большие группы
в данных возможны. Текущая маленькая family-витрина ограничена входным graph.

## Найденные нестыковки

1. В рабочем `notebooks/03_matching_comparison.ipynb` сейчас выставлен
   `DEDUP_FP_COST` default `4`, а архитектурные документы всё ещё описывают
   стартовый `FP_COST = 5`. Это влияет на выбранный threshold/run и перед
   следующим финальным прогоном должно быть приведено к одному решению.
2. В рабочем `notebooks/04_fusion_pack_grouping.ipynb` сейчас явно задан
   `MY_FUSION_METHOD = "reranker_qwen3_4b"`. На текущем summary auto-selection
   выбирает тот же method, поэтому это не объясняет маленькие family, но перед
   чистым reproducible прогоном лучше либо оставить явный выбор осознанно, либо
   вернуть `None`.

## Что делать дальше

Следующий технический шаг - не тюнить graph на gold-set, а сделать full
candidate inference:

1. Взять `research/dedup/data/candidates_sauces.csv` как graph-кандидаты.
2. Проставить score выбранным scorer/reranker для всех 60 000 candidate pairs
   или начать с дешёвого smoke-slice.
3. Применить тот же `threshold_same` / strategy.
4. Построить `fusion_family_id` и `fusion_pack_id` уже по full candidate graph.
5. Только после этого смотреть `05_grouped_sku_demo.ipynb` как витрину
   реальных групп, где family size может быть больше 3.

До этого текущие family sizes нужно читать как diagnostics по benchmark
gold-set, а не как результат дедупликации всей категории.
