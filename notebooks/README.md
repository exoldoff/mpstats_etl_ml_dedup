# Research notebooks

Эта папка — конкурсный research-рассказ по SKU-дедупликации. Ноутбуки лучше
читать и запускать по порядку: каждый следующий шаг использует результат
предыдущего.

| Notebook | Роль |
| --- | --- |
| `00_eda.ipynb` | Входной EDA категории: качество данных, brand, weight/pack, title-токены. |
| `01_candidate_generation.ipynb` | Пары-кандидаты: embeddings, FAISS top-k, supplemental pairs. |
| `02_labeling_dataset.ipynb` | CSV для ручной разметки: balanced strata, сохранение уже заполненных labels. |
| `03_matching_comparison.ipynb` | Сравнение matching-моделей: score sources, dev threshold, test readout. |
| `04_fusion_pack_grouping.ipynb` | Превращение pairwise решений в `family` и `pack` graph groups. |
| `05_grouped_sku_demo.ipynb` | Наглядное demo: DuckDB slice -> candidates -> FAISS -> cross-encoder -> дерево SKU. |

## Маршрут для жюри

| Шаг | Что доказывает | Что смотреть |
| --- | --- | --- |
| `00` | Данные пригодны для SKU-дедупа: есть brand, вес, фасовка, title-сигналы. | Таблицы качества данных, распределения веса/pack, примеры сложных названий. |
| `01` | Полный перебор пар заменён retrieval-этапом: embeddings + FAISS дают короткий список кандидатов. | Recall@k, источники candidate pairs, cross-marketplace пары. |
| `02` | Разметка не случайная, а сбалансированная по сложным случаям. | Страты, сохранение старых labels, CSV для ручной проверки. |
| `03` | ML-вклад измеряется отдельно от правил: сравниваются baseline, bi-encoder и reranker. | Precision/recall/F1, false merges, false splits, dev/test threshold. |
| `04` | Pairwise-решения превращаются в стабильные группы товара и фасовки. | Graph metrics, ошибки склейки/разделения, 2D/3D diagnostics. |
| `05` | Выбранная логика переносится в понятный end-to-end сценарий на данных куба. | Дерево canonical SKU -> входящие SKU и итоговые агрегаты. |

## Как запускать

1. Установите общий список зависимостей:

```bash
python3 -m pip install -r requirements.txt
```

`requirements-research.txt` оставлен как compatibility alias и сейчас просто
указывает на общий `requirements.txt`.

2. Укажите путь к локальному DuckDB в первой code-ячейке нужного notebook
   через `MY_DUCKDB_PATH` или через `MPSTATS_DUCKDB_PATH`.
3. Меняйте только `MY_*` настройки в первой code-ячейке. Секреты и model
   cache остаются в `.env`.
4. Запускайте notebook сверху вниз.

Outputs в git не хранятся специально: это делает diff читаемым и не тащит в
репозиторий локальные данные.

## Где смотреть результаты

- Candidate/labeling CSV: `research/dedup/data/` или
  `research/dedup/data/<category>/`.
- Benchmark reports: `artifacts/reports/` или
  `artifacts/reports/fine_tuning/`.
- Fusion/demo exports: reports-папка соответствующего category-run.

Эти папки обычно ignored. Для проверки результата смотрите текущие локальные
CSV/DB artifacts, а не только сохранённый notebook.
