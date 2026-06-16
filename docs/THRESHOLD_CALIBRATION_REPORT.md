# SKU Matching Threshold Calibration

Этот отчёт описывает production-критерий для benchmark SKU matching.
Актуальные числа появляются после запуска `notebooks/03_matching_comparison.ipynb`
и сохраняются в `artifacts/reports/`.

## Главный критерий auto-merge

Модель больше не выбирает один общий threshold по `macro_f1`.

Для каждого `method` на `dev` подбираются два порога:

- `threshold_auto_same`: `score >= threshold_auto_same` даёт
  `auto_same_base_product`;
- `threshold_auto_diff`: `score <= threshold_auto_diff` даёт
  `auto_different_product`;
- всё между ними уходит в `manual_review`.

`threshold_auto_same` выбирается как максимальный recall среди порогов, где:

- `auto_same_precision >= 0.97`;
- `false_merge_count <= 0` на dev.

Если таких порогов нет, notebook ставит порог выше максимального score и
пишет `passed_auto_same_constraints=false`. Это значит: метод не прошёл
safe auto-merge calibration при текущих правилах.

`threshold_auto_diff` выбирается по максимальному покрытию среди порогов, где
`auto_diff_precision >= 0.95`.

## Почему не macro-F1

`macro_f1` симметрично усредняет качество по классам. Для дедупликации SKU это
не production-критерий: ложная склейка разных товаров портит группу и может
заразить downstream-отчёты, а пропущенный дубль можно отправить в
`manual_review` или обработать позже. Поэтому главный критерий для
auto-merge — безопасность: высокая precision и ноль false merges на dev.

## Где смотреть прошедшие модели

Открой `artifacts/reports/threshold_calibration_dev.csv`.

Модели, прошедшие порог безопасности, это строки, где:

- `passed_auto_same_constraints=true`;
- `false_merge_count <= MAX_FALSE_MERGES_ON_DEV`;
- `auto_same_precision >= TARGET_AUTO_SAME_PRECISION`.

Для выбора кандидата смотри сначала `passed_auto_same_constraints`, затем
`auto_coverage`, `auto_same_recall` и `manual_review_rate`.

`artifacts/reports/threshold_evaluation_test.csv` нужен только для финальной
проверки выбранных на dev порогов. Test split нельзя использовать для
подбора threshold или выбора модели.

## Manual review

Доля ручной проверки записана в колонке `manual_review_rate`.

Практическое чтение:

- высокая `manual_review_rate` при нуле false merges обычно допустима для
  первого безопасного production-кандидата;
- низкая `manual_review_rate` с false merges опасна и не должна побеждать
  только за счёт красивого coverage;
- конкретные пары для чтения глазами лежат в
  `manual_review_pairs_dev.csv` и `manual_review_pairs_test.csv`.

## Основные артефакты

- `artifacts/reports/threshold_calibration_dev.csv`
- `artifacts/reports/threshold_evaluation_test.csv`
- `artifacts/reports/false_merges_on_dev.csv`
- `artifacts/reports/false_merges_on_test.csv`
- `artifacts/reports/false_rejects_on_dev.csv`
- `artifacts/reports/false_rejects_on_test.csv`
- `artifacts/reports/manual_review_pairs_dev.csv`
- `artifacts/reports/manual_review_pairs_test.csv`
- `artifacts/reports/threshold_confusion_matrices.csv`
- `artifacts/reports/*precision_recall*.png`
- `artifacts/reports/*threshold_false_merges*.png`
- `artifacts/reports/*threshold_manual_review*.png`
- `artifacts/reports/*score_distribution*.png`
