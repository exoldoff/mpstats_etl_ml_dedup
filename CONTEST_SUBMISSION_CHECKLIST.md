# Contest Submission Checklist

* [ ] README актуален
* [ ] Docker собирается
* [ ] Проект запускается с нуля
* [ ] Ноутбуки открываются
* [ ] Данные не содержат приватной информации
* [ ] Секреты не попали в репозиторий
* [ ] Основные результаты ML описаны
* [ ] Есть инструкция для жюри
* [ ] Есть список известных ограничений

## Перед отправкой

* [ ] Отправляется clean git archive или clean clone, не вся рабочая папка с `data/`, `artifacts/`, `.env`, `.venv`, `mpstats.duckdb` и `research/dedup/models/`.
* [ ] Решено, включать ли tracked `outputs/mpstats_fmcg_etl_pitch.pptx`.
* [ ] Проверено состояние `notebooks/00_eda.ipynb`: нет случайных локальных outputs и приватных путей.
* [ ] При необходимости подготовлен sanitized sample справочника категорий и правил классификатора.
* [ ] Выполнены `docker compose build` и `docker compose up` на машине с Docker daemon.
