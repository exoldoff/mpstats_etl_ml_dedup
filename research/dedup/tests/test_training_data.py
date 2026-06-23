from __future__ import annotations

import sqlite3

import pandas as pd

from research.dedup.training.data import (
    build_pair_text,
    component_aware_split,
    freeze_labeling_dataset,
    pair_key,
    write_freeze_outputs,
    write_split_outputs,
)


def _pair_row(left: str, right: str, label: str, *, category_run: str = "soap") -> dict[str, object]:
    return {
        "raw_record_id_a": left,
        "raw_record_id_b": right,
        "brand_a": f"Brand {left}",
        "brand_b": f"Brand {right}",
        "title_a": f"Title {left}",
        "title_b": f"Title {right}",
        "subcategory_a": "Subcat",
        "subcategory_b": "Subcat",
        "marketplace_a": "wb",
        "marketplace_b": "ozon",
        "unit_amount_a": 0.2,
        "unit_amount_b": 0.2,
        "total_amount_a": 0.2,
        "total_amount_b": 0.6,
        "multipack_count_a": 1,
        "multipack_count_b": 3,
        "category_run": category_run,
        "label": label,
    }


def _write_telegram_state(path, rows: list[tuple[int, str]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "create table row_states (row_index integer primary key, status text, final_label text, updated_at text)"
        )
        connection.executemany(
            "insert into row_states (row_index, status, final_label, updated_at) values (?, 'done', ?, '2026-06-29')",
            rows,
        )


def test_freeze_labeling_dataset_merges_sources_and_separates_conflicts(tmp_path):
    labeling_path = tmp_path / "labeling.csv"
    telegram_path = tmp_path / "labeling.csv.telegram_state.sqlite"
    old_path = tmp_path / "labeling_sauces.csv"

    pd.DataFrame(
        [
            _pair_row("A", "B", "exact_duplicate", category_run="soap"),
            _pair_row("C", "D", "", category_run="soap"),
            _pair_row("E", "F", "exact_duplicate", category_run="soap"),
            _pair_row("G", "H", "uncertain", category_run="soap"),
        ]
    ).to_csv(labeling_path, index=False)
    _write_telegram_state(
        telegram_path,
        [
            (1, "different_product"),
            (2, "different_product"),
        ],
    )
    pd.DataFrame(
        [
            _pair_row("C", "D", "different_product", category_run="sauces"),
            _pair_row("I", "J", "same_product_different_pack", category_run="sauces"),
        ]
    ).to_csv(old_path, index=False)

    result = freeze_labeling_dataset(
        labeling_path=labeling_path,
        telegram_state_path=telegram_path,
        old_sauces_path=old_path,
    )

    assert len(result.pairs) == 3
    assert len(result.conflicts) == 1
    assert len(result.excluded) == 1
    assert set(result.pairs["same_base_product"]) == {0, 1}
    assert pair_key("C", "D") in set(result.pairs["pair_key"])
    assert result.conflicts.iloc[0]["conflict_type"] == "csv_vs_telegram"
    assert result.excluded.iloc[0]["exclude_reason"] == "uncertain_label"
    assert "raw_record_id" not in result.pairs.iloc[0]["sentence_A"]
    assert "brand:" in result.pairs.iloc[0]["sentence_A"]


def test_component_aware_split_has_no_raw_id_leakage(tmp_path):
    frame = pd.DataFrame(
        [
            {**_pair_row("A", "B", "exact_duplicate"), "same_base_product": 1, "pair_key": pair_key("A", "B")},
            {**_pair_row("B", "C", "different_product"), "same_base_product": 0, "pair_key": pair_key("B", "C")},
            {**_pair_row("D", "E", "different_product"), "same_base_product": 0, "pair_key": pair_key("D", "E")},
            {**_pair_row("F", "G", "exact_duplicate"), "same_base_product": 1, "pair_key": pair_key("F", "G")},
        ]
    )

    split = component_aware_split(frame, seed=7, large_component_strategy="strict")
    raw_id_to_splits: dict[str, set[str]] = {}
    for _, row in split.pairs.iterrows():
        raw_id_to_splits.setdefault(row["raw_record_id_a"], set()).add(row["split"])
        raw_id_to_splits.setdefault(row["raw_record_id_b"], set()).add(row["split"])

    assert split.manifest["rows"] == 4
    assert set(split.pairs["split"]).issubset({"train", "dev", "test"})
    assert all(len(splits) == 1 for splits in raw_id_to_splits.values())


def test_pair_stratified_split_keeps_all_pairs_for_benchmark() -> None:
    rows = []
    for idx in range(30):
        rows.append(
            {
                **_pair_row(f"N{idx}A", f"N{idx}B", "different_product", category_run="soap"),
                "same_base_product": 0,
                "pair_key": pair_key(f"N{idx}A", f"N{idx}B"),
            }
        )
    for idx in range(12):
        rows.append(
            {
                **_pair_row(f"P{idx}A", f"P{idx}B", "exact_duplicate", category_run="soap"),
                "same_base_product": 1,
                "pair_key": pair_key(f"P{idx}A", f"P{idx}B"),
            }
        )
    frame = pd.DataFrame(rows)

    split = component_aware_split(frame, seed=7, large_component_strategy="pair_stratified")

    assert split.manifest["splitter"] == "pair_stratified"
    assert split.manifest["rows"] == len(frame)
    assert split.manifest["dropped_rows"] == 0
    assert split.dropped_pairs is not None and split.dropped_pairs.empty
    assert set(split.pairs["split"]) == {"train", "dev", "test"}
    assert split.pairs.groupby(["split", "same_base_product"]).size().unstack(fill_value=0).loc["test", 0] > 1


def test_write_outputs_create_csv_and_manifest(tmp_path):
    labeling_path = tmp_path / "labeling.csv"
    telegram_path = tmp_path / "labeling.csv.telegram_state.sqlite"
    old_path = tmp_path / "labeling_sauces.csv"
    output_dir = tmp_path / "training"
    pd.DataFrame([_pair_row("A", "B", "exact_duplicate")]).to_csv(labeling_path, index=False)
    _write_telegram_state(telegram_path, [])
    pd.DataFrame([_pair_row("C", "D", "different_product")]).to_csv(old_path, index=False)

    freeze = freeze_labeling_dataset(
        labeling_path=labeling_path,
        telegram_state_path=telegram_path,
        old_sauces_path=old_path,
    )
    freeze_paths = write_freeze_outputs(freeze, output_dir, prefix="check")
    split_paths = write_split_outputs(component_aware_split(freeze.pairs), output_dir, prefix="check")

    assert freeze_paths["pairs"].exists()
    assert freeze_paths["manifest"].exists()
    assert split_paths["split_pairs"].exists()
    assert split_paths["split_manifest"].exists()
    split_frame = pd.read_csv(split_paths["split_pairs"])
    assert "sentence_A" in split_frame.columns
    assert "sentence_B" in split_frame.columns
    assert "split" in split_frame.columns
    assert "csv_label" not in split_frame.columns
    assert "sqlite_label" not in split_frame.columns
    assert "source_priority" not in split_frame.columns


def test_build_pair_text_uses_structured_fields_without_ids():
    text = build_pair_text(_pair_row("SKU-1", "SKU-2", "exact_duplicate"), "a")
    assert "brand: Brand SKU-1" in text
    assert "title: Title SKU-1" in text
    assert "unit_weight_kg: 0.2" in text
    assert "raw_record_id" not in text
