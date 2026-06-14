from __future__ import annotations

import argparse
import curses
from pathlib import Path
import textwrap

import pandas as pd


DEFAULT_LABELING_PATH = Path(__file__).resolve().parent / "data" / "labeling_sauces.csv"

LABEL_BY_KEY = {
    "w": "exact_duplicate",
    "a": "same_product_different_pack",
    "s": "different_product",
    "d": "uncertain",
}

LABEL_HINTS = {
    "w": "same SKU and same pack",
    "a": "same product, different pack",
    "s": "different product",
    "d": "uncertain",
}


def normalize_label_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except TypeError:
        return ""
    return str(value).strip()


def load_labeling_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Labeling CSV not found: {path}")
    frame = pd.read_csv(path)
    for column in ("label", "notes"):
        if column not in frame.columns:
            frame.insert(0 if column == "label" else 1, column, "")
    frame["label"] = frame["label"].map(normalize_label_value)
    frame["notes"] = frame["notes"].map(normalize_label_value)
    return frame


def save_labeling_file(frame: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def is_labeled(value: object) -> bool:
    return normalize_label_value(value) != ""


def labeled_count(frame: pd.DataFrame) -> int:
    return int(frame["label"].map(is_labeled).sum())


def next_unlabeled_index(frame: pd.DataFrame, start: int = 0) -> int | None:
    if frame.empty:
        return None
    start = max(0, min(start, len(frame) - 1))
    for idx in range(start, len(frame)):
        if not is_labeled(frame.at[idx, "label"]):
            return idx
    for idx in range(0, start):
        if not is_labeled(frame.at[idx, "label"]):
            return idx
    return None


def apply_label(frame: pd.DataFrame, row_index: int, key: str) -> str:
    label = LABEL_BY_KEY[key.lower()]
    frame.at[row_index, "label"] = label
    return label


def clear_label(frame: pd.DataFrame, row_index: int) -> None:
    frame.at[row_index, "label"] = ""


def _cell(row: pd.Series, column: str) -> str:
    return normalize_label_value(row.get(column, ""))


def _first_cell(row: pd.Series, *columns: str) -> str:
    for column in columns:
        value = _cell(row, column)
        if value:
            return value
    return "-"


def _item_summary(row: pd.Series, side: str) -> tuple[str, str]:
    marketplace = _first_cell(row, f"marketplace_{side}")
    sku = _first_cell(row, f"sku_{side}")
    brand = _first_cell(row, f"brand_{side}")
    unit = _first_cell(row, f"unit_amount_{side}")
    total = _first_cell(row, f"total_amount_{side}")
    pack = _first_cell(row, f"multipack_count_{side}")
    identity = f"sku={sku} | marketplace={marketplace} | brand={brand}"
    pack_info = f"unit={unit} | total={total} | multipack={pack}"
    return identity, pack_info


def _add_line(screen: curses.window, y: int, text: str, attr: int = 0) -> int:
    height, width = screen.getmaxyx()
    if y >= height - 1:
        return y
    screen.addnstr(y, 0, text.ljust(max(0, width - 1)), max(0, width - 1), attr)
    return y + 1


def _add_wrapped(screen: curses.window, y: int, prefix: str, value: str, attr: int = 0) -> int:
    _, width = screen.getmaxyx()
    text = f"{prefix}{value}"
    wrapped = textwrap.wrap(text, width=max(20, width - 1), replace_whitespace=False) or [""]
    for line in wrapped:
        y = _add_line(screen, y, line, attr)
    return y


def _render(screen: curses.window, frame: pd.DataFrame, row_index: int, csv_path: Path, message: str) -> None:
    screen.erase()
    height, width = screen.getmaxyx()
    if height < 20 or width < 70:
        _add_line(screen, 0, "Terminal is too small. Resize to at least 70x20.")
        screen.refresh()
        return

    row = frame.iloc[row_index]
    done = labeled_count(frame)
    current_label = _cell(row, "label") or "<empty>"
    title_attr = curses.A_BOLD
    section_attr = curses.A_BOLD

    y = 0
    y = _add_line(
        screen,
        y,
        f"SKU dedup annotator | row {row_index + 1}/{len(frame)} | labeled {done}/{len(frame)} | label {current_label}",
        title_attr,
    )
    y = _add_line(screen, y, f"CSV: {csv_path}")
    y = _add_line(screen, y, "-" * (width - 1))

    y = _add_line(screen, y, "PAIR", section_attr)
    y = _add_wrapped(screen, y, "  A title: ", _cell(row, "title_a"), title_attr)
    y = _add_wrapped(screen, y, "  B title: ", _cell(row, "title_b"), title_attr)
    y = _add_line(screen, y, "")

    a_identity, a_pack = _item_summary(row, "a")
    b_identity, b_pack = _item_summary(row, "b")
    y = _add_line(screen, y, "ITEMS", section_attr)
    y = _add_wrapped(screen, y, "  A: ", a_identity)
    y = _add_wrapped(screen, y, "     ", a_pack)
    y = _add_wrapped(screen, y, "  B: ", b_identity)
    y = _add_wrapped(screen, y, "     ", b_pack)
    y = _add_line(screen, y, "")

    y = _add_line(screen, y, "RETRIEVAL", section_attr)
    y = _add_wrapped(
        screen,
        y,
        "  ",
        (
            f"source={_first_cell(row, 'candidate_source')} | "
            f"rank={_first_cell(row, 'candidate_rank')} | "
            f"embedding_score={_first_cell(row, 'embedding_similarity_score', 'baseline_similarity_score')} | "
            f"baseline_score={_first_cell(row, 'baseline_similarity_score')}"
        ),
    )
    y = _add_line(screen, y, "FLAGS", section_attr)
    y = _add_wrapped(
        screen,
        y,
        "  ",
        (
            f"stratum={_cell(row, 'labeling_stratum')} | "
            f"cross_marketplace={_cell(row, 'is_cross_marketplace_pair')} | "
            f"hard_negative={_cell(row, 'is_hard_negative_candidate')} | "
            f"pack_variant={_cell(row, 'is_pack_variant_candidate')}"
        ),
    )
    notes = _cell(row, "notes")
    if notes:
        y = _add_wrapped(screen, y, "Notes: ", notes)

    while y < height - 5:
        y = _add_line(screen, y, "")

    y = _add_line(screen, y, "Keys: w exact | a same-different-pack | s different | d uncertain | c clear")
    y = _add_line(screen, y, "Nav: Space/Right next | Left previous | g next empty | q quit")
    _add_line(screen, y, message[: max(0, width - 1)])
    screen.refresh()


def _run_curses(screen: curses.window, frame: pd.DataFrame, csv_path: Path) -> None:
    curses.curs_set(0)
    screen.keypad(True)
    try:
        curses.use_default_colors()
    except curses.error:
        pass

    row_index = next_unlabeled_index(frame, 0)
    if row_index is None:
        row_index = 0
        message = "All rows already have labels. Use arrows to review or c to clear."
    else:
        message = "Ready. Press w/a/s/d to label this pair."

    while True:
        _render(screen, frame, row_index, csv_path, message)
        key_code = screen.getch()
        if key_code == -1:
            continue
        try:
            key = chr(key_code).lower()
        except ValueError:
            key = ""

        if key == "q":
            save_labeling_file(frame, csv_path)
            return
        if key in LABEL_BY_KEY:
            label = apply_label(frame, row_index, key)
            save_labeling_file(frame, csv_path)
            next_index = next_unlabeled_index(frame, row_index + 1)
            if next_index is None:
                message = f"Saved {label}. All rows are labeled."
            else:
                row_index = next_index
                message = f"Saved {label}. Moved to next empty row."
            continue
        if key == "c":
            clear_label(frame, row_index)
            save_labeling_file(frame, csv_path)
            message = "Cleared current label."
            continue
        if key in {" ", "n"} or key_code == curses.KEY_RIGHT:
            row_index = min(len(frame) - 1, row_index + 1)
            message = "Moved next."
            continue
        if key in {"p", "b"} or key_code == curses.KEY_LEFT:
            row_index = max(0, row_index - 1)
            message = "Moved previous."
            continue
        if key == "g":
            next_index = next_unlabeled_index(frame, row_index + 1)
            if next_index is None:
                message = "No empty labels left."
            else:
                row_index = next_index
                message = "Moved to next empty row."
            continue
        message = "Unknown key. Use w/a/s/d, arrows, c, g, q."


def run(csv_path: Path) -> None:
    frame = load_labeling_file(csv_path)
    if frame.empty:
        raise ValueError(f"Labeling CSV is empty: {csv_path}")
    curses.wrapper(_run_curses, frame, csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Row-by-row WASD annotator for dedup labeling CSV.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LABELING_PATH,
        help=f"Path to labeling CSV. Default: {DEFAULT_LABELING_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.csv_path.expanduser().resolve())


if __name__ == "__main__":
    main()
