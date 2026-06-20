from __future__ import annotations

import argparse
import curses
from pathlib import Path

import pandas as pd

from .category_runs import resolve_category_run, resolve_run_paths

try:
    from wcwidth import wcwidth, wcswidth
except ImportError:  # pragma: no cover - wcwidth is present in the notebook env
    wcwidth = None
    wcswidth = None


def default_labeling_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return resolve_run_paths(project_root, resolve_category_run()).labeling_path


DEFAULT_LABELING_PATH = default_labeling_path()

LABEL_BY_KEY = {
    "w": "exact_duplicate",
    "s": "different_product",
    "d": "uncertain",
}

LABEL_HINTS = {
    "w": "same base product",
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


def _yes_no(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return "да"
    if normalized in {"false", "0", "no", "n"}:
        return "нет"
    return value or "-"


def _item_summary(row: pd.Series, side: str) -> str:
    marketplace = _first_cell(row, f"marketplace_{side}")
    sku = _first_cell(row, f"sku_{side}")
    brand = _first_cell(row, f"brand_{side}")
    subcategory = _first_cell(row, f"subcategory_{side}")
    unit = _first_cell(row, f"unit_amount_{side}")
    total = _first_cell(row, f"total_amount_{side}")
    pack = _first_cell(row, f"multipack_count_{side}")
    return f"{marketplace} | sku {sku} | brand {brand} | subcat {subcategory} | unit {unit} | total {total} | x{pack}"


def _signal_summary(row: pd.Series) -> str:
    source = _first_cell(row, "candidate_source")
    rank = _first_cell(row, "candidate_rank")
    score = _first_cell(row, "embedding_similarity_score", "baseline_similarity_score")
    blocking_scope = _first_cell(row, "blocking_scope")
    subcategory = _first_cell(row, "subcategory_relation")
    stratum = _first_cell(row, "labeling_stratum")
    cross = _yes_no(_cell(row, "is_cross_marketplace_pair"))
    hard_negative = _yes_no(_cell(row, "is_hard_negative_candidate"))
    pack_variant = _yes_no(_cell(row, "is_pack_variant_candidate"))
    return (
        f"источник={source} | scope={blocking_scope} | subcat={subcategory} | "
        f"rank={rank} | score={score} | стратегия={stratum} | "
        f"межмаркетплейс={cross} | сложный негатив={hard_negative} | вариант упаковки={pack_variant}"
    )


def _display_width(text: str) -> int:
    if wcswidth is not None:
        width = wcswidth(text)
        if width >= 0:
            return width
    total = 0
    for char in text:
        if wcwidth is None:
            total += 1
            continue
        char_width = wcwidth(char)
        total += max(0, char_width)
    return total


def _fit_display_width(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    result: list[str] = []
    used = 0
    for char in text:
        char_width = _display_width(char)
        if used + char_width > max_width:
            break
        result.append(char)
        used += char_width
    return "".join(result)


def _wrap_plain_display(text: str, max_width: int) -> list[str]:
    if not text:
        return [""]
    width = max(8, max_width)
    words = str(text).split()
    lines: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current:
            lines.append(current)
            current = ""

    for word in words:
        while _display_width(word) > width:
            flush_current()
            chunk = _fit_display_width(word, width)
            lines.append(chunk)
            word = word[len(chunk) :]
        candidate = word if not current else f"{current} {word}"
        if _display_width(candidate) <= width:
            current = candidate
        else:
            flush_current()
            current = word

    flush_current()
    return lines or [""]


def _wrap_display(prefix: str, value: str, max_width: int) -> list[str]:
    width = max(12, max_width)
    prefix_width = _display_width(prefix)
    body_width = max(8, width - prefix_width)
    body_lines = _wrap_plain_display(value, body_width)
    continuation_prefix = " " * len(prefix)
    lines = [f"{prefix}{body_lines[0]}"]
    lines.extend(f"{continuation_prefix}{line}" for line in body_lines[1:])
    return lines


def _add_line(screen: curses.window, y: int, text: str, attr: int = 0) -> int:
    height, width = screen.getmaxyx()
    if y >= height - 1:
        return y
    screen.move(y, 0)
    screen.clrtoeol()
    line = _fit_display_width(text, max(0, width - 1))
    if line:
        screen.addstr(y, 0, line, attr)
    return y + 1


def _add_wrapped(screen: curses.window, y: int, prefix: str, value: str, attr: int = 0) -> int:
    _, width = screen.getmaxyx()
    wrapped = _wrap_display(prefix, value, max(20, width - 1))
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

    y = 0
    y = _add_line(
        screen,
        y,
        f"SKU dedup annotator | row {row_index + 1}/{len(frame)} | labeled {done}/{len(frame)} | label {current_label}",
        title_attr,
    )
    y = _add_line(screen, y, f"CSV: {csv_path}")
    y = _add_line(screen, y, "-" * (width - 1))

    y = _add_wrapped(screen, y, "A  ", _cell(row, "title_a"), title_attr)
    y = _add_wrapped(screen, y, "   ", _item_summary(row, "a"))
    y = _add_line(screen, y, "")
    y = _add_wrapped(screen, y, "B  ", _cell(row, "title_b"), title_attr)
    y = _add_wrapped(screen, y, "   ", _item_summary(row, "b"))
    y = _add_line(screen, y, "")

    y = _add_wrapped(screen, y, "Сигналы: ", _signal_summary(row))
    notes = _cell(row, "notes")
    if notes:
        y = _add_wrapped(screen, y, "Заметка: ", notes)

    while y < height - 5:
        y = _add_line(screen, y, "")

    y = _add_line(screen, y, "Keys: w exact/same-product | s different | d uncertain | c clear")
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
        message = "Ready. Press w/s/d to label this pair."

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
        message = "Unknown key. Use w/s/d, arrows, c, g, q."


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
