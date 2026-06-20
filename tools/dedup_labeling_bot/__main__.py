from __future__ import annotations

import argparse
import logging
from pathlib import Path

from telegram import Update

from .config import ConfigError, load_config
from .handlers import create_application
from .service import LabelingBotService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram bot for SKU dedup labeling CSV files.")
    parser.add_argument("csv_path", type=Path, help="Path to labeling_*.csv")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    args = parse_args()
    try:
        config = load_config(args.csv_path)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    service = LabelingBotService(config)
    application = create_application(service)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
