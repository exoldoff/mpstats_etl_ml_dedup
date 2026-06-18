from __future__ import annotations

import os

from research.dedup.env import load_research_dotenv, parse_dotenv


def test_parse_dotenv_handles_quotes_exports_and_comments() -> None:
    values = parse_dotenv(
        """
        # ignored
        DEDUP_CATEGORY_RUN=soap
        export DEDUP_FAISS_TOP_K=30
        POLZA_BASE_URL="https://example.test/api/v1" # comment
        MPSTATS_DUCKDB_PATH="/tmp/мыло/mpstats.duckdb"
        LITERAL_HASH='value # kept'
        BAD KEY=ignored
        """
    )

    assert values == {
        "DEDUP_CATEGORY_RUN": "soap",
        "DEDUP_FAISS_TOP_K": "30",
        "POLZA_BASE_URL": "https://example.test/api/v1",
        "MPSTATS_DUCKDB_PATH": "/tmp/мыло/mpstats.duckdb",
        "LITERAL_HASH": "value # kept",
    }


def test_load_research_dotenv_does_not_override_shell_values(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("DEDUP_CATEGORY_RUN=soap\nDEDUP_FAISS_TOP_K=30\n", encoding="utf-8")
    monkeypatch.setenv("DEDUP_CATEGORY_RUN", "sauces")
    monkeypatch.delenv("DEDUP_FAISS_TOP_K", raising=False)

    loaded = load_research_dotenv(env_path=env_path)

    assert loaded == env_path
    assert os.environ["DEDUP_CATEGORY_RUN"] == "sauces"
    assert os.environ["DEDUP_FAISS_TOP_K"] == "30"


def test_load_research_dotenv_can_override_when_requested(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("DEDUP_CATEGORY_RUN=coconut_oil\n", encoding="utf-8")
    monkeypatch.setenv("DEDUP_CATEGORY_RUN", "sauces")

    load_research_dotenv(env_path=env_path, override=True)

    assert os.environ["DEDUP_CATEGORY_RUN"] == "coconut_oil"
