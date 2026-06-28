#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/app"
RUNTIME_DIR="${MPSTATS_RUNTIME_DIR:-$ROOT_DIR/runtime}"
RULES_PATH="${MPSTATS_RULES_PATH:-$RUNTIME_DIR/classifiers/rules.csv}"
MANUAL_OVERRIDES_PATH="${MPSTATS_MANUAL_OVERRIDES_PATH:-$RUNTIME_DIR/classifiers/manual_overrides.csv}"
CATEGORY_CATALOG_PATH="${MPSTATS_CATEGORY_CATALOG_PATH:-$RUNTIME_DIR/catalog/category_catalog.csv}"
CONFIG_PATH="${MPSTATS_CONFIG_PATH:-$RUNTIME_DIR/pipeline/step1_export_config.json}"
DUCKDB_TEMP="${DUCKDB_TEMP_DIRECTORY:-$RUNTIME_DIR/duckdb_tmp}"
HOST="${MPSTATS_APP_HOST:-0.0.0.0}"
PORT="${MPSTATS_APP_PORT:-8055}"

mkdir -p \
  "$RUNTIME_DIR" \
  "$(dirname "$RULES_PATH")" \
  "$(dirname "$MANUAL_OVERRIDES_PATH")" \
  "$(dirname "$CATEGORY_CATALOG_PATH")" \
  "$(dirname "$CONFIG_PATH")" \
  "$DUCKDB_TEMP" \
  "${HF_HOME:-$RUNTIME_DIR/huggingface}"

if [[ ! -f "$RULES_PATH" ]]; then
  if [[ -f "$ROOT_DIR/classifiers/rules.csv" ]]; then
    cp "$ROOT_DIR/classifiers/rules.csv" "$RULES_PATH"
  else
    printf "active;priority;category;target_column;match_field;match_type;pattern;set_value;mode;comment;conditions_json\n" > "$RULES_PATH"
  fi
fi

if [[ ! -f "$MANUAL_OVERRIDES_PATH" ]]; then
  printf "active;priority;match_field;match_value;target_column;set_value;mode;comment\n" > "$MANUAL_OVERRIDES_PATH"
fi

if [[ ! -f "$CATEGORY_CATALOG_PATH" ]]; then
  source_catalog="$(find "$ROOT_DIR" -maxdepth 1 -type f -name 'Справочник категори*MP STATS.csv' | head -n 1 || true)"
  if [[ -n "$source_catalog" ]]; then
    cp "$source_catalog" "$CATEGORY_CATALOG_PATH"
  fi
fi

if [[ "$#" -eq 0 ]]; then
  set -- python -m uvicorn mpstats_app.main:app --host "$HOST" --port "$PORT"
fi

exec "$@"
