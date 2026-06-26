#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="$ROOT_DIR/web"
SYSTEM_PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${MPSTATS_VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
HOST="${MPSTATS_APP_HOST:-127.0.0.1}"
PORT_START="${MPSTATS_APP_PORT:-8000}"
PORT_END="${MPSTATS_APP_PORT_END:-8010}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# faiss-cpu, scikit-learn and torch wheels can ship separate libomp.dylib copies on macOS.
# Keep the local app process alive when ML dedup loads more than one of them.
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

cd "$ROOT_DIR"

log() {
  printf "\033[1;32m[MPStats]\033[0m %s\n" "$1"
}

fail() {
  printf "\033[1;31m[MPStats]\033[0m %s\n" "$1" >&2
  printf "\nНажми Enter, чтобы закрыть окно..." >&2
  read -r _ || true
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Не найдена команда '$1'."
}

ensure_venv() {
  if [[ -x "$PYTHON_BIN" ]]; then
    return 0
  fi
  need_command "$SYSTEM_PYTHON_BIN"
  log "Создаю локальное Python-окружение .venv..."
  "$SYSTEM_PYTHON_BIN" -m venv "$VENV_DIR" || fail "Не удалось создать .venv через $SYSTEM_PYTHON_BIN."
}

dependencies_missing() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

modules = [
    "duckdb",
    "faiss",
    "fastapi",
    "numpy",
    "pandas",
    "sentence_transformers",
    "torch",
    "uvicorn",
]
missing = [module for module in modules if importlib.util.find_spec(module) is None]
if missing:
    print("Не найдены Python-модули: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
}

ensure_python_dependencies() {
  local marker="$VENV_DIR/.requirements-installed"
  if dependencies_missing && [[ -f "$marker" && "$marker" -nt "$ROOT_DIR/requirements.txt" ]]; then
    return 0
  fi
  log "Устанавливаю Python-зависимости в .venv из requirements.txt..."
  "$PYTHON_BIN" -m pip install -r requirements.txt || fail "python -m pip install -r requirements.txt завершился с ошибкой."
  dependencies_missing || fail "После установки Python-зависимости всё ещё недоступны."
  touch "$marker"
}

port_is_free() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

pick_port() {
  local port
  for port in $(seq "$PORT_START" "$PORT_END"); do
    if port_is_free "$port"; then
      printf "%s" "$port"
      return 0
    fi
  done
  return 1
}

frontend_needs_build() {
  if [[ ! -f "$WEB_DIR/dist/index.html" ]]; then
    return 0
  fi
  if [[ "$WEB_DIR/package.json" -nt "$WEB_DIR/dist/index.html" ]]; then
    return 0
  fi
  if [[ -f "$WEB_DIR/package-lock.json" && "$WEB_DIR/package-lock.json" -nt "$WEB_DIR/dist/index.html" ]]; then
    return 0
  fi
  if find "$WEB_DIR/src" -type f -newer "$WEB_DIR/dist/index.html" -print -quit | grep -q .; then
    return 0
  fi
  return 1
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    log "Останавливаю backend..."
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

need_command npm
ensure_venv
log "Проверяю Python-зависимости в .venv..."
ensure_python_dependencies

if [[ ! -d "$WEB_DIR/node_modules" ]]; then
  log "Устанавливаю frontend-зависимости..."
  (cd "$WEB_DIR" && npm install) || fail "npm install завершился с ошибкой."
fi

if frontend_needs_build; then
  log "Собираю frontend..."
  (cd "$WEB_DIR" && npm run build) || fail "npm run build завершился с ошибкой."
fi

PORT="$(pick_port)" || fail "Не нашёл свободный порт в диапазоне $PORT_START-$PORT_END."
URL="http://$HOST:$PORT"

log "Запускаю backend на $URL"
"$PYTHON_BIN" -m uvicorn mpstats_app.main:app --host "$HOST" --port "$PORT" &
SERVER_PID="$!"

READY=0
for _ in $(seq 1 40); do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    fail "Backend завершился раньше времени."
  fi
  if "$PYTHON_BIN" - "$URL" <<'PY' >/dev/null 2>&1
from urllib.request import urlopen
import sys

with urlopen(sys.argv[1] + "/api/health", timeout=0.5) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
  then
    READY=1
    break
  fi
  sleep 0.25
done

if [[ "$READY" != "1" ]]; then
  fail "Backend не ответил на /api/health."
fi

if [[ "${MPSTATS_APP_NO_BROWSER:-0}" != "1" ]]; then
  log "Открываю приложение..."
  open "$URL"
fi

printf "\nMPStats Local App работает: %s\n" "$URL"
printf "Закрой это окно или нажми Ctrl+C, чтобы остановить backend.\n\n"

wait "$SERVER_PID"
