#!/bin/bash

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
DB_PATH="data/zebrafish_esm.db"
HOST="127.0.0.1"
PRIMARY_PORT="8000"
FALLBACK_PORT="3000"
SERVER_PID=""

print_error() {
  echo
  echo "ERROR: $1"
  echo
  echo "Close this window when you are done reading this message."
}

dashboard_ready() {
  local port="$1"
  local status
  status=$(/usr/bin/curl -fsS --max-time 1 "http://${HOST}:${port}/api/status" 2>/dev/null) || return 1
  printf '%s' "$status" | "$VENV_DIR/bin/python" -c '
import json, sys
try:
    status = json.load(sys.stdin)
    sys.exit(0 if isinstance(status, dict) and status.get("ok") is True else 1)
except (ValueError, TypeError):
    sys.exit(1)
'
}

port_busy() {
  local port="$1"
  /usr/sbin/lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo
    echo "Stopping Zebrafish ESM Dashboard..."
    kill "$SERVER_PID" >/dev/null 2>&1
    wait "$SERVER_PID" >/dev/null 2>&1
  fi
}
trap cleanup EXIT INT TERM

clear
echo "Starting Zebrafish ESM Dashboard..."
echo

cd "$PROJECT_DIR" || {
  print_error "Could not open project folder: $PROJECT_DIR"
  read -r -p "Press Return to close this window. "
  exit 1
}

if [ ! -d "$VENV_DIR" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
  print_error "The virtual environment is missing or incomplete: $VENV_DIR"
  echo "Create it first, then install requirements:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  read -r -p "Press Return to close this window. "
  exit 1
fi

if [ ! -f "$DB_PATH" ]; then
  print_error "The SQLite database is missing: $PROJECT_DIR/$DB_PATH"
  echo "Build the database first with build_database.py using your existing embeddings and metadata."
  echo "This launcher will not rebuild the database automatically."
  read -r -p "Press Return to close this window. "
  exit 1
fi

PORT="$PRIMARY_PORT"
if port_busy "$PRIMARY_PORT"; then
  if dashboard_ready "$PRIMARY_PORT"; then
    URL="http://${HOST}:${PRIMARY_PORT}"
    echo "The dashboard is already running on $URL"
    echo "Opening it in your browser..."
    /usr/bin/open "$URL"
    echo
    echo "This launcher did not start a new server."
    echo "Close the original dashboard launcher window to stop the dashboard."
    read -r -p "Press Return to close this window. "
    exit 0
  fi

  echo "Port $PRIMARY_PORT is already busy, so I will try port $FALLBACK_PORT instead."
  PORT="$FALLBACK_PORT"
fi

if port_busy "$PORT"; then
  if dashboard_ready "$PORT"; then
    URL="http://${HOST}:${PORT}"
    echo "The dashboard is already running on $URL"
    echo "Opening it in your browser..."
    /usr/bin/open "$URL"
    echo
    echo "Close the original dashboard launcher window to stop it."
    read -r -p "Press Return to close this window. "
    exit 0
  fi

  print_error "Port $PORT is also busy, and it does not look like the dashboard."
  echo "Stop the app using port $PORT, then run this launcher again."
  read -r -p "Press Return to close this window. "
  exit 1
fi

source "$VENV_DIR/bin/activate"

URL="http://${HOST}:${PORT}"
echo "Using database: $PROJECT_DIR/$DB_PATH"
echo "Starting local server at $URL"
echo
echo "When you are finished, close or quit this launcher window to stop the dashboard."
echo

PYTHONUNBUFFERED=1 "$VENV_DIR/bin/python" app.py --db "$DB_PATH" --host "$HOST" --port "$PORT" &
SERVER_PID="$!"

echo "Waiting for the dashboard to become ready..."
for _ in $(seq 1 120); do
  if dashboard_ready "$PORT"; then
    echo "Dashboard is ready. Opening browser..."
    /usr/bin/open "$URL"
    echo
    echo "Dashboard is running at $URL"
    echo "Close or quit this launcher window to stop it."
    wait "$SERVER_PID"
    exit $?
  fi

  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    print_error "The dashboard server stopped before it became ready."
    read -r -p "Press Return to close this window. "
    exit 1
  fi

  sleep 0.5
done

print_error "The dashboard did not become ready within 60 seconds."
echo "The server process is still running; check the messages above for details."
read -r -p "Press Return to stop the server and close this window. "
exit 1
