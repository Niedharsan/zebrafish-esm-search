# Mac Launcher

This folder includes a local macOS launcher for the Zebrafish ESM Dashboard. It uses the existing virtual environment in `.venv/` and the existing SQLite database at `data/zebrafish_esm.db`.

It does not rebuild the database and does not move or modify the original ESM embedding chunks.

## Run

Double-click either:

```text
start_dashboard.command
```

or:

```text
Zebrafish ESM Dashboard.app
```

The launcher will:

- open the project folder
- activate `.venv/`
- run `python app.py --db data/zebrafish_esm.db --host 127.0.0.1 --port 8000`
- open `http://127.0.0.1:8000` in your browser
- use `http://127.0.0.1:3000` if port `8000` is busy with something other than this dashboard

The dashboard only binds to `127.0.0.1`, so it stays local to your Mac.

## Stop

Close or quit the Terminal window opened by the launcher. That stops the local dashboard server.

If the browser tab stays open afterward, refresh will fail until you start the launcher again.

## If macOS blocks it

If macOS says the command or app cannot be opened:

1. Open **System Settings**.
2. Go to **Privacy & Security**.
3. Scroll to the security message for this launcher and choose **Open Anyway**.

You can also Control-click the launcher, choose **Open**, then confirm.

If macOS says the command file is not executable, run this once from Terminal:

```bash
chmod +x start_dashboard.command
chmod +x "Zebrafish ESM Dashboard.app/Contents/MacOS/Zebrafish ESM Dashboard"
```

## Rebuild the database

Only rebuild `data/zebrafish_esm.db` if the embeddings or metadata change.

Use `build_database.py` with the same option that matches your source data:

```bash
source .venv/bin/activate
python build_database.py --help
```

The launcher will never rebuild the database automatically.
