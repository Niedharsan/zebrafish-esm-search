import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LauncherReadinessTests(unittest.TestCase):
    def test_readiness_uses_successful_status_json_not_homepage_wording(self):
        launcher = (ROOT / "start_dashboard.command").read_text()
        # Source only the real function definitions, without launching the app.
        functions = launcher[launcher.index("dashboard_ready() {"):launcher.index("port_busy() {")]
        responses = [
            (200, '{"ok": true}', True),
            (200, '{\n  "ok" : true, "title": "Any future dashboard title"\n}', True),
            (200, '{"ok": false}', False),
            (200, '{"ok": "true"}', False),
            (200, '{"ok": 1}', False),
            (200, '{}', False),
            (200, '[]', False),
            (200, 'not JSON', False),
            (200, '<title>Zebrafish ESM Similarity</title>', False),
            (503, '{"ok": true}', False),
        ]
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                self.send_response(self.server.response_status)
                self.end_headers()
                self.wfile.write(self.server.response_body.encode())

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory)
            (venv / "bin").mkdir()
            (venv / "bin/python").symlink_to(sys.executable)
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for status, body, expected in responses:
                    with self.subTest(status=status, body=body):
                        server.response_status = status
                        server.response_body = body
                        result = subprocess.run(
                            ["bash", "-c", functions + '\nHOST=127.0.0.1\nVENV_DIR="$1"\ndashboard_ready "$2"',
                             "launcher-test", str(venv), str(server.server_port)],
                            capture_output=True, text=True, timeout=5,
                        )
                        self.assertEqual(result.returncode == 0, expected, result.stderr)
                self.assertEqual(requests, ["/api/status"] * len(responses))
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
