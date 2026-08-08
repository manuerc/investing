"""Serve docs/index.html on localhost so anyone in the Discord can read the model.

    python3 serve.py            -> http://localhost:8000
    python3 serve.py 8080       -> http://localhost:8080
"""

import http.server
import socketserver
import sys
import webbrowser
from functools import partial
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "docs"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"[SERVE] {fmt % args}")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = partial(Handler, directory=str(DOCS))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"[SERVE] documentación del modelo en {url}  (Ctrl+C para cortar)")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[SERVE] listo")


if __name__ == "__main__":
    main()
