#!/usr/bin/env python3
"""Serve remote evaluation files with bounded stalled-client connections."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class EvaluationHandler(SimpleHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(15)
        super().setup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18879)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    if not directory.is_dir():
        parser.error("evaluation directory must exist")
    handler = partial(EvaluationHandler, directory=str(directory))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        server.daemon_threads = True
        server.serve_forever()


if __name__ == "__main__":
    main()
