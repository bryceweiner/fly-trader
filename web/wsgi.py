"""WSGI entry point required by Gandi Simple Hosting (Python instance).

Serves the static site in ./site/ as-is, including HTTP Range support so
browsers can seek/stream the hero videos.
"""
import mimetypes
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
CHUNK = 256 * 1024


def _resolve(path_info):
    path = path_info.split("?", 1)[0]
    if path in ("", "/"):
        path = "/index.html"
    rel = os.path.normpath(path).lstrip("/")
    full = os.path.abspath(os.path.join(ROOT, rel))
    if full != ROOT and not full.startswith(ROOT + os.sep):
        return None
    return full


class _BoundedFile:
    def __init__(self, f, length):
        self.f = f
        self.remaining = length

    def read(self, size=-1):
        if self.remaining <= 0:
            return b""
        n = CHUNK if size is None or size < 0 else min(size, CHUNK)
        n = min(n, self.remaining)
        data = self.f.read(n)
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


def application(environ, start_response):
    full = _resolve(environ.get("PATH_INFO", "/"))
    if not full or not os.path.isfile(full):
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not Found"]

    size = os.path.getsize(full)
    content_type = mimetypes.guess_type(full)[0] or "application/octet-stream"
    wrapper = environ.get("wsgi.file_wrapper")
    match = RANGE_RE.match(environ.get("HTTP_RANGE", ""))

    if match:
        start = int(match.group(1)) if match.group(1) else 0
        end = int(match.group(2)) if match.group(2) else size - 1
        end = min(end, size - 1)
        length = max(end - start + 1, 0)
        f = open(full, "rb")
        f.seek(start)
        start_response(
            "206 Partial Content",
            [
                ("Content-Type", content_type),
                ("Content-Length", str(length)),
                ("Content-Range", "bytes %d-%d/%d" % (start, end, size)),
                ("Accept-Ranges", "bytes"),
            ],
        )
        bounded = _BoundedFile(f, length)
        return wrapper(bounded) if wrapper else iter(lambda: bounded.read(CHUNK), b"")

    f = open(full, "rb")
    start_response(
        "200 OK",
        [
            ("Content-Type", content_type),
            ("Content-Length", str(size)),
            ("Accept-Ranges", "bytes"),
        ],
    )
    return wrapper(f) if wrapper else iter(lambda: f.read(CHUNK), b"")
