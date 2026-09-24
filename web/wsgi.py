"""WSGI entry point required by Gandi Simple Hosting (Python instance).

/api/* goes to the vault relay (relay/app.py, imported lazily so a relay failure never takes the site down).
Everything else is the static site in ./site/, with single-range support so browsers can seek the hero videos,
conditional GETs, and security headers on every response. Optional ./site-headers.json
({"html": {...}, "all": {...}}) adds or overrides headers; its "html" Content-Security-Policy replaces the default.
"""
import email.utils
import json
import mimetypes
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "site")
HEADERS_PATH = os.path.join(HERE, "site-headers.json")
CHUNK = 256 * 1024
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
HEADER_NAME_RE = re.compile(r"[A-Za-z0-9!#$%&'*+.^_`|~-]+")

SECURITY_HEADERS = [
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "strict-origin-when-cross-origin"),
    ("Strict-Transport-Security", "max-age=31536000"),
    ("X-Frame-Options", "DENY"),
]
DEFAULT_CSP = ("default-src 'self'; img-src 'self' data: https:; style-src 'self' 'unsafe-inline' "
               "https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; connect-src 'self'; "
               "frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
# Host mime tables differ (Python 3.8 lacks webp/woff2); with nosniff a wrong script/style type breaks the page.
TYPES = {
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json", ".map": "application/json", ".webmanifest": "application/manifest+json",
    ".txt": "text/plain; charset=utf-8", ".xml": "application/xml", ".svg": "image/svg+xml",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".avif": "image/avif", ".ico": "image/x-icon",
    ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf", ".otf": "font/otf",
    ".mp4": "video/mp4", ".webm": "video/webm", ".wasm": "application/wasm", ".pdf": "application/pdf",
}


def application(environ, start_response):
    start_response = _secured(start_response)
    path = environ.get("PATH_INFO") or "/"
    if path == "/api" or path.startswith("/api/"):
        return _api(environ, start_response)
    return _static(environ, start_response)


# ---- /api ----
_relay = {"app": None, "failed_at": None}


def _api(environ, start_response):
    if _relay["app"] is None:
        failed = _relay["failed_at"]
        if failed is None or time.time() - failed > 60:
            try:
                if HERE not in sys.path:
                    sys.path.insert(0, HERE)
                from relay import app as relay_app
                _relay["app"], _relay["failed_at"] = relay_app, None
            except Exception:
                _relay["failed_at"] = time.time()
                sys.stderr.write("[wsgi] relay import failed; static site still served\n")
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
        if _relay["app"] is None:
            body = b'{"error":"relay unavailable"}'
            start_response("503 Service Unavailable", [("Content-Type", "application/json"),
                                                       ("Cache-Control", "no-store"),
                                                       ("Content-Length", str(len(body)))])
            return [body]
    return _relay["app"].application(environ, start_response)


# ---- headers ----
_site_cache = {"key": None, "value": ([], [])}


def _site_headers():
    """(html, all) header lists from site-headers.json, re-read when the file changes."""
    try:
        st = os.stat(HEADERS_PATH)
    except OSError:
        return [], []
    key = (HEADERS_PATH, st.st_mtime_ns, st.st_size)
    if _site_cache["key"] != key:
        value = ([], [])
        try:
            with open(HEADERS_PATH, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
            for out, section in zip(value, ("html", "all")):
                for name, val in (data.get(section) or {}).items():
                    if HEADER_NAME_RE.fullmatch(name) and isinstance(val, str) and not re.search(r"[\r\n\0]", val):
                        out.append((name, val))
                    else:
                        sys.stderr.write("[wsgi] site-headers.json: skipped bad header %r\n" % (name,))
        except (OSError, ValueError, AttributeError) as e:
            sys.stderr.write("[wsgi] site-headers.json ignored: %s\n" % e)
            value = ([], [])
        _site_cache.update(key=key, value=value)
    return _site_cache["value"]


def _secured(start_response):
    """Adds security and site headers. Headers the handler set itself (type, length, cache...) always win."""
    def wrapped(status, headers, exc_info=None):
        present = {n.lower() for n, _ in headers}
        ctype = next((v for n, v in headers if n.lower() == "content-type"), "")
        html_extra, all_extra = _site_headers()
        extra = SECURITY_HEADERS + all_extra
        if ctype.startswith("text/html"):
            extra = extra + [("Content-Security-Policy", DEFAULT_CSP)] + html_extra
        merged = {}
        for name, value in extra:  # later entries (site file) override earlier ones (defaults)
            merged[name.lower()] = (name, value)
        out = list(headers) + [nv for low, nv in merged.items() if low not in present]
        return start_response(status, out, exc_info) if exc_info else start_response(status, out)
    return wrapped


# ---- static ----
def _text(start_response, status, text, headers=(), head=False):
    body = text.encode("utf-8")
    start_response(status, [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body))),
                            ("Cache-Control", "no-cache")] + list(headers))
    return [] if head else [body]


def _resolve(path_info):
    """Absolute file path inside ROOT, or None."""
    try:
        path = path_info.encode("latin-1").decode("utf-8")  # WSGI hands us UTF-8 bytes as latin-1
    except UnicodeError:
        return None
    if "\0" in path:
        return None
    if path in ("", "/"):
        path = "/index.html"
    rel = os.path.normpath(path).lstrip("/")
    if any(p.startswith(".") and p != ".well-known" for p in rel.split("/")):
        return None
    root = os.path.realpath(ROOT)
    full = os.path.realpath(os.path.join(root, rel))
    if full != root and not full.startswith(root + os.sep):
        return None
    if os.path.isdir(full):
        full = os.path.join(full, "index.html")
    elif not os.path.exists(full) and "." not in os.path.basename(full):
        full += ".html"  # /vault -> vault.html
    return full if os.path.isfile(full) else None


def _content_type(full):
    ext = os.path.splitext(full)[1].lower()
    return TYPES.get(ext) or mimetypes.guess_type(full)[0] or "application/octet-stream"


def _cache_control(path_info, ctype):
    if ctype.startswith("text/html"):
        return "no-cache"
    if path_info.startswith("/static/"):  # Vite's content-hashed bundles
        return "public, max-age=31536000, immutable"
    return "public, max-age=3600"         # /assets/ originals and the rest keep their names across deploys


def _parse_range(header, size):
    """None to ignore the header (serve 200), 'bad' for 416, else (start, end) inclusive."""
    m = RANGE_RE.fullmatch(header.strip())
    if not m or (not m.group(1) and not m.group(2)):
        return None  # malformed or multi-range: ignoring Range is allowed (RFC 9110 §14.2)
    if not m.group(1):  # suffix: the last N bytes
        n = int(m.group(2))
        if n == 0 or size == 0:
            return "bad"
        return max(0, size - n), size - 1
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else size - 1
    if start >= size or end < start:
        return "bad"
    return start, min(end, size - 1)


def _not_modified(environ, etag, mtime):
    inm = environ.get("HTTP_IF_NONE_MATCH")
    if inm:
        tags = [t.strip() for t in inm.split(",")]
        return "*" in tags or any(t.replace("W/", "", 1) == etag for t in tags)
    ims = environ.get("HTTP_IF_MODIFIED_SINCE")
    if ims:
        try:
            return int(mtime) <= email.utils.mktime_tz(email.utils.parsedate_tz(ims))
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _static(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET")
    head = method == "HEAD"
    if method not in ("GET", "HEAD"):
        return _text(start_response, "405 Method Not Allowed", "Method Not Allowed", [("Allow", "GET, HEAD")])
    path_info = environ.get("PATH_INFO") or "/"
    full = _resolve(path_info)
    status = "200 OK"
    if not full:
        page = os.path.join(ROOT, "404.html")
        if not os.path.isfile(page):
            return _text(start_response, "404 Not Found", "Not Found", head=head)
        full, status = page, "404 Not Found"

    st = os.stat(full)
    size = st.st_size
    ctype = _content_type(full)
    etag = '"%x-%x"' % (st.st_mtime_ns // 1000, size)
    last_modified = email.utils.formatdate(st.st_mtime, usegmt=True)
    headers = [("Cache-Control", "no-cache" if status != "200 OK" else _cache_control(path_info, ctype))]
    if status == "200 OK":
        headers += [("ETag", etag), ("Last-Modified", last_modified)]
        if _not_modified(environ, etag, st.st_mtime):
            start_response("304 Not Modified", headers)
            return []
        headers.append(("Accept-Ranges", "bytes"))

    rng = None
    if status == "200 OK" and not head and environ.get("HTTP_RANGE"):
        if_range = environ.get("HTTP_IF_RANGE")
        if not if_range or if_range in (etag, last_modified):
            rng = _parse_range(environ["HTTP_RANGE"], size)
    if rng == "bad":
        return _text(start_response, "416 Range Not Satisfiable", "Range Not Satisfiable",
                     [("Content-Range", "bytes */%d" % size)])

    if rng:
        start, end = rng
        length = end - start + 1
        start_response("206 Partial Content", [("Content-Type", ctype), ("Content-Length", str(length)),
                                               ("Content-Range", "bytes %d-%d/%d" % (start, end, size))] + headers)
    else:
        start, length = 0, size
        start_response(status, [("Content-Type", ctype), ("Content-Length", str(size))] + headers)
    if head:
        return []
    f = open(full, "rb")
    wrapper = environ.get("wsgi.file_wrapper")
    if rng:
        f.seek(start)
        bounded = _BoundedFile(f, length)
        # uWSGI sendfile()s objects with a fileno from offset 0, so ranges go through read().
        return wrapper(bounded, CHUNK) if wrapper else _FileIter(bounded)
    return wrapper(f, CHUNK) if wrapper else _FileIter(f)


class _BoundedFile:
    def __init__(self, f, length):
        self.f = f
        self.remaining = length

    def read(self, size=-1):
        if self.remaining <= 0:
            return b""
        n = CHUNK if size is None or size < 0 else min(size, CHUNK)
        data = self.f.read(min(n, self.remaining))
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


class _FileIter:
    """Body iterable that closes its file (the WSGI server calls close())."""

    def __init__(self, f):
        self.f = f

    def __iter__(self):
        while True:
            data = self.f.read(CHUNK)
            if not data:
                return
            yield data

    def close(self):
        self.f.close()
