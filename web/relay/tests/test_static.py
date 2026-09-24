"""Static serving in web/wsgi.py: ranges, methods, traversal, caching and headers."""
from __future__ import annotations

import json
import os

import pytest

import wsgi

DATA = bytes(range(256)) * 4   # 1024 bytes


@pytest.fixture
def site(client):
    (client.site / "index.html").write_text("<!doctype html><title>fly</title>")
    (client.site / "vault.html").write_text("<p>vault</p>")
    (client.site / "assets").mkdir()
    (client.site / "assets" / "hero.mp4").write_bytes(DATA)
    (client.site / "static").mkdir()
    (client.site / "static" / "index-Bx3k9aZq.js").write_text("console.log(1)")
    (client.site / ".env").write_text("SECRET=1")
    (client.tmp / "secret.txt").write_text("outside")
    return client


def test_full_get(site):
    r = site.get("/assets/hero.mp4")
    assert r.status == 200 and r.body == DATA and r.header("Content-Type") == "video/mp4"
    assert r.header("Content-Length") == "1024" and r.header("Accept-Ranges") == "bytes"


@pytest.mark.parametrize("rng,start,end", [("bytes=0-9", 0, 9), ("bytes=1000-", 1000, 1023),
                                           ("bytes=-10", 1014, 1023), ("bytes=-5000", 0, 1023),
                                           ("bytes=1020-5000", 1020, 1023)])
def test_ranges(site, rng, start, end):
    r = site.get("/assets/hero.mp4", headers={"Range": rng})
    assert r.status == 206 and r.body == DATA[start:end + 1]
    assert r.header("Content-Range") == "bytes %d-%d/1024" % (start, end)
    assert r.header("Content-Length") == str(end - start + 1)


@pytest.mark.parametrize("rng", ["bytes=1024-", "bytes=5000-6000", "bytes=-0", "bytes=9-3"])
def test_unsatisfiable_range_is_416(site, rng):
    r = site.get("/assets/hero.mp4", headers={"Range": rng})
    assert r.status == 416 and r.header("Content-Range") == "bytes */1024"


@pytest.mark.parametrize("rng", ["bytes=0-1,5-6", "items=0-1", "bytes=-"])
def test_unsupported_range_serves_whole_file(site, rng):
    r = site.get("/assets/hero.mp4", headers={"Range": rng})
    assert r.status == 200 and r.body == DATA


def test_if_range_mismatch_serves_whole_file(site):
    r = site.get("/assets/hero.mp4", headers={"Range": "bytes=0-9", "If-Range": '"stale"'})
    assert r.status == 200 and len(r.body) == 1024
    etag = site.get("/assets/hero.mp4").header("ETag")
    assert site.get("/assets/hero.mp4", headers={"Range": "bytes=0-9", "If-Range": etag}).status == 206


def test_head_has_headers_without_body(site):
    r = site.request("HEAD", "/assets/hero.mp4", headers={"Range": "bytes=0-9"})
    assert r.status == 200 and r.body == b"" and r.header("Content-Length") == "1024"
    r = site.request("HEAD", "/")
    assert r.status == 200 and r.body == b"" and r.header("Content-Security-Policy")


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "OPTIONS"])
def test_other_methods_405(site, method):
    r = site.request(method, "/index.html")
    assert r.status == 405 and r.header("Allow") == "GET, HEAD"


@pytest.mark.parametrize("path", ["/../secret.txt", "/assets/../../secret.txt", "/%2e%2e/secret.txt",
                                  "/..%2fsecret.txt", "/.env", "/assets/\x00x", "/nope.html"])
def test_traversal_and_missing_are_404(site, path):
    r = site.get(path)
    assert r.status == 404 and b"outside" not in r.body and b"SECRET" not in r.body


def test_symlink_out_of_root_refused(site):
    os.symlink(str(site.tmp / "secret.txt"), str(site.site / "link.txt"))
    assert site.get("/link.txt").status == 404


def test_extensionless_and_index(site):
    assert site.get("/vault").body == b"<p>vault</p>"
    assert site.get("/vault.html").body == b"<p>vault</p>"
    assert site.get("/").body.startswith(b"<!doctype")


def test_custom_404_page(site):
    (site.site / "404.html").write_text("<h1>lost</h1>")
    r = site.get("/missing")
    assert r.status == 404 and r.body == b"<h1>lost</h1>" and r.header("Content-Security-Policy")


def test_cache_headers(site):
    assert site.get("/").header("Cache-Control") == "no-cache"
    assert site.get("/static/index-Bx3k9aZq.js").header("Cache-Control") == "public, max-age=31536000, immutable"
    assert site.get("/static/index-Bx3k9aZq.js").header("Content-Type").startswith("text/javascript")
    assert site.get("/assets/hero.mp4").header("Cache-Control") == "public, max-age=3600"


def test_conditional_get_304(site):
    r = site.get("/vault.html")
    r2 = site.get("/vault.html", headers={"If-None-Match": r.header("ETag")})
    assert r2.status == 304 and r2.body == b"" and r2.header("Content-Type") is None
    r3 = site.get("/vault.html", headers={"If-Modified-Since": r.header("Last-Modified")})
    assert r3.status == 304
    assert site.get("/vault.html", headers={"If-None-Match": '"other"'}).status == 200


def test_security_headers_everywhere(site):
    for path in ("/", "/assets/hero.mp4", "/nope"):
        r = site.get(path)
        assert r.header("X-Content-Type-Options") == "nosniff"
        assert r.header("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert r.header("Strict-Transport-Security") == "max-age=31536000"
        assert r.header("X-Frame-Options") == "DENY"
    assert site.get("/").header("Content-Security-Policy") == wsgi.DEFAULT_CSP
    assert site.get("/assets/hero.mp4").header("Content-Security-Policy") is None


def test_site_headers_json_applies_csp_to_html_only(site):
    with open(wsgi.HEADERS_PATH, "w") as f:
        json.dump({"html": {"Content-Security-Policy": "default-src 'self'; connect-src 'self' https://rpc.x"},
                   "all": {"Permissions-Policy": "camera=()", "Cache-Control": "public, max-age=99",
                           "Bad\nName": "x"}}, f)
    html = site.get("/")
    assert html.header("Content-Security-Policy") == "default-src 'self'; connect-src 'self' https://rpc.x"
    assert html.header("Permissions-Policy") == "camera=()"
    assert html.header("Cache-Control") == "no-cache"          # the file cannot override protocol headers
    media = site.get("/assets/hero.mp4")
    assert media.header("Content-Security-Policy") is None and media.header("Permissions-Policy") == "camera=()"
    api = site.get("/api/stats")
    assert api.header("Cache-Control") == "no-store" and api.header("Content-Security-Policy") is None


def test_broken_site_headers_json_falls_back(site):
    with open(wsgi.HEADERS_PATH, "w") as f:
        f.write("{nope")
    assert site.get("/").header("Content-Security-Policy") == wsgi.DEFAULT_CSP


def test_relay_import_failure_keeps_static(site, monkeypatch):
    monkeypatch.setitem(wsgi._relay, "app", None)
    monkeypatch.setitem(wsgi._relay, "failed_at", 10 ** 12)  # recent failure: no retry yet
    r = site.get("/api/stats")
    assert r.status == 503 and r.json() == {"error": "relay unavailable"}
    assert site.get("/").status == 200
