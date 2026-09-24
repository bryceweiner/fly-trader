"""The $FLY vault relay (docs/vault/SPEC.md §3): passes stats and claims between the fly and browsers.

Standard library only; runs under Gandi's uWSGI next to the static site (web/wsgi.py dispatches /api/ here).
"""
