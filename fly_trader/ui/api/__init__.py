"""Every topic and action the console serves, one module per page (importing a module registers its topics and
actions with ui/hub.py). Topic names are ``<page>.<panel>``; action names ``<page>.<verb>``."""
from . import (brain, data, inspector, kalshi_data, kalshi_model, kalshi_overview, kalshi_trades, logs, meta, model, notifications,  # noqa: F401
               overview, processes, rh, safety, search, status, ticker, trades, vault)
