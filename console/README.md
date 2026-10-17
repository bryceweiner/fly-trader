# fly-trader console (dashboard source)

The React dashboard that `fly-trader ui` serves. The built copy is committed in `fly_trader/ui/static/console`, so an
install runs the console without Node; rebuild it after changing anything here.

```bash
cd console
npm ci                # once (.npmrc sets legacy-peer-deps for @vitejs/plugin-react's optional peers)
npm run build         # typecheck, then write ../fly_trader/ui/static/console
npm test              # the formatters and helpers
```

## Develop against a running system

```bash
uv run fly-trader ui --observe --port 8599   # a console that hosts no workers (controls are refused)
cd console && npm run dev                    # http://localhost:5173, proxying /api, /ws and /static/brain to :8599
```

`CONSOLE_URL=http://127.0.0.1:8501 npm run dev` proxies to the real console instead (its controls work).

## How it fits together

- **Topics** (reads) are registered in `fly_trader/ui/api/*.py` with `@topic("page.panel", every=s)`. A page subscribes with
  `useTopic('page.panel', params)`; the server computes each (topic, params) once for every open tab and pushes it over the
  WebSocket only when it changed (`fly_trader/ui/hub.py`).
- **Actions** (writes) are `@action("page.verb")` behind `POST /api/action/<name>`. In the browser every action goes through
  `ActionButton` (or a form `Modal`), which asks for confirmation first; the server enforces the same guards (for example,
  no reset while the engine runs).
- **Security**: there is no login. The server binds 127.0.0.1, refuses a foreign `Host`, and requires the `X-Fly-Console`
  header and a same-origin `Origin` on every write and WebSocket (`fly_trader/ui/security.py`).
- **Layout and look**: Dashtreme's horizontal layout (`src/vendor/dashtreme`, a subset of the purchased template, see its
  README) with the fly-trader brand on top (`src/theme`).
- **The brain view** (`src/brain`) draws the connectome geometry from `/static/brain` (generated into
  `fly_trader/ui/static/brain` on first use) with three.js, bundled.

Pages live in `src/pages`, shared components in `src/components`, the fly's inspector and ticker in `src/fly`.

Charts are ApexCharts, except the trades-by-hour wheel, which is Highcharts' variable-radius pie (`highcharts`). Highcharts
is free for non-commercial use; commercial use needs a licence from Highsoft (highcharts.com/license).
