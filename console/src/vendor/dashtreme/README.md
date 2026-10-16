# Dashtreme (vendored subset)

The parts of **Dashtreme – Multipurpose Bootstrap 5 Admin Template** (codervent, ThemeForest) that the console uses:

| File | From | Change |
|---|---|---|
| `bootstrap.min.css` | `horizontal/assets/css/bootstrap.min.css` (Bootstrap 5.3.0-alpha1, MIT) | source-map comment removed; the alpha's invalid `.form-floating::before:not(...)` rule removed (the CSS minifier rejects it) |
| `bootstrap-extended.css` | `horizontal/assets/css/bootstrap-extended.css` | none |
| `app.css` | `horizontal/assets/css/app.css` | rules that load the template's background and login images removed |
| `boxicons.css`, `fonts/boxicons.woff2` | `horizontal/assets/css/icons.css` (its Boxicons part, MIT) and `horizontal/assets/fonts` | `@font-face` points at the woff2 copy only |

These files are used under the purchaser's Envato license. They are **not** covered by this repository's MIT license.
The console's own look (the fly-trader brand: colours, fonts, the status strip) is in `console/src/theme/`.
