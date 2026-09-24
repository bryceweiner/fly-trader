/// <reference types="vitest/config" />
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig, loadEnv, type Plugin } from 'vite'
import { devApi } from './dev/mock-api.ts'

const root = import.meta.dirname
const PAGES = ['index', 'trading', 'vault', 'terms', 'privacy']

/** Replaces `<!-- @include name -->` with partials/name.html. On index.html the in-page anchors stay
 *  bare (`#about`), elsewhere they point back to index.html; the current page's nav link gets aria-current. */
function includePartials(): Plugin {
  return {
    name: 'fly-include-partials',
    transformIndexHtml: {
      order: 'pre',
      handler(html, ctx) {
        const page = (ctx.filename.split('/').pop() || 'index.html').replace(/\.html$/, '')
        const isIndex = page === 'index'
        return html.replace(/<!--\s*@include\s+([\w-]+)\s*-->/g, (_, name: string) =>
          readFileSync(resolve(root, 'partials', `${name}.html`), 'utf8')
            .replaceAll('{{index}}', isIndex ? '' : 'index.html')
            .replaceAll('{{home}}', isIndex ? '#home' : 'index.html')
            .replaceAll(`data-nav="${page}"`, `data-nav="${page}" aria-current="page"`),
        )
      },
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, root, '')
  const proxy = env.VITE_RELAY_PROXY
  return {
    root,
    appType: 'mpa',
    plugins: [includePartials(), proxy ? null : devApi(env.VITE_NETWORK === 'testnet' ? 'testnet' : 'mainnet')],
    server: {
      port: 5173,
      proxy: proxy ? { '/api': { target: proxy, changeOrigin: true } } : undefined,
    },
    build: {
      outDir: 'dist',
      assetsDir: 'static',
      emptyOutDir: true,
      target: 'es2022',
      chunkSizeWarningLimit: 4096,
      rollupOptions: {
        input: Object.fromEntries(PAGES.map((p) => [p, resolve(root, `${p}.html`)])),
      },
    },
    test: {
      include: ['src/**/*.test.ts'],
      environment: 'node',
    },
  }
})
