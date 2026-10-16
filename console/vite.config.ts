/// <reference types="vitest/config" />
import { resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const root = import.meta.dirname
// `npm run dev` serves the dashboard on :5173 and proxies the API and WebSocket to a console on CONSOLE_URL
// (default a preview console: `fly-trader ui --observe --port 8599`, which hosts no workers).
const target = process.env.CONSOLE_URL || 'http://127.0.0.1:8599'

export default defineConfig({
  root,
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target, changeOrigin: true },
      '/static/brain': { target, changeOrigin: true },
      // the console refuses a WebSocket whose Origin is not its own (fly_trader/ui/security.py): present the target's
      '/ws': { target: target.replace(/^http/, 'ws'), ws: true, changeOrigin: true,
        configure: (proxy) => proxy.on('proxyReqWs', (req) => req.setHeader('origin', target)) },
    },
  },
  build: {
    // committed: installs run the dashboard without Node (the release ships `git ls-files`)
    outDir: resolve(root, '../fly_trader/ui/static/console'),
    assetsDir: 'assets',
    emptyOutDir: true,
    target: 'es2022',
    chunkSizeWarningLimit: 2048,
  },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
})
