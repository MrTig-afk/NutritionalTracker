import { defineConfig } from 'vite'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

// PUBLIC DEMO build: VITE_DEMO=1 swaps the network layer (src/lib/api.js) and
// the push helpers (src/lib/push.js) for the synthetic in-memory ones under
// src/demo/ at resolve time, so every importer picks them up with zero source
// changes. The PWA plugin is skipped entirely (no service worker, no
// manifest), and the Google Fonts links are replaced by self-hosted subsets —
// the demo makes zero off-origin requests. Normal builds keep an empty alias
// list and are byte-for-byte unaffected.
const DEMO = process.env.VITE_DEMO === '1'

function demoHtml() {
  return {
    name: 'demo-html',
    transformIndexHtml(html) {
      if (!DEMO) return html
      return html
        .replace(/^\s*<link rel="preconnect" href="https:\/\/fonts[^>]*>\s*\r?\n/gm, '')
        .replace(/^\s*<link href="https:\/\/fonts\.googleapis\.com[^>]*>\s*\r?\n/gm, '')
        .replace(/^\s*<noscript>[\s\S]*?<\/noscript>\s*\r?\n/m, '')
    },
  }
}

export default defineConfig({
  // Drop console/debugger from production bundles (keep console.error).
  esbuild: { drop: ['debugger'], pure: ['console.log', 'console.debug', 'console.info'] },
  // Always define the flag ('' when off) so demo-gated branches are statically
  // constant-folded and tree-shaken out of normal builds.
  define: { 'import.meta.env.VITE_DEMO': JSON.stringify(DEMO ? '1' : '') },
  resolve: {
    alias: DEMO
      ? [
          {
            find: /^(\.\.?\/)+lib\/api$/,
            replacement: fileURLToPath(new URL('./src/demo/apiDemo.js', import.meta.url)),
          },
          {
            find: /^(\.\.?\/)+lib\/push$/,
            replacement: fileURLToPath(new URL('./src/demo/pushDemo.js', import.meta.url)),
          },
        ]
      : [],
  },
  plugins: [
    react(),
    demoHtml(),
    ...(DEMO
      ? []
      : [
          VitePWA({
            strategies: 'injectManifest',
            srcDir: 'src',
            filename: 'sw.js',
            registerType: 'prompt',
            includeAssets: ['favicon.svg', 'icon-192.png', 'icon-512.png'],
            devOptions: { enabled: true, type: 'module' },
            manifest: {
              name: 'NutriScan',
              short_name: 'NutriScan',
              description: 'AI-powered nutrition label scanner and macro tracker',
              theme_color: '#006D77',
              background_color: '#ffffff',
              display: 'standalone',
              orientation: 'portrait',
              start_url: '/',
              icons: [
                {
                  src: '/icon-192.png',
                  sizes: '192x192',
                  type: 'image/png',
                },
                {
                  src: '/icon-512.png',
                  sizes: '512x512',
                  type: 'image/png',
                },
                {
                  src: '/icon-512.png',
                  sizes: '512x512',
                  type: 'image/png',
                  purpose: 'maskable',
                },
              ],
            },
            injectManifest: {
              globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
            },
          }),
        ]),
  ],
})
