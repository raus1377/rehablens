/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { iwsdkDev } from '@iwsdk/vite-plugin-dev';
import { defineConfig } from 'vite';
// Dev-only endpoint: the headset POSTs each recorded take here and it lands in ./recordings/
function recorder() {
    return {
        name: 'rehablens-recorder',
        configureServer(server) {
            server.middlewares.use('/__save', (req, res) => {
                if (req.method !== 'POST') {
                    res.statusCode = 405;
                    return res.end();
                }
                const name = String(req.headers['x-filename'] || 'take.json').replace(/[^\w.-]/g, '_');
                const chunks = [];
                req.on('data', (c) => chunks.push(c));
                req.on('end', () => {
                    mkdirSync('recordings', { recursive: true });
                    writeFileSync(join('recordings', name), Buffer.concat(chunks));
                    res.statusCode = 200;
                    res.end('ok');
                });
            });
        },
    };
}
export default defineConfig({
    plugins: [iwsdkDev(), recorder()],
    server: { host: '0.0.0.0', port: 8081, open: false },
    build: {
        outDir: 'dist',
        sourcemap: process.env.NODE_ENV !== 'production',
        target: 'esnext',
        rollupOptions: { input: './index.html' },
    },
    esbuild: { target: 'esnext' },
    // @drawcall/uikitml otherwise pulls a second three/@pmndrs/uikit graph
    // (its own three vs the app's super-three alias). Duplicate Component classes break
    // instanceof checks → "Only pmndrs/uikit components can be added as children".
    resolve: {
        dedupe: [
            'three',
            '@pmndrs/uikit',
            '@pmndrs/uikit-horizon',
            '@pmndrs/uikit-lucide',
        ],
    },
    optimizeDeps: {
        exclude: ['@babylonjs/havok'],
        include: [
            'three',
            '@pmndrs/uikit',
            '@pmndrs/uikit-horizon',
            '@pmndrs/uikit-lucide',
            '@drawcall/uikitml',
        ],
        esbuildOptions: { target: 'esnext' },
    },
    publicDir: 'public',
    base: './',
});
