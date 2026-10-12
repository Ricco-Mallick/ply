import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'node:path';

// Served from ricco.codes/ply (GitHub Pages project repo), so every asset path is under /ply/.
export default defineConfig({
  base: '/ply/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } }, // the capture guide imports ../docs/SOP.md
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'),
        walk: resolve(__dirname, 'walk.html'),
        capture: resolve(__dirname, 'capture.html'),
      },
    },
  },
});
