import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';
import { networkInterfaces } from 'node:os';
import { resolve } from 'node:path';

// Served from ricco.codes/ply (GitHub Pages project repo), so every asset path is under /ply/.
// `npm run dev:phone` serves https on the LAN so a phone can open the capture app (camera + sensors need a secure page).
const PHONE = process.env.PLY_PHONE === '1', PORT = 5199;
const lan = Object.values(networkInterfaces()).flat().find(i => i && i.family === 'IPv4' && !i.internal)?.address;

export default defineConfig({
  base: '/ply/',
  plugins: [react(), ...(PHONE ? [basicSsl()] : [])],
  define: { __LAN_URL__: JSON.stringify(PHONE && lan ? `https://${lan}:${PORT}/` : '') },
  server: { port: PORT, host: PHONE, fs: { allow: ['..'] } }, // the guide imports ../docs/SOP.md
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'),
        capture: resolve(__dirname, 'capture.html'),
        walk: resolve(__dirname, 'walk.html'),
        guide: resolve(__dirname, 'guide.html'),
      },
    },
  },
});
