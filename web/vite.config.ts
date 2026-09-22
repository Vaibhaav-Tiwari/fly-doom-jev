import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {proxy: {'/live-api': {target: 'http://127.0.0.1:8420', changeOrigin: true, rewrite: path => path.replace(/^\/live-api/, '')}}},
  build: {outDir: 'dist', assetsDir: 'assets'},
});
