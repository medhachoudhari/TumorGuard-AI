import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  oxc: { jsx: { runtime: 'automatic' } },
  esbuild: { jsx: 'automatic' },
  server: {
    port: 3000,
    strictPort: true
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.js'],
    include: ['src/__tests__/**/*.test.{js,jsx}'],
  },
})
