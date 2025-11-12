import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { // Add this 'server' configuration block
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
        // Optional: you can add a rewrite rule if your Flask endpoints
        // don't start with /api in their definition. For this specific case,
        // since your Flask route is `/api/home`, you don't need 'rewrite'.
        // rewrite: (path) => path.replace(/^\/api/, ''), 
      },
    },
    // Optional: You can specify the port for the React dev server if needed
    // port: 3000, 
  },
});
