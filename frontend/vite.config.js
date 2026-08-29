import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The API runs separately on :8000. Proxying keeps the frontend
    // origin-agnostic, so no CORS config is needed in development.
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true,
                rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
});
