import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    preserveSymlinks: true,
  },
  server: {
    proxy: {
      "/chat": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
});
