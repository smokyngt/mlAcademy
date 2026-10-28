import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwind from "@tailwindcss/vite";
import path from "node:path";

export default defineConfig({
  root: "src/web",
  plugins: [react(), tailwind()],
  resolve: { alias: { "@shared": path.resolve(__dirname, "src/shared") } },
  build: { outDir: "../../dist", emptyOutDir: true },
  server: { port: 5173, strictPort: true, proxy: { "/api/": "http://localhost:3001" } },
});
