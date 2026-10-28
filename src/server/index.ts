import { serveStatic } from "hono/bun";
import { Hono } from "hono";
import fs from "node:fs";
import path from "node:path";
import { openDb, ACADEMY_ROOT } from "./db";
import { createApp } from "./app";

const { db, sqlite } = openDb();
const api = createApp(db, sqlite);

const root = new Hono();
root.route("/", api);

// Production: serve the built SPA (vite build -> dist/). In development Vite serves it and proxies /api here.
const dist = path.join(ACADEMY_ROOT, "dist");
if (fs.existsSync(dist)) {
  root.use("/*", serveStatic({ root: path.relative(process.cwd(), dist) || "." }));
  root.get("*", (c) => c.html(fs.readFileSync(path.join(dist, "index.html"), "utf8")));
}

const port = Number(process.env.PORT ?? 3001);
console.log(`ML Academy API on http://localhost:${port}`);
export default { port, fetch: root.fetch, hostname: "127.0.0.1" };
