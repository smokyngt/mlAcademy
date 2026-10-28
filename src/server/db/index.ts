import { Database } from "bun:sqlite";
import { drizzle } from "drizzle-orm/bun-sqlite";
import { migrate } from "drizzle-orm/bun-sqlite/migrator";
import path from "node:path";
import fs from "node:fs";
import * as schema from "./schema";

export const ACADEMY_ROOT = path.resolve(import.meta.dir, "../../..");
const DB_PATH = process.env.ACADEMY_DB ?? path.join(ACADEMY_ROOT, "data", "academy.db");

export function openDb(file: string = DB_PATH) {
  if (file !== ":memory:") fs.mkdirSync(path.dirname(file), { recursive: true });
  const sqlite = new Database(file);
  sqlite.exec("PRAGMA journal_mode = WAL; PRAGMA foreign_keys = OFF;");
  const db = drizzle(sqlite, { schema });
  migrate(db, { migrationsFolder: path.join(ACADEMY_ROOT, "drizzle") });
  // Full-text index over lessons. Not managed by drizzle (virtual table); rebuilt by `bun run sync`.
  sqlite.exec(
    "CREATE VIRTUAL TABLE IF NOT EXISTS lesson_fts USING fts5(lesson_id UNINDEXED, title, body, tokenize='porter unicode61')",
  );
  return { db, sqlite };
}

export type Db = ReturnType<typeof openDb>["db"];
export type Sqlite = ReturnType<typeof openDb>["sqlite"];
