/** Prints every concept id (core file + content/concepts.d/*.json) with its title. */
import fs from "node:fs";
import path from "node:path";

const dir = path.resolve(import.meta.dir, "..", "content");
const all: { id: string; title: string; stage: string }[] = JSON.parse(fs.readFileSync(path.join(dir, "concepts.json"), "utf8"));
const d = path.join(dir, "concepts.d");
if (fs.existsSync(d)) for (const f of fs.readdirSync(d).filter((x) => x.endsWith(".json"))) all.push(...JSON.parse(fs.readFileSync(path.join(d, f), "utf8")));
for (const c of all) console.log(`${c.id.padEnd(28)} [${c.stage}] ${c.title}`);
