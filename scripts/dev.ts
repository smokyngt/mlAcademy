/** Starts the API (watch mode) and Vite together. Replaces `concurrently`, which cannot spawn cmd.exe from Bun on some Windows setups. */
export {};
const run = (name: string, cmd: string[]) =>
  Bun.spawn(cmd, { stdout: "inherit", stderr: "inherit", stdin: "inherit", env: process.env, onExit: (_p, code) => { console.log(`[${name}] exited (${code})`); } });

const sync = Bun.spawnSync([process.execPath, "scripts/build-catalog.ts"], { stdout: "inherit", stderr: "inherit" });
if (sync.exitCode !== 0) process.exit(sync.exitCode ?? 1);

const api = run("api", [process.execPath, "--watch", "src/server/index.ts"]);
const web = run("web", [process.execPath, "x", "vite"]);
console.log("\nOpen http://localhost:5173\n");

const killTree = (p: { pid: number; kill: () => void }) => {
  // On Windows, kill() leaves grandchildren (vite, the watch child) running; /T kills the whole tree.
  if (process.platform === "win32") Bun.spawnSync(["taskkill", "/PID", String(p.pid), "/T", "/F"], { stdout: "ignore", stderr: "ignore" });
  else p.kill();
};
const shutdown = () => { killTree(api); killTree(web); process.exit(0); };
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
await Promise.race([api.exited, web.exited]);
shutdown();
