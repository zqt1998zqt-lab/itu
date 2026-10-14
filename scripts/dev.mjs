import { spawn } from "node:child_process";

const jobs = [
  spawn("bash", ["scripts/api.sh"], { stdio: "inherit", detached: true }),
  spawn("npm", ["run", "dev"], { stdio: "inherit", detached: true }),
];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const job of jobs) { try { process.kill(-job.pid, "SIGTERM"); } catch {} }
  process.exitCode = code;
}
jobs.forEach(job => { job.on("exit", code => stop(code ?? 1)); job.on("error", () => stop(1)); });
process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());
