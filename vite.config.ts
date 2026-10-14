import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { existsSync, readFileSync } from "node:fs";
import { resolve, sep } from "node:path";

const cacheRoot = resolve("public/evaluation-cache");
const ledgerRoot = resolve("public/evaluation/itut-current");
const ledgerFiles = new Set(["itut-ledger.json", "itut-ledger.md", "summary.json", "plan.json", "publication.json"]);
const frontendRoot = resolve("public/evaluation/vehicle-frontend-current");
const frontendFiles = new Set(["summary.json", "report.json", "report.md", "field-quality.json", "publication.json", "native-report.json", "method-report.json", "fir-report.json", "streams-report.json"]);
const activeRoot = resolve("public/evaluation/no-passenger-current");
const activeFiles = new Set(["scope.json", "inputs-manifest.json", "matrix-report.json", "operations.jsonl", "publication.json"]);
const ledgerMiddleware = (request: { url?: string }, response: { setHeader: (name: string, value: string) => void; statusCode: number; end: (body?: string | Buffer) => void }, next: () => void) => {
  const pathname = new URL(request.url ?? "/", "http://localhost").pathname;
  const isFrontend = pathname.startsWith("/evaluation/vehicle-frontend-current/");
  const isActive = pathname.startsWith("/evaluation/no-passenger-current/");
  if (!isActive && !isFrontend && !pathname.startsWith("/evaluation/itut-current/")) return next();
  const base = isActive ? "/evaluation/no-passenger-current/" : isFrontend ? "/evaluation/vehicle-frontend-current/" : "/evaluation/itut-current/";
  const name = pathname.slice(base.length);
  const root = isActive ? activeRoot : isFrontend ? frontendRoot : ledgerRoot;
  const allowed = isActive ? activeFiles : isFrontend ? frontendFiles : ledgerFiles;
  if (!allowed.has(name) || !existsSync(resolve(root, name))) {
    response.statusCode = 404; response.end("Report unavailable"); return;
  }
  response.setHeader("Content-Type", name.endsWith(".json") ? "application/json; charset=utf-8" : "text/markdown; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.end(readFileSync(resolve(root, name)));
};
const bypassMetadata = (request: { url?: string }) => {
  const pathname = new URL(request.url ?? "/", "http://localhost").pathname;
  const suffix = pathname.replace(/^\/evaluation\//, "");
  const file = resolve(cacheRoot, suffix);
  if (file.startsWith(cacheRoot + sep) && existsSync(file)) return `/evaluation-cache/${suffix}`;
};

const evaluationProxy = {
  "/evaluation/v3-native": { target: "http://127.0.0.1:18879", bypass: bypassMetadata, rewrite: (path: string) => path.replace(/^\/evaluation\/v3-native/, "/published") },
  "/evaluation/vehicle-stream-v1": { target: "http://127.0.0.1:18879", rewrite: (path: string) => path.replace(/^\/evaluation\/vehicle-stream-v1/, "/vehicle-stream-paced-v1") },
  "/evaluation/vehicle-stream-v2": { target: "http://127.0.0.1:18879", bypass: bypassMetadata, rewrite: (path: string) => path.replace(/^\/evaluation\/vehicle-stream-v2/, "/vehicle-stream-paced-v2") },
  "/evaluation/vehicle-regression-v2": { target: "http://127.0.0.1:18879", bypass: bypassMetadata, rewrite: (path: string) => path.replace(/^\/evaluation\/vehicle-regression-v2/, "/vehicle-stream-40-v2") },
  "/evaluation/vehicle-array-v2": { target: "http://127.0.0.1:18879", bypass: bypassMetadata, rewrite: (path: string) => path.replace(/^\/evaluation\/vehicle-array-v2/, "/vehicle-array-probe-v2") },
};

export default defineConfig({
  plugins: [react(), { name: "current-itut-report", configureServer(server) { server.middlewares.use(ledgerMiddleware); },
    configurePreviewServer(server) { server.middlewares.use(ledgerMiddleware); } }],
  server: {
    proxy: { "/api": "http://127.0.0.1:8001", ...evaluationProxy },
    watch: { ignored: ["**/datasets/**", "**/public/evaluation/**", "**/public/evaluation-cache/**", "**/build/**", "**/third_party/**", "**/android/**"] },
  },
  preview: { proxy: { "/api": "http://127.0.0.1:8001", ...evaluationProxy } },
  build: { target: "es2022" },
});
