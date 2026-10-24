import { createServer, request as httpRequest } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { resolve } from "node:path";
import { stat } from "node:fs/promises";
import { parseArgs } from "node:util";
import { historicalRoute, serveEvaluation, serveFile, verifyWebAssets } from "./evaluation-assets.ts";

export function proxyRequest(request: IncomingMessage, response: ServerResponse, origin: string, path: string) {
  const target = new URL(path, origin);
  const upstream = httpRequest(target, { method: request.method, headers: { ...request.headers, host: target.host } }, incoming => {
    response.writeHead(incoming.statusCode ?? 502, incoming.headers);
    incoming.on("error", () => response.destroy());
    incoming.pipe(response);
  });
  upstream.setTimeout(300000, () => upstream.destroy(Error("Upstream timed out")));
  upstream.on("error", () => {
    if (!response.headersSent) response.writeHead(502, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Upstream service unavailable");
  });
  request.on("aborted", () => upstream.destroy());
  response.on("close", () => upstream.destroy());
  request.pipe(upstream);
}

export function webServer(options: { webRoot: string; assetsRoot: string; apiOrigin?: string; historicalOrigin?: string }) {
  return createServer((request, response) => {
    void (async () => {
      const url = new URL(request.url ?? "/", "http://localhost");
      if (url.pathname === "/api" || url.pathname.startsWith("/api/")) {
        proxyRequest(request, response, options.apiOrigin ?? "http://127.0.0.1:8001", request.url!); return;
      }
      if (await serveEvaluation(request, response, options.assetsRoot)) return;
      const route = historicalRoute(url.pathname);
      if (route) {
        proxyRequest(request, response, options.historicalOrigin ?? "http://127.0.0.1:18879", route.upstream + request.url!.slice(route.prefix.length)); return;
      }
      if (await serveFile(request, response, options.webRoot, url.pathname.slice(1) || "index.html")) return;
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" }); response.end("Not found");
    })().catch(error => {
      console.error(error);
      if (!response.headersSent) response.writeHead(500);
      response.end("Web service error");
    });
  });
}

async function main() {
  const { values } = parseArgs({ options: {
    host: { type: "string", default: "127.0.0.1" }, port: { type: "string", default: "5173" },
    "web-root": { type: "string", default: "dist" }, "assets-root": { type: "string" },
  } });
  const webRoot = resolve(values["web-root"]!);
  const configuredAssetsRoot = values["assets-root"] ?? process.env.ITU_EVALUATION_ASSETS_ROOT;
  const assetsRoot = resolve(configuredAssetsRoot ?? "public");
  await stat(resolve(webRoot, "index.html"));
  console.log("Verifying evaluation asset release and file hashes…");
  const manifest = await verifyWebAssets(webRoot, assetsRoot, { allowUnpublished: configuredAssetsRoot === undefined });
  const server = webServer({ webRoot, assetsRoot, apiOrigin: process.env.ITU_API_ORIGIN, historicalOrigin: process.env.ITU_HISTORICAL_ORIGIN });
  server.on("error", error => { console.error(error); process.exitCode = 1; });
  server.listen(Number(values.port), values.host, () => console.log(`ITU production web: http://${values.host}:${values.port} · assets ${manifest.release_id}`));
  for (const signal of ["SIGINT", "SIGTERM"] as const) process.on(signal, () => server.close());
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.dirname, "serve.ts")) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
}
