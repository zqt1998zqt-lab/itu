import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { cp, mkdir, readdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { extname, resolve, sep } from "node:path";

export const historicalRoutes = [
  { prefix: "/evaluation/v3-native", upstream: "/published", cached: true },
  { prefix: "/evaluation/vehicle-stream-v1", upstream: "/vehicle-stream-paced-v1", cached: false },
  { prefix: "/evaluation/vehicle-stream-v2", upstream: "/vehicle-stream-paced-v2", cached: true },
  { prefix: "/evaluation/vehicle-regression-v2", upstream: "/vehicle-stream-40-v2", cached: true },
  { prefix: "/evaluation/vehicle-array-v2", upstream: "/vehicle-array-probe-v2", cached: true },
] as const;

export const historicalRoute = (pathname: string) => historicalRoutes.find(route =>
  pathname === route.prefix || pathname.startsWith(`${route.prefix}/`));

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".jsonl": "application/x-ndjson; charset=utf-8", ".svg": "image/svg+xml",
  ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg",
  ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".ico": "image/x-icon",
  ".md": "text/markdown; charset=utf-8", ".csv": "text/csv; charset=utf-8",
  ".txt": "text/plain; charset=utf-8", ".woff2": "font/woff2", ".glb": "model/gltf-binary",
  ".gltf": "model/gltf+json", ".pdf": "application/pdf",
};

function inside(root: string, name: string) {
  const base = resolve(root);
  const file = resolve(base, name);
  if (!file.startsWith(base + sep)) throw Error("Asset path escapes its root");
  return file;
}

// Streams large audio and implements byte ranges so seeking works in production.
export async function serveFile(request: IncomingMessage, response: ServerResponse, root: string, name: string): Promise<boolean> {
  let file: string;
  try { file = inside(root, decodeURIComponent(name)); } catch { return false; }
  const info = await stat(file).catch(() => null);
  if (!info?.isFile()) return false;
  if (request.method !== "GET" && request.method !== "HEAD") {
    response.writeHead(405, { Allow: "GET, HEAD" }); response.end(); return true;
  }
  const etag = `W/"${info.size.toString(16)}-${info.mtimeMs.toString(16)}"`;
  response.setHeader("Content-Type", MIME[extname(file).toLowerCase()] ?? "application/octet-stream");
  response.setHeader("X-Content-Type-Options", "nosniff");
  response.setHeader("Cache-Control", name.startsWith("assets/") ? "public, max-age=31536000, immutable" : "no-cache");
  response.setHeader("ETag", etag);
  response.setHeader("Accept-Ranges", "bytes");
  if (request.headers["if-none-match"] === etag) {
    response.writeHead(304); response.end(); return true;
  }
  let start = 0, end = info.size - 1;
  // A weak ETag cannot validate If-Range; send the whole representation instead.
  const range = request.headers["if-range"] ? undefined : request.headers.range;
  if (range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (match && (match[1] || match[2])) {
      start = match[1] ? Number(match[1]) : Math.max(0, info.size - Number(match[2]));
      end = match[1] && match[2] ? Math.min(Number(match[2]), end) : end;
    }
    if (!match || !(match[1] || match[2]) || !Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= info.size) {
      response.writeHead(416, { "Content-Range": `bytes */${info.size}` }); response.end(); return true;
    }
    response.statusCode = 206;
    response.setHeader("Content-Range", `bytes ${start}-${end}/${info.size}`);
  }
  response.setHeader("Content-Length", Math.max(0, end - start + 1));
  if (request.method === "HEAD" || info.size === 0) { response.end(); return true; }
  const stream = createReadStream(file, { start, end });
  stream.on("error", () => response.destroy());
  response.on("close", () => stream.destroy());
  stream.pipe(response);
  return true;
}

export async function serveEvaluation(request: IncomingMessage, response: ServerResponse, publicRoot: string): Promise<boolean> {
  const path = new URL(request.url ?? "/", "http://localhost").pathname;
  if (path.startsWith("/evaluation-cache/")) {
    if (await serveFile(request, response, resolve(publicRoot, "evaluation-cache"), path.slice(18))) return true;
  } else if (path.startsWith("/evaluation/")) {
    const suffix = path.slice(12);
    const route = historicalRoute(path);
    if (route) {
      if (route.cached && await serveFile(request, response, resolve(publicRoot, "evaluation-cache"), suffix)) return true;
      return false; // Preserve the historical remote audio proxy.
    }
    if (await serveFile(request, response, resolve(publicRoot, "evaluation"), suffix)) return true;
  } else return false;
  // Never turn a missing image/JSON into the HTML application shell.
  response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
  response.end("Evaluation asset unavailable");
  return true;
}

export type Asset = { url: string; path: string; bytes: number; sha256: string };
export type AssetManifest = { schema_version: 1; release_id: string; files: number; bytes: number; assets: Asset[] };

export async function sha256(file: string) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

async function walk(root: string, prefix: string): Promise<string[]> {
  const entries = await readdir(resolve(root, prefix), { withFileTypes: true });
  const paths = await Promise.all(entries.map(async entry => {
    const path = `${prefix}/${entry.name}`;
    // Published datasets use regular directories and files. Do not silently omit
    // symbolic links from an identity manifest.
    if (entry.isSymbolicLink()) throw Error(`Publish a materialized asset, not a symlink: ${path}`);
    return entry.isDirectory() ? walk(root, path) : entry.isFile() ? [path] : [];
  }));
  return paths.flat();
}

const manifestId = (assets: Asset[]) => createHash("sha256").update(JSON.stringify(assets)).digest("hex");

export async function buildAssetManifest(publicRoot: string): Promise<AssetManifest> {
  const paths = (await Promise.all(["evaluation", "evaluation-cache"].map(prefix => walk(publicRoot, prefix)))).flat().sort();
  if (!paths.length) throw Error("No evaluation assets to publish");
  const assets: Asset[] = [];
  // Bound open files and hash memory independently of dataset size.
  for (let i = 0; i < paths.length; i += 8) {
    assets.push(...await Promise.all(paths.slice(i, i + 8).map(async path => {
      const file = inside(publicRoot, path);
      const before = await stat(file);
      const hash = await sha256(file);
      const after = await stat(file);
      if (before.size !== after.size || before.mtimeMs !== after.mtimeMs) throw Error(`Asset changed while hashing: ${path}`);
      return { path, url: "/" + path.split("/").map(encodeURIComponent).join("/"), bytes: after.size, sha256: hash };
    })));
  }
  return { schema_version: 1, release_id: manifestId(assets), files: assets.length,
    bytes: assets.reduce((sum, asset) => sum + asset.bytes, 0), assets };
}

export function validateManifest(manifest: AssetManifest) {
  if (!manifest || manifest.schema_version !== 1 || !manifest.assets?.length || manifest.files !== manifest.assets.length ||
    manifest.bytes !== manifest.assets.reduce((sum, asset) => sum + asset.bytes, 0) ||
    manifest.release_id !== manifestId(manifest.assets)) throw Error("Invalid evaluation asset manifest");
  const seen = new Set<string>();
  for (const asset of manifest.assets) {
    if (!/^(evaluation|evaluation-cache)\//.test(asset.path) || seen.has(asset.path) || !/^[a-f0-9]{64}$/.test(asset.sha256) ||
      !Number.isSafeInteger(asset.bytes) || asset.bytes < 0 || asset.path.split("/").some(part => !part || part === "." || part === ".."))
      throw Error(`Invalid asset identity: ${asset.path}`);
    seen.add(asset.path);
  }
}

export async function verifyAssets(root: string, manifest: AssetManifest) {
  validateManifest(manifest);
  for (const asset of manifest.assets) {
    const file = inside(root, asset.path);
    if ((await stat(file)).size !== asset.bytes || await sha256(file) !== asset.sha256) throw Error(`Asset hash mismatch: ${asset.path}`);
  }
}

export async function verifyWebAssets(webRoot: string, assetsRoot: string, options: { allowUnpublished?: boolean } = {}) {
  const manifest = await readManifest(resolve(webRoot, "evaluation-assets.json"));
  validateManifest(manifest);
  let published: AssetManifest | undefined;
  try {
    published = await readManifest(resolve(assetsRoot, "evaluation-assets.json"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    if (!options.allowUnpublished) throw Error(`Missing evaluation asset release manifest: ${assetsRoot}/evaluation-assets.json`);
  }
  if (published !== undefined) {
    validateManifest(published);
    if (manifest.release_id !== published.release_id) throw Error("Web and evaluation asset releases do not match");
  }
  // A matching manifest alone does not prove that its files are still intact.
  // The local source directory also has to match the built web release in full.
  await verifyAssets(assetsRoot, manifest);
  return manifest;
}

export async function publishAssets(root: string, target: string, manifest: AssetManifest) {
  validateManifest(manifest);
  if (await stat(target).catch(() => null)) throw Error("Asset release target must be a new directory");
  const stage = `${target}.partial-${process.pid}`;
  await mkdir(stage, { recursive: false });
  try {
    for (const asset of manifest.assets) {
      const output = inside(stage, asset.path);
      await mkdir(resolve(output, ".."), { recursive: true });
      await cp(inside(root, asset.path), output);
    }
    await verifyAssets(stage, manifest);
    await writeFile(resolve(stage, "evaluation-assets.json"), JSON.stringify(manifest));
    await rename(stage, target);
  } catch (error) {
    await rm(stage, { recursive: true, force: true });
    throw error;
  }
}

export async function copyPublicShell(publicRoot: string, distRoot: string) {
  await Promise.all((await readdir(publicRoot)).filter(name => !["evaluation", "evaluation-cache"].includes(name) &&
    !/^car-3d\.(backup|day|snap)-/.test(name)).map(name => cp(resolve(publicRoot, name), resolve(distRoot, name), { recursive: true })));
}

export const readManifest = async (file: string): Promise<AssetManifest> => JSON.parse(await readFile(file, "utf8"));
