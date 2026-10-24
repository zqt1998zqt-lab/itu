import { test } from "node:test";
import type { TestContext } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { execFile } from "node:child_process";
import type { Server } from "node:http";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { buildAssetManifest, publishAssets, verifyAssets, verifyWebAssets } from "../scripts/evaluation-assets.ts";
import { webServer } from "../scripts/serve.ts";

async function listen(server: Server) {
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  return `http://127.0.0.1:${address.port}`;
}

async function fixture(t: TestContext) {
  const root = await mkdtemp(resolve(tmpdir(), "itu-web-assets-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const webRoot = resolve(root, "web");
  const assetsRoot = resolve(root, "public");
  await Promise.all([mkdir(webRoot), mkdir(resolve(assetsRoot, "evaluation/pipeline-cases"), { recursive: true }),
    mkdir(resolve(assetsRoot, "evaluation-cache/v3-native"), { recursive: true })]);
  await Promise.all([
    writeFile(resolve(webRoot, "index.html"), "<!doctype html><title>ITU</title>"),
    writeFile(resolve(assetsRoot, "evaluation/pipeline-cases/sample.svg"), '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'),
    writeFile(resolve(assetsRoot, "evaluation/sample.wav"), "RIFF0123456789"),
    writeFile(resolve(assetsRoot, "evaluation-cache/v3-native/catalog.json"), '{"cached":true}'),
  ]);
  return { root, webRoot, assetsRoot };
}

test("production media returns its real MIME and missing assets never return an HTML success", async t => {
  const dirs = await fixture(t);
  const server = webServer(dirs);
  t.after(() => server.close());
  const origin = await listen(server);
  const figure = await fetch(`${origin}/evaluation/pipeline-cases/sample.svg`);
  assert.equal(figure.status, 200);
  assert.equal(figure.headers.get("content-type"), "image/svg+xml");
  assert.match(await figure.text(), /^<svg/);
  for (const path of ["/evaluation/pipeline-cases/missing.svg", "/evaluation/missing.json", "/public/evaluation/pipeline-cases/sample.svg"]) {
    const missing = await fetch(origin + path);
    assert.equal(missing.status, 404, path);
    assert.doesNotMatch(missing.headers.get("content-type")!, /text\/html/);
  }
});

test("audio GET, HEAD, ranges, conditional requests and traversal are handled correctly", async t => {
  const dirs = await fixture(t);
  const server = webServer(dirs);
  t.after(() => server.close());
  const origin = await listen(server);
  const url = `${origin}/evaluation/sample.wav`;
  const whole = await fetch(url);
  assert.equal(whole.headers.get("content-type"), "audio/wav");
  assert.equal(await whole.text(), "RIFF0123456789");
  const head = await fetch(url, { method: "HEAD" });
  assert.equal(head.headers.get("content-length"), "14");
  assert.equal(await head.text(), "");
  const range = await fetch(url, { headers: { Range: "bytes=4-7" } });
  assert.equal(range.status, 206);
  assert.equal(range.headers.get("content-range"), "bytes 4-7/14");
  assert.equal(await range.text(), "0123");
  const suffix = await fetch(url, { headers: { Range: "bytes=-3" } });
  assert.equal(await suffix.text(), "789");
  for (const header of ["bytes=99-100", "bytes=-0", "bytes=8-3", "bytes=0-1,4-5"]) {
    assert.equal((await fetch(url, { headers: { Range: header } })).status, 416, header);
  }
  assert.equal((await fetch(url, { headers: { "If-None-Match": whole.headers.get("etag")! } })).status, 304);
  const traversal = await fetch(`${origin}/evaluation/%2e%2e%2fweb%2findex.html`);
  assert.equal(traversal.status, 404);
});

test("same-origin API forwarding preserves multipart bytes and historical audio uses its original rewrite", async t => {
  const dirs = await fixture(t);
  const seen: { path: string; type?: string; body: string }[] = [];
  const upstream = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    seen.push({ path: req.url!, type: req.headers["content-type"], body: Buffer.concat(chunks).toString() });
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ path: req.url }));
  });
  t.after(() => upstream.close());
  const target = await listen(upstream);
  const server = webServer({ ...dirs, apiOrigin: target, historicalOrigin: target });
  t.after(() => server.close());
  const origin = await listen(server);
  const form = new FormData(); form.append("microphone", new Blob(["pcm"], { type: "audio/wav" }), "mic.wav");
  assert.equal((await fetch(`${origin}/api/synthesize`, { method: "POST", body: form })).status, 200);
  assert.match(seen[0].type!, /multipart\/form-data; boundary=/);
  assert.match(seen[0].body, /filename="mic.wav"/);
  assert.match(seen[0].body, /pcm/);
  const remote = await fetch(`${origin}/evaluation/v3-native/audio/case.wav?channel=0`);
  assert.equal((await remote.json()).path, "/published/audio/case.wav?channel=0");
  const cached = await fetch(`${origin}/evaluation/v3-native/catalog.json`);
  assert.deepEqual(await cached.json(), { cached: true });
  assert.equal(seen.length, 2, "cached metadata must not require the historical server");
});

test("asset publication round-trips with hashes and rejects a changed source or empty manifest", async t => {
  const dirs = await fixture(t);
  const manifest = await buildAssetManifest(dirs.assetsRoot);
  assert.equal(manifest.files, 3);
  const target = resolve(dirs.root, "release");
  await publishAssets(dirs.assetsRoot, target, manifest);
  await verifyAssets(target, manifest);
  assert.deepEqual(JSON.parse(await readFile(resolve(target, "evaluation-assets.json"), "utf8")), manifest);
  await assert.rejects(publishAssets(dirs.assetsRoot, target, manifest), /new directory/);
  await writeFile(resolve(dirs.assetsRoot, "evaluation/sample.wav"), "RIFF9876543210");
  await assert.rejects(publishAssets(dirs.assetsRoot, resolve(dirs.root, "bad-release"), manifest), /hash mismatch/);
  await assert.rejects(verifyAssets(target, { ...manifest, assets: [], files: 0, bytes: 0 }), /Invalid/);
});

test("production startup rejects different web and asset releases even when both manifests are valid", async t => {
  const dirs = await fixture(t);
  const original = await buildAssetManifest(dirs.assetsRoot);
  await writeFile(resolve(dirs.webRoot, "evaluation-assets.json"), JSON.stringify(original));
  await writeFile(resolve(dirs.assetsRoot, "evaluation/sample.wav"), "RIFF9876543210");
  const changed = await buildAssetManifest(dirs.assetsRoot);
  assert.equal(changed.bytes, original.bytes, "negative control changes content without changing file sizes");
  assert.notEqual(changed.release_id, original.release_id);
  const target = resolve(dirs.root, "different-release");
  await publishAssets(dirs.assetsRoot, target, changed);
  await assert.rejects(promisify(execFile)(process.execPath, ["--experimental-strip-types",
    fileURLToPath(new URL("../scripts/serve.ts", import.meta.url)), "--web-root", dirs.webRoot,
    "--assets-root", target, "--port", "0"], { timeout: 5000 }), /Web and evaluation asset releases do not match/);
});

test("explicit asset roots without a release manifest fail in production and preview", async t => {
  const dirs = await fixture(t);
  const manifest = await buildAssetManifest(dirs.assetsRoot);
  await writeFile(resolve(dirs.webRoot, "evaluation-assets.json"), JSON.stringify(manifest));
  await assert.rejects(verifyWebAssets(dirs.webRoot, dirs.assetsRoot), /Missing evaluation asset release manifest/);
  const run = promisify(execFile);
  await assert.rejects(run(process.execPath, ["--experimental-strip-types",
    fileURLToPath(new URL("../scripts/serve.ts", import.meta.url)), "--web-root", dirs.webRoot,
    "--assets-root", dirs.assetsRoot, "--port", "0"], { timeout: 10000 }), /Missing evaluation asset release manifest/);
  await assert.rejects(run(process.execPath, [
    fileURLToPath(new URL("../node_modules/vite/bin/vite.js", import.meta.url)),
    "preview", "--outDir", dirs.webRoot, "--port", "0",
  ], { timeout: 10000, env: { ...process.env, ITU_EVALUATION_ASSETS_ROOT: dirs.assetsRoot } }), /Missing evaluation asset release manifest/);
});

test("local source assets without a manifest require a complete hash match", async t => {
  const dirs = await fixture(t);
  const manifest = await buildAssetManifest(dirs.assetsRoot);
  await writeFile(resolve(dirs.webRoot, "evaluation-assets.json"), JSON.stringify(manifest));
  assert.equal((await verifyWebAssets(dirs.webRoot, dirs.assetsRoot, { allowUnpublished: true })).release_id, manifest.release_id);
  await writeFile(resolve(dirs.assetsRoot, "evaluation/sample.wav"), "RIFF9876543210");
  await assert.rejects(verifyWebAssets(dirs.webRoot, dirs.assetsRoot, { allowUnpublished: true }), /Asset hash mismatch/);
});

test("false-like JSON release manifests cannot opt out of release validation", async t => {
  const dirs = await fixture(t);
  const manifest = await buildAssetManifest(dirs.assetsRoot);
  await writeFile(resolve(dirs.webRoot, "evaluation-assets.json"), JSON.stringify(manifest));
  for (const invalid of [null, false, 0, ""]) {
    await writeFile(resolve(dirs.assetsRoot, "evaluation-assets.json"), JSON.stringify(invalid));
    await assert.rejects(verifyWebAssets(dirs.webRoot, dirs.assetsRoot), /Invalid evaluation asset manifest/);
    await assert.rejects(verifyWebAssets(dirs.webRoot, dirs.assetsRoot, { allowUnpublished: true }), /Invalid evaluation asset manifest/);
  }
});

test("matching release manifests cannot hide a same-size corrupted asset", async t => {
  const dirs = await fixture(t);
  const manifest = await buildAssetManifest(dirs.assetsRoot);
  await writeFile(resolve(dirs.webRoot, "evaluation-assets.json"), JSON.stringify(manifest));
  const target = resolve(dirs.root, "release");
  await publishAssets(dirs.assetsRoot, target, manifest);
  await verifyWebAssets(dirs.webRoot, target);
  await writeFile(resolve(target, "evaluation/sample.wav"), "RIFF9876543210");
  await assert.rejects(verifyWebAssets(dirs.webRoot, target), /Asset hash mismatch/);
});
