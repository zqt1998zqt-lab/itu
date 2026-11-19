import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { spawn } from "node:child_process";

const require = createRequire(import.meta.url);
const { chromium } = require("/home/zhangqingtian/.npm/_npx/9833c18b2d85bc59/node_modules/playwright");
const root = "/home/zhangqingtian/project/itu";
const artifacts = fs.mkdtempSync(path.join(root, "build/dataset-revision-20261117/ui-"));
const release = JSON.parse(fs.readFileSync(path.join(root, "src/features/scene-dataset-revision-r2-ids.json"), "utf8"));
const activation = JSON.parse(fs.readFileSync(path.join(root, "build/dataset-revision-20261117/ACTIVATION-r2.json"), "utf8"));
const browserProcess = spawn("/home/zhangqingtian/.cache/ms-playwright/chromium_headless_shell-1234/chrome-linux/headless_shell", [
  "--headless", "--no-sandbox", "--disable-dev-shm-usage", "--remote-debugging-port=0",
  "--user-data-dir=" + artifacts + "/profile", "about:blank",
], { stdio: ["ignore", "ignore", fs.openSync(artifacts + "/browser.log", "ax")] });
let browser;
try {
  const portFile = artifacts + "/profile/DevToolsActivePort";
  for (let attempt = 0; attempt < 100 && !fs.existsSync(portFile); attempt++) {
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert(fs.existsSync(portFile));
  browser = await chromium.connectOverCDP("http://127.0.0.1:" + fs.readFileSync(portFile, "utf8").split("\n")[0]);
  const page = await browser.contexts()[0].newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  await page.setViewportSize({ width: 1600, height: 1050 });
  const scenes = [["sce1", "background-noise-idle"], ["sce2", "background-noise-urban"],
    ["sce3", "background-noise-highway"], ["sce4", "background-noise-hvac"]];
  const checked = [];
  for (const [scene, route] of scenes) {
    await page.goto("http://localhost:5173/#" + route, { waitUntil: "domcontentloaded" });
    const modeSelect = page.locator('select[id$="-mode"]:visible').first();
    await modeSelect.waitFor({ timeout: 30000 });
    await modeSelect.selectOption("WB-wired");
    const api = `/api/alignment/acceptance/scene-condition-datasets?scope=background_noise&mode=WB-wired&scene=${scene}`;
    const response = await page.request.get("http://localhost:5173" + api);
    assert.equal(response.status(), 200);
    const body = await response.json();
    assert.equal(body.datasets[0].dataset_id, release.scenes[scene]["WB-wired"]);
    await page.waitForFunction(() => document.body.innerText.includes("100"));
    await page.screenshot({ path: artifacts + "/" + route + ".png", fullPage: false });
    assert(!/场景工程预览数据集未绑定|未注册的场景条件数据集/.test(await page.locator("body").innerText()));
    checked.push({ route, dataset_id: body.datasets[0].dataset_id });
  }
  for (const [scope, route] of [["echo", "echo"], ["double_talk", "double-talk"]]) {
    const expected = activation.datasets.find(row => row.kind === "stress" && row.scope === scope && row.mode === "WB-wired");
    assert(expected);
    await page.goto("http://localhost:5173/#" + route, { waitUntil: "domcontentloaded" });
    const response = await page.request.get(`http://localhost:5173/api/alignment/acceptance/stress-datasets?scope=${scope}&mode=WB-wired`);
    assert.equal(response.status(), 200);
    assert.equal((await response.json()).dataset_id, expected.dataset_id);
    await page.screenshot({ path: artifacts + "/" + route + ".png", fullPage: false });
    checked.push({ route, dataset_id: expected.dataset_id });
  }
  const runtime = JSON.parse(fs.readFileSync(path.join(root, "build/dataset-revision-20261117/runtime-r2/scene-sce1-WB-wired.json"), "utf8"));
  const url = `http://localhost:5173/api/alignment/acceptance/scene-condition-batches/${runtime.batch_id}/cases/${encodeURIComponent(runtime.case_id)}/outputs/tx`;
  await page.evaluate(async source => {
    const audio = document.createElement("audio");
    audio.id = "revised-dataset-output-playback";
    audio.src = source;
    document.body.appendChild(audio);
    await audio.play();
  }, url);
  await page.waitForTimeout(300);
  const playback = await page.locator("#revised-dataset-output-playback").evaluate(audio => {
    audio.pause();
    return { currentTime: audio.currentTime, duration: audio.duration, error: audio.error?.message ?? null };
  });
  assert.equal(playback.error, null);
  assert(playback.currentTime > 0);
  assert.equal(playback.duration, 12);
  assert.deepEqual(errors, []);
  const result = { artifacts, checked, playback, errors };
  fs.writeFileSync(artifacts + "/result.json", JSON.stringify(result, null, 2), { flag: "wx" });
  console.log(JSON.stringify(result));
} finally {
  if (browser) await browser.close();
  browserProcess.kill("SIGTERM");
}
