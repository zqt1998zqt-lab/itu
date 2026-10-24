import { defineConfig } from "vite";
import type { Connect, ResolvedConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import { writeFile } from "node:fs/promises";
import { buildAssetManifest, copyPublicShell, historicalRoutes, serveEvaluation, verifyWebAssets } from "./scripts/evaluation-assets.ts";

const publicRoot = resolve(process.env.ITU_EVALUATION_ASSETS_ROOT ?? "public");
const evaluationMiddleware: Connect.NextHandleFunction = (request, response, next) => {
  void serveEvaluation(request, response, publicRoot).then(handled => { if (!handled) next(); }).catch(next);
};
const evaluationProxy = Object.fromEntries(historicalRoutes.map(route => [route.prefix, {
  target: process.env.ITU_HISTORICAL_ORIGIN ?? "http://127.0.0.1:18879",
  rewrite: (path: string) => route.upstream + path.slice(route.prefix.length),
}]));
let thisConfig: ResolvedConfig;

export default defineConfig({
  plugins: [react(), {
    name: "evaluation-assets",
    configureServer(server) { server.middlewares.use(evaluationMiddleware); },
    async configurePreviewServer(server) {
      await verifyWebAssets(resolve(server.config.root, server.config.build.outDir), publicRoot, {
        allowUnpublished: process.env.ITU_EVALUATION_ASSETS_ROOT === undefined,
      });
      server.middlewares.use(evaluationMiddleware);
    },
  }, {
    name: "publish-web-shell",
    configResolved(config) { thisConfig = config; },
    async writeBundle() {
      const distRoot = resolve(thisConfig.root, thisConfig.build.outDir);
      await copyPublicShell(thisConfig.publicDir, distRoot);
      const manifest = await buildAssetManifest(publicRoot);
      await writeFile(resolve(distRoot, "evaluation-assets.json"), JSON.stringify(manifest));
      console.log(`Evaluation assets kept separate: ${manifest.files} files, ${manifest.bytes} bytes, sha256 ${manifest.release_id}`);
    },
  }],
  server: {
    proxy: { "/api": "http://127.0.0.1:8001", ...evaluationProxy },
    watch: { ignored: ["**/datasets/**", "**/public/evaluation/**", "**/public/evaluation-cache/**", "**/build/**", "**/third_party/**", "**/android/**"] },
  },
  preview: { proxy: { "/api": "http://127.0.0.1:8001", ...evaluationProxy } },
  build: { target: "es2022", copyPublicDir: false },
});
