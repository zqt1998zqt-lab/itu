import { parseArgs } from "node:util";
import { resolve } from "node:path";
import { publishAssets, readManifest, verifyAssets } from "./evaluation-assets.ts";

const { values } = parseArgs({ options: {
  root: { type: "string", default: "public" }, manifest: { type: "string", default: "dist/evaluation-assets.json" },
  to: { type: "string" }, verify: { type: "boolean", default: false },
} });
const manifest = await readManifest(resolve(values.manifest!));
if (values.verify) {
  await verifyAssets(resolve(values.root!), manifest);
  console.log(`Verified ${manifest.files} files / ${manifest.bytes} bytes · ${manifest.release_id}`);
} else {
  if (!values.to) throw Error("Specify a new release directory with --to <directory>");
  await publishAssets(resolve(values.root!), resolve(values.to), manifest);
  console.log(`Published and verified ${manifest.files} files · ${manifest.release_id}`);
}
