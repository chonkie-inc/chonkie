import { mkdir, writeFile, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { externalDocSources } from "./external-docs.config.mjs";

const envPath = path.resolve(".env");
if (existsSync(envPath)) {
  const envContent = await readFile(envPath, "utf-8");
  for (const line of envContent.split("\n")) {
    const match = line.match(/^([^#=]+)=(.*)$/);
    if (match && !process.env[match[1].trim()]) {
      process.env[match[1].trim()] = match[2].trim();
    }
  }
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.text();
}

function applyTransforms(content, transforms = []) {
  let next = content;
  for (const transform of transforms) {
    for (const [from, to] of transform.replacements ?? []) {
      next = next.replace(from, to);
    }
  }
  return next;
}

async function syncSource(source) {
  const baseUrl = process.env[source.envVar];
  if (!baseUrl) {
    console.error(`${source.envVar} env variable is not set (check .env file)`);
    process.exit(1);
  }

  const base = baseUrl.replace(/\/$/, "");
  const outDir = path.resolve(source.outDir);

  console.log(`Syncing ${source.id} docs from ${base}...`);

  const manifestUrl = `${base}/manifest.json`;
  const manifest = JSON.parse(await fetchText(manifestUrl));

  for (const filePath of manifest.files) {
    const url = `${base}/${filePath}`;
    let content = await fetchText(url);

    const fileTransforms = source.transforms?.filter(
      (transform) => transform.file === filePath,
    );
    content = applyTransforms(content, fileTransforms);

    const dest = path.join(outDir, filePath);
    await mkdir(path.dirname(dest), { recursive: true });
    await writeFile(dest, content);
    console.log(`  ${filePath}`);
  }

  console.log(`Done syncing ${source.id}.`);
}

async function syncAll() {
  for (const source of externalDocSources) {
    await syncSource(source);
  }
}

syncAll();
