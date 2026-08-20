import { copyFile, mkdir, readFile, readdir, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDir, "..");
const sourceRoot = join(
  repositoryRoot,
  "docs",
  "articles",
  "fugue-agentic-software-factory",
);
const manifest = JSON.parse(
  await readFile(join(sourceRoot, "series.json"), "utf8"),
);

for (const entry of manifest.entries) {
  const filmRoot = join(sourceRoot, entry.slug, "media", "film");
  const spec = JSON.parse(await readFile(join(filmRoot, "film-spec.json"), "utf8"));
  const checkpointRoot = join(filmRoot, "checkpoints");
  await mkdir(checkpointRoot, { recursive: true });
  const expectedCheckpoints = new Set(
    spec.checkpoints.map((checkpoint) => `${checkpoint.name}.png`),
  );
  for (const existingCheckpoint of await readdir(checkpointRoot)) {
    if (
      existingCheckpoint.endsWith(".png")
      && !expectedCheckpoints.has(existingCheckpoint)
    ) {
      await rm(join(checkpointRoot, existingCheckpoint));
    }
  }
  for (const checkpoint of spec.checkpoints) {
    const looseCheckpoint = join(filmRoot, `${checkpoint.name}.png`);
    try {
      await copyFile(
        looseCheckpoint,
        join(checkpointRoot, `${checkpoint.name}.png`),
      );
      await rm(looseCheckpoint);
    } catch (error) {
      if (error.code !== "ENOENT") {
        throw error;
      }
    }
  }
  await rm(join(filmRoot, "render.log"), { force: true });
  process.stdout.write(`packaged checkpoints for ${entry.slug}\n`);
}
