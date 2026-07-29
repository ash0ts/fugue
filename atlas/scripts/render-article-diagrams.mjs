import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const atlasRoot = resolve(scriptDir, "..");
const repositoryRoot = resolve(atlasRoot, "..");
const sourceRoot = join(
  repositoryRoot,
  "docs",
  "articles",
  "fugue-agentic-software-factory",
);
const outputRoot = join(atlasRoot, "dist", "articles");
const manifest = JSON.parse(
  readFileSync(join(sourceRoot, "series.json"), "utf8"),
);
const mermaidCli = join(atlasRoot, "node_modules", ".bin", "mmdc");
const mermaidConfig = join(atlasRoot, "mermaid-config.json");

const chromeCandidates = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  process.env.CHROME_BIN,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
].filter(Boolean);
const chromePath = chromeCandidates.find((candidate) => existsSync(candidate));

if (!chromePath) {
  throw new Error(
    "Article diagram rendering requires Chrome. Set PUPPETEER_EXECUTABLE_PATH.",
  );
}
if (!existsSync(mermaidCli)) {
  throw new Error("Mermaid CLI is missing. Run npm ci in atlas/.");
}

const blocks = (source) =>
  [...source.matchAll(/```mermaid\s*\n([\s\S]*?)```/g)].map((match) =>
    match[1].trim(),
  );
const sha256 = (value) =>
  createHash("sha256").update(value).digest("hex");

const sanitizeSvg = (source, label) => {
  if (
    /<script\b/i.test(source) ||
    /\son[a-z]+\s*=/i.test(source) ||
    /(?:href|src)\s*=\s*["']https?:/i.test(source)
  ) {
    throw new Error(`unsafe SVG output for ${label}`);
  }
  return source
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/\s+xmlns:xlink="[^"]*"/g, "")
    .replace(/<\?xml[\s\S]*?\?>/g, "")
    .trim();
};

const temporaryRoot = mkdtempSync(join(tmpdir(), "fugue-article-diagrams-"));
try {
  const puppeteerConfig = join(temporaryRoot, "puppeteer.json");
  writeFileSync(
    puppeteerConfig,
    `${JSON.stringify({
      executablePath: chromePath,
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    })}\n`,
  );

  for (const entry of manifest.entries) {
    if (entry.publication_state === "planned") continue;
    const article = readFileSync(
      join(sourceRoot, entry.slug, "article.md"),
      "utf8",
    );
    const diagrams = blocks(article);
    const figureRoot = join(outputRoot, entry.slug, "media", "figures");
    mkdirSync(figureRoot, { recursive: true });
    const receipt = [];

    diagrams.forEach((diagram, index) => {
      const digest = sha256(diagram);
      const basename = `figure-${String(index + 1).padStart(2, "0")}-${digest.slice(0, 10)}.svg`;
      const input = join(temporaryRoot, `${entry.slug}-${index + 1}.mmd`);
      const rawOutput = join(temporaryRoot, `${entry.slug}-${index + 1}.svg`);
      writeFileSync(input, `${diagram}\n`);
      const render = spawnSync(
        mermaidCli,
        [
          "--input",
          input,
          "--output",
          rawOutput,
          "--configFile",
          mermaidConfig,
          "--puppeteerConfigFile",
          puppeteerConfig,
          "--backgroundColor",
          "transparent",
          "--quiet",
        ],
        { cwd: atlasRoot, encoding: "utf8" },
      );
      if (render.status !== 0) {
        throw new Error(
          `Mermaid failed for ${entry.id} figure ${index + 1}:\n${render.stderr || render.stdout}`,
        );
      }
      const sanitized = sanitizeSvg(
        readFileSync(rawOutput, "utf8"),
        `${entry.id} figure ${index + 1}`,
      );
      writeFileSync(join(figureRoot, basename), `${sanitized}\n`);
      receipt.push({
        number: index + 1,
        file: basename,
        source_digest: `sha256:${digest}`,
      });
    });

    writeFileSync(
      join(figureRoot, "manifest.json"),
      `${JSON.stringify({ schema_version: 1, article_id: entry.id, figures: receipt }, null, 2)}\n`,
    );
  }
} finally {
  rmSync(temporaryRoot, { recursive: true, force: true });
}

console.log("article diagrams rendered to sanitized local SVG");
