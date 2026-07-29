import { createHash } from "node:crypto";
import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import MarkdownIt from "markdown-it";
import footnote from "markdown-it-footnote";

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
const manifestPath = join(sourceRoot, "series.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const entriesById = new Map(manifest.entries.map((entry) => [entry.id, entry]));

const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const slugify = (value) =>
  value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/[\s_]+/g, "-")
    .replace(/-+/g, "-");

const md = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: true,
})
  .use(footnote)
  .use((parser) => {
    parser.core.ruler.push("fugue_heading_ids", (state) => {
      const seen = new Map();
      state.env.headings = [];
      for (let index = 0; index < state.tokens.length - 1; index += 1) {
        const token = state.tokens[index];
        if (token.type !== "heading_open") continue;
        const label = state.tokens[index + 1].content;
        const base = slugify(label) || "section";
        const count = seen.get(base) ?? 0;
        seen.set(base, count + 1);
        const id = count === 0 ? base : `${base}-${count + 1}`;
        token.attrSet("id", id);
        state.env.headings.push({
          level: Number(token.tag.slice(1)),
          label,
          id,
        });
      }
    });
  });

const defaultFence =
  md.renderer.rules.fence ??
  ((tokens, index, options, _env, renderer) =>
    renderer.renderToken(tokens, index, options));

md.renderer.rules.fence = (tokens, index, options, env, renderer) => {
  const token = tokens[index];
  if (token.info.trim() !== "mermaid") {
    return defaultFence(tokens, index, options, env, renderer);
  }
  const diagramNumber = (env.diagramCount = (env.diagramCount ?? 0) + 1);
  const figure = env.figures?.[diagramNumber - 1];
  if (!figure) {
    throw new Error(`missing rendered SVG for figure ${diagramNumber}`);
  }
  const description = token.content
    .replace(/%%.*$/gm, "")
    .replace(/\b(?:flowchart|graph|sequenceDiagram|stateDiagram-v2|quadrantChart|subgraph|end)\b/gi, " ")
    .replace(/[-=<>|()[\]{}:"']/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 360);
  return [
    `<figure class="article-diagram" aria-labelledby="diagram-${diagramNumber}-caption">`,
    `<img src="media/figures/${escapeHtml(figure.file)}" alt="${escapeHtml(description)}">`,
    `<figcaption id="diagram-${diagramNumber}-caption">Figure ${diagramNumber}. `,
    "A locally rendered relationship diagram; the surrounding section explains the argument and limits.",
    "</figcaption>",
    `<details class="diagram-source"><summary>Text description and diagram source</summary><p>${escapeHtml(description)}</p><pre><code>${escapeHtml(token.content)}</code></pre></details>`,
    "</figure>\n",
  ].join("");
};

const defaultLinkOpen =
  md.renderer.rules.link_open ??
  ((tokens, index, options, _env, renderer) =>
    renderer.renderToken(tokens, index, options));
md.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
  const href = tokens[index].attrGet("href") ?? "";
  if (href.startsWith("http://") || href.startsWith("https://")) {
    tokens[index].attrSet("rel", "noreferrer");
  }
  return defaultLinkOpen(tokens, index, options, env, renderer);
};

const sha256 = (value) =>
  `sha256:${createHash("sha256").update(value).digest("hex")}`;

const isVisible = (entry) =>
  Boolean(entry) && entry.publication_state !== "planned";
const isReleased = (entry) => entry?.publication_state === "published";

const loadSources = (entry) => {
  const path = join(sourceRoot, entry.slug, "sources.json");
  if (!existsSync(path)) return [];
  const document = JSON.parse(readFileSync(path, "utf8"));
  return document.sources ?? [];
};

const citation = (source) => {
  const date = source.date ? `, ${source.date}` : "";
  return `${source.author}. “${source.title}.” ${source.publication}${date}. ${source.url}`;
};

const prepareArticleSource = (entry, source) => {
  const sources = loadSources(entry);
  const sourceById = new Map(sources.map((item) => [item.id, item]));
  const cited = new Set(
    [...source.matchAll(/\[@([a-z0-9][a-z0-9-]*)\]/g)].map((match) => match[1]),
  );
  for (const id of cited) {
    if (!sourceById.has(id)) {
      throw new Error(`${entry.id} cites missing source ${id}`);
    }
  }
  const unused = sources.filter((item) => !cited.has(item.id));
  if (unused.length) {
    throw new Error(
      `${entry.id} has unused structured sources: ${unused.map((item) => item.id).join(", ")}`,
    );
  }
  const definitions = [...cited]
    .map((id) => `[^${id}]: ${citation(sourceById.get(id))}`)
    .join("\n\n");
  return `${source.replace(/\[@([a-z0-9][a-z0-9-]*)\]/g, "[^$1]")}\n\n${definitions}\n`;
};

const figureReceipt = (entry) => {
  const path = join(
    outputRoot,
    entry.slug,
    "media",
    "figures",
    "manifest.json",
  );
  if (!existsSync(path)) {
    throw new Error(`missing diagram receipt for ${entry.id}`);
  }
  return JSON.parse(readFileSync(path, "utf8")).figures ?? [];
};

const stripSourceHeader = (source) => {
  const lines = source.split("\n");
  if (lines[0]?.startsWith("# ")) lines.shift();
  while (lines[0] === "") lines.shift();
  if (lines[0]?.startsWith("> ")) {
    while (lines[0]?.startsWith("> ") || lines[0] === "") {
      lines.shift();
      if (!lines.length) break;
    }
  }
  return lines.join("\n");
};

const renderSiteHeader = (current) => `
  <a class="skip-link" href="#main">Skip to article</a>
  <header class="site-header">
    <a class="wordmark" href="/fugue/">Fugue <span>Evidence Atlas</span></a>
    <nav aria-label="Primary navigation">
      <a href="/fugue/">Product</a>
      <a href="/fugue/experiments.html">Studies</a>
      <a${current === "articles" ? ' aria-current="page"' : ""} href="/fugue/articles/">Articles</a>
      <a href="https://github.com/ash0ts/fugue" target="_blank" rel="noreferrer">GitHub</a>
    </nav>
  </header>`;

const renderFooter = () => `
  <footer>
    <span>Fugue field notes</span>
    <span><a href="/fugue/methods.html">Evidence policy</a> · Claims follow locked evidence</span>
  </footer>`;

const articleUrl = (entry) => `/fugue/articles/${entry.slug}/`;

const renderSeriesIndex = () => {
  const entries = manifest.entries
    .map((entry, index) => {
      const label = `Fugue ${entry.part}`;
      const title = isVisible(entry)
        ? `<a href="${articleUrl(entry)}">${escapeHtml(entry.title)}</a>`
        : escapeHtml(entry.title);
      return `
        <li class="article-index-entry is-${escapeHtml(entry.publication_state.replace("_", "-"))}">
          <div class="entry-sequence" aria-hidden="true">${String(index + 1).padStart(2, "0")}</div>
          <div>
            <p class="entry-label">${escapeHtml(label)} · ${escapeHtml(entry.publication_state.replaceAll("_", " "))}</p>
            <h2>${title}</h2>
            <p>${escapeHtml(entry.context)}</p>
          </div>
          <span class="entry-state">${escapeHtml(entry.evidence_state.replaceAll("_", " "))}</span>
        </li>`;
    })
    .join("");
  const canonical = manifest.canonical_root;
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="${escapeHtml(manifest.subtitle)}">
    <link rel="canonical" href="${escapeHtml(canonical)}">
    <link rel="stylesheet" href="/fugue/articles/article.css">
    <title>${escapeHtml(manifest.title)}</title>
  </head>
  <body>
    ${renderSiteHeader("articles")}
    <main id="main" class="series-main">
      <header class="series-hero">
        <p class="eyebrow">Nine standalone field notes / open for review</p>
        <h1>${escapeHtml(manifest.title)}</h1>
        <p class="series-deck">${escapeHtml(manifest.subtitle)}</p>
        <p class="series-contract">Each article defines its own terms, carries its own evidence boundary, and remains useful without opening another installment. Mutable drafts are public for review and excluded from search indexing; only published work is a release.</p>
      </header>
      <ol class="article-index">${entries}</ol>
    </main>
    ${renderFooter()}
  </body>
</html>`;
};

const resultFiles = (entry) => {
  const root = join(sourceRoot, entry.slug, "results");
  if (!existsSync(root)) return [];
  return readdirSync(root)
    .filter((name) => name.endsWith(".md") && name !== "README.md")
    .sort()
    .map((name) => ({
      name,
      content: readFileSync(join(root, name), "utf8"),
    }));
};

const loadFilmSpec = (entry) => {
  const path = join(
    sourceRoot,
    entry.slug,
    "media",
    "film",
    "film-spec.json",
  );
  if (!existsSync(path)) {
    throw new Error(`missing film spec for ${entry.id}`);
  }
  return JSON.parse(readFileSync(path, "utf8"));
};

const chapterLabel = (value) =>
  value
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

const renderFilmBlock = (entry, afterHeading) => {
  const film = entry.animation;
  const spec = loadFilmSpec(entry);
  if (spec.duration !== film.duration_seconds) {
    throw new Error(
      `${entry.id} manifest duration ${film.duration_seconds} does not match film spec ${spec.duration}`,
    );
  }
  const checkpointByName = new Map(
    spec.checkpoints.map((checkpoint) => [checkpoint.name, checkpoint]),
  );
  const chapters = film.chapters
    .map((name, index) => {
      const checkpoint = checkpointByName.get(name);
      if (!checkpoint) {
        throw new Error(`${entry.id} chapter ${name} is absent from film spec`);
      }
      const timelineUrl = `media/film/${escapeHtml(film.html.split("/").at(-1))}#t=${checkpoint.time.toFixed(2)}`;
      return `<li>
        <button type="button" data-film-seek="${checkpoint.time.toFixed(2)}" data-film-target="film-${escapeHtml(entry.id)}">
          <span>${String(index + 1).padStart(2, "0")} · ${escapeHtml(chapterLabel(name))}</span>
          <time>${Math.floor(checkpoint.time / 60)}:${String(Math.floor(checkpoint.time % 60)).padStart(2, "0")}</time>
        </button>
        <a href="${timelineUrl}" aria-label="Open ${escapeHtml(chapterLabel(name))} in the interactive timeline">Interactive ↗</a>
      </li>`;
    })
    .join("");
  const playerControls = (allowDialog) => `
    <div class="film-player-controls" aria-label="Film playback controls">
      <button type="button" data-film-toggle>Play</button>
      <button type="button" data-film-restart>Restart</button>
      <label class="film-scrub-label">
        <span class="sr-only">Film position</span>
        <input data-film-scrub type="range" min="0" max="${spec.duration}" step="0.01" value="0">
      </label>
      <output data-film-clock>0:00 / ${Math.floor(spec.duration / 60)}:${String(spec.duration % 60).padStart(2, "0")}</output>
      ${allowDialog ? `<button type="button" data-film-open="film-${escapeHtml(entry.id)}-dialog">View full size</button>` : ""}
      <button type="button" data-film-screen>Fullscreen</button>
    </div>`;
  return `<section class="film-block" data-after-heading="${escapeHtml(film.after_heading)}" data-bridge-heading="${escapeHtml(film.bridge_to_heading)}" aria-labelledby="film-${escapeHtml(entry.id)}-heading">
    <div class="film-copy">
      <p class="eyebrow">Analytical film / silent-first / ${spec.duration} seconds</p>
      <h2 id="film-${escapeHtml(entry.id)}-heading">Visual synthesis: ${escapeHtml(entry.title)}</h2>
      <p>${escapeHtml(film.what_to_watch)}</p>
      <p class="film-boundary">Placed after “${escapeHtml(afterHeading.label)},” this film synthesizes ${spec.sourceHeadingIds.length} cited sections above. It adds no evidence beyond the article and claim ledger.</p>
      <p class="film-links">
        <a href="media/film/${escapeHtml(film.html.split("/").at(-1))}">Open interactive timeline</a>
        <a href="media/film/${escapeHtml(film.transcript.split("/").at(-1))}">Read transcript and evidence status</a>
      </p>
      <p class="film-mobile-note">On a phone, use “View full size,” then “Fullscreen,” to inspect compact evidence labels.</p>
    </div>
    <div class="film-player" data-film-player id="film-${escapeHtml(entry.id)}-player">
      <video id="film-${escapeHtml(entry.id)}" preload="metadata" muted playsinline tabindex="0" aria-label="Analytical film: ${escapeHtml(entry.title)}" poster="media/film/${escapeHtml(film.poster.split("/").at(-1))}">
        <source src="media/film/${escapeHtml(film.mp4.split("/").at(-1))}" type="video/mp4">
        The film is available through its transcript and interactive timeline.
      </video>
      ${playerControls(true)}
    </div>
    <details class="film-chapters">
      <summary>Film chapters and interactive checkpoints</summary>
      <nav aria-label="Film chapters"><ol>${chapters}</ol></nav>
    </details>
    <p class="film-continue">Continue with <a href="#${escapeHtml(film.bridge_to_heading)}">${escapeHtml(film.bridge_to_heading.split("-").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" "))}</a>.</p>
    <dialog class="film-dialog" id="film-${escapeHtml(entry.id)}-dialog" aria-labelledby="film-${escapeHtml(entry.id)}-dialog-title">
      <div class="film-dialog-header">
        <h3 id="film-${escapeHtml(entry.id)}-dialog-title">${escapeHtml(entry.title)}</h3>
        <button type="button" data-film-close>Close</button>
      </div>
      <div class="film-player is-dialog" data-film-player data-film-dialog-player>
        <video preload="metadata" muted playsinline tabindex="0" aria-label="Full-size analytical film: ${escapeHtml(entry.title)}" poster="media/film/${escapeHtml(film.poster.split("/").at(-1))}">
          <source src="media/film/${escapeHtml(film.mp4.split("/").at(-1))}" type="video/mp4">
        </video>
        ${playerControls(false)}
      </div>
    </dialog>
  </section>`;
};

const insertFilmAfterSection = (entry, body, headings) => {
  const afterHeading = headings.find(
    ({ level, id }) => level === 2 && id === entry.animation.after_heading,
  );
  const bridgeHeading = headings.find(
    ({ level, id }) => level === 2 && id === entry.animation.bridge_to_heading,
  );
  if (!afterHeading || !bridgeHeading) {
    throw new Error(
      `${entry.id} film placement headings are absent: ${entry.animation.after_heading} → ${entry.animation.bridge_to_heading}`,
    );
  }
  const afterMarker = `<h2 id="${escapeHtml(afterHeading.id)}">`;
  const bridgeMarker = `<h2 id="${escapeHtml(bridgeHeading.id)}">`;
  const afterIndex = body.indexOf(afterMarker);
  const bridgeIndex = body.indexOf(bridgeMarker);
  if (afterIndex < 0 || bridgeIndex <= afterIndex) {
    throw new Error(
      `${entry.id} film placement is not sequential: ${afterHeading.id} → ${bridgeHeading.id}`,
    );
  }
  return `${body.slice(0, bridgeIndex)}${renderFilmBlock(entry, afterHeading)}${body.slice(bridgeIndex)}`;
};

const renderArticle = (entry, source, digest) => {
  const previous = entriesById.get(entry.previous_id);
  const next = entriesById.get(entry.next_id);
  const bodyEnvironment = {
    diagramCount: 0,
    figures: figureReceipt(entry),
    headings: [],
  };
  let body = md.render(
    stripSourceHeader(prepareArticleSource(entry, source)),
    bodyEnvironment,
  );
  body = insertFilmAfterSection(entry, body, bodyEnvironment.headings);
  const results = resultFiles(entry)
    .map(({ name, content }) => `
      <section class="result-appendix" data-result-file="${escapeHtml(name)}">
        ${md.render(content, { diagramCount: 0, figures: [], headings: [] })}
      </section>`)
    .join("");
  const previousLink = isVisible(previous)
    ? `<a href="${articleUrl(previous)}">← Fugue ${escapeHtml(previous.part)}: ${escapeHtml(previous.title)}</a>`
    : "<span></span>";
  const nextLink = isVisible(next)
    ? `<a href="${articleUrl(next)}">Fugue ${escapeHtml(next.part)}: ${escapeHtml(next.title)} →</a>`
    : next
      ? `<span class="next-planned">Next: Fugue ${escapeHtml(next.part)} — ${escapeHtml(next.title)} <em>planned</em></span>`
      : "<span>Series close</span>";
  const statusLabel = entry.evidence_state.replaceAll("_", " ");
  const draftDesign = entry.evidence_state === "draft_preregistration";
  const bannerLabel = isReleased(entry)
    ? "Published"
    : draftDesign
      ? "Draft preregistration — no result"
      : "Working draft";
  const bannerBoundary = draftDesign
    ? "The design remains mutable and has no accepted preview or result. Nothing on this page is a treatment conclusion."
    : isReleased(entry)
      ? "This essay is released. Any later empirical result must arrive as a dated appendix."
      : "This public review draft is mutable, search-excluded, and must not be cited as a released Fugue result.";
  const tocEntries = bodyEnvironment.headings.filter(
    ({ level, label }) => level === 2 && label !== "References",
  );
  const toc = tocEntries
    .map(
      ({ id, label }) =>
        `<li><a href="#${escapeHtml(id)}">${escapeHtml(label)}</a></li>`,
    )
    .join("");
  const canonical = `${manifest.canonical_root}${entry.slug}/`;
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="${escapeHtml(entry.context)}">
    ${isReleased(entry) ? "" : '<meta name="robots" content="noindex, nofollow">'}
    <link rel="canonical" href="${escapeHtml(canonical)}">
    <link rel="stylesheet" href="/fugue/articles/article.css">
    <script src="/fugue/articles/article.js" defer></script>
    <title>Fugue ${escapeHtml(entry.part)} — ${escapeHtml(entry.title)}</title>
  </head>
  <body>
    <div class="reading-progress" aria-hidden="true"><span></span></div>
    ${renderSiteHeader("articles")}
    <main id="main" class="article-main">
      <section class="publication-banner ${isReleased(entry) ? "is-released" : "is-draft"}" aria-label="Publication status">
        <strong>${escapeHtml(bannerLabel)}</strong>
        <span>${escapeHtml(bannerBoundary)}</span>
      </section>
      <header class="article-hero">
        <div>
          <p class="eyebrow">Fugue ${escapeHtml(entry.part)} / ${escapeHtml(statusLabel)}</p>
          <h1>${escapeHtml(entry.title)}</h1>
          <p class="article-deck">${escapeHtml(entry.context)}</p>
        </div>
        <dl class="evidence-rail">
          <div><dt>Publication</dt><dd>${escapeHtml(entry.publication_state.replaceAll("_", " "))}</dd></div>
          <div><dt>Evidence</dt><dd>${escapeHtml(statusLabel)}</dd></div>
          <div><dt>Revised</dt><dd>${escapeHtml(entry.updated_at ?? "not yet")}</dd></div>
          <div><dt>Source lock</dt><dd><code>${escapeHtml(digest.slice(0, 19))}…</code></dd></div>
          <div><dt>Source</dt><dd><a href="article.md">Review Markdown</a></dd></div>
        </dl>
      </header>
      <div class="article-layout">
        <aside class="article-aside" aria-label="Article contract">
          <p class="aside-label">Reading contract</p>
          <p>Definitions live here. External links add evidence; they do not supply missing context.</p>
          <nav class="article-toc" aria-label="Table of contents">
            <p class="aside-label">On this page</p>
            <ol>${toc}</ol>
          </nav>
          <a href="/fugue/articles/">All field notes</a>
        </aside>
        <article class="article-prose">
          <details class="mobile-toc">
            <summary>Table of contents</summary>
            <ol>${toc}</ol>
          </details>
          ${body}
          ${results}
        </article>
      </div>
      <nav class="article-pagination" aria-label="Series pagination">
        ${previousLink}
        <a class="series-home-link" href="/fugue/articles/">Series index</a>
        ${nextLink}
      </nav>
    </main>
    ${renderFooter()}
  </body>
</html>`;
};

const copyFont = (packageName, sourceName, targetName) => {
  const source = join(
    atlasRoot,
    "node_modules",
    "@fontsource",
    packageName,
    "files",
    sourceName,
  );
  if (!existsSync(source)) {
    throw new Error(`missing reviewed article font: ${relative(repositoryRoot, source)}`);
  }
  copyFileSync(source, join(outputRoot, "fonts", targetName));
};

mkdirSync(outputRoot, { recursive: true });
mkdirSync(join(outputRoot, "fonts"), { recursive: true });
copyFileSync(join(atlasRoot, "src", "article.css"), join(outputRoot, "article.css"));
copyFileSync(join(atlasRoot, "src", "article.js"), join(outputRoot, "article.js"));
copyFont(
  "source-serif-4",
  "source-serif-4-latin-400-normal.woff2",
  "source-serif-4-regular.woff2",
);
copyFont(
  "source-serif-4",
  "source-serif-4-latin-600-normal.woff2",
  "source-serif-4-semibold.woff2",
);
copyFont(
  "ibm-plex-sans-condensed",
  "ibm-plex-sans-condensed-latin-500-normal.woff2",
  "ibm-plex-sans-condensed-medium.woff2",
);
copyFont(
  "ibm-plex-mono",
  "ibm-plex-mono-latin-400-normal.woff2",
  "ibm-plex-mono-regular.woff2",
);
copyFont(
  "ibm-plex-mono",
  "ibm-plex-mono-latin-600-normal.woff2",
  "ibm-plex-mono-semibold.woff2",
);

writeFileSync(join(outputRoot, "index.html"), renderSeriesIndex());

for (const entry of manifest.entries) {
  const articlePath = join(sourceRoot, entry.slug, "article.md");
  if (!existsSync(articlePath)) {
    throw new Error(`missing article source for ${entry.id}: ${articlePath}`);
  }
  const source = readFileSync(articlePath, "utf8");
  const digest = sha256(source);
  if (entry.source_digest !== digest) {
    throw new Error(
      `${entry.id} source digest mismatch: manifest=${entry.source_digest} actual=${digest}`,
    );
  }
  if (!isVisible(entry)) continue;
  const articleOutput = join(outputRoot, entry.slug);
  mkdirSync(articleOutput, { recursive: true });
  writeFileSync(join(articleOutput, "index.html"), renderArticle(entry, source, digest));
  copyFileSync(articlePath, join(articleOutput, "article.md"));
  const structuredSources = join(sourceRoot, entry.slug, "sources.json");
  if (existsSync(structuredSources)) {
    copyFileSync(structuredSources, join(articleOutput, "sources.json"));
  }
  const runbook = join(sourceRoot, entry.slug, "runbook.md");
  if (existsSync(runbook)) {
    copyFileSync(runbook, join(articleOutput, "runbook.md"));
  }
  const filmSource = join(sourceRoot, entry.slug, "media", "film");
  if (!existsSync(filmSource)) {
    throw new Error(`published article ${entry.id} is missing its film package`);
  }
  cpSync(filmSource, join(articleOutput, "media", "film"), { recursive: true });
}

writeFileSync(
  join(outputRoot, "publication.json"),
  `${JSON.stringify(
    {
      schema_version: manifest.schema_version,
      series_id: manifest.id,
      canonical_root: manifest.canonical_root,
      visible: manifest.entries
        .filter(isVisible)
        .map(({ id, slug, publication_state: publicationState, source_digest: sourceDigest }) => ({
          id,
          slug,
          publication_state: publicationState,
          source_digest: sourceDigest,
        })),
      released: manifest.entries
        .filter(isReleased)
        .map(({ id, slug, source_digest: sourceDigest }) => ({
          id,
          slug,
          source_digest: sourceDigest,
        })),
    },
    null,
    2,
  )}\n`,
);

console.log(
  `articles built: ${manifest.entries.filter(isVisible).length} visible / ${manifest.entries.filter(isReleased).length} released`,
);
