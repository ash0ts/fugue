import { el, safeExternalLink } from "./common.js";
import { experimentById, productInfo } from "./data.js";

const product = productInfo();
const featured = experimentById(product.featured_study_id);

renderFeaturedStudy(featured);
renderQuickstart(product.quickstart);
renderReferences(product.references);

document.documentElement.classList.add("receipt-ready");

function renderFeaturedStudy(study) {
  const root = document.querySelector("#featured-study");
  if (!study) {
    root.append(el("p", { className: "empty-copy", text: "The featured study is unavailable." }));
    return;
  }

  const harnessResults = study.groups
    .map((group) => ({
      harness: group.harness,
      resolved: group.metrics.passed_predictions,
      total: group.metrics.scored_predictions
    }))
    .sort((left, right) => right.resolved - left.resolved);

  root.append(
    el("article", { className: "proof-receipt" }, [
      el("div", { className: "proof-question" }, [
        el("p", { className: "eyebrow", text: "Completed baseline benchmark" }),
        el("h3", { text: study.question }),
        el("p", { text: study.findings[1] })
      ]),
      el("div", { className: "proof-denominator" }, [
        el("div", {}, [
          el("span", { text: "Published" }),
          el("strong", { text: `${study.metrics.predictions}/${study.metrics.expected_predictions}` }),
          el("small", { text: "exact planned denominator" })
        ]),
        el("div", {}, [
          el("span", { text: "Resolved" }),
          el("strong", { text: `${study.metrics.passed_predictions}/${study.metrics.scored_predictions}` }),
          el("small", { text: "official verifier" })
        ])
      ]),
      el("div", { className: "proof-harnesses" },
        harnessResults.map((result) => el("div", {}, [
          el("span", { text: result.harness }),
          el("span", { className: "proof-bar", style: `--resolved: ${result.resolved}; --total: ${result.total}` }, el("i")),
          el("strong", { text: `${result.resolved}/${result.total}` })
        ]))
      ),
      el("div", { className: "proof-limitation" }, [
        el("span", { text: "LIMITATION" }),
        el("p", { text: study.caveats[0] })
      ]),
      el("a", {
        className: "button-link secondary",
        href: `./experiment.html?id=${encodeURIComponent(study.id)}`,
        text: "Open the complete study"
      })
    ])
  );
}

function renderQuickstart(quickstart) {
  const status = document.querySelector("#quickstart-status");
  const root = document.querySelector("#quickstart-content");
  const shortRevision = quickstart.revision.slice(0, 12);

  status.append(
    el("span", { className: "preview-badge", text: "Preview checkout" }),
    document.createTextNode(` Pinned to ${shortRevision}; not yet released on main.`)
  );

  const code = el("code", { text: quickstart.commands.join("\n") });
  root.append(
    el("div", { className: "quickstart-grid" }, [
      el("div", { className: "terminal-card" }, [
        el("div", { className: "terminal-title" }, [
          el("span", { text: "No-key replay" }),
          el("span", { text: shortRevision })
        ]),
        el("pre", {}, code),
        el("div", { className: "quickstart-links" }, [
          safeExternalLink("Inspect the exact source", quickstart.source_url),
          safeExternalLink("Follow draft PR #36", quickstart.pr_url)
        ])
      ]),
      el("div", { className: "claim-boundary" }, [
        claimList("This proves", quickstart.proves, "verified"),
        claimList("This does not prove", quickstart.does_not_prove, "pending")
      ])
    ]),
    el("p", {
      className: "quickstart-receipt",
      text: "Expected replay receipt: 16 aligned rows · baseline 2/8 · candidate 6/8 · five improved · one regressed · two unchanged · mechanism evidence unavailable."
    })
  );
}

function claimList(title, values, state) {
  const list = el("ul");
  for (const value of values) list.append(el("li", { text: value }));
  return el("section", { className: `claim-list claim-${state}` }, [
    el("h3", { text: title }),
    list
  ]);
}

function renderReferences(references) {
  const root = document.querySelector("#research-references");
  for (const reference of references) {
    root.append(
      el("article", { className: "reference-card" }, [
        el("span", { text: reference.author }),
        el("h3", {}, safeExternalLink(reference.title, reference.url)),
        el("p", { text: reference.relevance })
      ])
    );
  }
}
