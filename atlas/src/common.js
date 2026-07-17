import "@fontsource/source-serif-4/latin-400.css";
import "@fontsource/source-serif-4/latin-600.css";
import "@fontsource/ibm-plex-sans-condensed/latin-500.css";
import "@fontsource/ibm-plex-sans-condensed/latin-600.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "./site.css";

export function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(options)) {
    if (key === "className") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (value !== undefined && value !== null) node.setAttribute(key, String(value));
  }
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child !== undefined && child !== null) node.append(child);
  }
  return node;
}

export function formatNumber(value, options = {}) {
  if (value === null || value === undefined) return "Unavailable";
  return new Intl.NumberFormat("en-US", options).format(value);
}

export function formatPercent(value) {
  return value === null || value === undefined
    ? "Unavailable"
    : `${formatNumber(value * 100, { maximumFractionDigits: 1 })}%`;
}

export function tierLabel(tier) {
  return {
    confirmed: "Confirmed",
    directional: "Directional",
    baseline: "Baseline benchmark",
    contract: "Contract evidence",
    active: "Active",
    blocked: "Blocked"
  }[tier] || tier;
}

export function metric(label, value, note) {
  const content = [
    el("span", { className: "metric-label", text: label }),
    el("strong", { text: value })
  ];
  if (note) content.push(el("small", { text: note }));
  return el("div", { className: "metric" }, content);
}

export function safeExternalLink(label, href) {
  return el(
    "a",
    {
      href,
      target: "_blank",
      rel: "noopener noreferrer",
      className: "evidence-link"
    },
    [el("span", { text: label }), el("span", { "aria-hidden": "true", text: "↗" })]
  );
}
