const $ = (id) => document.getElementById(id);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const TABS = {
  overview: "Overview",
  setup: "Setup",
  run: "Run matrix",
  jobs: "Jobs",
  results: "Results",
};

const DEFAULT_MANIFEST = "datasets/pilot.yaml";

const state = {
  activeTab: "overview",
  activeJob: null,
  activeJobId: null,
  eventSource: null,
  initializedControls: false,
  manifest: null,
  modelTouched: false,
  resultFilter: "all",
  results: { rows: [], summary: {} },
  selectedConditions: new Set(),
  selectedHarnesses: new Set(),
  status: null,
  summary: null,
};

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

async function postJSON(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

function boot() {
  bindNavigation();
  bindActions();
  setActiveTab(tabFromHash());
  refreshAll().catch(showError);
}

function bindNavigation() {
  $$("[data-tab-target]").forEach((button) => {
    button.addEventListener("click", () => {
      location.hash = button.dataset.tabTarget || "overview";
    });
  });
  $$("[data-tab-link]").forEach((button) => {
    button.addEventListener("click", () => {
      location.hash = button.dataset.tabLink || "overview";
    });
  });
  window.addEventListener("hashchange", () => setActiveTab(tabFromHash()));
}

function bindActions() {
  $("refreshBtn").addEventListener("click", () => refreshAll().catch(showError));
  $("preflightBtn").addEventListener("click", () => startJob("preflight"));
  $("bridgeBtn").addEventListener("click", () => startJob("bridge"));
  $("prepareBtn").addEventListener("click", () => startJob("prepare"));
  $("runBtn").addEventListener("click", () => startJob("run"));
  $("exportBtn").addEventListener("click", () => startJob("export"));

  ["modelInput", "taskInput", "attemptInput", "concurrentInput", "dryRunInput"].forEach((id) => {
    $(id).addEventListener("input", () => {
      if (id === "modelInput") state.modelTouched = true;
      renderCommandPreview();
    });
    $(id).addEventListener("change", renderCommandPreview);
  });

  document.addEventListener("click", (event) => {
    const harness = event.target.closest("[data-toggle-harness]");
    if (harness) {
      toggleSetValue(state.selectedHarnesses, harness.dataset.toggleHarness);
      renderRun();
      return;
    }

    const condition = event.target.closest("[data-toggle-condition]");
    if (condition) {
      toggleSetValue(state.selectedConditions, condition.dataset.toggleCondition);
      renderRun();
      return;
    }

    const filter = event.target.closest("[data-result-filter]");
    if (filter) {
      state.resultFilter = filter.dataset.resultFilter || "all";
      renderResults();
      return;
    }

    const jobButton = event.target.closest("[data-job-id]");
    if (jobButton) {
      loadJob(jobButton.dataset.jobId).catch(showError);
      return;
    }

    const copyButton = event.target.closest("[data-copy]");
    if (copyButton) {
      copyText(copyButton.dataset.copy || "");
    }
  });
}

async function refreshAll() {
  const [summary, jobs, results] = await Promise.all([
    getJSON("/api/summary"),
    getJSON("/api/jobs"),
    getJSON("/api/results"),
  ]);
  state.summary = summary;
  state.status = summary.status || null;
  state.manifest = summary.manifest || null;
  state.jobs = jobs || [];
  state.results = results || { rows: [], summary: {} };
  syncControls();
  renderAll();
}

function renderAll() {
  renderHeader();
  renderOverview();
  renderSetup();
  renderRun();
  renderJobs();
  renderResults();
}

function tabFromHash() {
  const name = location.hash.replace("#", "");
  return TABS[name] ? name : "overview";
}

function setActiveTab(tab) {
  state.activeTab = TABS[tab] ? tab : "overview";
  $("pageTitle").textContent = TABS[state.activeTab];
  $$("[data-tab-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.tabPanel !== state.activeTab;
  });
  $$("[data-tab-target]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tabTarget === state.activeTab);
  });
}

function syncControls() {
  const harnesses = manifestHarnesses();
  const conditions = manifestConditions();
  if (!state.initializedControls) {
    state.selectedHarnesses = new Set(harnesses);
    state.selectedConditions = new Set(conditions);
    $("attemptInput").value = state.manifest?.k || 1;
    $("concurrentInput").value = state.manifest?.n_concurrent || 2;
    if (!state.modelTouched) {
      $("modelInput").value =
        state.status?.route?.model || state.manifest?.model || state.status?.default_model || "";
    }
    state.initializedControls = true;
    return;
  }
  pruneSelection(state.selectedHarnesses, harnesses);
  pruneSelection(state.selectedConditions, conditions);
}

function pruneSelection(selection, available) {
  const allowed = new Set(available);
  Array.from(selection).forEach((item) => {
    if (!allowed.has(item)) selection.delete(item);
  });
  if (selection.size === 0) available.forEach((item) => selection.add(item));
}

function renderHeader() {
  const status = state.status || {};
  const route = status.route || {};
  $("providerPill").textContent = route.error ? "route error" : route.provider || "provider";
  $("providerPill").className = `status-pill ${route.error ? "danger" : "neutral"}`;
  $("modelPill").textContent = route.model || "model";
  setOptionalLink($("projectPill"), status.wandb_project_url, status.trace_project || "W&B project");
  setOptionalLink($("openWandbBtn"), status.wandb_project_url, "Open W&B");
  setOptionalLink($("sidebarWeaveLink"), status.weave_project_url, status.trace_project || "Not configured");
}

function renderOverview() {
  renderReadinessCards();
  renderMatrix();
  renderLatestJob();
  renderSummaryMetrics();
  renderOverviewLinks();
}

function renderReadinessCards() {
  const readiness = state.summary?.readiness || {};
  const manifest = state.summary?.manifest || {};
  const status = state.status || {};
  const cards = [
    {
      label: "Trace",
      value: readiness.trace ? "Ready" : "Missing",
      note: status.trace_project || "Set WANDB_ENTITY and WANDB_PROJECT",
      ok: readiness.trace,
    },
    {
      label: "Model key",
      value: readiness.model ? "Ready" : "Missing",
      note: status.route?.api_key_env || "Select a provider-prefixed model",
      ok: readiness.model,
    },
    {
      label: "Bridge",
      value: readiness.bridge ? "Online" : "Offline",
      note: readiness.bridge ? "127.0.0.1:4000" : "Start bridge before bridged runs",
      ok: readiness.bridge,
      warn: !readiness.bridge,
    },
    {
      label: "Manifest",
      value: readiness.manifest ? "Loaded" : "Missing",
      note: `${manifest.counts?.tasks || 0} tasks, ${manifest.counts?.matrix_cells || 0} cells`,
      ok: readiness.manifest,
    },
  ];
  $("readinessCards").innerHTML = cards
    .map((card) => {
      const badgeClass = card.ok ? "ok" : card.warn ? "warn" : "danger";
      return `
        <article class="stat-card">
          <div class="stat-top">
            <span class="stat-label">${escapeHTML(card.label)}</span>
            <span class="badge ${badgeClass}">${escapeHTML(card.value)}</span>
          </div>
          <div class="stat-value">${escapeHTML(card.value)}</div>
          <div class="stat-note">${escapeHTML(card.note)}</div>
        </article>
      `;
    })
    .join("");
}

function renderMatrix() {
  const manifest = state.manifest || {};
  const conditions = manifestConditions();
  const matrix = state.summary?.matrix || [];
  $("matrixCount").textContent = `${(manifest.counts || {}).matrix_cells || 0} cells`;
  $("matrixStave").style.setProperty("--matrix-cols", String(Math.max(conditions.length, 1)));
  if (!matrix.length || !conditions.length) {
    $("matrixStave").innerHTML = `<div class="empty-state">No matrix yet. Add harnesses and conditions to ${escapeHTML(DEFAULT_MANIFEST)}.</div>`;
    return;
  }
  const header = `
    <div class="matrix-row matrix-header">
      <div class="matrix-voice">Harness</div>
      ${conditions.map((condition) => `<div class="matrix-cell">${escapeHTML(condition)}</div>`).join("")}
    </div>
  `;
  const rows = matrix
    .map((voice) => {
      const cells = (voice.cells || [])
        .map((cell) => matrixCellHTML(cell))
        .join("");
      return `
        <div class="matrix-row">
          <div class="matrix-voice">${escapeHTML(voice.harness)}</div>
          ${cells}
        </div>
      `;
    })
    .join("");
  $("matrixStave").innerHTML = header + rows;
}

function matrixCellHTML(cell) {
  const status = String(cell.status || "not run");
  const className = status.replace(/\s+/g, "-");
  const count = cell.total ? `${cell.passed || 0}/${cell.total}` : "";
  return `
    <div class="matrix-cell status-${escapeHTML(className)}">
      <span>${escapeHTML(status)}</span>
      <span class="cell-count">${escapeHTML(count)}</span>
    </div>
  `;
}

function renderLatestJob() {
  const latest = state.summary?.jobs?.latest;
  if (!latest) {
    $("latestJobCard").innerHTML = `<div class="empty-state">No jobs yet. Run preflight or launch a dry matrix.</div>`;
    return;
  }
  const failed = latest.status === "failed";
  $("latestJobCard").innerHTML = `
    ${failed ? `<div class="error-callout">Job failed. Open the log stream and rerun the failing step after fixing the command output.</div>` : ""}
    ${detailRow("Kind", latest.kind, latest.status)}
    ${detailRow("Job id", latest.id, latest.started_at)}
    ${detailRow("Command", commandText(latest.command), "Click Jobs to inspect output")}
  `;
}

function renderSummaryMetrics() {
  const summary = state.summary?.results || {};
  const tokens = summary.tokens || {};
  const tokenTotal = Number(tokens.input || 0) + Number(tokens.cache || 0) + Number(tokens.output || 0);
  const metrics = [
    ["Rows", formatInteger(summary.total || 0), `${summary.passed || 0} passed`],
    ["Pass rate", formatPercent(summary.pass_rate), `${summary.failed || 0} failed`],
    ["Cost", formatCurrency(summary.cost_usd), "reported by agents"],
    ["Tokens", formatInteger(tokenTotal), `${formatInteger(tokens.output || 0)} output`],
  ];
  $("summaryMetrics").innerHTML = metrics.map(metricHTML).join("");
}

function renderOverviewLinks() {
  const status = state.status || {};
  const links = [
    ["Project", status.wandb_project_url, status.trace_project || "Set WANDB_ENTITY/WANDB_PROJECT"],
    ["Weave", status.weave_project_url, status.weave_project_url ? "Open traces" : "Set W&B tracing"],
  ];
  $("overviewLinks").innerHTML = links
    .map(([label, href, detail]) => `
      <div class="link-row">
        <span>
          <span class="detail-title">${escapeHTML(label)}</span>
          <span class="detail-subtitle">${escapeHTML(detail)}</span>
        </span>
        ${href ? `<a class="button cyan small" href="${escapeAttr(href)}" target="_blank" rel="noreferrer">Open</a>` : `<span class="badge warn">Missing</span>`}
      </div>
    `)
    .join("");
}

function renderSetup() {
  renderKeys();
  renderBridge();
  renderRouteDetails();
  renderManifestDetails();
}

function renderKeys() {
  const keys = state.status?.keys || {};
  $("setupKeys").innerHTML = Object.entries(keys)
    .map(([key, present]) => `
      <div class="key-item">
        <span class="key-dot ${present ? "present" : "missing"}" aria-hidden="true"></span>
        <span>
          <span class="detail-title">${escapeHTML(key)}</span>
          <span class="detail-subtitle">${present ? "present" : "missing"}</span>
        </span>
      </div>
    `)
    .join("");
}

function renderBridge() {
  const bridge = state.status?.bridge || {};
  const ok = Boolean(bridge.ok);
  $("setupBridge").innerHTML = `
    ${detailRow("Status", ok ? "Online" : "Offline", ok ? "Bridge is healthy" : "Bridge is offline. Start bridge before Claude/Codex bridged runs.")}
    ${detailRow("URL", "http://127.0.0.1:4000", "Containers use http://host.docker.internal:4000")}
    ${detailRow("Detail", ok ? bridge.body || "OK" : bridge.error || bridge.body || "Not reachable", "Local only")}
  `;
}

function renderRouteDetails() {
  const status = state.status || {};
  const route = status.route || {};
  $("setupRoute").innerHTML = definitionRows({
    Provider: route.provider || "unknown",
    Model: route.model || status.default_model || "unknown",
    "Key env": route.api_key_env || "unknown",
    Error: route.error || "",
    CWD: status.cwd || "",
  });
}

function renderManifestDetails() {
  const manifest = state.manifest || {};
  $("setupManifest").innerHTML = definitionRows({
    Dataset: manifest.dataset?.ref || "unknown",
    Version: manifest.dataset?.version || "latest",
    Tasks: manifest.counts?.tasks || 0,
    Harnesses: manifestHarnesses().join(", "),
    Conditions: manifestConditions().join(", "),
    Attempts: manifest.k || 1,
    Concurrent: manifest.n_concurrent || 1,
    "Jobs dir": manifest.jobs_dir || "",
  });
}

function renderRun() {
  renderToggleGroup("harnessToggles", manifestHarnesses(), state.selectedHarnesses, "harness");
  renderToggleGroup("conditionToggles", manifestConditions(), state.selectedConditions, "condition");
  $("runManifestSummary").innerHTML = `
    ${detailRow("Dataset", state.manifest?.dataset?.ref || "unknown", `${state.manifest?.counts?.tasks || 0} tasks`)}
    ${detailRow("Cells selected", String(selectedCellCount()), `${state.selectedHarnesses.size} harnesses x ${state.selectedConditions.size} conditions`)}
    ${detailRow("Trace", state.status?.trace_project || "Not configured", state.status?.weave_project_url ? "Weave link ready" : "Set W&B tracing")}
  `;
  renderCommandPreview();
}

function renderToggleGroup(id, values, selection, type) {
  $(id).innerHTML = values
    .map((value) => {
      const active = selection.has(value);
      const dataName = type === "harness" ? "data-toggle-harness" : "data-toggle-condition";
      return `
        <button class="toggle-chip ${active ? "active" : ""}" ${dataName}="${escapeAttr(value)}" aria-pressed="${String(active)}" type="button">
          ${escapeHTML(value)}
        </button>
      `;
    })
    .join("");
}

function renderCommandPreview() {
  $("commandPreview").textContent = buildRunCommand().join(" ");
}

function renderJobs() {
  const jobs = state.jobs || [];
  if (!jobs.length) {
    $("jobList").innerHTML = `<div class="empty-state">No web jobs yet. Run preflight, start bridge, or launch a dry matrix.</div>`;
    $("jobDetail").innerHTML = "";
    $("activeJobBadge").textContent = "No job selected";
    $("activeJobBadge").className = "badge neutral";
    return;
  }
  if (!state.activeJobId) state.activeJobId = jobs[0].id;
  $("jobList").innerHTML = jobs.map(jobItemHTML).join("");
  renderJobDetail();
}

function jobItemHTML(job) {
  const active = job.id === state.activeJobId;
  return `
    <button class="job-item ${active ? "active" : ""}" data-job-id="${escapeAttr(job.id)}" type="button">
      <span class="job-main">
        <span class="job-title">${escapeHTML(job.kind || "job")}</span>
        <span class="job-subtitle">${escapeHTML(job.id || "")}</span>
      </span>
      ${statusBadge(job.status)}
    </button>
  `;
}

function renderJobDetail() {
  const job = state.activeJob || (state.jobs || []).find((item) => item.id === state.activeJobId);
  if (!job) return;
  $("activeJobBadge").outerHTML = statusBadge(job.status, "activeJobBadge");
  const failed = job.status === "failed";
  $("jobDetail").innerHTML = `
    ${failed ? `<div class="error-callout">Job failed. Inspect the final log lines, fix the failing system, then rerun this step.</div>` : ""}
    ${detailRow("Job", job.id, job.kind)}
    ${detailRow("Status", job.status || "unknown", job.returncode === undefined ? "running" : `exit ${job.returncode}`)}
    ${detailRow("Command", commandText(job.command), "")}
  `;
  if (job.log_tail && $("logView").textContent === "") {
    $("logView").textContent = job.log_tail;
  }
}

function renderResults() {
  const rows = state.results?.rows || [];
  const summary = state.results?.summary || {};
  renderResultCards(summary);
  renderResultFilters(rows);
  const filtered = rows.filter(rowMatchesFilter);
  $("resultEmpty").hidden = rows.length > 0;
  $("resultRows").innerHTML = filtered.slice(0, 300).map(resultRowHTML).join("");
  setOptionalLink($("resultsWeaveLink"), state.status?.weave_project_url, "Open Weave");
}

function renderResultCards(summary) {
  const groups = [
    { scope: "All", name: "Trials", ...summary },
    ...groupCards("Provider", summary.by_provider || []),
    ...groupCards("Harness", summary.by_harness || []),
    ...groupCards("Condition", summary.by_condition || []),
  ].slice(0, 8);
  if (!groups.length) {
    $("resultCards").innerHTML = metricHTML(["Rows", "0", "No exported trials"]);
    return;
  }
  $("resultCards").innerHTML = groups
    .map((group) => `
      <article class="result-card">
        <div class="result-card-label">${escapeHTML(group.scope)}</div>
        <div class="result-card-value">${escapeHTML(formatPercent(group.pass_rate))}</div>
        <div class="result-card-note">${escapeHTML(group.name)} - ${escapeHTML(group.passed || 0)}/${escapeHTML(group.total || 0)} passed</div>
      </article>
    `)
    .join("");
}

function groupCards(scope, groups) {
  return groups.map((group) => ({
    scope,
    name: group.name,
    total: group.total,
    passed: group.passed,
    pass_rate: group.pass_rate,
  }));
}

function renderResultFilters(rows) {
  const filters = [
    ["all", `All ${rows.length}`],
    ["passed", `Passed ${rows.filter((row) => row.pass === true).length}`],
    ["failed", `Failed ${rows.filter((row) => row.pass === false).length}`],
    ["exceptions", `Exceptions ${rows.filter((row) => row.exception_class).length}`],
  ];
  $("resultFilters").innerHTML = filters
    .map(([key, label]) => `
      <button class="filter-chip ${state.resultFilter === key ? "active" : ""}" data-result-filter="${escapeAttr(key)}" type="button">
        ${escapeHTML(label)}
      </button>
    `)
    .join("");
}

function resultRowHTML(row) {
  const tokens = Number(row.n_input_tokens || 0) + Number(row.n_cache_tokens || 0) + Number(row.n_output_tokens || 0);
  const weaveUrl = row.weave_url || buildWeaveUrl(row.trace_project);
  return `
    <tr>
      <td>${escapeHTML(row.harness || "")}</td>
      <td>${escapeHTML(row.condition || "")}</td>
      <td>${escapeHTML(row.model_provider || "")}</td>
      <td class="code-cell">${escapeHTML(row.model || "")}</td>
      <td>${passBadge(row)}</td>
      <td>${escapeHTML(formatValue(row.reward))}</td>
      <td>${escapeHTML(formatInteger(tokens))}</td>
      <td>${escapeHTML(formatCurrency(row.cost_usd))}</td>
      <td>${escapeHTML(formatDuration(row.wall_time_sec))}</td>
      <td class="code-cell"><button class="copy-key" data-copy="${escapeAttr(row.run_key || "")}" type="button">${escapeHTML(row.run_key || "")}</button></td>
      <td>${weaveUrl ? `<a href="${escapeAttr(weaveUrl)}" target="_blank" rel="noreferrer">Open Weave</a>` : ""}</td>
    </tr>
  `;
}

function rowMatchesFilter(row) {
  if (state.resultFilter === "passed") return row.pass === true;
  if (state.resultFilter === "failed") return row.pass === false;
  if (state.resultFilter === "exceptions") return Boolean(row.exception_class);
  return true;
}

async function startJob(kind) {
  try {
    setActionBusy(true);
    const config = jobConfig(kind);
    const job = await postJSON(config.url, config.body);
    state.activeJob = job;
    state.activeJobId = job.id;
    $("logView").textContent = "";
    showToast(`${config.label} started`);
    await refreshJobsOnly();
    location.hash = "jobs";
    streamJob(job.id);
  } catch (error) {
    showError(error);
  } finally {
    setActionBusy(false);
  }
}

function jobConfig(kind) {
  const model = $("modelInput").value.trim();
  if (kind === "preflight") {
    return { label: "Run preflight", url: "/api/preflight", body: { model } };
  }
  if (kind === "bridge") {
    return { label: "Start bridge", url: "/api/bridge/up", body: { model } };
  }
  if (kind === "prepare") {
    return {
      label: "Prepare memory",
      url: "/api/prepare",
      body: { manifest: DEFAULT_MANIFEST, conditions: Array.from(state.selectedConditions) },
    };
  }
  if (kind === "export") {
    return { label: "Export results", url: "/api/export", body: {} };
  }
  return { label: "Run matrix", url: "/api/run", body: runBody() };
}

function runBody() {
  return {
    manifest: DEFAULT_MANIFEST,
    model: $("modelInput").value.trim(),
    harnesses: Array.from(state.selectedHarnesses),
    conditions: Array.from(state.selectedConditions),
    n_tasks: $("taskInput").value,
    n_attempts: $("attemptInput").value,
    n_concurrent: $("concurrentInput").value,
    dry_run: $("dryRunInput").checked,
  };
}

async function refreshJobsOnly() {
  state.jobs = await getJSON("/api/jobs");
  renderJobs();
}

async function loadJob(jobId) {
  if (!jobId) return;
  closeStream();
  state.activeJobId = jobId;
  state.activeJob = await getJSON(`/api/jobs/${encodeURIComponent(jobId)}`);
  $("logView").textContent = state.activeJob.log_tail || "";
  renderJobs();
  if (state.activeJob.status === "running") streamJob(jobId);
}

function streamJob(jobId) {
  closeStream();
  state.activeJobId = jobId;
  state.eventSource = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
  state.eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.chunk) appendLog(data.chunk);
    if (data.done) {
      closeStream();
      state.activeJob = data.status || state.activeJob;
      refreshAll().catch(showError);
    }
  };
  state.eventSource.onerror = () => {
    closeStream();
    showToast("Log stream disconnected");
  };
}

function closeStream() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
}

function appendLog(chunk) {
  $("logView").textContent += chunk;
  $("logView").scrollTop = $("logView").scrollHeight;
}

function buildRunCommand() {
  const body = runBody();
  const parts = ["fugue", "run", "--manifest", body.manifest];
  if (body.model) parts.push("--model", body.model);
  if (body.harnesses.length) parts.push("--harnesses", body.harnesses.join(","));
  if (body.conditions.length) parts.push("--conditions", body.conditions.join(","));
  if (body.n_tasks) parts.push("-l", String(body.n_tasks));
  if (body.n_attempts) parts.push("-k", String(body.n_attempts));
  if (body.n_concurrent) parts.push("-n", String(body.n_concurrent));
  if (body.dry_run) parts.push("--dry-run");
  return parts.map(shellQuote);
}

function manifestHarnesses() {
  return (state.manifest?.harnesses || []).map((harness) => harness.name).filter(Boolean);
}

function manifestConditions() {
  return (state.manifest?.conditions || []).filter(Boolean);
}

function selectedCellCount() {
  return state.selectedHarnesses.size * state.selectedConditions.size;
}

function toggleSetValue(selection, value) {
  if (!value) return;
  if (selection.has(value) && selection.size > 1) {
    selection.delete(value);
  } else {
    selection.add(value);
  }
}

function setActionBusy(busy) {
  ["preflightBtn", "bridgeBtn", "prepareBtn", "runBtn", "exportBtn"].forEach((id) => {
    $(id).disabled = busy;
  });
}

function setOptionalLink(element, href, label) {
  element.textContent = label;
  if (href) {
    element.href = href;
    element.classList.remove("disabled");
    element.setAttribute("aria-disabled", "false");
  } else {
    element.removeAttribute("href");
    element.classList.add("disabled");
    element.setAttribute("aria-disabled", "true");
  }
}

function detailRow(title, value, subtitle) {
  return `
    <div class="detail-row">
      <span class="detail-main">
        <span class="detail-title">${escapeHTML(title)}</span>
        <span class="detail-subtitle">${escapeHTML(value || "")}</span>
      </span>
      ${subtitle ? `<span class="badge neutral">${escapeHTML(subtitle)}</span>` : ""}
    </div>
  `;
}

function definitionRows(values) {
  return Object.entries(values)
    .filter(([, value]) => value !== "")
    .map(([key, value]) => `<dt>${escapeHTML(key)}</dt><dd>${escapeHTML(value)}</dd>`)
    .join("");
}

function metricHTML([label, value, note]) {
  return `
    <article class="metric">
      <div class="metric-label">${escapeHTML(label)}</div>
      <div class="metric-value">${escapeHTML(value)}</div>
      <div class="metric-note">${escapeHTML(note)}</div>
    </article>
  `;
}

function statusBadge(status, id = "") {
  const value = status || "unknown";
  const kind = value === "succeeded" ? "ok" : value === "failed" ? "danger" : value === "running" ? "warn" : "neutral";
  const idAttr = id ? ` id="${escapeAttr(id)}"` : "";
  return `<span${idAttr} class="badge ${kind}">${escapeHTML(value)}</span>`;
}

function passBadge(row) {
  if (row.exception_class) return `<span class="badge danger">exception</span>`;
  if (row.pass === true) return `<span class="badge ok">passed</span>`;
  if (row.pass === false) return `<span class="badge danger">failed</span>`;
  return `<span class="badge neutral">not run</span>`;
}

function buildWeaveUrl(traceProject) {
  if (!traceProject) return "";
  const base = (state.status?.wandb_app_base_url || "https://wandb.ai").replace(/\/+$/, "");
  return `${base}/${traceProject}/weave`;
}

function commandText(command) {
  return Array.isArray(command) ? command.map(shellQuote).join(" ") : String(command || "");
}

function shellQuote(value) {
  const text = String(value);
  if (/^[A-Za-z0-9_./:=,@+-]+$/.test(text)) return text;
  return `'${text.replaceAll("'", "'\\''")}'`;
}

function formatPercent(value) {
  if (value === null || value === undefined) return "n/a";
  return `${Math.round(Number(value) * 100)}%`;
}

function formatCurrency(value) {
  const number = Number(value || 0);
  return `$${number.toFixed(number >= 1 ? 2 : 4)}`;
}

function formatInteger(value) {
  return new Intl.NumberFormat().format(Number(value || 0));
}

function formatDuration(value) {
  if (value === null || value === undefined || value === "") return "";
  const seconds = Number(value);
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function formatValue(value) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(value) {
  return escapeHTML(value).replaceAll("'", "&#39;");
}

function copyText(text) {
  if (!text) return;
  navigator.clipboard?.writeText(text).then(
    () => showToast("Run key copied"),
    () => showToast("Could not copy run key"),
  );
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => {
    toast.hidden = true;
  }, 2400);
}

function showError(error) {
  console.error(error);
  showToast(error.message || String(error));
}

boot();
