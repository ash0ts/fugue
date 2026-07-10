const $ = (id) => document.getElementById(id);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const TABS = { run: "Run", compare: "Compare", setup: "Setup" };
const DEFAULT_MANIFEST = "datasets/pilot.yaml";
const MEMORY_OPTIONS = ["none", "agentsmd", "openwiki", "semsearch", "deepwiki"];

const state = {
  activeJob: null,
  activeJobId: null,
  activeTab: "run",
  editor: { kind: "prompt", variantId: null },
  advancedVariantId: null,
  eventSource: null,
  experiment: null,
  experimentId: "pilot",
  initializedControls: false,
  library: { prompts: [], skills: [], experiments: [] },
  manifest: null,
  preview: null,
  previewTimer: null,
  resultFilter: "all",
  results: { rows: [], summary: {} },
  selectedHarnesses: new Set(),
  status: null,
  summary: null,
};

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

async function putJSON(url, body = {}) {
  const response = await fetch(url, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorText(response));
  return response.json();
}

async function postJSON(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorText(response));
  return response.json();
}

async function errorText(response) {
  try {
    const data = await response.json();
    return data.detail || `${response.url} returned ${response.status}`;
  } catch {
    return `${response.url} returned ${response.status}`;
  }
}

function boot() {
  bindNavigation();
  bindActions();
  if (!location.hash || !TABS[location.hash.slice(1)]) location.hash = "run";
  setActiveTab(tabFromHash());
  refreshAll().catch(showError);
}

function bindNavigation() {
  $$("[data-tab-target]").forEach((button) => {
    button.addEventListener("click", () => {
      location.hash = button.dataset.tabTarget || "run";
    });
  });
  window.addEventListener("hashchange", () => setActiveTab(tabFromHash()));
}

function bindActions() {
  $("refreshBtn").addEventListener("click", () => refreshAll().catch(showError));
  $("preflightBtn").addEventListener("click", () => startJob("preflight"));
  $("bridgeBtn").addEventListener("click", () => startJob("bridge"));
  $("renderBtn").addEventListener("click", () => renderPreview(true).catch(showError));
  $("runBtn").addEventListener("click", () => startJob("run"));
  $("exportBtn").addEventListener("click", () => startJob("export"));
  $("saveExperimentBtn").addEventListener("click", saveExperiment);
  $("addVariantBtn").addEventListener("click", addVariant);
  $("experimentSelect").addEventListener("change", () => loadExperiment($("experimentSelect").value));
  $("closeEditorBtn").addEventListener("click", closeEditor);
  $("saveEditorBtn").addEventListener("click", saveEditor);
  $("closeAdvancedBtn").addEventListener("click", closeAdvanced);
  $("saveAdvancedBtn").addEventListener("click", saveAdvanced);

  ["experimentIdInput", "experimentNameInput", "modelInput", "manifestInput", "taskInput", "attemptInput", "concurrentInput", "dryRunInput"].forEach((id) => {
    $(id).addEventListener("input", () => {
      syncExperimentFromControls();
      renderCommandPreview();
      schedulePreview();
    });
    $(id).addEventListener("change", () => {
      syncExperimentFromControls();
      renderCommandPreview();
      schedulePreview();
    });
  });

  document.addEventListener("input", (event) => {
    const field = event.target.closest("[data-variant-field]");
    if (field) {
      updateVariantField(field);
      renderRun();
      schedulePreview();
    }
  });
  document.addEventListener("change", (event) => {
    const field = event.target.closest("[data-variant-field]");
    if (field) {
      updateVariantField(field);
      renderRun();
      schedulePreview();
      return;
    }
    const harness = event.target.closest("[data-toggle-kind='harness']");
    if (harness) {
      toggleHarness(harness.dataset.toggleValue);
      renderRun();
      schedulePreview();
    }
  });
  document.addEventListener("click", (event) => {
    const toggle = event.target.closest("[data-toggle-kind='harness']");
    if (toggle) {
      toggleHarness(toggle.dataset.toggleValue);
      renderRun();
      schedulePreview();
      return;
    }
    const action = event.target.closest("[data-variant-action]");
    if (action) {
      handleVariantAction(action.dataset.variantAction, action.dataset.variantId);
      return;
    }
    const filter = event.target.closest("[data-result-filter]");
    if (filter) {
      state.resultFilter = filter.dataset.resultFilter || "all";
      renderCompare();
      return;
    }
    const jobButton = event.target.closest("[data-job-id]");
    if (jobButton) {
      loadJob(jobButton.dataset.jobId).catch(showError);
      return;
    }
    const copyButton = event.target.closest("[data-copy]");
    if (copyButton) copyText(copyButton.dataset.copy || "");
  });
}

async function refreshAll() {
  const [summary, jobs, results, library] = await Promise.all([
    getJSON("/api/summary"),
    getJSON("/api/jobs"),
    getJSON("/api/results"),
    getJSON("/api/library"),
  ]);
  state.summary = summary;
  state.status = summary.status || null;
  state.manifest = summary.manifest || null;
  state.jobs = jobs || [];
  state.results = results || { rows: [], summary: {} };
  state.library = normalizeLibrary(library);
  const firstExperiment = state.library.experiments[0]?.id || "pilot";
  await loadExperiment(state.experimentId || firstExperiment, { renderAfter: false });
  syncControls();
  renderAll();
  schedulePreview();
}

function normalizeLibrary(library) {
  return {
    prompts: library?.prompts || [],
    skills: library?.skills || [],
    experiments: library?.experiments || [],
  };
}

async function loadExperiment(experimentId, options = {}) {
  if (!experimentId) return;
  state.experimentId = experimentId;
  try {
    const payload = await getJSON(`/api/experiments/${encodeURIComponent(experimentId)}`);
    state.experiment = payload.experiment;
    state.experiment.variants = normalizeVariants(state.experiment.variants);
    state.initializedControls = false;
    if (options.renderAfter !== false) {
      syncControls();
      renderAll();
      schedulePreview();
    }
  } catch (error) {
    state.experiment = defaultExperiment(experimentId);
    state.initializedControls = false;
    if (options.renderAfter !== false) {
      syncControls();
      renderAll();
      schedulePreview();
    }
    if (!options.quiet) showToast(`Created local draft for ${experimentId}`);
  }
}

function defaultExperiment(experimentId = "pilot") {
  return {
    id: experimentId,
    title: experimentId,
    run_name: experimentId,
    manifest: DEFAULT_MANIFEST,
    model: state.status?.route?.model || state.manifest?.model || state.status?.default_model || "",
    harnesses: manifestHarnesses(),
    variants: normalizeVariants([]),
    n_attempts: state.manifest?.k || 1,
    n_concurrent: state.manifest?.n_concurrent || 2,
  };
}

function normalizeVariants(variants) {
  if (Array.isArray(variants) && variants.length) {
    return variants.map((variant, index) => ({
      id: variant.id || `variant-${index + 1}`,
      label: variant.label || variant.id || `Variant ${index + 1}`,
      prompt_id: variant.prompt_id || "",
      skill_ids: Array.isArray(variant.skill_ids) ? variant.skill_ids : [],
      memory: variant.memory || "none",
      agent_kwargs: variant.agent_kwargs || {},
      agent_env: variant.agent_env || {},
      mcp_servers: variant.mcp_servers || [],
      environment: variant.environment || {},
      verifier: variant.verifier || {},
      retry: variant.retry || {},
      artifacts: variant.artifacts || [],
      enabled: variant.enabled !== false,
    }));
  }
  return [
    { id: "baseline", label: "Baseline", prompt_id: "", skill_ids: [], memory: "none", enabled: true },
    { id: "prompt-skill", label: "Prompt + skill", prompt_id: state.library.prompts[0]?.id || "", skill_ids: state.library.skills[0]?.id ? [state.library.skills[0].id] : [], memory: "none", enabled: true },
    { id: "memory-skill", label: "Memory + skill", prompt_id: state.library.prompts[0]?.id || "", skill_ids: state.library.skills[0]?.id ? [state.library.skills[0].id] : [], memory: "agentsmd", enabled: true },
  ];
}

function syncControls() {
  const experiment = state.experiment || {};
  if (!state.initializedControls) {
    $("experimentSelect").value = state.experimentId;
    $("experimentIdInput").value = experiment.id || state.experimentId || "pilot";
    $("experimentNameInput").value = experiment.run_name || experiment.title || "";
    $("modelInput").value = experiment.model || state.status?.route?.model || state.manifest?.model || state.status?.default_model || "";
    $("manifestInput").value = experiment.manifest || DEFAULT_MANIFEST;
    $("attemptInput").value = experiment.n_attempts || state.manifest?.k || 1;
    $("concurrentInput").value = experiment.n_concurrent || state.manifest?.n_concurrent || 2;
    $("taskInput").value = experiment.n_tasks || "";
    state.selectedHarnesses = new Set(experiment.harnesses?.length ? experiment.harnesses : manifestHarnesses());
    state.initializedControls = true;
  }
}

function syncExperimentFromControls() {
  if (!state.experiment) state.experiment = {};
  state.experiment.id = $("experimentIdInput").value.trim() || state.experimentId || "pilot";
  state.experiment.title = $("experimentNameInput").value.trim() || state.experiment.id;
  state.experiment.run_name = $("experimentNameInput").value.trim();
  state.experiment.model = $("modelInput").value.trim();
  state.experiment.manifest = $("manifestInput").value.trim() || DEFAULT_MANIFEST;
  state.experiment.n_attempts = numberOrNull($("attemptInput").value);
  state.experiment.n_concurrent = numberOrNull($("concurrentInput").value);
  state.experiment.n_tasks = numberOrNull($("taskInput").value);
  state.experiment.harnesses = Array.from(state.selectedHarnesses);
  state.experiment.variants = normalizeVariants(state.experiment.variants);
}

function renderAll() {
  renderHeader();
  renderRun();
  renderCompare();
  renderSetup();
}

function tabFromHash() {
  const name = location.hash.replace("#", "");
  return TABS[name] ? name : "run";
}

function setActiveTab(tab) {
  state.activeTab = TABS[tab] ? tab : "run";
  $("pageTitle").textContent = TABS[state.activeTab];
  $$("[data-tab-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.tabPanel !== state.activeTab;
  });
  $$("[data-tab-target]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tabTarget === state.activeTab);
  });
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

function renderRun() {
  fillSelect($("experimentSelect"), state.library.experiments, state.experimentId);
  renderHarnessToggles();
  renderVariantRows();
  renderRunSummary();
  renderCommandPreview();
  renderRenderedConfigs();
  renderActivity();
}

function renderHarnessToggles() {
  const harnesses = manifestHarnesses();
  if (!state.selectedHarnesses.size) harnesses.forEach((value) => state.selectedHarnesses.add(value));
  $("harnessToggles").innerHTML = harnesses.map((value) => {
    const active = state.selectedHarnesses.has(value);
    return `<button class="toggle-chip ${active ? "active" : ""}" data-toggle-kind="harness" data-toggle-value="${escapeAttr(value)}" aria-pressed="${String(active)}" type="button">${escapeHTML(value)}</button>`;
  }).join("");
}

function renderVariantRows() {
  if (!state.experiment) state.experiment = defaultExperiment(state.experimentId || "pilot");
  const variants = normalizeVariants(state.experiment?.variants);
  state.experiment.variants = variants;
  $("variantRows").innerHTML = variants.map((variant) => `
    <tr data-variant-id="${escapeAttr(variant.id)}">
      <td><input data-variant-field="enabled" data-variant-id="${escapeAttr(variant.id)}" type="checkbox" ${variant.enabled ? "checked" : ""} /></td>
      <td>
        <input class="table-input strong" data-variant-field="label" data-variant-id="${escapeAttr(variant.id)}" value="${escapeAttr(variant.label)}" />
        <input class="table-input code" data-variant-field="id" data-variant-id="${escapeAttr(variant.id)}" value="${escapeAttr(variant.id)}" />
      </td>
      <td>
        <select class="table-input" data-variant-field="prompt_id" data-variant-id="${escapeAttr(variant.id)}">
          <option value="">No prompt</option>
          ${state.library.prompts.map((item) => `<option value="${escapeAttr(item.id)}" ${item.id === variant.prompt_id ? "selected" : ""}>${escapeHTML(item.title || item.id)}</option>`).join("")}
        </select>
        <button class="inline-link" data-variant-action="edit-prompt" data-variant-id="${escapeAttr(variant.id)}" type="button">Edit prompt</button>
      </td>
      <td>
        <input class="table-input code" data-variant-field="skill_ids" data-variant-id="${escapeAttr(variant.id)}" value="${escapeAttr((variant.skill_ids || []).join(","))}" placeholder="skill-a,skill-b" />
        <button class="inline-link" data-variant-action="edit-skill" data-variant-id="${escapeAttr(variant.id)}" type="button">Edit skills</button>
      </td>
      <td>
        <select class="table-input" data-variant-field="memory" data-variant-id="${escapeAttr(variant.id)}">
          ${MEMORY_OPTIONS.map((value) => `<option value="${escapeAttr(value)}" ${value === (variant.memory || "none") ? "selected" : ""}>${escapeHTML(value)}</option>`).join("")}
        </select>
      </td>
      <td><span class="badge ${hasAdvancedConfig(variant) ? "ok" : "neutral"}">${hasAdvancedConfig(variant) ? "custom" : "default"}</span></td>
      <td class="table-actions">
        <button class="icon-button" data-variant-action="advanced" data-variant-id="${escapeAttr(variant.id)}" type="button">Config</button>
        <button class="icon-button" data-variant-action="duplicate" data-variant-id="${escapeAttr(variant.id)}" type="button">Duplicate</button>
      </td>
    </tr>
  `).join("");
}

function renderRunSummary() {
  const preview = state.preview?.summary;
  const variants = enabledVariants();
  const taskCount = numberOrNull($("taskInput").value) || state.manifest?.counts?.tasks || 0;
  const attempts = numberOrNull($("attemptInput").value) || 1;
  const cells = preview?.cells || state.selectedHarnesses.size * variants.length;
  const trials = preview?.estimated_trials || cells * taskCount * attempts;
  $("runSummaryCards").innerHTML = [
    ["Cells", String(cells), `${state.selectedHarnesses.size} harnesses x ${variants.length} variants`],
    ["Estimated trials", String(trials), `${taskCount} tasks x ${attempts} trials`],
    ["Prompts", String(new Set(variants.map((v) => v.prompt_id).filter(Boolean)).size), "selected prompt files"],
    ["Skills", String(new Set(variants.flatMap((v) => v.skill_ids || [])).size), "selected Harbor skills"],
  ].map(metricHTML).join("");
  $("readinessStrip").innerHTML = readinessItems().map(([label, ok, note]) => `<span class="badge ${ok ? "ok" : "danger"}">${escapeHTML(label)} ${ok ? "ready" : note}</span>`).join("");
}

function renderCommandPreview() {
  $("commandPreview").textContent = buildRunCommand().join(" ");
}

function renderRenderedConfigs() {
  const commands = state.preview?.commands || [];
  $("previewBadge").textContent = commands.length ? `${commands.length} configs` : "Not rendered";
  $("previewBadge").className = `badge ${commands.length ? "ok" : "neutral"}`;
  $("renderedConfigs").innerHTML = commands.length ? commands.slice(0, 8).map((item) => `
    <div class="config-item">
      <div><strong>${escapeHTML(item.harness)}</strong><span>${escapeHTML(item.variant_label || item.variant_id)}</span></div>
      <code>${escapeHTML(item.config_path)}</code>
    </div>
  `).join("") : `<div class="empty-state compact">Render configs to inspect generated Harbor files.</div>`;
}

function renderActivity() {
  const latest = state.jobs?.[0];
  $("activityPanel").innerHTML = latest ? `
    ${detailRow("Latest job", latest.id, latest.kind)}
    ${detailRow("Status", latest.status || "unknown", latest.returncode === undefined ? "running" : `exit ${latest.returncode}`)}
    ${detailRow("Command", commandText(latest.command), "")}
  ` : `<div class="empty-state compact">No jobs yet. Run preflight or launch a dry experiment.</div>`;
}

function renderCompare() {
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
    ...groupCards("Variant", summary.by_variant_id || summary.by_variant || []),
    ...groupCards("Harness", summary.by_harness || []),
    ...groupCards("Prompt", summary.by_prompt || []),
    ...groupCards("Skill", summary.by_skill || []),
    ...groupCards("Provider", summary.by_provider || []),
  ].slice(0, 10);
  $("resultCards").innerHTML = groups.length ? groups.map((group) => `
    <article class="result-card">
      <div class="result-card-label">${escapeHTML(group.scope)}</div>
      <div class="result-card-value">${escapeHTML(formatPercent(group.pass_rate))}</div>
      <div class="result-card-note">${escapeHTML(group.name)} - ${escapeHTML(group.passed || 0)}/${escapeHTML(group.total || 0)} passed</div>
    </article>
  `).join("") : metricHTML(["Rows", "0", "No exported trials"]);
}

function renderResultFilters(rows) {
  const filters = [
    ["all", `All ${rows.length}`],
    ["passed", `Passed ${rows.filter((row) => row.pass === true).length}`],
    ["failed", `Failed ${rows.filter((row) => row.pass === false).length}`],
    ["exceptions", `Exceptions ${rows.filter((row) => row.exception_class).length}`],
  ];
  $("resultFilters").innerHTML = filters.map(([key, label]) => `<button class="filter-chip ${state.resultFilter === key ? "active" : ""}" data-result-filter="${escapeAttr(key)}" type="button">${escapeHTML(label)}</button>`).join("");
}

function resultRowHTML(row) {
  const tokens = Number(row.n_input_tokens || 0) + Number(row.n_cache_tokens || 0) + Number(row.n_output_tokens || 0);
  const weaveUrl = row.weave_url || buildWeaveUrl(row.trace_project);
  return `
    <tr>
      <td>${escapeHTML(row.experiment_id || row.run_name || "")}</td>
      <td class="code-cell">${escapeHTML(row.variant_id || "")}</td>
      <td>${escapeHTML(row.prompt_id || "")}</td>
      <td>${escapeHTML((row.skill_ids || []).join(", "))}</td>
      <td>${escapeHTML(row.harness || "")}</td>
      <td class="code-cell">${escapeHTML(row.model || "")}</td>
      <td>${passBadge(row)}</td>
      <td>${escapeHTML(formatValue(row.reward))}</td>
      <td>${escapeHTML(formatInteger(tokens))}</td>
      <td>${escapeHTML(formatCurrency(row.cost_usd))}</td>
      <td class="code-cell"><button class="copy-key" data-copy="${escapeAttr(row.run_key || "")}" type="button">${escapeHTML(row.run_key || "")}</button></td>
      <td>${weaveUrl ? `<a href="${escapeAttr(weaveUrl)}" target="_blank" rel="noreferrer">Open Weave</a>` : ""}</td>
    </tr>
  `;
}

function renderSetup() {
  renderReadinessCards();
  const keys = state.status?.keys || {};
  $("setupKeys").innerHTML = Object.entries(keys).map(([key, present]) => `
    <div class="key-row"><span class="code-cell">${escapeHTML(key)}</span><span class="badge ${present ? "ok" : "danger"}">${present ? "present" : "missing"}</span></div>
  `).join("");
  const bridge = state.status?.bridge || {};
  $("setupBridge").innerHTML = `
    ${detailRow("Status", bridge.ok ? "Online" : "Offline", bridge.url || "127.0.0.1:4000")}
    ${detailRow("Next action", bridge.ok ? "Run experiment" : "Start bridge", bridge.error || "")}
  `;
  const route = state.status?.route || {};
  $("setupRoute").innerHTML = definitionRows({
    Provider: route.provider || "unknown",
    Model: route.model || "",
    "Model key": route.api_key_env || "",
  });
  const manifest = state.manifest || {};
  $("setupManifest").innerHTML = definitionRows({
    Dataset: manifest.dataset?.ref || "",
    Tasks: manifest.counts?.tasks || 0,
    Harnesses: manifestHarnesses().join(", "),
    "Jobs dir": manifest.jobs_dir || "",
  });
}

function renderReadinessCards() {
  const readiness = state.summary?.readiness || {};
  const status = state.status || {};
  const cards = [
    ["Trace", readiness.trace ? "Ready" : "Missing", status.trace_project || "Set W&B tracing", readiness.trace],
    ["Model key", readiness.model ? "Ready" : "Missing", status.route?.api_key_env || "Select a model", readiness.model],
    ["Bridge", readiness.bridge ? "Online" : "Offline", readiness.bridge ? "127.0.0.1:4000" : "Start bridge for bridged runs", readiness.bridge],
    ["Manifest", readiness.manifest ? "Loaded" : "Missing", `${state.manifest?.counts?.tasks || 0} tasks`, readiness.manifest],
  ];
  $("readinessCards").innerHTML = cards.map(([label, value, note, ok]) => `
    <article class="stat-card">
      <div class="stat-top"><span class="stat-label">${escapeHTML(label)}</span><span class="badge ${ok ? "ok" : "danger"}">${escapeHTML(value)}</span></div>
      <div class="stat-value">${escapeHTML(value)}</div>
      <div class="stat-note">${escapeHTML(note)}</div>
    </article>
  `).join("");
}

function addVariant() {
  syncExperimentFromControls();
  const base = `variant-${state.experiment.variants.length + 1}`;
  state.experiment.variants.push({
    id: uniqueVariantId(base),
    label: `Variant ${state.experiment.variants.length + 1}`,
    prompt_id: state.library.prompts[0]?.id || "",
    skill_ids: [],
    memory: "none",
    enabled: true,
  });
  renderRun();
  schedulePreview();
}

function duplicateVariant(variantId) {
  const variant = findVariant(variantId);
  if (!variant) return;
  const copy = JSON.parse(JSON.stringify(variant));
  copy.id = uniqueVariantId(`${variant.id}-copy`);
  copy.label = `${variant.label} copy`;
  state.experiment.variants.push(copy);
  renderRun();
  schedulePreview();
}

function updateVariantField(field) {
  const variant = findVariant(field.dataset.variantId);
  if (!variant) return;
  const key = field.dataset.variantField;
  if (key === "enabled") variant.enabled = field.checked;
  else if (key === "skill_ids") variant.skill_ids = csv(field.value);
  else if (key === "id") variant.id = sanitizeId(field.value);
  else variant[key] = field.value;
}

function handleVariantAction(action, variantId) {
  if (action === "duplicate") duplicateVariant(variantId);
  if (action === "edit-prompt") openPromptEditor(variantId);
  if (action === "edit-skill") openSkillEditor(variantId);
  if (action === "advanced") openAdvanced(variantId);
}

async function openPromptEditor(variantId) {
  const variant = findVariant(variantId);
  state.editor = { kind: "prompt", variantId };
  const promptId = variant?.prompt_id || uniquePromptId(variantId);
  $("editorKicker").textContent = "Prompt";
  $("editorTitle").textContent = "Edit prompt";
  $("editorIdInput").value = promptId;
  $("editorBody").value = promptId ? await loadText(`/api/prompts/${encodeURIComponent(promptId)}`) : "# New prompt\n";
  $("editorModal").hidden = false;
}

async function openSkillEditor(variantId) {
  const variant = findVariant(variantId);
  state.editor = { kind: "skill", variantId };
  const skillId = variant?.skill_ids?.[0] || uniqueSkillId(variantId);
  $("editorKicker").textContent = "Skill";
  $("editorTitle").textContent = "Edit Harbor skill";
  $("editorIdInput").value = skillId;
  $("editorBody").value = skillId ? await loadText(`/api/skills/${encodeURIComponent(skillId)}`) : "# New skill\n";
  $("editorModal").hidden = false;
}

async function loadText(url) {
  try {
    const data = await getJSON(url);
    return data.body || "# New file\n";
  } catch {
    return "# New file\n";
  }
}

function closeEditor() {
  $("editorModal").hidden = true;
}

async function saveEditor() {
  const id = sanitizeId($("editorIdInput").value);
  const body = $("editorBody").value;
  const variant = findVariant(state.editor.variantId);
  if (state.editor.kind === "prompt") {
    await putJSON(`/api/prompts/${encodeURIComponent(id)}`, { body });
    if (variant) variant.prompt_id = id;
    showToast("Prompt saved");
  } else {
    await putJSON(`/api/skills/${encodeURIComponent(id)}`, { body });
    if (variant && !variant.skill_ids.includes(id)) variant.skill_ids = [id, ...variant.skill_ids];
    showToast("Skill saved");
  }
  closeEditor();
  await refreshLibraryOnly();
  renderRun();
  schedulePreview();
}

function openAdvanced(variantId) {
  const variant = findVariant(variantId);
  if (!variant) return;
  state.advancedVariantId = variantId;
  $("advancedTitle").textContent = `Advanced config: ${variant.label}`;
  $("advancedBody").value = JSON.stringify({
    agent_kwargs: variant.agent_kwargs || {},
    agent_env: variant.agent_env || {},
    mcp_servers: variant.mcp_servers || [],
    environment: variant.environment || {},
    verifier: variant.verifier || {},
    retry: variant.retry || {},
    artifacts: variant.artifacts || [],
  }, null, 2);
  $("advancedModal").hidden = false;
}

function closeAdvanced() {
  $("advancedModal").hidden = true;
}

function saveAdvanced() {
  const variant = findVariant(state.advancedVariantId);
  if (!variant) return;
  try {
    const data = JSON.parse($("advancedBody").value || "{}");
    ["agent_kwargs", "agent_env", "environment", "verifier", "retry"].forEach((key) => {
      variant[key] = data[key] || {};
    });
    variant.mcp_servers = Array.isArray(data.mcp_servers) ? data.mcp_servers : [];
    variant.artifacts = Array.isArray(data.artifacts) ? data.artifacts : [];
    closeAdvanced();
    renderRun();
    schedulePreview();
    showToast("Variant config saved");
  } catch (error) {
    showError(error);
  }
}

async function saveExperiment() {
  syncExperimentFromControls();
  const id = sanitizeId($("experimentIdInput").value || state.experimentId || "pilot");
  state.experiment.id = id;
  const payload = await putJSON(`/api/experiments/${encodeURIComponent(id)}`, { experiment: state.experiment });
  state.experiment = payload.experiment;
  state.experimentId = id;
  showToast("Experiment saved");
  await refreshLibraryOnly();
  renderAll();
}

async function refreshLibraryOnly() {
  state.library = normalizeLibrary(await getJSON("/api/library"));
}

function schedulePreview() {
  clearTimeout(state.previewTimer);
  state.previewTimer = setTimeout(() => renderPreview(false).catch(() => {}), 400);
}

async function renderPreview(toast = false) {
  syncExperimentFromControls();
  const data = await postJSON(toast ? "/api/render" : "/api/preview", runBody());
  state.preview = data;
  renderRunSummary();
  renderRenderedConfigs();
  if (toast) showToast("Configs rendered");
}

async function startJob(kind) {
  try {
    setActionBusy(true);
    const config = jobConfig(kind);
    const job = await postJSON(config.url, config.body);
    state.activeJob = job;
    state.activeJobId = job.id;
    showToast(`${config.label} started`);
    await refreshJobsOnly();
    if (kind === "run") streamJob(job.id);
  } catch (error) {
    showError(error);
  } finally {
    setActionBusy(false);
  }
}

function jobConfig(kind) {
  const model = $("modelInput").value.trim();
  if (kind === "preflight") return { label: "Run preflight", url: "/api/preflight", body: { model } };
  if (kind === "bridge") return { label: "Start bridge", url: "/api/bridge/up", body: { model } };
  if (kind === "export") return { label: "Export results", url: "/api/export", body: {} };
  return { label: "Run experiment", url: "/api/run", body: runBody() };
}

function runBody() {
  syncExperimentFromControls();
  return {
    experiment_id: state.experiment.id || state.experimentId || "pilot",
    experiment: state.experiment,
    manifest: state.experiment.manifest || DEFAULT_MANIFEST,
    model: state.experiment.model || "",
    run_name: state.experiment.run_name || state.experiment.title || state.experiment.id || "",
    harnesses: Array.from(state.selectedHarnesses),
    variant_ids: enabledVariants().map((variant) => variant.id),
    n_tasks: state.experiment.n_tasks,
    n_attempts: state.experiment.n_attempts,
    n_concurrent: state.experiment.n_concurrent,
    dry_run: $("dryRunInput").checked,
  };
}

async function refreshJobsOnly() {
  state.jobs = await getJSON("/api/jobs");
  renderActivity();
}

async function loadJob(jobId) {
  if (!jobId) return;
  closeStream();
  state.activeJobId = jobId;
  state.activeJob = await getJSON(`/api/jobs/${encodeURIComponent(jobId)}`);
  renderActivity();
  if (state.activeJob.status === "running") streamJob(jobId);
}

function streamJob(jobId) {
  closeStream();
  state.eventSource = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
  state.eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
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

function buildRunCommand() {
  const body = runBody();
  const parts = ["fugue", "run", "--experiment", body.experiment_id, "--manifest", body.manifest];
  if (body.model) parts.push("--model", body.model);
  if (body.run_name) parts.push("--run-name", body.run_name);
  if (body.harnesses.length) parts.push("--harnesses", body.harnesses.join(","));
  if (body.variant_ids.length) parts.push("--variants", body.variant_ids.join(","));
  if (body.n_tasks) parts.push("-l", String(body.n_tasks));
  if (body.n_attempts) parts.push("-k", String(body.n_attempts));
  if (body.n_concurrent) parts.push("-n", String(body.n_concurrent));
  if (body.dry_run) parts.push("--dry-run");
  return parts.map(shellQuote);
}

function enabledVariants() {
  return normalizeVariants(state.experiment?.variants).filter((variant) => variant.enabled);
}

function findVariant(variantId) {
  return normalizeVariants(state.experiment?.variants).find((variant) => variant.id === variantId);
}

function toggleHarness(value) {
  if (!value) return;
  if (state.selectedHarnesses.has(value) && state.selectedHarnesses.size > 1) state.selectedHarnesses.delete(value);
  else state.selectedHarnesses.add(value);
  syncExperimentFromControls();
}

function hasAdvancedConfig(variant) {
  return Boolean(
    Object.keys(variant.agent_kwargs || {}).length ||
    Object.keys(variant.agent_env || {}).length ||
    (variant.mcp_servers || []).length ||
    Object.keys(variant.environment || {}).length ||
    Object.keys(variant.verifier || {}).length ||
    Object.keys(variant.retry || {}).length ||
    (variant.artifacts || []).length
  );
}

function readinessItems() {
  const readiness = state.summary?.readiness || {};
  return [
    ["trace", readiness.trace, "missing"],
    ["model", readiness.model, "missing"],
    ["manifest", readiness.manifest, "missing"],
  ];
}

function rowMatchesFilter(row) {
  if (state.resultFilter === "passed") return row.pass === true;
  if (state.resultFilter === "failed") return row.pass === false;
  if (state.resultFilter === "exceptions") return Boolean(row.exception_class);
  return true;
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

function manifestHarnesses() {
  return (state.manifest?.harnesses || []).map((harness) => harness.name).filter(Boolean);
}

function numberOrNull(value) {
  return value === "" || value === null || value === undefined ? null : Number(value);
}

function csv(value) {
  return String(value || "").split(",").map((part) => part.trim()).filter(Boolean);
}

function sanitizeId(value) {
  const id = String(value || "").trim().replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-+|-+$/g, "");
  return id || "item";
}

function uniqueVariantId(base) {
  const existing = new Set(normalizeVariants(state.experiment?.variants).map((variant) => variant.id));
  let id = sanitizeId(base);
  let index = 2;
  while (existing.has(id)) {
    id = `${sanitizeId(base)}-${index}`;
    index += 1;
  }
  return id;
}

function uniquePromptId(variantId) {
  return sanitizeId(`${variantId || "variant"}-prompt`);
}

function uniqueSkillId(variantId) {
  return sanitizeId(`${variantId || "variant"}-skill`);
}

function setActionBusy(busy) {
  ["preflightBtn", "bridgeBtn", "renderBtn", "runBtn", "exportBtn", "saveExperimentBtn", "addVariantBtn", "saveEditorBtn", "saveAdvancedBtn"].forEach((id) => {
    const el = $(id);
    if (el) el.disabled = busy;
  });
}

function fillSelect(element, items, selected) {
  if (!element) return;
  const options = [...(items || [])];
  if (selected && !options.some((item) => item.id === selected)) {
    options.unshift({ id: selected, title: `${selected} (draft)` });
  }
  element.innerHTML = options.map((item) => `<option value="${escapeAttr(item.id)}">${escapeHTML(item.title || item.id)}</option>`).join("");
  if (selected && options.some((item) => item.id === selected)) element.value = selected;
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
  return `<div class="detail-row"><span class="detail-main"><span class="detail-title">${escapeHTML(title)}</span><span class="detail-subtitle">${escapeHTML(value || "")}</span></span>${subtitle ? `<span class="badge neutral">${escapeHTML(subtitle)}</span>` : ""}</div>`;
}

function definitionRows(values) {
  return Object.entries(values).filter(([, value]) => value !== "").map(([key, value]) => `<dt>${escapeHTML(key)}</dt><dd>${escapeHTML(value)}</dd>`).join("");
}

function metricHTML([label, value, note]) {
  return `<article class="metric"><div class="metric-label">${escapeHTML(label)}</div><div class="metric-value">${escapeHTML(value)}</div><div class="metric-note">${escapeHTML(note)}</div></article>`;
}

function passBadge(row) {
  if (row.pass === true) return `<span class="badge ok">pass</span>`;
  if (row.pass === false) return `<span class="badge danger">fail</span>`;
  if (row.exception_class) return `<span class="badge danger">error</span>`;
  return `<span class="badge neutral">unknown</span>`;
}

function commandText(command) {
  return Array.isArray(command) ? command.map(shellQuote).join(" ") : String(command || "");
}

function shellQuote(value) {
  const text = String(value);
  return /^[A-Za-z0-9_/:=.,@+-]+$/.test(text) ? text : `'${text.replaceAll("'", "'\\''")}'`;
}

function buildWeaveUrl(traceProject) {
  if (!traceProject) return "";
  const base = state.status?.wandb_app_base_url || "https://wandb.ai";
  return `${base.replace(/\/$/, "")}/${traceProject}/weave`;
}

function formatPercent(value) {
  return value === null || value === undefined ? "n/a" : `${Math.round(Number(value) * 100)}%`;
}

function formatInteger(value) {
  return Number(value || 0).toLocaleString();
}

function formatCurrency(value) {
  const number = Number(value || 0);
  return number ? `$${number.toFixed(4)}` : "$0";
}

function formatValue(value) {
  return value === null || value === undefined ? "" : String(value);
}

function escapeHTML(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHTML(value);
}

function copyText(text) {
  navigator.clipboard?.writeText(text).then(
    () => showToast("Run key copied"),
    () => showToast("Copy failed"),
  );
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    toast.hidden = true;
  }, 2600);
}

function showError(error) {
  console.error(error);
  showToast(error.message || String(error));
}

document.addEventListener("DOMContentLoaded", boot);
