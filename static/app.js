const state = {
  documents: [],
  documentId: null,
  pipeline: "relay",
  structure: null,
  tab: "retrieve",
};

const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

function fmtJson(obj) {
  return JSON.stringify(obj, null, 2);
}

function pipelineClass(id) {
  return id;
}

async function refreshDocs() {
  state.documents = await api("/api/documents");
  const box = $("docs");
  box.innerHTML = "";
  for (const doc of state.documents) {
    const el = document.createElement("div");
    el.className = "doc" + (doc.id === state.documentId ? " active" : "");
    const runs = (doc.runs || []).map((r) => `${r.pipeline_id}:${r.status}`).join(" · ") || "not processed";
    const route = doc.route || (doc.metadata && doc.metadata.route) || {};
    const tier = route.tier || "";
    el.innerHTML = `<div class="doc-row"><div><strong>${doc.title || doc.filename}</strong> ${tier ? `<span class="badge-tier pill ${route.pipeline || ""}">${esc(tier)} · ${esc(route.pipeline || "")}</span>` : ""}<small>${doc.filename} · ${doc.page_count} pages<br>${runs}</small></div>
      <button type="button" class="doc-del secondary" data-del="${doc.id}" title="Remove this PDF">Remove</button></div>`;
    el.onclick = (ev) => {
      if (ev.target.closest("[data-del]")) return;
      state.documentId = doc.id;
      applyRoute(doc);
      refreshDocs();
      loadStructure();
    };
    el.querySelector("[data-del]").onclick = (ev) => {
      ev.stopPropagation();
      removeDocument(doc.id, doc.filename);
    };
    box.appendChild(el);
  }
  if (!state.documentId && state.documents[0]) {
    state.documentId = state.documents[0].id;
    applyRoute(state.documents[0]);
    refreshDocs();
    loadStructure();
  }
}

async function removeDocument(id, filename) {
  if (!confirm(`Remove ${filename || id} and its indexes?`)) return;
  $("status").textContent = `Removing ${filename || id}…`;
  try {
    await api(`/api/documents/${id}`, { method: "DELETE" });
    if (state.documentId === id) {
      state.documentId = null;
      state.structure = null;
      $("structure").innerHTML = `<p class="muted">Document removed. Upload a PDF or pick another.</p>`;
      $("workspaceTitle").textContent = "Document";
    }
    $("file").value = "";
    $("status").textContent = `Removed ${filename || id}. Upload a replacement anytime.`;
    await refreshDocs();
  } catch (err) {
    $("status").textContent = String(err);
  }
}

async function uploadPdf() {
  const file = $("file").files[0];
  if (!file) {
    $("file").click();
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  $("status").textContent = `Uploading ${file.name}…`;
  $("uploadBtn").disabled = true;
  try {
    const doc = await api("/api/documents", { method: "POST", body: fd });
    state.documentId = doc.id;
    applyRoute(doc);
    const rec = (doc.route || {}).pipeline || "relay";
    $("status").textContent = `Uploaded ${doc.filename}. Classifier: ${(doc.route || {}).tier || "light"} → ${rec}. Running that pipeline…`;
    await refreshDocs();
    await loadStructure();
    await runSelected();
  } catch (err) {
    $("status").textContent = String(err);
  } finally {
    $("uploadBtn").disabled = false;
  }
}

function applyRoute(doc) {
  const route = doc.route || (doc.metadata && doc.metadata.route);
  const box = $("routeBox");
  if (!route) {
    box.className = "route-box muted";
    box.innerHTML = "Classifier picks <strong>light = Relay</strong> or <strong>heavy = Prism</strong> from layout (no LLM). Baseline is the bake-off control.";
    return;
  }
  box.className = `route-box ${route.tier || ""}`;
  const reasons = (route.reasons || []).map((r) => `• ${esc(r)}`).join("<br>");
  box.innerHTML = `<strong>${esc(route.tier)} → ${esc(route.pipeline)}</strong><br>${reasons}<br><span class="muted">${esc((route.legend || {})[route.tier] || "")}</span>`;
  document.querySelectorAll("input[name=pipe]").forEach((el) => {
    el.checked = el.value === route.pipeline;
  });
}

function showProgress(job) {
  const card = $("progressCard");
  if (!card) return;
  card.hidden = false;
  const pct = job.percent ?? 0;
  $("progressPct").textContent = `${pct}%`;
  $("progressStage").textContent = job.detail ? `${job.stage} — ${job.detail}` : job.stage || "";
  const bar = $("progressBar");
  bar.style.width = `${pct}%`;
  bar.classList.toggle("done", job.status === "ok");
  bar.classList.toggle("failed", job.status === "error");
  $("progressPipes").innerHTML = (job.pipelines || [])
    .map((p) => {
      const cls = p.status === "ok" ? "ok" : p.status === "error" ? "error" : p.status === "running" ? "running" : "";
      return `<li class="${cls}"><span>${esc(p.pipeline_id)} · ${esc(p.stage || p.status)}</span><span>${p.percent}%</span></li>`;
    })
    .join("");
}

async function pollJob(jobId) {
  const started = Date.now();
  while (Date.now() - started < 15 * 60 * 1000) {
    const job = await api(`/api/jobs/${jobId}`);
    showProgress(job);
    $("status").textContent = job.status === "running" || job.status === "queued"
      ? `${job.percent}% · ${job.stage}`
      : job.status === "ok"
        ? `Done (${job.percent}%). ${job.detail || ""}`
        : `Failed: ${job.error || job.detail || "unknown error"}`;
    if (job.status === "ok" || job.status === "error") {
      return job;
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error("Pipeline timed out");
}

async function runSelected() {
  if (!state.documentId) return;
  const names = [...document.querySelectorAll("input[name=pipe]:checked")].map((x) => x.value);
  if (!names.length) {
    $("status").textContent = "Select at least one pipeline.";
    return;
  }
  $("status").textContent = `Starting ${names.join(", ")}…`;
  $("runBtn").disabled = true;
  $("uploadBtn").disabled = true;
  showProgress({
    percent: 0,
    status: "queued",
    stage: "queued",
    pipelines: names.map((id) => ({ pipeline_id: id, status: "queued", percent: 0, stage: "queued" })),
  });
  try {
    const started = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: state.documentId, pipelines: names }),
    });
    const job = await pollJob(started.job_id);
    if (job.status !== "ok") {
      throw new Error(job.error || "pipeline failed");
    }
    await refreshDocs();
    await loadStructure();
  } catch (err) {
    $("status").textContent = String(err);
  } finally {
    $("runBtn").disabled = false;
    $("uploadBtn").disabled = false;
  }
}

async function loadStructure() {
  if (!state.documentId) return;
  const current = state.documents.find((d) => d.id === state.documentId) || {};
  $("workspaceTitle").textContent = current.filename || current.title || "Document";
  try {
    state.structure = await api(`/api/documents/${state.documentId}/structure?pipeline_id=${state.pipeline}`);
  } catch (err) {
    state.structure = null;
    $("structure").innerHTML = `<p class="muted">No ${state.pipeline} run yet. Select the pipeline and click Run.</p>`;
    $("json").textContent = "";
    return;
  }
  render();
}

function render() {
  const s = state.structure;
  if (s) {
    if (state.tab === "structure") renderStructure(s);
    if (state.tab === "inspect") renderInspect(s);
    if (state.tab === "json") $("json").textContent = fmtJson(downstreamView(s));
  }
  $("jsonWrap").hidden = state.tab !== "json";
  $("structure").hidden = state.tab !== "structure" && state.tab !== "inspect";
  $("retrieve").hidden = state.tab !== "retrieve";
  $("bench").hidden = state.tab !== "bench";
  $("designs").hidden = state.tab !== "designs";
}

function renderStructure(s) {
  const run = s.run;
  const stats = run?.stats || {};
  const warnings = run?.warnings || [];
  const sections = s.sections || [];
  const byParent = {};
  for (const sec of sections) {
    const key = sec.parent_id || "_root";
    (byParent[key] ||= []).push(sec);
  }
  const walk = (parent, depth = 0) =>
    (byParent[parent] || [])
      .map((sec) => {
        const kids = walk(sec.id, depth + 1);
        return `<li><strong>${esc(sec.title)}</strong> <span class="meta">L${sec.level} · p.${sec.page_start}–${sec.page_end}</span>${kids}</li>`;
      })
      .join("");
  const roots = (byParent._root || []).map((sec) => sec.id);
  // also include orphans whose parent is missing
  let tree = `<ul class="tree">${walk("_root")}</ul>`;
  if (!sections.length) tree = `<p class="muted">No sections stored for this pipeline yet.</p>`;

  const pages = (s.pages || [])
    .map((p) => `<span class="pill">${p.page_number}: ${p.label || "?"} (${p.char_count}c)</span>`)
    .join(" ");
  const assets = (s.assets || [])
    .map((a) => {
      if (a.asset_type === "figure" && a.blob_uri) {
        return `<div><img class="figure" src="/api/assets/${a.id}" alt="${esc(a.caption || a.id)}"><div class="meta">${esc(a.caption || "figure")} · p.${a.page_number}</div></div>`;
      }
      const extra = a.extra || a.extra_json;
      const md = extra && extra.markdown ? extra.markdown : "";
      return `<div><div class="meta">${a.asset_type} · p.${a.page_number} · ${esc(a.caption || a.id)}</div><pre>${esc(md || JSON.stringify(extra, null, 2) || "")}</pre></div>`;
    })
    .join("");

  $("structure").innerHTML = `
    <div class="card">
      <h3>${esc(s.document.title)} <span class="pill ${state.pipeline}">${state.pipeline}</span></h3>
      <div class="meta">${esc(s.document.filename)} · ${s.document.page_count} pages · sha ${s.document.sha256.slice(0, 12)}</div>
      <p>${run ? `Status <span class="ok">${run.status}</span> · chunks ${stats.chunks ?? "—"} · sections ${stats.sections ?? "—"}` : "Not run"}</p>
      ${stats.vectorprism ? `<div class="meta">VectorPrism ${(stats.vectorprism.channels || []).length} channels · ${(stats.vectorprism.channels || []).join(", ")} · ${stats.vectorprism.chunks ?? stats.chunks} × 6 embeddings</div>` : ""}
      ${warnings.length ? `<div class="warn">${warnings.map(esc).join("<br>")}</div>` : ""}
    </div>
    <div class="grid2">
      <div class="card"><h3>Section tree</h3>${tree}</div>
      <div class="card"><h3>Page router labels</h3><div class="pills">${pages}</div></div>
    </div>
    <div class="card"><h3>Figures & tables</h3>${assets || '<p class="muted">None extracted.</p>'}</div>
    ${renderHeadingExplain(s)}
  `;
}

function renderHeadingExplain(s) {
  const route = (s.document && (s.document.metadata || s.document.metadata_json) && (s.document.metadata || {}).route) ||
    (state.documents.find((d) => d.id === state.documentId) || {}).route;
  const heading = route && route.heading;
  if (!heading) {
    return `<div class="card"><h3>Titles vs paragraphs</h3>
      <p>A text block is a <strong>heading</strong> if it is short, larger than the page’s dominant body font (or bold + slightly larger, or numbered <code>1.</code> / <code>1.1</code> / Chapter). Longer blocks at the body size are <strong>paragraphs</strong>. Repeated top/bottom lines are headers/footers, not titles.</p></div>`;
  }
  const rows = (heading.headings || [])
    .map((h) => `<tr><td>H${h.level}</td><td>${esc(h.text)}</td><td>${h.font_size}pt${h.bold ? " bold" : ""}</td><td>p.${h.page}</td></tr>`)
    .join("");
  return `<div class="card">
    <h3>Titles vs paragraphs</h3>
    <p class="muted">${esc(heading.rule || "")} Body font ≈ <strong>${heading.body_font_size}pt</strong>. ${heading.heading_count || 0} headings (${heading.numbered_headings || 0} numbered).</p>
    ${rows ? `<table class="compare"><tr><th>Lvl</th><th>Text</th><th>Font</th><th>Page</th></tr>${rows}</table>` : ""}
  </div>`;
}

function renderInspect(s) {
  const chunks = s.chunks || [];
  const params = s.parameters || [];
  const edges = s.graph_edges || [];
  $("structure").innerHTML = `
    <div class="card">
      <h3>Chunks for downstream retrieval (${chunks.length})</h3>
      ${chunks
        .map(
          (c) => `<div class="hit"><div class="meta">p.${c.page_start}–${c.page_end} · ${c.id}</div><div>${esc((c.retrieval_text || "").slice(0, 500))}</div></div>`
        )
        .join("") || '<p class="muted">No chunks.</p>'}
    </div>
    <div class="card">
      <h3>Signed parameters (Prism)</h3>
      ${
        params.length
          ? `<table class="compare"><tr><th>Name</th><th>Value</th><th>Page</th><th>Sig</th></tr>${params
              .map(
                (p) =>
                  `<tr><td>${esc(p.parameter_name)}</td><td>${esc(p.raw_string_value)}</td><td>${p.provenance_page}</td><td>${p.manifest_signature ? p.manifest_signature.slice(0, 12) + "…" : "—"}</td></tr>`
              )
              .join("")}</table>`
          : '<p class="muted">Only the Prism pipeline writes document_parameters.</p>'
      }
    </div>
    <div class="card">
      <h3>ChorusGraph edges</h3>
      ${
        edges.length
          ? edges
              .slice(0, 40)
              .map((e) => `<div class="meta">${e.relationship_type} · ${e.source_node} → ${e.target_node} (${e.weight})</div>`)
              .join("")
          : '<p class="muted">No graph edges for this document (run Prism on 2+ docs for overlaps_with).</p>'
      }
    </div>
  `;
}

function downstreamView(s) {
  return {
    document: {
      id: s.document.id,
      title: s.document.title,
      source: { filename: s.document.filename, sha256: s.document.sha256, page_count: s.document.page_count },
      metadata: s.document.metadata || s.document.metadata_json,
    },
    pipeline: state.pipeline,
    run: s.run,
    outline: (s.sections || []).map((sec) => ({
      id: sec.id,
      parent_id: sec.parent_id,
      level: sec.level,
      title: sec.title,
      page_start: sec.page_start,
      page_end: sec.page_end,
      summary: sec.summary,
    })),
    chunks: (s.chunks || []).map((c) => ({
      id: c.id,
      section_id: c.section_id,
      page_start: c.page_start,
      page_end: c.page_end,
      retrieval_text: c.retrieval_text,
      context: c.context || c.context_json,
      asset_ids: c.asset_ids || c.asset_ids_json,
    })),
    assets: s.assets,
    parameters: s.parameters,
    graph_edges: s.graph_edges,
    pages: s.pages,
  };
}

function splitRetrieval(h) {
  const raw = h.retrieval_text || h.text || "";
  const cut = raw.indexOf("\n\n");
  if (cut === -1) return { prefix: "", body: raw };
  return { prefix: raw.slice(0, cut), body: raw.slice(cut + 2) };
}

function renderShieldPanel(p, r) {
  const s = r.shield || {};
  if (p !== "prism") {
    return `<div class="shield-panel off">
      <h4>PrismShield</h4>
      <p class="meta">Not applied. ${esc(s.role || "This stack does not sign metrics.")}</p>
    </div>`;
  }
  const tokens = (items, cls) =>
    items.length
      ? items.map((x) => `<span class="token ${cls}">${esc(x)}</span>`).join("")
      : `<span class="meta">none</span>`;
  return `<div class="shield-panel">
    <h4>PrismShield ${s.applied ? "· applied" : "· off"}</h4>
    <p class="meta">${esc(s.role || "Query-time gate. Does not rewrite prose.")}</p>
    <div class="shield-row"><strong class="ok">Verified</strong> ${tokens(s.verified || [], "ok")}</div>
    <div class="shield-row"><strong>Unsigned</strong> ${tokens(s.unsigned || [], "muted")}</div>
    <div class="shield-row"><strong class="warn">Drifted</strong> ${tokens(s.drifted || [], "warn")}</div>
  </div>`;
}

function renderCompareHit(h, pipelineId) {
  const ctx = h.context && typeof h.context === "object" ? h.context : {};
  const path = [].concat(ctx.section_path || ctx.inherited_header || []).filter(Boolean).join(" > ");
  const { prefix, body } = splitRetrieval(h);
  const assets = [].concat(h.asset_ids || []).filter(Boolean);
  const bits = [
    esc(h.filename || docName(h.document_id)),
    `p.${h.page_start}${h.page_end && h.page_end !== h.page_start ? "–" + h.page_end : ""}`,
    h.section_id ? `section ${h.section_id}` : "",
    h.page_label ? `label ${h.page_label}` : "",
    `score ${Number(h.score || 0).toFixed(3)}`,
  ].filter(Boolean);
  const extras = [];
  if (path) extras.push(`path ${esc(path)}`);
  if ((h.channels || []).length) extras.push("ch " + h.channels.join("+"));
  if (h.graph_edge) extras.push(`graph ${h.graph_edge} ${h.graph_weight ?? ""}`);
  if (assets.length) extras.push(`assets ${assets.length}`);
  if ((h.verified_parameters || []).length) {
    extras.push("signed params " + h.verified_parameters.map((p) => p.raw || p.name).slice(0, 3).join(", "));
  }
  const hitShield = h.shield || {};
  if ((hitShield.verified_parameters || []).length) extras.push("shield verified " + hitShield.verified_parameters.slice(0, 3).join(", "));
  if ((hitShield.drifted || []).length) extras.push("shield drift " + hitShield.drifted.map((d) => d.raw || d).join(", "));
  return `<div class="hit ${esc(pipelineId)}">
    <div class="meta">${bits.join(" · ")}</div>
    ${extras.filter(Boolean).length ? `<div class="meta">${extras.filter(Boolean).join(" · ")}</div>` : ""}
    ${prefix ? `<pre class="hit-prefix">${esc(prefix)}</pre>` : ""}
    <div>${esc(body.slice(0, 280))}</div>
  </div>`;
}

function designChips(p, r) {
  const chips = {
    baseline: ["1 channel", "path prefix", "no graph", "no assets"],
    prism: ["6 channels", "intent mix", "Shield", "ChorusGraph"],
    relay: ["leaf section", "page label", "asset ids", "no graph"],
  }[p] || [];
  if (p === "prism" && r.vectorprism) {
    chips.push(`${(r.vectorprism.channels || []).length}ch live`);
  }
  return chips.map((c) => `<span class="design-chip">${esc(c)}</span>`).join("");
}

function renderDesignStrip(p, r) {
  const d = r.design || {};
  const mix = r.vectorprism
    ? `<div class="meta">mix ${Object.entries(r.vectorprism.weights || {})
        .map(([k, v]) => `${k} ${v}`)
        .join(" · ")}</div>`
    : "";
  return `<div class="design-strip">
    <div class="meta">${esc(d.name || p)} · ${esc(r.index_name || d.index || "")} · ${r.latency_ms ?? "—"} ms</div>
    <div class="chip-row">${designChips(p, r)}</div>
    <div>${esc(d.chunk || "")}</div>
    <div class="meta"><strong>useful</strong> ${esc(d.useful || "")}</div>
    <div class="meta"><strong>trace</strong> ${esc(d.trace || "")}</div>
    <div class="meta"><strong>context</strong> ${esc(d.context || "")}</div>
    ${mix}
  </div>`;
}

async function runQuery() {
  const query = $("query").value.trim();
  if (!query) return;
  const box = $("hits");
  const searchAll = !$("searchAll") || $("searchAll").checked;
  const documentId = searchAll ? null : state.documentId;
  const scope = documentId
    ? (state.documents.find((d) => d.id === documentId) || {}).filename || documentId
    : "all documents";
  box.innerHTML = `Searching three indexes in <strong>${esc(scope)}</strong>…`;
  const intents = { baseline: null, prism: $("intent").value || null, relay: null };
  const results = {};
  for (const p of ["baseline", "prism", "relay"]) {
    try {
      results[p] = await api("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          pipeline_id: p,
          k: 4,
          intent: intents[p],
          apply_shield: true,
          document_id: documentId,
        }),
      });
    } catch (err) {
      results[p] = { error: String(err), hits: [], design: {} };
    }
  }
  const cols = ["baseline", "prism", "relay"]
    .map((p) => {
      const r = results[p];
      const hits = (r.hits || []).map((h) => renderCompareHit(h, p)).join("");
      return `<div class="card stack-card">
        <h3 class="${p}">${p}</h3>
        ${renderDesignStrip(p, r)}
        ${renderShieldPanel(p, r)}
        ${r.error ? `<p class="warn">${esc(r.error)}</p>` : hits || '<p class="muted">No hits. Run this pipeline first.</p>'}
      </div>`;
    })
    .join("");
  box.innerHTML = `<div class="retrieve-cols">${cols}</div>`;
}

function scoreClass(n) {
  if (n >= 75) return "good";
  if (n >= 50) return "mid";
  return "bad";
}

function renderChecks(checks) {
  return (checks || [])
    .map((c) => `<div class="${c.pass ? "check-pass" : "check-fail"}">${c.pass ? "pass" : "fail"} · ${esc(c.name)} — ${esc(c.detail)}</div>`)
    .join("");
}

function renderGrades(grades) {
  if (!grades || !grades.length) return "";
  return (
    `<div class="meta">` +
    grades
      .map((g) => `r${g.rank} rel ${g.rel} · ${esc(g.filename || "—")} — ${esc(g.why || "")}`)
      .join("<br>") +
    `</div>`
  );
}

async function runBench() {
  $("benchOut").textContent = "Scoring retrieval quality, structure, and model/OCR/embed calls…";
  const data = await api("/api/benchmark", { method: "POST" });
  const leakBy = Object.fromEntries(
    (data.structure || []).map((s) => [s.pipeline_id, s.avg_boilerplate_leak])
  );
  const boardRows = (data.leaderboard || [])
    .map((r) => {
      const overall = r.overall ?? r.avg_retrieval_quality;
      const leak = leakBy[r.pipeline_id];
      let struct = r.structure_score ?? "—";
      if (r.structure_score != null && leak != null) {
        struct = leak === 0 ? `${r.structure_score} (no header leak)` : `${r.structure_score} (${Math.round(leak * 100)}% leak)`;
      }
      return `<tr>
        <td class="${r.pipeline_id}">${esc(r.pipeline_id)}</td>
        <td><span class="score ${scoreClass(overall)}">${overall}</span></td>
        <td>${r.avg_retrieval_quality ?? "—"}</td>
        <td>${struct}</td>
        <td>${r.ndcg ?? "—"}</td>
        <td>${r.precision ?? "—"}</td>
        <td class="meta">${r.mrr ?? "—"}</td>
      </tr>`;
    })
    .join("");
  const costRows = (data.cost || [])
    .map(
      (c) => `<tr>
        <td>${c.pipeline_id}</td>
        <td>${c.llm_calls}</td>
        <td>${c.vision_pages}</td>
        <td>${c.ocr_pages}</td>
        <td>${c.embed_docs}</td>
        <td>${c.ed25519_signs}</td>
        <td>$${c.estimated_usd_actual}</td>
        <td>$${c.estimated_usd_if_every_page_vision}</td>
        <td>$${c.usd_saved_vs_per_page_vision}</td>
      </tr>`
    )
    .join("");
  const structRows = (data.structure || [])
    .map((s) => {
      const rec = s.avg_heading_recall == null ? "—" : `${Math.round(s.avg_heading_recall * 100)}%`;
      const leak = s.avg_boilerplate_leak == null ? "—" : `${Math.round(s.avg_boilerplate_leak * 100)}%`;
      const docs = (s.documents || [])
        .map((d) => `${esc(d.filename)}: L${(d.heading_levels || []).join("/")} · ${d.sections} sections · ctx ${Math.round((d.section_context_coverage || 0) * 100)}%`)
        .join("<br>");
      const vp = (s.vectorprism_channels || []).length
        ? `<div>VectorPrism: ${s.vectorprism_channels.join(", ")}</div>`
        : "";
      return `<tr><td>${s.pipeline_id}</td><td>${rec}</td><td>${leak}</td><td class="meta">${docs || "—"}${vp}</td></tr>`;
    })
    .join("");
  const queryBlocks = (data.results || [])
    .map((block) => {
      const q = block.query;
      const rows = (block.pipelines || [])
        .map((p) => {
          const top = (p.hits || [])[0];
          const qy = p.quality || {};
          return `<tr>
            <td>${p.pipeline_id}</td>
            <td><span class="score ${scoreClass(qy.score || 0)}">${qy.score ?? "—"}</span> <span class="meta">nDCG ${qy.ndcg ?? "—"} · MRR ${qy.mrr ?? "—"} · P@5 ${qy.precision ?? "—"}</span></td>
            <td>${p.latency_ms} ms</td>
            <td class="meta">${esc(p.index_name || "")}${(top && top.channels) ? "<br>ch " + top.channels.join("+") : ""}${p.vectorprism ? "<br>VP " + (p.vectorprism.channels || []).join(", ") : ""}<br>embed queries: ${p.usage?.embed_queries ?? 0} · llm: ${p.usage?.llm_calls ?? 0}</td>
            <td>${top ? esc((top.retrieval_text || "").slice(0, 160)) : "—"}</td>
          </tr>
          <tr><td></td><td colspan="4">${renderGrades(qy.grades)}${renderChecks(qy.checks)}</td></tr>`;
        })
        .join("");
      return `<h3>${esc(q.query)}</h3><p class="muted">${esc(q.notes || "")} Must contain: ${(q.must_contain || []).join(", ") || "—"}</p>
        <table class="compare"><tr><th>Pipeline</th><th>Quality</th><th>Latency</th><th>Index / calls</th><th>Top hit</th></tr>${rows}</table>`;
    })
    .join("");
  $("benchOut").innerHTML = `
    <div class="card">
      <h3>Graded leaderboard</h3>
      <table class="compare">
        <tr>
          <th></th>
          <th>Overall</th>
          <th>Retrieval</th>
          <th>Structure</th>
          <th>nDCG</th>
          <th>P@5</th>
          <th>MRR</th>
        </tr>
        ${boardRows}
      </table>
      <p class="muted">${esc(data.rubric?.document_understanding || "")}. ${esc(data.rubric?.system_design || "")}</p>
    </div>
    <div class="card">
      <h3>Model / OCR / embed calls (latest successful run per document)</h3>
      <p class="muted">${esc(data.rubric?.cost || "")} Vision-$ is the naïve “send every page to a multimodal model” we did not do.</p>
      <table class="compare">
        <tr><th>Pipeline</th><th>LLM</th><th>Vision pages</th><th>OCR pages</th><th>Embed texts</th><th>Ed25519</th><th>USD actual</th><th>USD if all-page vision</th><th>USD saved</th></tr>
        ${costRows}
      </table>
    </div>
    <div class="card">
      <h3>Structure quality</h3>
      <p class="muted">Heading recall vs gold outlines on seed PDFs. Context = section path on the chunk. Leak = running headers left in retrieval text.</p>
      <table class="compare"><tr><th>Pipeline</th><th>Heading recall</th><th>Boilerplate leak</th><th>Per document</th></tr>${structRows}</table>
    </div>
    <div class="card">
      <h3>Per-query retrieval</h3>
      ${queryBlocks}
    </div>
  `;
  renderBonus(data.bonus);
}

function renderBonus(bonus) {
  const box = $("bonusOut");
  if (!box) return;
  if (!bonus) {
    box.innerHTML = "";
    return;
  }
  const pairs = (bonus.document_pairs || [])
    .map(
      (p) =>
        `<div class="meta">${esc(p.document_a.filename)} ↔ ${esc(p.document_b.filename)} · ${p.edge_count} edges · max ${p.max_weight} · mean ${p.mean_weight} (section-level)</div>`
    )
    .join("") || '<p class="muted">Run Prism on 2+ PDFs to build pairs.</p>';
  const rel = (bonus.related_sections || [])
    .slice(0, 12)
    .map(
      (r) =>
        `<div class="hit"><div class="meta">${esc(r.relationship)} · ${r.weight} · ${esc(r.source.filename)} → ${esc(r.target.filename)}</div>${esc(r.why)}<div class="meta">${esc(r.source.label)} ↔ ${esc(r.target.label)}</div></div>`
    )
    .join("");
  const uniq = (bonus.unique_sections || [])
    .slice(0, 10)
    .map((u) => `<div class="meta">${esc(u.filename)} · ${esc(u.title)} · p.${u.page_start}</div>`)
    .join("");
  box.innerHTML = `
    <div class="card">
      <h3>Bonus — cross-document</h3>
      <p class="muted">${esc(bonus.note || "")} ${bonus.related_count || 0} related edges · ${bonus.unique_count || 0} unique Prism sections.</p>
      <h4>Document pairs</h4>${pairs}
      <h4>Why sections were linked</h4>${rel || '<p class="muted">No cross-doc edges yet.</p>'}
      <h4>Unique to one document</h4>${uniq || '<p class="muted">None.</p>'}
    </div>`;
}

function renderMarkdown(md) {
  const lines = String(md || "").replaceAll("\r\n", "\n").split("\n");
  const out = [];
  let inCode = false;
  let inTable = false;
  let listOpen = false;
  const closeList = () => {
    if (listOpen) {
      out.push("</ul>");
      listOpen = false;
    }
  };
  const closeTable = () => {
    if (inTable) {
      out.push("</table>");
      inTable = false;
    }
  };
  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  for (const line of lines) {
    if (line.startsWith("```")) {
      closeList();
      closeTable();
      if (!inCode) {
        out.push("<pre>");
        inCode = true;
      } else {
        out.push("</pre>");
        inCode = false;
      }
      continue;
    }
    if (inCode) {
      out.push(esc(line) + "\n");
      continue;
    }
    if (line.startsWith("|") && line.includes("|", 1)) {
      closeList();
      if (/^\|?\s*:?-/.test(line.replace(/\|/g, "").trim()) || /---/.test(line)) continue;
      const cells = line.split("|").slice(1, -1).map((c) => c.trim());
      if (!inTable) {
        out.push("<table>");
        out.push("<tr>" + cells.map((c) => `<th>${inline(c)}</th>`).join("") + "</tr>");
        inTable = true;
      } else {
        out.push("<tr>" + cells.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>");
      }
      continue;
    }
    closeTable();
    const heading = /^(#{1,3})\s+(.*)$/.exec(line);
    if (heading) {
      closeList();
      const n = heading[1].length;
      out.push(`<h${n}>${inline(heading[2])}</h${n}>`);
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      if (!listOpen) {
        out.push("<ul>");
        listOpen = true;
      }
      out.push(`<li>${inline(line.replace(/^[-*]\s+/, ""))}</li>`);
      continue;
    }
    closeList();
    if (!line.trim()) continue;
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  closeTable();
  if (inCode) out.push("</pre>");
  return out.join("");
}

async function openDesign(id) {
  setTab("designs");
  const card = $("designDocCard");
  const body = $("designDocBody");
  $("designDocTitle").textContent = "Loading…";
  card.hidden = false;
  try {
    const doc = await api(`/api/designs/${id}`);
    $("designDocTitle").textContent = doc.title;
    $("designDocRaw").href = `/design-docs/${id}`;
    $("designDocRaw").textContent = doc.file;
    body.innerHTML = renderMarkdown(doc.markdown);
    card.scrollIntoView({ behavior: "smooth", block: "start" });
    document.querySelectorAll(".design-card").forEach((el) => {
      el.style.outline = el.id === `design-card-${id}` ? "1px solid var(--accent)" : "";
    });
  } catch (err) {
    body.innerHTML = `<p class="warn">${esc(String(err))}</p>`;
  }
}

function docName(id) {
  const d = state.documents.find((x) => x.id === id);
  return d ? d.filename : id;
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function _shieldMeta(h) {
  const s = h.shield || {};
  const bits = [];
  const signed = s.verified_parameters || [];
  const drifted = (s.drifted || []).map((d) => d.raw || d);
  if (signed.length) bits.push("signed " + signed.slice(0, 4).join(", "));
  if (drifted.length) bits.push("drift " + drifted.join(", "));
  return bits.length ? " · " + bits.join(" · ") : "";
}

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".nav-item").forEach((el) => el.classList.toggle("active", el.dataset.tab === tab));
  render();
  if (tab === "bench") loadBonus();
}

async function loadBonus() {
  try {
    renderBonus(await api("/api/bonus"));
  } catch {
    /* bench run will fill this */
  }
}

function setPipeline(id) {
  state.pipeline = id;
  document.querySelectorAll("[data-pipe]").forEach((el) => el.classList.toggle("active", el.dataset.pipe === id));
  loadStructure();
}

window.addEventListener("DOMContentLoaded", async () => {
  $("uploadBtn").onclick = uploadPdf;
  $("runBtn").onclick = runSelected;
  $("queryBtn").onclick = runQuery;
  $("query").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") runQuery();
  });
  document.querySelectorAll("[data-sample-q]").forEach((el) => {
    el.onclick = () => {
      $("query").value = el.getAttribute("data-sample-q") || "";
      const intent = el.getAttribute("data-sample-intent") || "";
      if ($("intent") && intent) $("intent").value = intent;
      runQuery();
    };
  });
  $("benchBtn").onclick = runBench;
  $("file").addEventListener("change", uploadPdf);
  document.querySelectorAll(".nav-item").forEach((el) => (el.onclick = () => setTab(el.dataset.tab)));
  document.querySelectorAll("[data-pipe]").forEach((el) => (el.onclick = () => setPipeline(el.dataset.pipe)));
  document.querySelectorAll("[data-open-design]").forEach((el) => {
    el.addEventListener("click", (ev) => {
      ev.preventDefault();
      openDesign(el.getAttribute("data-open-design"));
    });
  });
  const health = await api("/api/health");
  $("health").textContent = `sqlite · ${health.indexes.map((i) => i.index + ":" + i.vectors).join(" · ")}`;
  await refreshDocs();
});
