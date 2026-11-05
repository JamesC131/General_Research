/* Simulation Lab: run the four simulations, compare with human data, replicate, inspect participants. */

const SOURCES = [
  ["paper", "From the paper", "Values published in Gomez & Sun (2024, 2025)."],
  ["calibrated", "Calibrated to human data", "Not in the papers; fitted so the model matches the human results."],
  ["chosen", "Chosen", "Not in the papers; set once on reasonable grounds and not fitted."],
];

let catalog = [];
let simId = null;
let values = {};       // current parameter values
let last = null;       // last run payload
let job = null;        // running replication job id
let poll = null;
let selectedPart = null;

const spec = () => catalog.find((s) => s.id === simId);

/* ---------------------------------------------------------------------------- */
/* side panel                                                                      */
/* ---------------------------------------------------------------------------- */

function renderSimList() {
  $("#sim-list").replaceChildren(...catalog.map((s) => el("button", {
    class: "sim-option", type: "button", role: "radio", "aria-checked": s.id === simId ? "true" : "false",
    onclick: () => chooseSim(s.id) }, el("b", {}, s.title), el("span", {}, `${s.study}, in ${s.paper}`))));
}

function chooseSim(id) {
  simId = id;
  const s = spec();
  values = Object.fromEntries(s.params.map((p) => [p.name, structuredClone(p.value)]));
  try { history.replaceState(null, "", `#${id}`); } catch (_) {}
  $("#title").textContent = s.title;
  $("#study").textContent = `${s.study}, simulated in ${s.paper}`;
  $("#summary").textContent = s.summary;
  $("#run-hint").textContent = `One run takes about ${s.seconds} seconds.`;
  renderSimList();
  renderParams();
  resetReplications();
  run();
}

function renderParams() {
  const s = spec();
  const wrap = $("#params");
  wrap.replaceChildren();
  for (const [src, title, hint] of SOURCES) {
    const ps = s.params.filter((p) => p.source === src);
    if (!ps.length) continue;
    const det = el("details", { class: "param-group", open: src !== "chosen" },
      el("summary", {}, title, el("span", { class: `tag ${src}` }, `${ps.length}`)),
      el("p", { class: "hint" }, hint));
    ps.forEach((p) => det.append(paramRow(p)));
    wrap.append(det);
  }
}

function paramRow(p) {
  const id = `p-${p.name}`;
  const row = el("div", { class: "param", title: p.note || "" });
  const mark = () => row.classList.toggle("changed", JSON.stringify(values[p.name]) !== JSON.stringify(p.value));
  let input;
  if (p.kind === "bool") {
    input = el("input", { type: "checkbox", id, checked: !!values[p.name],
      onchange: (e) => { values[p.name] = e.target.checked; mark(); } });
  } else if (p.kind === "pair") {
    input = el("div", { class: "pair" }, ...values[p.name].map((v, i) => el("input", {
      type: "number", value: v, step: Number.isInteger(v) ? 1 : 0.05, "aria-label": `${p.label} ${i + 1}`,
      oninput: (e) => { values[p.name][i] = parseFloat(e.target.value); mark(); } })));
  } else if (p.kind === "text") {
    input = el("input", { type: "text", id, value: values[p.name], oninput: (e) => { values[p.name] = e.target.value; mark(); } });
  } else {
    input = el("input", { type: "number", id, value: values[p.name], step: p.kind === "int" ? 1 : stepFor(p.value),
      oninput: (e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) { values[p.name] = v; mark(); } } });
  }
  row.append(el("label", { for: id }, p.label), input);
  if (p.note) row.append(el("span", { class: "note" }, p.note));
  return row;
}

function stepFor(v) {
  const a = Math.abs(v);
  return a >= 10 ? 1 : a >= 1 ? 0.1 : a >= 0.1 ? 0.01 : 0.001;
}

/* ---------------------------------------------------------------------------- */
/* single run                                                                    */
/* ---------------------------------------------------------------------------- */

async function run() {
  const btn = $("#run");
  btn.disabled = true; btn.textContent = "Running…";
  document.body.classList.add("busy");
  const err = $("#error");
  try {
    last = await api(`/api/sims/${simId}/run`, { params: values, seed: +$("#seed").value || 0 });
    err.hidden = true;
    renderResult();
    renderParticipants();
  } catch (e) {
    err.textContent = e.message; err.hidden = false;
  } finally {
    btn.disabled = false; btn.textContent = "Run simulation";
    document.body.classList.remove("busy");
  }
}

const CHECK = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 8.5l3.2 3L13 4.5"/></svg>';
const CROSS = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4l8 8M12 4l-8 8"/></svg>';

/* Draws human (diamond) and published (ring) reference markers over bar charts. */
const refPlugin = {
  id: "refs",
  afterDatasetsDraw(c) {
    const o = c.options.plugins.refs;
    if (!o) return;
    const meta = c.getDatasetMeta(0);
    const { ctx } = c;
    meta.data.forEach((bar, j) => {
      const half = (bar.width || 30) / 2;
      if (o.published && o.published[j] !== null && o.published[j] !== undefined) {
        const y = c.scales.y.getPixelForValue(o.published[j]);
        ctx.save(); ctx.strokeStyle = css("--published"); ctx.lineWidth = 2.5; ctx.fillStyle = css("--surface");
        ctx.beginPath(); ctx.arc(bar.x + half + 10, y, 5.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.restore();
      }
      if (o.human && o.human[j] !== null && o.human[j] !== undefined) {
        const y = c.scales.y.getPixelForValue(o.human[j]);
        ctx.save(); ctx.fillStyle = css("--human"); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(bar.x, y - 8); ctx.lineTo(bar.x + 8, y); ctx.lineTo(bar.x, y + 8); ctx.lineTo(bar.x - 8, y);
        ctx.closePath(); ctx.stroke(); ctx.fill(); ctx.restore();
      }
    });
  },
};

function renderResult() {
  const s = spec();
  $("#run-meta").textContent = `Seed ${last.seed}, ${last.participants.length} simulated participants.`;
  const ok = last.pattern.achieved;
  $("#verdict").className = `verdict ${ok ? "yes" : "no"}`;
  $("#verdict").innerHTML = `${ok ? CHECK : CROSS}<span>${ok ? "Human pattern reproduced" : "Human pattern not reproduced"} in this run</span>`;
  $("#verdict").title = last.pattern.text;

  // panels
  const P = $("#panels");
  P.replaceChildren();
  last.panels.forEach((pn, i) => {
    const id = `c-panel-${i}`;
    P.append(el("div", {}, el("p", { class: "chart-title" }, pn.title), el("div", { class: "chart-box" }, el("canvas", { id, "aria-label": pn.title }))));
    const ext = [...pn.model, ...pn.human, ...(pn.published || [])].filter((v) => v !== null);
    const sdTop = pn.model_sd ? Math.max(...pn.model.map((m, j) => m + pn.model_sd[j])) : -Infinity;
    const lo = pn.ymin < 0 ? Math.min(pn.ymin, Math.floor(Math.min(...ext) * 10) / 10) : 0;
    const hi = Math.max(Math.min(pn.ymax, Math.max(...ext, sdTop) * 1.12), Math.max(...ext) * 1.05);
    chart(id, {
      type: "bar",
      data: { labels: pn.categories, datasets: [{ label: "This model", data: pn.model, whiskers: pn.model_sd,
        backgroundColor: css("--explicit"), maxBarThickness: 64, categoryPercentage: 0.55 }] },
      options: {
        scales: { x: { grid: { display: false } },
                  y: { min: lo, max: hi < 1.5 ? Math.ceil(hi * 10) / 10 : Math.ceil(hi / 5) * 5, title: { display: true, text: pn.ylabel } } },
        plugins: { refs: { human: pn.human, published: pn.published },
          tooltip: { callbacks: { label: (c) => {
            const j = c.dataIndex;
            return [`This model: ${fmt(c.parsed.y)}${pn.model_sd ? ` (SD ${fmt(pn.model_sd[j], 1)})` : ""}`,
                    `Human: ${fmt(pn.human[j])}`, `Published simulation: ${fmt(pn.published[j])}`];
          } } } },
      },
      plugins: [whiskerPlugin, refPlugin],
    });
  });

  // comparison table
  const T = $("#compare");
  T.replaceChildren(el("thead", {}, el("tr", {}, el("th", {}, "Measure"), el("th", {}, "Human"), el("th", {}, "Published sim"), el("th", {}, "This run"))));
  const tb = el("tbody");
  let group = null;
  last.rows.forEach((r) => {
    if (r.group !== group) { group = r.group; tb.append(el("tr", { class: "group" }, el("td", { colspan: 4 }, group))); }
    tb.append(el("tr", {}, el("td", {}, r.label), el("td", {}, r.human), el("td", { class: "muted" }, r.published), el("td", { class: "model" }, r.model)));
  });
  T.append(tb);

  // mechanism
  $("#mech-title").textContent = last.mechanism.title;
  chart("c-mech", {
    type: "bar",
    data: { labels: last.mechanism.categories, datasets: [{ data: last.mechanism.values, backgroundColor: css("--explicit"),
      maxBarThickness: 26 }] },
    options: { indexAxis: "y", scales: { x: { min: 0, max: 1, title: { display: true, text: "Explicitness" } }, y: { grid: { display: false } } },
      plugins: { tooltip: { callbacks: { label: (c) => fmt(c.parsed.x) } } } },
  });

  // Friese slopes
  $("#slopes-wrap").hidden = !last.slopes;
  if (last.slopes) {
    for (const [id, m, name] of [["c-slope-exp", "explicit", "Explicit measure"], ["c-slope-imp", "implicit", "Implicit measure"]]) {
      chart(id, {
        type: "line",
        data: { labels: ["−1 SD", "+1 SD"], datasets: [
          { label: "High capacity", data: last.slopes.high[m], borderColor: css("--explicit"), backgroundColor: css("--explicit"), pointRadius: 5 },
          { label: "Low capacity", data: last.slopes.low[m], borderColor: css("--implicit"), backgroundColor: css("--implicit"), pointRadius: 5, pointStyle: "rect" }] },
        options: { scales: { x: { title: { display: true, text: name }, grid: { display: false } },
                             y: { title: { display: true, text: "Chocolates chosen" }, suggestedMin: 2, suggestedMax: 3.5 } },
          plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y)}` } } } },
      });
    }
  }
}

/* ---------------------------------------------------------------------------- */
/* replications                                                                   */
/* ---------------------------------------------------------------------------- */

function resetReplications() {
  if (poll) clearInterval(poll);
  poll = null; job = null;
  $("#progress").hidden = true;
  $("#cancel").hidden = true;
  $("#replicate").disabled = false;
  $("#rep-out").className = "empty";
  $("#rep-out").textContent = "Run replications to see the distribution of results.";
}

async function replicate() {
  resetReplications();
  try {
    job = (await api(`/api/sims/${simId}/replicate`, { params: values, reps: +$("#reps").value })).job;
  } catch (e) {
    $("#rep-out").className = "error"; $("#rep-out").textContent = e.message; return;
  }
  $("#replicate").disabled = true;
  $("#cancel").hidden = false;
  $("#progress").hidden = false;
  const thisJob = job;
  poll = setInterval(async () => {
    if (job !== thisJob) return;
    try {
      const st = await api(`/api/jobs/${thisJob}`);
      $("#progress > div").style.width = `${(100 * st.done) / st.total}%`;
      if (st.n) renderReplications(st);
      if (st.error) { $("#rep-out").className = "error"; $("#rep-out").textContent = `A replication failed: ${st.error}`; }
      if (st.finished || st.error) finishPolling();
    } catch (e) {
      $("#rep-out").className = "error"; $("#rep-out").textContent = e.message; finishPolling();
    }
  }, 800);
}

function finishPolling() {
  clearInterval(poll); poll = null;
  $("#replicate").disabled = false;
  $("#cancel").hidden = true;
}

function renderReplications(st) {
  const s = spec();
  const out = $("#rep-out");
  if (out.className !== "rep") {
    out.className = "rep";
    out.replaceChildren(
      el("p", { class: "rep-head", id: "rep-head" }),
      el("div", { class: "rep-grid" },
        el("div", {}, el("p", { class: "chart-title", id: "hist-title" }),
          el("div", { class: "legend" }, el("span", {}, el("i", { style: "background:var(--explicit)" }), "Replications"),
            s.key_human !== null ? el("span", {}, el("i", { class: "diamond", style: "background:var(--human)" }), "Human value") : null),
          el("div", { class: "chart-box" }, el("canvas", { id: "c-hist" }))),
        el("div", {}, el("div", { class: "table-scroll", style: "max-height:300px" }, el("table", { class: "data", id: "rep-table" })),
          el("button", { class: "btn small", type: "button", style: "margin-top:10px", id: "csv-reps" }, "Download replications CSV"))));
  }
  const k = Math.round(st.pattern_rate * st.n);
  $("#rep-head").innerHTML = `The human pattern (${s.pattern_text.toLowerCase()}) appeared in <b>${k} of ${st.n}</b> replications <b>(${Math.round(st.pattern_rate * 100)}%)</b>${st.finished ? "." : ", still running…"}`;
  $("#hist-title").textContent = `${st.key.label} across replications`;

  // histogram
  const v = st.key.values, lo = Math.min(...v, s.key_human ?? Infinity), hi = Math.max(...v, s.key_human ?? -Infinity);
  const nb = Math.min(16, Math.max(6, Math.round(Math.sqrt(v.length) * 1.6)));
  const w = (hi - lo) / nb || 1;
  const counts = Array(nb).fill(0);
  v.forEach((x) => { counts[Math.min(nb - 1, Math.floor((x - lo) / w))]++; });
  const labels = counts.map((_, i) => fmt(lo + (i + 0.5) * w));
  const humanBin = s.key_human === null ? null : Math.min(nb - 1, Math.floor((s.key_human - lo) / w));
  chart("c-hist", {
    type: "bar",
    data: { labels, datasets: [{ data: counts, backgroundColor: css("--explicit"), categoryPercentage: 0.94, barPercentage: 0.96 }] },
    options: { animation: false, scales: { x: { title: { display: true, text: st.key.label }, grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 8 } },
                         y: { title: { display: true, text: "Replications" }, ticks: { precision: 0 } } },
      plugins: { tooltip: { callbacks: { title: (c) => `${fmt(lo + c[0].dataIndex * w)} to ${fmt(lo + (c[0].dataIndex + 1) * w)}`,
        label: (c) => `${c.parsed.y} replications` } },
        refs: humanBin === null ? null : { human: counts.map((_, i) => (i === humanBin ? Math.max(...counts) * 0.04 + 0.5 : null)) } } },
    plugins: [refPlugin],
  });

  // metric table
  const T = $("#rep-table");
  T.replaceChildren(el("thead", {}, el("tr", {}, el("th", {}, "Measure"), el("th", {}, "Mean"), el("th", {}, "Middle 95%"))));
  const tb = el("tbody");
  Object.entries(st.metrics).forEach(([key, m]) => tb.append(el("tr", {},
    el("td", {}, metricLabel(key)), el("td", { class: "model" }, fmt(m.mean)), el("td", { class: "muted" }, `${fmt(m.lo)} to ${fmt(m.hi)}`))));
  T.append(tb);
  $("#csv-reps").onclick = () => downloadCSV(`${simId}_replications.csv`, st.rows.map((r, i) => ({ replication: i + 1, ...r })));
}

function metricLabel(k) {
  return k.replace(/_/g, " ").replace(/test(\d)/, "Test $1")
    .replace(/\bt\b/, "t-test").replace(/\bF\b/, "interaction F").replace(/\br exp imp\b/, "r explicit x implicit")
    .replace(/\br exp choc\b/, "r explicit x chocolates").replace(/\br imp choc\b/, "r implicit x chocolates")
    .replace(/^./, (c) => c.toUpperCase());
}

/* ---------------------------------------------------------------------------- */
/* participants                                                                  */
/* ---------------------------------------------------------------------------- */

function renderParticipants() {
  const conds = [...new Set(last.participants.map((p) => p.condition + (p.test ? `, ${p.test}` : "")))];
  const f = $("#cond-filter");
  const prev = f.value;
  f.replaceChildren(el("option", { value: "" }, "All conditions"), ...conds.map((c) => el("option", { value: c, selected: c === prev }, c)));
  drawPartTable();
  selectedPart = null;
  $("#detail").replaceChildren(el("p", { class: "empty" }, "No participant selected."));
}

function drawPartTable() {
  const filter = $("#cond-filter").value;
  const rows = last.participants.filter((p) => !filter || p.condition + (p.test ? `, ${p.test}` : "") === filter);
  const cols = last.columns;
  const T = $("#parts");
  T.replaceChildren(el("thead", {}, el("tr", {}, ...cols.map((c) => el("th", {}, prettyName(c))))));
  const tb = el("tbody");
  rows.forEach((p) => tb.append(el("tr", { class: `clickable${selectedPart === p.id ? " selected" : ""}`, tabindex: 0,
    onclick: () => showPart(p), onkeydown: (e) => { if (e.key === "Enter") showPart(p); } },
    ...cols.map((c) => el("td", { class: typeof p[c] === "number" ? "num" : "" }, cell(c, p[c]))))));
  T.append(tb);
}

const COUNT_COLUMNS = new Set(["id", "score", "chocolates"]);
function cell(col, v) {
  if (v === null || v === undefined) return "–";
  if (typeof v !== "number") return v;
  return COUNT_COLUMNS.has(col) ? String(v) : fmt(v, 2);
}

function showPart(p) {
  selectedPart = p.id;
  drawPartTable();
  const D = $("#detail");
  D.replaceChildren(el("h3", {}, `Participant ${p.id}`), el("p", { class: "muted" }, p.condition + (p.test ? `, ${p.test}` : "")));
  if (last.detail === "timeline") {
    D.append(el("p", { class: "chart-title" }, "During the video: deficit and explicitness"),
      el("div", { class: "legend" }, el("span", {}, el("i", { style: "background:var(--drive-1)" }), "Achievement deficit"),
        el("span", {}, el("i", { style: "background:var(--explicit)" }), "Top level acted"),
        el("span", {}, el("i", { style: "background:var(--implicit)" }), "Bottom level acted")),
      el("div", { class: "chart-box" }, el("canvas", { id: "c-part" })),
      el("dl", { class: "kv", style: "margin-top:12px" },
        el("dt", {}, "Memory score"), el("dd", {}, p.score),
        el("dt", {}, "Explicitness in the memory test"), el("dd", {}, fmt(p.explicitness_memory)),
        el("dt", {}, "Reading words when told to ignore them"), el("dd", {}, p.control_failures === null ? "–" : `${Math.round(p.control_failures * 100)}% of moments with words`)));
    const tl = p.timeline;
    chart("c-part", {
      type: "line",
      data: { labels: tl.map((_, i) => i + 1), datasets: [
        { label: "Deficit", data: tl.map((e) => e.deficit), borderColor: css("--drive-1"), pointRadius: 0 },
        { label: "Explicitness", data: tl.map((e) => e.p_explicit), showLine: false, pointRadius: 4.5,
          pointBackgroundColor: tl.map((e) => (e.level === "top" ? css("--explicit") : css("--implicit"))),
          pointBorderColor: css("--surface") }] },
      options: { scales: { x: { title: { display: true, text: "Video moment" }, grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 10 } },
                           y: { min: 0, max: Math.max(1.05, ...tl.map((e) => e.deficit)) * 1.02 } },
        plugins: { tooltip: { callbacks: { label: (c) => c.datasetIndex === 0 ? `Deficit ${fmt(c.parsed.y)}` :
          `Explicitness ${fmt(c.parsed.y)}: ${prettyName(tl[c.dataIndex].action)}${tl[c.dataIndex].words ? " (words on screen)" : ""}` } } } },
    });
  } else if (last.detail === "events") {
    const t = el("table", { class: "data" }, el("thead", {}, el("tr", {}, el("th", {}, "Step"), el("th", {}, "Explicitness"), el("th", {}, "Level"), el("th", {}, "Outcome"))));
    const tb = el("tbody");
    p.events.forEach((e) => tb.append(el("tr", {},
      el("td", {}, prettyName(e.step)), el("td", {}, fmt(e.explicitness)),
      el("td", { class: `level-${e.level}` }, e.level === "top" ? "top (explicit)" : "bottom (implicit)"),
      el("td", {}, e.rating !== undefined ? `rating ${fmt(e.rating, 1)}${e.rule ? ` via rule ${e.rule}` : ""}` :
        e.goal ? `goal: ${prettyName(e.goal)}` : e.action ? `${prettyName(e.action)}${e.rule ? ` via rule ${e.rule}` : ""}` : ""))));
    t.append(tb);
    D.append(el("dl", { class: "kv" }, el("dt", {}, "Granola rating"), el("dd", {}, fmt(p.granola, 1)),
      el("dt", {}, "Chocolate rating"), el("dd", {}, fmt(p.chocolate, 1)), el("dt", {}, "Chose"), el("dd", {}, p.choice),
      el("dt", {}, "Active goal"), el("dd", {}, p.goal)), el("div", { class: "table-scroll", style: "margin-top:12px" }, t));
  } else if (last.detail === "picks") {
    D.append(el("dl", { class: "kv" },
        el("dt", {}, "Explicitness (set by capacity)"), el("dd", {}, fmt(p.explicitness)),
        el("dt", {}, "Explicit attitude measure"), el("dd", {}, fmt(p.explicit_measure)),
        el("dt", {}, "Implicit attitude measure"), el("dd", {}, fmt(p.implicit_measure)),
        el("dt", {}, "Chocolates chosen"), el("dd", {}, `${p.chocolates} of 5`)),
      el("p", { class: "chart-title", style: "margin-top:14px" }, "The five picks, in order"),
      el("div", { class: "legend" }, el("span", {}, el("i", { style: "background:var(--explicit)" }), "Picked by the top level (explicit rules)"),
        el("span", {}, el("i", { style: "background:var(--implicit)" }), "Picked by the bottom level (implicit network)")),
      el("div", { class: "picks" }, ...p.picks.map((k, i) => el("span", { class: `pick ${k.level}` }, `${i + 1}. ${prettyName(k.item)}`))));
  }
}

/* ---------------------------------------------------------------------------- */

async function init() {
  initTheme();
  try {
    catalog = await api("/api/sims");
  } catch (e) {
    $("#error").textContent = `Could not load the simulations: ${e.message}`; $("#error").hidden = false; return;
  }
  $("#run").addEventListener("click", run);
  $("#reset").addEventListener("click", () => chooseSim(simId));
  $("#replicate").addEventListener("click", replicate);
  $("#cancel").addEventListener("click", async () => { if (job) await api(`/api/jobs/${job}`, undefined, "DELETE").catch(() => {}); finishPolling(); });
  $("#cond-filter").addEventListener("change", drawPartTable);
  $("#csv-parts").addEventListener("click", () => last && downloadCSV(`${simId}_seed${last.seed}_participants.csv`, last.participants, last.columns));
  window.addEventListener("themechange", () => { if (last) { renderResult(); if (selectedPart) showPart(last.participants.find((p) => p.id === selectedPart)); } });
  const fromHash = location.hash.slice(1);
  chooseSim(catalog.some((s) => s.id === fromHash) ? fromHash : catalog[0].id);
}

init();
