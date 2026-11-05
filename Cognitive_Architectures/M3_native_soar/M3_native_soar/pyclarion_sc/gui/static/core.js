/* Core Studio: edit a scenario, run the Clarion cycle, inspect each step. */

const DRIVE_COLORS = ["--drive-1", "--drive-2", "--drive-3", "--drive-4", "--published"];
const MCS_FIELDS = [
  ["alpha", "Benefit coefficient α", 0.05],
  ["cost", "Cost coefficient c", 0.005],
  ["temperature", "Temperature τ (explicitness)", 0.05],
  ["n_levels", "Explicitness levels", 1],
  ["goal_temperature", "Temperature (goal choice)", 0.05],
  ["dt", "Time on task per cycle", 0.005],
];

let presets = [];
let sc = null;          // the scenario being edited
let result = null;      // last run
let cur = 0;            // selected step index
let playing = null;

/* ---------------------------------------------------------------------------- */
/* editor                                                                        */
/* ---------------------------------------------------------------------------- */

function num(value, onChange, step = 0.05, attrs = {}) {
  return el("input", { type: "number", value, step, ...attrs,
    oninput: (e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) { onChange(v); changed(); } } });
}

function renderEditor() {
  $("#preset-desc").textContent = sc.description || "";

  // drives
  const dt = $("#drives");
  dt.replaceChildren(el("tr", {}, el("th", {}, "Drive"), el("th", {}, "Stimulus"), el("th", {}, "Deficit"),
    el("th", {}, "Decline"), el("th", {})));
  sc.drives.forEach((d, i) => dt.append(el("tr", {},
    el("td", {}, el("input", { type: "text", value: d.name, "aria-label": "Drive name",
      onchange: (e) => renameDrive(i, e.target.value) })),
    el("td", {}, num(d.stimulus, (v) => (d.stimulus = v), 0.05, { "aria-label": `${d.name} stimulus` })),
    el("td", {}, num(d.deficit, (v) => (d.deficit = v), 0.05, { "aria-label": `${d.name} deficit` })),
    el("td", {}, num(d.k || 0, (v) => (d.k = v), 0.02, { "aria-label": `${d.name} decline rate`, min: 0 })),
    el("td", {}, el("button", { class: "icon-btn", type: "button", "aria-label": `Remove ${d.name}`,
      onclick: () => { sc.drives.splice(i, 1); renderEditor(); changed(); } }, "✕")))));

  // goals
  const gWrap = $("#goals");
  gWrap.replaceChildren();
  sc.goals.forEach((g, gi) => {
    const table = el("table", { class: "mini-table" },
      el("tr", {}, el("th", {}, "Drive"), el("th", {}, "Relevance"), el("th", {}, "Satisfaction")));
    sc.drives.forEach((d) => table.append(el("tr", {}, el("td", {}, prettyName(d.name)),
      el("td", {}, num(g.relevance[d.name] ?? 0, (v) => (g.relevance[d.name] = v), 0.1, { "aria-label": `${g.name} relevance of ${d.name}` })),
      el("td", {}, num(g.satisfaction[d.name] ?? 0, (v) => (g.satisfaction[d.name] = v), 0.05, { "aria-label": `${g.name} satisfaction of ${d.name}` })))));
    gWrap.append(el("div", { class: "goal-card" },
      el("div", { class: "row" },
        el("input", { type: "text", value: g.name, "aria-label": "Goal name", onchange: (e) => renameGoal(gi, e.target.value) }),
        el("button", { class: "icon-btn", type: "button", "aria-label": `Remove goal ${g.name}`,
          onclick: () => { sc.goals.splice(gi, 1); renderEditor(); changed(); } }, "✕")),
      table));
  });

  // metacognition
  const m = sc.mcs;
  const mWrap = $("#mcs");
  mWrap.replaceChildren(
    el("div", { class: "field" }, el("label", { for: "mode" }, "Explicitness set by"),
      el("select", { id: "mode", onchange: (e) => { m.mode = e.target.value; renderEditor(); changed(); } },
        el("option", { value: "utility", selected: m.mode !== "fixed" }, "utility"),
        el("option", { value: "fixed", selected: m.mode === "fixed" }, "situation"))),
  );
  if (m.mode === "fixed") {
    mWrap.append(el("div", { class: "field" }, el("label", {}, "Fixed explicitness (0-1)"),
      num(m.fixed ?? 0.8, (v) => (m.fixed = v), 0.05, { min: 0, max: 1 })));
  }
  MCS_FIELDS.forEach(([k, label, step]) => mWrap.append(el("div", { class: "field" }, el("label", {}, label),
    num(m[k], (v) => (m[k] = v), step, { "aria-label": label }))));

  // state
  const sWrap = $("#state");
  sWrap.replaceChildren();
  Object.keys(sc.state).forEach((f) => sWrap.append(el("div", { class: "feature" },
    el("input", { type: "checkbox", id: `f-${f}`, checked: !!sc.state[f],
      onchange: (e) => { sc.state[f] = e.target.checked; changed(); } }),
    el("label", { for: `f-${f}` }, prettyName(f)),
    el("button", { class: "icon-btn", type: "button", "aria-label": `Remove ${f}`,
      onclick: () => { delete sc.state[f]; sc.rules.forEach((r) => delete r.if[f]); renderEditor(); changed(); } }, "✕"))));

  // actions
  const aT = $("#actions");
  aT.replaceChildren(el("tr", {}, el("th", {}, "Action"), el("th", {}, "Q-value"), el("th", {})));
  sc.actions.forEach((a, i) => aT.append(el("tr", {},
    el("td", {}, prettyName(a)),
    el("td", {}, num(sc.q_values[a] ?? 0, (v) => (sc.q_values[a] = v), 0.05, { "aria-label": `Q-value of ${a}` })),
    el("td", {}, el("button", { class: "icon-btn", type: "button", "aria-label": `Remove ${a}`,
      onclick: () => { sc.actions.splice(i, 1); delete sc.q_values[a]; sc.rules = sc.rules.filter((r) => r.then !== a); renderEditor(); changed(); } }, "✕")))));
  aT.append(el("tr", {}, el("td", {}, "Temperature"), el("td", {}, num(sc.bottom_temperature, (v) => (sc.bottom_temperature = v), 0.01)), el("td", {})));

  // rules
  const rWrap = $("#rules");
  rWrap.replaceChildren();
  sc.rules.forEach((r, i) => {
    const conds = el("div", { class: "conds" });
    Object.keys(sc.state).forEach((f) => {
      const v = r.if[f] === undefined ? "any" : r.if[f] ? "yes" : "no";
      conds.append(el("label", {}, `${prettyName(f)} `, el("select", {
        onchange: (e) => { if (e.target.value === "any") delete r.if[f]; else r.if[f] = e.target.value === "yes"; delete r.text; renderEditor(); changed(); } },
        ...["any", "yes", "no"].map((o) => el("option", { value: o, selected: o === v }, o)))));
    });
    rWrap.append(el("div", { class: "rule-card" },
      el("div", { class: "row" },
        el("input", { type: "text", value: r.name, style: "max-width:70px", "aria-label": "Rule name",
          onchange: (e) => { r.name = e.target.value; changed(); } }),
        el("span", { class: "hint" }, "goal"),
        el("select", { "aria-label": "Rule goal", onchange: (e) => { r.goal = e.target.value || null; delete r.text; renderEditor(); changed(); } },
          el("option", { value: "", selected: !r.goal }, "any"),
          ...sc.goals.map((g) => el("option", { value: g.name, selected: r.goal === g.name }, prettyName(g.name)))),
        el("button", { class: "icon-btn", type: "button", "aria-label": `Remove rule ${r.name}`,
          onclick: () => { sc.rules.splice(i, 1); renderEditor(); changed(); } }, "✕")),
      conds,
      el("div", { class: "row" }, el("span", { class: "hint" }, "then"),
        el("select", { "aria-label": "Rule action", onchange: (e) => { r.then = e.target.value; delete r.text; renderEditor(); changed(); } },
          ...sc.actions.map((a) => el("option", { value: a, selected: r.then === a }, prettyName(a)))),
        el("span", { class: "hint" }, "support"),
        num(r.support ?? 1, (v) => (r.support = v), 0.1, { style: "max-width:70px", "aria-label": "Rule support" })),
      el("p", { class: "rule-text" }, ruleText(r))));
  });
}

function ruleText(r) {
  if (r.text) return r.text;
  const parts = Object.entries(r.if).map(([f, v]) => `${prettyName(f)}${v ? "" : " is false"}`);
  if (r.goal) parts.unshift(`goal = ${prettyName(r.goal)}`);
  return `IF ${parts.length ? parts.join(" AND ") : "always"} THEN ${prettyName(r.then)}`;
}

function renameDrive(i, name) {
  name = name.trim().replace(/\s+/g, "_");
  if (!name) return;
  const old = sc.drives[i].name;
  sc.drives[i].name = name;
  sc.goals.forEach((g) => {
    for (const key of ["relevance", "satisfaction"]) if (old in g[key]) { g[key][name] = g[key][old]; delete g[key][old]; }
  });
  renderEditor(); changed();
}

function renameGoal(i, name) {
  name = name.trim().replace(/\s+/g, "_");
  if (!name) return;
  const old = sc.goals[i].name;
  sc.goals[i].name = name;
  sc.rules.forEach((r) => { if (r.goal === old) r.goal = name; });
  renderEditor(); changed();
}

/* ---------------------------------------------------------------------------- */
/* run                                                                            */
/* ---------------------------------------------------------------------------- */

const changed = debounce(runModel, 200);

async function runModel() {
  const err = $("#error");
  try {
    document.body.classList.add("busy");
    const steps = +$("#steps").value || 40, seed = +$("#seed").value || 0;
    result = window.CORE_LOCAL ? window.CoreEngine.run(sc, steps, seed)       // no server: run in the browser
                               : await api("/api/core/run", { scenario: sc, steps, seed });
    err.hidden = true;
    cur = Math.min(cur, result.steps.length - 1);
    $("#t").max = result.steps.length;
    $("#t-max").textContent = result.steps.length;
    renderTimeline();
    select(cur, false);
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    document.body.classList.remove("busy");
  }
}

function select(i, animate = true) {
  cur = i;
  $("#t").value = i + 1;
  $("#t-label").textContent = i + 1;
  drawArch(result.steps[i], animate);
  renderInspector(result.steps[i]);
  for (const id of ["c-drives", "c-expl"]) {
    const c = charts[id];
    if (c) { c.options.plugins.cursor = i; c.update("none"); }
  }
}

/* ---------------------------------------------------------------------------- */
/* architecture diagram                                                          */
/* ---------------------------------------------------------------------------- */

const SVGNS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}, text) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}

function wrapText(str, max = 16) {
  const words = prettyName(str).split(" ");
  const lines = [""];
  for (const w of words) {
    if ((lines[lines.length - 1] + " " + w).trim().length > max && lines[lines.length - 1]) lines.push(w);
    else lines[lines.length - 1] = (lines[lines.length - 1] + " " + w).trim();
  }
  return lines;
}

let horizonNow = null;

function drawArch(st, animate) {
  const svg = $("#arch");
  svg.replaceChildren(s("title", { id: "arch-title" }, `Clarion architecture at cycle ${st.t}`));
  const defs = s("defs");
  const mk = s("marker", { id: "arrowhead", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" });
  mk.append(s("path", { d: "M0,0 L10,5 L0,10 z", fill: css("--ink-3") }));
  defs.append(mk);
  svg.append(defs);

  const Y = 20, H = 290;
  const box = (x, w, title, sub) => {
    svg.append(s("rect", { x, y: Y, width: w, height: H, rx: 8, class: "box" }));
    svg.append(s("text", { x: x + 14, y: Y + 24, class: "label" }, title));
    if (sub) svg.append(s("text", { x: x + 14, y: Y + 41, class: "sub" }, sub));
  };
  const arrow = (x1, x2, y = Y + H / 2) => svg.append(s("path", { d: `M${x1},${y} L${x2},${y}`, class: "arrow" }));

  // 1. situation
  box(0, 150, "Situation", "state features");
  Object.entries(sc.state).slice(0, 8).forEach(([f, v], i) => {
    const y = Y + 70 + i * 26;
    svg.append(s("text", { x: 14, y, class: "sub", style: `fill:${v ? css("--ink") : css("--ink-3")}` }, `${v ? "✓" : "–"}  ${prettyName(f)}`.slice(0, 22)));
  });
  arrow(150, 172);

  // 2. MS
  box(175, 200, "Motivational subsystem", "drive strength = stimulus × deficit");
  const drives = Object.entries(st.drives);
  drives.slice(0, 4).forEach(([name, d], i) => {
    const y = Y + 70 + i * 50;
    const color = css(DRIVE_COLORS[i % DRIVE_COLORS.length]);
    svg.append(s("text", { x: 189, y, class: "sub" }, prettyName(name)));
    svg.append(s("rect", { x: 189, y: y + 6, width: 150, height: 9, rx: 4, fill: css("--sunken") }));
    svg.append(s("rect", { x: 189, y: y + 6, width: Math.max(2, 150 * Math.min(d.strength, 1)), height: 9, rx: 4, fill: color }));
    svg.append(s("text", { x: 345, y: y + 15, class: "sub num" }, fmt(d.strength)));
    svg.append(s("text", { x: 189, y: y + 30, class: "sub" }, `${fmt(d.stimulus_level)} × ${fmt(d.deficit)}`));
  });
  arrow(375, 397);

  // 3. MCS
  box(400, 200, "Metacognitive subsystem", "goal, value(g), explicitness");
  svg.append(s("text", { x: 414, y: Y + 74, class: "sub" }, "goal"));
  svg.append(s("text", { x: 414, y: Y + 92, class: "label serif" }, prettyName(st.goal).slice(0, 24)));
  svg.append(s("text", { x: 414, y: Y + 108, class: "sub" }, `chosen with p = ${fmt(st.goal_probs[st.goal])}`));
  svg.append(s("text", { x: 414, y: Y + 138, class: "sub" }, "value(g)"));
  svg.append(s("text", { x: 500, y: Y + 138, class: "label num" }, fmt(st.value, 3)));
  const net = sc.mcs.alpha * st.value - sc.mcs.cost;
  svg.append(s("text", { x: 414, y: Y + 158, class: "sub" }, "α·value − c"));
  svg.append(s("text", { x: 500, y: Y + 158, class: "label num", style: `fill:${net >= 0 ? css("--explicit") : css("--implicit")}` }, fmt(net, 3)));
  // gauge
  const gx = 414, gy = Y + 190, gw = 170;
  svg.append(s("text", { x: gx, y: gy - 8, class: "sub" }, st.level >= 0 ? `explicitness level ${st.level} of ${sc.mcs.n_levels - 1}` : "explicitness set by situation"));
  svg.append(s("rect", { x: gx, y: gy, width: gw, height: 10, rx: 5, fill: css("--implicit-soft") }));
  svg.append(s("rect", { x: gx, y: gy, width: gw * st.p_explicit, height: 10, rx: 5, fill: css("--explicit") }));
  svg.append(s("text", { x: gx, y: gy + 28, class: "sub" }, "p(top level acts)"));
  svg.append(s("text", { x: gx + gw, y: gy + 28, class: "label num", "text-anchor": "end" }, fmt(st.p_explicit)));
  if (st.expected_explicitness !== null)
    svg.append(s("text", { x: gx, y: gy + 48, class: "sub" }, `expected over levels: ${fmt(st.expected_explicitness)}`));
  arrow(600, 622);

  // 4. ACS with moving horizon
  const ax = 625, aw = 240, top = Y, inner = H;
  svg.append(s("text", { x: ax, y: 12, class: "sub" }, "Action-centered subsystem"));
  const target = top + inner * st.p_explicit;
  const from = horizonNow === null ? target : horizonNow;
  const topBand = s("rect", { x: ax, y: top, width: aw, height: from - top, class: "top-band", rx: 0 });
  const botBand = s("rect", { x: ax, y: from, width: aw, height: top + inner - from, class: "bottom-band" });
  const line = s("line", { x1: ax - 6, x2: ax + aw + 6, y1: from, y2: from, class: "horizon" });
  const frame = s("rect", { x: ax, y: top, width: aw, height: inner, rx: 8, fill: "none", stroke: css("--rule-strong") });
  svg.append(topBand, botBand, line, frame);

  const topActs = st.level_used === "top";
  const tgrp = s("g", { class: topActs ? "" : "dim" });
  tgrp.append(s("text", { x: ax + 14, y: top + 22, class: "label", style: `fill:${css("--explicit")}` }, "Top level: explicit rules"));
  tgrp.append(s("text", { x: ax + 14, y: top + 40, class: "sub" },
    st.top_rule ? `rule ${st.top_rule} → ${prettyName(st.top_choice)}`.slice(0, 36) : "no rule applies"));
  tgrp.append(s("text", { x: ax + 14, y: top + 56, class: "sub" }, `${st.applicable_rules.length} rule(s) applicable`));
  svg.append(tgrp);

  const bgrp = s("g", { class: topActs ? "dim" : "" });
  bgrp.append(s("text", { x: ax + 14, y: top + inner - 92, class: "label", style: `fill:${css("--implicit")}` }, "Bottom level: implicit network"));
  const qs = Object.entries(st.bottom_probs);
  qs.slice(0, 3).forEach(([a, p], i) => {
    const y = top + inner - 74 + i * 22;
    bgrp.append(s("text", { x: ax + 14, y: y + 9, class: "sub" }, prettyName(a).slice(0, 18)));
    bgrp.append(s("rect", { x: ax + 140, y, width: 80, height: 9, rx: 4, fill: css("--surface"), opacity: 0.7 }));
    bgrp.append(s("rect", { x: ax + 140, y, width: Math.max(2, 80 * p), height: 9, rx: 4,
      fill: a === st.bottom_choice ? css("--implicit") : css("--ink-3") }));
  });
  svg.append(bgrp);
  if (st.conflict) svg.append(s("text", { x: ax + aw, y: top + inner + 16, class: "sub", "text-anchor": "end" },
    "the two levels proposed different actions"));

  // 5. action
  arrow(865, 885);
  svg.append(s("rect", { x: 888, y: Y + H / 2 - 52, width: 90, height: 104, rx: 8, fill: css("--surface"),
    stroke: topActs ? css("--explicit") : css("--implicit"), "stroke-width": 2.5 }));
  svg.append(s("text", { x: 898, y: Y + H / 2 - 30, class: "sub" }, "action"));
  wrapText(st.action, 11).slice(0, 4).forEach((ln, i) =>
    svg.append(s("text", { x: 898, y: Y + H / 2 - 10 + i * 16, class: "label", style: "font-size:12.5px" }, ln)));
  svg.append(s("text", { x: 898, y: Y + H / 2 + 44, class: "sub", style: `fill:${topActs ? css("--explicit") : css("--implicit")}` },
    topActs ? "by top level" : "by bottom level"));

  // the one orchestrated motion: the horizon glides to the new explicitness
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!animate || reduce || from === target) { setHorizon(target); return; }
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / 260), e = 1 - Math.pow(1 - k, 3);
    setHorizon(from + (target - from) * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);

  function setHorizon(y) {
    horizonNow = y;
    topBand.setAttribute("height", Math.max(0, y - top));
    botBand.setAttribute("y", y); botBand.setAttribute("height", Math.max(0, top + inner - y));
    line.setAttribute("y1", y); line.setAttribute("y2", y);
  }
}

/* ---------------------------------------------------------------------------- */
/* timeline + inspector                                                          */
/* ---------------------------------------------------------------------------- */

const cursorPlugin = {
  id: "cursor",
  afterDraw(c) {
    const i = c.options.plugins.cursor;
    if (i === undefined) return;
    const x = c.scales.x.getPixelForValue(i);
    const { ctx, chartArea } = c;
    ctx.save(); ctx.strokeStyle = css("--ink-3"); ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke(); ctx.restore();
  },
};

function renderTimeline() {
  const steps = result.steps;
  const labels = steps.map((x) => x.t);
  const names = Object.keys(steps[0].drives);
  $("#drive-legend").replaceChildren(...names.map((n, i) =>
    el("span", {}, el("i", { style: `background:var(${DRIVE_COLORS[i % DRIVE_COLORS.length]})` }), prettyName(n))));
  const onClick = (evt, els, c) => {
    const x = c.scales.x.getValueForPixel(evt.x);
    if (x !== undefined) select(Math.max(0, Math.min(steps.length - 1, Math.round(x))));
  };
  chart("c-drives", {
    type: "line",
    data: { labels, datasets: names.map((n, i) => ({ label: prettyName(n), data: steps.map((x) => x.drives[n].strength),
      borderColor: css(DRIVE_COLORS[i % DRIVE_COLORS.length]), pointRadius: 0, tension: 0.2 })) },
    options: { onClick, interaction: { mode: "index", intersect: false },
      scales: { x: { title: { display: true, text: "Decision cycle" }, grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 12 } },
                y: { min: 0, title: { display: true, text: "Strength" } } },
      plugins: { cursor: cur, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y)}` } } } },
    plugins: [cursorPlugin],
  });
  chart("c-expl", {
    type: "line",
    data: { labels, datasets: [
      { label: "p(top level acts)", data: steps.map((x) => x.p_explicit), borderColor: css("--rule-strong"), borderWidth: 1.5,
        pointRadius: steps.map((x) => (x.conflict ? 6 : 5)),
        pointBackgroundColor: steps.map((x) => (x.level_used === "top" ? css("--explicit") : css("--implicit"))),
        pointBorderColor: steps.map((x) => (x.conflict ? css("--ink") : css("--surface"))),
        pointBorderWidth: steps.map((x) => (x.conflict ? 2 : 1)), stepped: false },
      { label: "Expected explicitness", data: steps.map((x) => x.expected_explicitness), borderColor: css("--ink-3"),
        borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 },
    ] },
    options: { onClick, interaction: { mode: "nearest", intersect: false, axis: "x" },
      scales: { x: { title: { display: true, text: "Decision cycle" }, grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 12 } },
                y: { min: 0, max: 1, title: { display: true, text: "Explicitness" } } },
      plugins: { cursor: cur, tooltip: { callbacks: {
        label: (c) => c.datasetIndex === 1 ? `Expected: ${fmt(c.parsed.y)}` :
          `p(top) ${fmt(c.parsed.y)}: ${steps[c.dataIndex].level_used} level → ${prettyName(steps[c.dataIndex].action)}` } } } },
    plugins: [cursorPlugin],
  });
  const sm = result.summary;
  $("#summary").replaceChildren(
    el("dt", {}, "Top level acted"), el("dd", {}, `${Math.round(sm.top_share * 100)}% of cycles`),
    el("dt", {}, "Levels disagreed"), el("dd", {}, `${sm.conflicts} cycles, top level won ${sm.conflicts_won_by_top}`));
}

function renderInspector(st) {
  $("#inspect-title").textContent = `Inside cycle ${st.t}`;
  const m = sc.mcs;
  const E = $("#equations");
  E.replaceChildren();
  const eq = (html) => E.append(el("p", { class: "eq", html }));
  E.append(el("h3", {}, "Drive strength (Eq. 5)"));
  Object.entries(st.drives).forEach(([n, d]) =>
    eq(`<var>ds</var><sub>${prettyName(n)}</sub> = ${fmt(d.stimulus_level)} × ${fmt(d.deficit)} = <span class="res">${fmt(d.strength, 3)}</span>`));
  E.append(el("h3", {}, "Goal strength (Eq. 7) and goal choice"));
  Object.entries(st.goal_strengths).forEach(([g, v]) =>
    eq(`<var>gs</var><sub>${prettyName(g)}</sub> = <span class="res">${fmt(v, 3)}</span>, chosen with p = ${fmt(st.goal_probs[g])}${g === st.goal ? " <b>(chosen)</b>" : ""}`));
  const goal = sc.goals.find((g) => g.name === st.goal);
  const terms = Object.entries(goal.satisfaction).filter(([, v]) => v).map(([d, v]) => `${fmt(st.drives[d]?.strength ?? 0, 3)} × ${fmt(v)}`);
  E.append(el("h3", {}, "Value of the goal and utility of explicitness (Eqs. 2-4)"));
  eq(`value(${prettyName(st.goal)}) = ${terms.join(" + ") || "0"} = <span class="res">${fmt(st.value, 3)}</span>`);
  const net = m.alpha * st.value - m.cost;
  eq(`<var>U</var><sub>j</sub> = level<sub>j</sub> × (${fmt(m.alpha)} × ${fmt(st.value, 3)} − ${fmt(m.cost, 3)}) = level<sub>j</sub> × <span class="res ${net >= 0 ? "explicit" : "implicit"}">${fmt(net, 3)}</span>`);
  eq(net >= 0 ? "Higher explicitness has higher utility, so explicit processing is favored."
              : "Higher explicitness has lower utility, so implicit processing is favored.");
  if (st.level >= 0) eq(`Chosen level ${st.level} → p(top level acts) = ${st.level}/${m.n_levels - 1} = <span class="res">${fmt(st.p_explicit)}</span>`);
  else eq(`Explicitness set by the situation: <span class="res">${fmt(st.p_explicit)}</span>`);
  E.append(el("h3", {}, "Action selection (ACS)"));
  eq(`<span class="explicit">Top level:</span> ${st.top_rule ? `rule ${st.top_rule} proposes <b>${prettyName(st.top_choice)}</b>` : "no rule applies"}`);
  eq(`<span class="implicit">Bottom level:</span> Q-values ${Object.entries(st.q_values).map(([a, q]) => `${prettyName(a)} ${fmt(q)}`).join(", ")} → proposes <b>${prettyName(st.bottom_choice)}</b>`);
  eq(`The <b class="${st.level_used === "top" ? "explicit" : "implicit"}">${st.level_used} level</b> acted: <b>${prettyName(st.action)}</b>${st.conflict ? " (the levels disagreed)" : ""}.`);

  const L = st.utilities.map((_, j) => j);
  chart("c-util", { type: "bar", data: { labels: L, datasets: [{ data: st.utilities,
      backgroundColor: st.utilities.map((u) => (u >= 0 ? css("--explicit") : css("--implicit"))), borderRadius: 2 }] },
    options: { scales: { x: { title: { display: true, text: "Explicitness level" }, grid: { display: false } }, y: { title: { display: true, text: "Utility" } } },
      plugins: { tooltip: { callbacks: { label: (c) => `U = ${fmt(c.parsed.y, 3)}` } } } } });
  const probs = st.level_probs || [];
  chart("c-prob", { type: "bar", data: { labels: L, datasets: [{ data: probs,
      backgroundColor: L.map((j) => (j === st.level ? css("--ink") : css("--rule-strong"))), borderRadius: 2 }] },
    options: { scales: { x: { title: { display: true, text: "Explicitness level (dark = chosen this cycle)" }, grid: { display: false } },
      y: { min: 0, title: { display: true, text: "Probability" } } },
      plugins: { tooltip: { callbacks: { label: (c) => `p = ${fmt(c.parsed.y, 3)}` } } } } });
}

/* ---------------------------------------------------------------------------- */
/* wiring                                                                         */
/* ---------------------------------------------------------------------------- */

function load(scenario) {
  sc = JSON.parse(JSON.stringify(scenario));
  sc.rules = sc.rules || []; sc.state = sc.state || {}; sc.q_values = sc.q_values || {};
  sc.goals.forEach((g) => { g.relevance = g.relevance || {}; g.satisfaction = g.satisfaction || {}; });
  $("#title").textContent = sc.name || "Custom scenario";
  horizonNow = null; cur = 0;
  renderEditor();
  runModel();
}

async function init() {
  initTheme();
  presets = window.CORE_LOCAL ? window.CORE_LOCAL.presets : await api("/api/core/presets");
  $("#preset").replaceChildren(...presets.map((p, i) => el("option", { value: i }, p.name)));
  $("#preset").addEventListener("change", (e) => load(presets[+e.target.value]));
  $("#t").addEventListener("input", (e) => select(+e.target.value - 1));
  $("#steps").addEventListener("change", changed);
  $("#seed").addEventListener("change", changed);
  $("#add-drive").addEventListener("click", () => { sc.drives.push({ name: `drive_${sc.drives.length + 1}`, stimulus: 0.5, deficit: 0.5, k: 0 }); renderEditor(); changed(); });
  $("#add-goal").addEventListener("click", () => { sc.goals.push({ name: `goal_${sc.goals.length + 1}`, relevance: {}, satisfaction: {} }); renderEditor(); changed(); });
  $("#add-rule").addEventListener("click", () => { sc.rules.push({ name: `R${sc.rules.length + 1}`, if: {}, goal: null, then: sc.actions[0], support: 1 }); renderEditor(); changed(); });
  $("#add-feature").addEventListener("click", () => { const n = $("#new-feature").value.trim().replace(/\s+/g, "_"); if (n && !(n in sc.state)) { sc.state[n] = true; $("#new-feature").value = ""; renderEditor(); changed(); } });
  $("#add-action").addEventListener("click", () => { const n = $("#new-action").value.trim().replace(/\s+/g, "_"); if (n && !sc.actions.includes(n)) { sc.actions.push(n); sc.q_values[n] = 0.5; $("#new-action").value = ""; renderEditor(); changed(); } });
  $("#play").addEventListener("click", () => {
    if (playing) { clearInterval(playing); playing = null; $("#play").textContent = "Play"; return; }
    $("#play").textContent = "Pause";
    if (cur >= result.steps.length - 1) select(0);
    playing = setInterval(() => {
      if (cur >= result.steps.length - 1) { clearInterval(playing); playing = null; $("#play").textContent = "Play"; return; }
      select(cur + 1);
    }, 450);
  });
  $("#export").addEventListener("click", () => {
    const a = el("a", { href: URL.createObjectURL(new Blob([JSON.stringify(sc, null, 2)], { type: "application/json" })),
      download: `${(sc.name || "scenario").replace(/[^a-z0-9]+/gi, "_").toLowerCase()}.json` });
    document.body.append(a); a.click(); a.remove();
  });
  $("#import").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    try { load(JSON.parse(await f.text())); } catch (_) { $("#error").textContent = "That file is not a valid scenario (JSON)."; $("#error").hidden = false; }
  });
  window.addEventListener("themechange", () => { if (result) { renderTimeline(); select(cur, false); } });
  load(presets[0]);
}

init();
