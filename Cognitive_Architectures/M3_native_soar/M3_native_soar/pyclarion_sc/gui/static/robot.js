/* Clarion Robot World: a cleaning robot controlled by the Clarion self-control architecture.
   Data sources:
     app       Python server (/api/robot/...), any seed
     static    files next to the page (window.ROBOT_STATIC = "data/"), pre-computed seeds
     embedded  everything inside the page (window.ROBOT_DATA), pre-computed seeds */

const MODE = window.ROBOT_DATA ? "embedded" : window.ROBOT_STATIC ? "static" : "app";
let INDEX = null;
let sel = { scenario: null, condition: null, controller: "clarion", seed: 1 };
let ep = null, i = 0, timer = null, dirt = null;
let levelAt = [], decAt = [], tally = [];
const DRIVE_VARS = { achievement: "--drive-1", energy: "--drive-2" };

/* ---------------------------------------------------------------- data */

async function loadIndex() {
  if (MODE === "embedded") return window.ROBOT_DATA.index;
  if (MODE === "static") {
    const r = await fetch(`${window.ROBOT_STATIC}index.json`);
    if (!r.ok) throw new Error("Could not load data/index.json. Upload the data folder next to index.html.");
    return r.json();
  }
  return api("/api/robot/index");
}

async function loadEpisode({ scenario, condition, controller, seed }) {
  const key = `${scenario}_${condition}_${controller}_${seed}`;
  if (MODE === "embedded") {
    const e = window.ROBOT_DATA.episodes[key];
    if (!e) throw new Error("This combination is not included in this preview.");
    return e;
  }
  if (MODE === "static") {
    const r = await fetch(`${window.ROBOT_STATIC}episodes/${key}.json`);
    if (!r.ok) throw new Error(`Episode file ${key}.json is missing from the data folder.`);
    return r.json();
  }
  return api("/api/robot/run", { scenario, condition, controller, seed });
}

/* ---------------------------------------------------------------- run */

async function run() {
  stop();
  const err = $("#error");
  try {
    ep = await loadEpisode(sel);
    err.hidden = true;
  } catch (e) {
    err.textContent = e.message; err.hidden = false; return;
  }
  prepare();
  renderStatic();
  goTo(0);
  play();
}

function prepare() {
  levelAt = []; decAt = []; tally = [];
  const rooms = ep.layout.rooms, assigned = ep.layout.assigned, tempting = ep.layout.tempting;
  const roomOf = (r, c) => Object.keys(rooms).find((n) => { const [a, b, x, y] = rooms[n]; return r >= a && r <= b && c >= x && c <= y; });
  const total = { assigned: 0, tempting: 0 };
  ep.layout.dirt.forEach((row, r) => row.forEach((d, c) => {
    const n = roomOf(r, c);
    if (assigned.includes(n)) total.assigned += d; else if (n === tempting) total.tempting += d;
  }));
  let lv = null, d = -1;
  const acc = { assigned: 0, tempting: 0, decisions: 0, plan: 0, conflicts: 0, violations: 0 };
  ep.steps.forEach((s) => {
    if (s.decision !== null && s.decision !== undefined) {
      d = s.decision;
      const dec = ep.decisions[d];
      lv = dec.level;
      if (!dec.reflex) { acc.decisions++; acc.plan += dec.level === "top"; acc.conflicts += dec.conflict; }
    }
    const n = roomOf(s.pos[0], s.pos[1]);
    if (s.cleaned > 0) { if (assigned.includes(n)) acc.assigned += s.cleaned; else if (n === tempting) acc.tempting += s.cleaned; }
    acc.violations += s.quiet_violation ? 1 : 0;
    levelAt.push(lv); decAt.push(d);
    tally.push({ ...acc, assignedPct: (100 * acc.assigned) / total.assigned, temptingPct: (100 * acc.tempting) / total.tempting });
  });
}

/* ---------------------------------------------------------------- canvas */

function dirtAt(step) {
  const m = ep.layout.dirt.map((row) => row.slice());
  for (let k = 0; k < step; k++) {
    const s = ep.steps[k];
    if (s.cleaned > 0) m[s.pos[0]][s.pos[1]] = Math.max(0, m[s.pos[0]][s.pos[1]] - s.cleaned);
  }
  return m;
}

function hash(r, c, k) { const x = Math.sin(r * 127.1 + c * 311.7 + k * 74.7) * 43758.5453; return x - Math.floor(x); }

function sizeCanvas() {
  const wrap = $("#canvas-wrap"), cv = $("#world");
  const rows = ep.layout.rows, H = rows.length, W = rows[0].length;
  const pad = 36;
  const availW = wrap.clientWidth - pad, availH = wrap.clientHeight - pad;
  const cs = Math.max(8, Math.floor(Math.min(availW / W, availH > 80 ? availH / H : availW / W)));
  const dpr = window.devicePixelRatio || 1;
  cv.style.width = `${cs * W}px`; cv.style.height = `${cs * H}px`;
  cv.width = Math.round(cs * W * dpr); cv.height = Math.round(cs * H * dpr);
  cv.dataset.cs = cs;
}

function drawWorld() {
  const cv = $("#world");
  const rows = ep.layout.rows, H = rows.length, W = rows[0].length;
  const cs = +cv.dataset.cs, dpr = window.devicePixelRatio || 1;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cs * W, cs * H);
  const font = (w, px) => `${w} ${px}px "Public Sans", system-ui, sans-serif`;

  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    ctx.fillStyle = rows[r][c] === "#" ? css("--wall") : css("--floor");
    ctx.fillRect(c * cs, r * cs, cs + 0.5, cs + 0.5);
  }
  const [r0, r1, c0, c1] = ep.layout.rooms[ep.layout.tempting];
  ctx.fillStyle = css("--floor-alt");
  ctx.fillRect(c0 * cs, r0 * cs, (c1 - c0 + 1) * cs, (r1 - r0 + 1) * cs);
  ctx.save(); ctx.setLineDash([6, 5]); ctx.strokeStyle = css("--ink-3"); ctx.lineWidth = 1.5;
  ctx.strokeRect(c0 * cs + 3, r0 * cs + 3, (c1 - c0 + 1) * cs - 6, (r1 - r0 + 1) * cs - 6); ctx.restore();
  if (ep.layout.quiet_zone) {
    const [q0, q1, p0, p1] = ep.layout.rooms[ep.layout.quiet_zone];
    ctx.save(); ctx.beginPath(); ctx.rect(p0 * cs, q0 * cs, (p1 - p0 + 1) * cs, (q1 - q0 + 1) * cs); ctx.clip();
    ctx.strokeStyle = css("--explicit"); ctx.globalAlpha = 0.22; ctx.lineWidth = 1;
    for (let x = p0 * cs - (q1 - q0 + 1) * cs; x < (p1 + 1) * cs; x += 10) {
      ctx.beginPath(); ctx.moveTo(x, q0 * cs); ctx.lineTo(x + (q1 - q0 + 1) * cs, (q1 + 1) * cs); ctx.stroke();
    }
    ctx.restore();
  }
  ctx.fillStyle = css("--dirt");
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const d = dirt[r][c];
    if (d <= 0.05) continue;
    const n = Math.ceil(d * 7);
    for (let k = 0; k < n; k++) {
      ctx.beginPath();
      ctx.arc(c * cs + cs * (0.15 + 0.7 * hash(r, c, k)), r * cs + cs * (0.15 + 0.7 * hash(c, r, k + 3)), Math.max(1.2, cs * 0.055), 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.font = font(600, Math.max(10, cs * 0.34));
  ctx.fillStyle = css("--ink-2");
  const label = (name, text) => { const [a, , b] = ep.layout.rooms[name]; ctx.fillText(text, b * cs + 6, a * cs + cs * 0.55); };
  ep.layout.assigned.forEach((n) => label(n, `Room ${n}, assigned${ep.layout.quiet_zone === n ? ", quiet zone" : ""}`));
  label(ep.layout.tempting, `Room ${ep.layout.tempting}, not assigned`);
  const [dr, dc] = ep.layout.dock;
  ctx.fillStyle = css("--good"); ctx.globalAlpha = 0.18; ctx.fillRect(dc * cs + 2, dr * cs + 2, cs - 4, cs - 4); ctx.globalAlpha = 1;
  ctx.strokeStyle = css("--good"); ctx.lineWidth = 2; ctx.strokeRect(dc * cs + 3, dr * cs + 3, cs - 6, cs - 6);
  ctx.fillStyle = css("--ink-2"); ctx.font = font(500, Math.max(9, cs * 0.3)); ctx.fillText("dock", dc * cs + 2, dr * cs - 4);

  for (let k = 0; k < i; k++) {
    const s = ep.steps[k];
    ctx.globalAlpha = i - k < 40 ? 0.85 : 0.3;
    ctx.fillStyle = levelAt[k] === "top" ? css("--explicit") : levelAt[k] === "bottom" ? css("--implicit") : css("--ink-3");
    ctx.beginPath(); ctx.arc((s.pos[1] + 0.5) * cs, (s.pos[0] + 0.5) * cs, cs * 0.1, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;
  const pos = i === 0 ? ep.layout.dock : ep.steps[i - 1].pos;
  const lv = i === 0 ? null : levelAt[i - 1];
  const cx = (pos[1] + 0.5) * cs, cy = (pos[0] + 0.5) * cs;
  ctx.beginPath(); ctx.arc(cx, cy, cs * 0.4, 0, Math.PI * 2);
  ctx.fillStyle = css("--ink"); ctx.fill();
  ctx.lineWidth = Math.max(2, cs * 0.1);
  ctx.strokeStyle = lv === "top" ? css("--explicit") : lv === "bottom" ? css("--implicit") : css("--ink-3");
  ctx.stroke();
  if (i > 1) {
    const prev = ep.steps[i - 2].pos, dx = pos[1] - prev[1], dy = pos[0] - prev[0];
    if (dx || dy) {
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dx * cs * 0.32, cy + dy * cs * 0.32);
      ctx.strokeStyle = css("--surface"); ctx.lineWidth = Math.max(1.5, cs * 0.08); ctx.stroke();
    }
  }
}

/* ---------------------------------------------------------------- live panel */

const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const optText = (o) => (ep.options && ep.options[o]) || prettyName(o);

function renderPanel() {
  const s = i === 0 ? null : ep.steps[i - 1];
  $("#clock").textContent = `step ${i} of ${ep.steps.length}`;
  $("#phase").textContent = s ? cap(s.phase) : "At the dock, about to start";
  const battery = s ? s.battery : 1;
  $("#battery-bar").style.width = `${battery * 100}%`;
  $("#battery-text").textContent = `${Math.round(battery * 100)}% charged`;
  const t = i === 0 ? null : tally[i - 1];
  $("#tally").replaceChildren(
    el("dt", {}, "Assigned dirt removed"), el("dd", {}, t ? `${fmt(t.assignedPct, 0)}%` : "0%"),
    el("dt", {}, "Room C dirt removed"), el("dd", {}, t ? `${fmt(t.temptingPct, 0)}%` : "0%"),
    el("dt", {}, "Plan in control"), el("dd", {}, t && t.decisions ? `${Math.round((100 * t.plan) / t.decisions)}% of ${t.decisions}` : "–"),
    el("dt", {}, "Plan and habit disagreed"), el("dd", {}, t ? `${t.conflicts}` : "0"),
    el("dt", {}, "Planning calls"), el("dd", {}, t ? `${t.plan}` : "0"),
    ...(ep.layout.quiet_zone ? [el("dt", {}, "Quiet-zone rule broken"), el("dd", {}, t ? `${t.violations}` : "0")] : []));
  const di = i === 0 ? -1 : decAt[i - 1];
  if (di < 0) return;
  const d = ep.decisions[di];
  if (d.reflex) {
    $("#decision").replaceChildren(el("p", { class: "acted" }, "Safety reflex: battery critically low, returning to the dock."));
    return;
  }
  $("#drives").replaceChildren(...Object.entries(d.drives).map(([n, v]) => el("div", { class: "drive" },
    el("div", { class: "top-line" }, el("span", {}, cap(n)), el("span", { class: "num" }, fmt(v.strength))),
    el("div", { class: "bar" }, el("div", { style: `width:${Math.min(100, v.strength * 100)}%;background:var(${DRIVE_VARS[n]})` })),
    el("span", { class: "hint num" }, `stimulus ${fmt(v.stimulus)} × deficit ${fmt(v.deficit)}`))));
  $("#goal").textContent = prettyName(d.goal);
  $("#value").textContent = fmt(d.value, 3);
  $("#net").textContent = fmt(d.net, 3);
  $("#net").style.color = d.net >= 0 ? "var(--explicit)" : "var(--implicit)";
  $("#h-top").style.flexBasis = `${Math.max(4, Math.min(96, d.p_explicit * 100))}%`;
  $("#expl-text").textContent =
    d.explicitness_source === "situation" ? `Computer busy: explicitness set to ${fmt(d.p_explicit)} by the situation.`
    : d.explicitness_source === "fixed" ? `This comparison robot uses fixed explicitness ${fmt(d.p_explicit)}.`
    : `Explicitness ${fmt(d.p_explicit)} (level ${d.explicitness_level} of 20), chosen from the utilities.`;
  $("#decision").replaceChildren(...[
    el("p", {}, el("span", { class: "level-top" }, "Plan: "), d.top_option ? `${optText(d.top_option)} (rule ${d.top_rule})` : "no rule applies"),
    el("p", {}, el("span", { class: "level-bottom" }, "Habit: "), optText(d.bottom_option)),
    el("p", { class: "acted" }, `${d.level === "top" ? "The plan" : "The habit"} acted: ${optText(d.option)}`),
    d.conflict ? el("p", { class: "hint" }, "Plan and habit disagreed: a self-control conflict.") : null].filter(Boolean));
}

/* ---------------------------------------------------------------- playback */

function syncCharts() {
  for (const id of ["c-expl", "c-drives"]) if (charts[id]) { charts[id].options.plugins.cursorStep = i; charts[id].update("none"); }
}

function goTo(k) {
  i = Math.max(0, Math.min(ep.steps.length, k));
  dirt = dirtAt(i);
  $("#scrub").value = i;
  drawWorld(); renderPanel(); syncCharts();
}

function advance() {
  if (i >= ep.steps.length) { stop(); return; }
  const s = ep.steps[i];
  if (s.cleaned > 0) dirt[s.pos[0]][s.pos[1]] = Math.max(0, dirt[s.pos[0]][s.pos[1]] - s.cleaned);
  i += 1;
  $("#scrub").value = i;
  drawWorld(); renderPanel();
  if (i % 3 === 0 || i === ep.steps.length) syncCharts();
}

function play() {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) { goTo(ep.steps.length); return; }
  stop();
  if (i >= ep.steps.length) goTo(0);
  $("#play").classList.add("playing"); $("#play").setAttribute("aria-label", "Pause");
  timer = setInterval(advance, 1000 / +$("#speed").value);
}

function stop() {
  if (timer) clearInterval(timer);
  timer = null;
  $("#play").classList.remove("playing"); $("#play").setAttribute("aria-label", "Play");
}

/* ---------------------------------------------------------------- drawers */

const DRAWERS = { settings: "Scenario and robot", about: "About this scenario", charts: "Over the episode",
                  plan: "The robot's plan and habit", useful: "Is self-control useful for a robot?" };
let openDrawer = null, lastFocus = null;

function toggleDrawer(name) {
  if (openDrawer === name) return closeDrawer();
  lastFocus = document.activeElement;
  openDrawer = name;
  $("#drawer-title").textContent = DRAWERS[name];
  $$(".drawer-body").forEach((b) => { b.hidden = b.dataset.panel !== name; });
  $$("[data-drawer]").forEach((b) => b.setAttribute("aria-expanded", b.dataset.drawer === name ? "true" : "false"));
  $("#drawer").hidden = false; $("#backdrop").hidden = false;
  if (name === "charts") renderCharts();
  $("#drawer-close").focus();
}

function closeDrawer() {
  openDrawer = null;
  $("#drawer").hidden = true; $("#backdrop").hidden = true;
  $$("[data-drawer]").forEach((b) => b.setAttribute("aria-expanded", "false"));
  if (lastFocus) lastFocus.focus();
}

/* ---------------------------------------------------------------- static content */

function renderStatic() {
  const sc = INDEX.scenarios.find((s) => s.id === sel.scenario);
  const cond = sc.conditions.find((c) => c.id === sel.condition);
  const ctl = INDEX.controllers.find((c) => c.id === sel.controller);
  $("#now").textContent = `${sc.title.split(":")[0]}: ${cond.label.charAt(0).toLowerCase()}${cond.label.slice(1)}. ${ctl.label}, seed ${sel.seed}`;
  $("#robot-name").textContent = ctl.label;
  $("#paper").textContent = `Based on ${sc.paper}`;
  $("#story").textContent = sc.story;
  $("#rules").replaceChildren(...ep.rules.map((r) => el("li", {}, r)));
  $("#options").replaceChildren(...Object.entries(ep.options).map(([k, v]) => el("li", {}, el("b", {}, prettyName(k)), `: ${v}`)));
  $("#scrub").max = ep.steps.length;
  sizeCanvas();
  renderSettings();
  renderUseful();
  if (openDrawer === "charts") renderCharts();
}

function radio(list, current, onPick) {
  return list.map((o) => el("button", { class: "option-btn", type: "button", role: "radio",
    "aria-checked": o.id === current ? "true" : "false", onclick: () => onPick(o.id) },
    el("b", {}, o.title || o.label), o.sub ? el("span", {}, o.sub) : null));
}

function renderSettings() {
  const sc = INDEX.scenarios.find((s) => s.id === sel.scenario);
  $("#scenario-list").replaceChildren(...radio(INDEX.scenarios.map((s) => ({ id: s.id, title: s.title.split(":")[0], sub: s.title.split(":").slice(1).join(":").trim() })),
    sel.scenario, (id) => { sel.scenario = id; sel.condition = INDEX.scenarios.find((s) => s.id === id).conditions[0].id; run(); }));
  $("#condition-list").replaceChildren(...radio(sc.conditions, sel.condition, (id) => { sel.condition = id; run(); }));
  $("#controller-list").replaceChildren(...radio(INDEX.controllers.map((c) => ({ id: c.id, title: c.label, sub: c.description.replace(/^[^(]*\(|\)$/g, "") })),
    sel.controller, (id) => { sel.controller = id; run(); }));
}

function renderUseful() {
  const cmp = INDEX.comparison[sel.scenario][sel.condition];
  const sc = INDEX.scenarios.find((s) => s.id === sel.scenario);
  const cond = sc.conditions.find((c) => c.id === sel.condition);
  const C = cmp.clarion, P = cmp.plan_only, H = cmp.habit_only;
  const pct = (x) => `${Math.round(x * 100)}%`;
  const rows = [
    ["steps", "Steps to finish (fewer is better)", (x) => fmt(x, 0), "min"],
    ["battery_used", "Battery used", pct, "min"],
    ["planning_calls", "Planning calls (computing effort)", (x) => fmt(x, 0), "min"],
    ["unassigned_cleaned_pct", "Room C cleaned, not its job", (x) => `${fmt(x, 0)}%`, "min"],
    ["quiet_violations", "Quiet-zone rule broken (times)", (x) => fmt(x, 1), "min"],
    ["finished", "Episodes finished", pct, "max"],
  ];
  const order = ["habit_only", "clarion", "plan_only"];
  const label = { habit_only: "Habit only", clarion: "Clarion", plan_only: "Plan only" };
  const table = el("table", { class: "data compare" },
    el("thead", {}, el("tr", {}, el("th", {}, ""), ...order.map((k) => el("th", {}, label[k])))),
    el("tbody", {}, ...rows.filter(([k]) => k !== "quiet_violations" || order.some((c) => cmp[c][k] > 0)).map(([k, name, f, best]) => {
      const vals = order.map((c) => cmp[c][k]);
      const target = best === "min" ? Math.min(...vals) : Math.max(...vals);
      return el("tr", {}, el("td", {}, name), ...order.map((c, j) => el("td", { class: vals[j] === target ? "best num" : "num" }, f(vals[j]))));
    })));
  const fewer = Math.round(100 * (1 - C.planning_calls / P.planning_calls));
  $("#useful").replaceChildren(
    el("p", { class: "drawer-lede" }, "Self-control here means one thing: when the plan and the habit want different actions, the plan can win. How often it wins is set by explicitness, which depends on the robot's drives."),
    el("h3", {}, `Three robots in this scenario (${cond.label.toLowerCase()})`),
    el("p", { class: "hint" }, `Average over the same ${cmp.seeds} worlds (seeds 1 to ${cmp.seeds}). Bold marks the best value in each row. Open "Scenario and robot" to watch each one.`),
    el("div", { class: "table-scroll" }, table),
    el("p", { class: "takeaway", html:
      `Without self-control, the habit-only robot took <b>${fmt(H.steps / P.steps, 1)} times as long</b> as the plan-only robot, used ${pct(H.battery_used)} of its battery, and cleaned ${fmt(H.unassigned_cleaned_pct, 0)}% of room C, which was not its job.` }),
    el("p", { class: "takeaway", html:
      `The Clarion robot planned <b>${fewer}% less</b> than the plan-only robot (${fmt(C.planning_calls, 0)} vs ${fmt(P.planning_calls, 0)} planning calls), but took ${fmt(C.steps - P.steps, 0)} more steps, mostly on detours.` }),
    el("h3", {}, "What this shows"),
    el("ol", { class: "plain-ol" },
      el("li", {}, "The ability to override habits is what keeps the robot on its task. Take it away and the robot follows the dirt, cleans a room that is not its job, and takes about twice as long. In the quiet-zone scenario it also breaks the quiet-zone rule."),
      el("li", {}, "The Clarion robot does not apply control evenly. Control is stronger when a choice is ahead, and weaker after a long constrained phase or while its computer is busy, as in the human studies. That saves deliberation where the drives say it matters less."),
      el("li", {}, "In this world, always planning still gives the best result. Planning here is almost free (0.1% battery per decision), the plan is always right, and the plan-only robot is never slowed by a busy computer. So this demonstration shows that the mechanism works; it does not yet show that adaptive self-control beats constant control.")),
    el("h3", {}, "When adaptive self-control should pay off"),
    el("ul", { class: "plain" },
      el("li", {}, "• Deliberation costs real time, energy or computing, for example a planner that pauses the robot or competes with mapping and perception."),
      el("li", {}, "• Capacity is limited, so constant planning is not possible. Here the plan-only robot ignores the busy computer, which favors it."),
      el("li", {}, "• Habits are sometimes right: fast reactions, learned shortcuts, or opportunities with real value."),
      el("li", {}, "• People work around the robot and benefit from predictable, human-like priorities.")),
    el("p", { class: "hint" }, "These are the conditions to add, on principled grounds, in the next experiments, with the plan-only and habit-only robots as fair baselines. The numbers above are descriptive averages, not hypothesis tests."));
}

/* ---------------------------------------------------------------- charts (drawer) */

function phaseSegments() {
  const seg = [];
  ep.steps.forEach((s, k) => {
    const last = seg[seg.length - 1];
    if (last && last.phase === s.phase) last.end = k + 1;
    else seg.push({ phase: s.phase, start: k, end: k + 1 });
  });
  return seg;
}

const bandsPlugin = {
  id: "bands",
  beforeDatasetsDraw(c) {
    const seg = c.options.plugins.bands;
    if (!seg) return;
    const { ctx, chartArea, scales } = c;
    ctx.save();
    seg.forEach((s, k) => {
      const x0 = scales.x.getPixelForValue(s.start), x1 = scales.x.getPixelForValue(s.end);
      if (k % 2) { ctx.fillStyle = css("--sunken"); ctx.fillRect(x0, chartArea.top, x1 - x0, chartArea.bottom - chartArea.top); }
    });
    ctx.restore();
  },
  afterDraw(c) {
    const step = c.options.plugins.cursorStep;
    if (step === undefined) return;
    const { ctx, chartArea, scales } = c;
    const x = scales.x.getPixelForValue(step);
    ctx.save(); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke(); ctx.restore();
  },
};

function renderCharts() {
  const decs = ep.decisions.filter((d) => !d.reflex);
  const seg = phaseSegments();
  const onClick = (evt, els, c) => goTo(Math.round(c.scales.x.getValueForPixel(evt.x)));
  const x = { type: "linear", min: 0, max: ep.steps.length, title: { display: true, text: "Step" }, grid: { display: false } };
  chart("c-expl", {
    type: "scatter",
    data: { datasets: [{ data: decs.map((d) => ({ x: d.t, y: d.p_explicit })), clip: false,
      pointRadius: decs.map((d) => (d.conflict ? 5 : 4)),
      pointBackgroundColor: decs.map((d) => (d.level === "top" ? css("--explicit") : css("--implicit"))),
      pointBorderColor: decs.map((d) => (d.conflict ? css("--ink") : css("--surface"))),
      pointBorderWidth: decs.map((d) => (d.conflict ? 1.8 : 1)) }] },
    options: { onClick, animation: false, scales: { x, y: { min: 0, max: 1, title: { display: true, text: "Explicitness" } } },
      plugins: { bands: seg, cursorStep: i, tooltip: { callbacks: { label: (c) => { const d = decs[c.dataIndex];
        return `step ${d.t}: ${fmt(d.p_explicit)}, ${d.level === "top" ? "plan" : "habit"} acted`; } } } } },
    plugins: [bandsPlugin],
  });
  chart("c-drives", {
    type: "scatter",
    data: { datasets: ["achievement", "energy"].map((n) => ({ label: cap(n), showLine: true, pointRadius: 0, borderWidth: 2,
      borderColor: css(DRIVE_VARS[n]), data: decs.map((d) => ({ x: d.t, y: d.drives[n].strength })) })) },
    options: { onClick, animation: false, scales: { x, y: { min: 0, title: { display: true, text: "Strength" } } },
      plugins: { bands: seg, cursorStep: i, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y)}` } } } },
    plugins: [bandsPlugin],
  });
  const s = ep.summary;
  const order = seg.map((x) => x.phase).filter((p, k, a) => a.indexOf(p) === k);
  $("#phases").replaceChildren(
    el("thead", {}, el("tr", {}, el("th", {}, "Phase"), el("th", {}, "Decisions"), el("th", {}, "Explicitness"), el("th", {}, "Plan acted"), el("th", {}, "Detour"))),
    el("tbody", {}, ...order.filter((p) => s.by_phase[p]).map((p) => { const b = s.by_phase[p];
      return el("tr", {}, el("td", {}, cap(p)), el("td", { class: "num" }, b.n), el("td", { class: "num" }, fmt(b.mean_explicitness)),
        el("td", { class: "num" }, `${Math.round(b.top_share * 100)}%`), el("td", { class: "num" }, `${Math.round(b.detour_share * 100)}%`)); })));
}

/* ---------------------------------------------------------------- init */

async function init() {
  initTheme();
  try { INDEX = await loadIndex(); } catch (e) { $("#error").textContent = e.message; $("#error").hidden = false; return; }
  sel.scenario = INDEX.scenarios[0].id;
  sel.condition = INDEX.scenarios[0].conditions[0].id;
  const fromHash = location.hash.slice(1);
  if (INDEX.scenarios.some((s) => s.id === fromHash)) { sel.scenario = fromHash; sel.condition = INDEX.scenarios.find((s) => s.id === fromHash).conditions[0].id; }

  let seedInput;
  if (MODE === "app") {
    seedInput = el("input", { type: "number", id: "seed", min: 0, value: 1 });
    $("#app-links").hidden = false;
  } else {
    seedInput = el("select", { id: "seed" }, ...INDEX.seeds.map((s) => el("option", { value: s }, `${s}`)));
    $("#seed-hint").textContent = `This version includes seeds ${INDEX.seeds[0]} to ${INDEX.seeds[INDEX.seeds.length - 1]}. Same seed, same world: compare robots fairly.`;
  }
  $("#seed-slot").replaceWith(seedInput);
  seedInput.addEventListener("change", () => { sel.seed = +seedInput.value || 0; run(); });

  $$("[data-drawer]").forEach((b) => { b.setAttribute("aria-expanded", "false"); b.addEventListener("click", () => toggleDrawer(b.dataset.drawer)); });
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#backdrop").addEventListener("click", closeDrawer);
  $("#play").addEventListener("click", () => (timer ? stop() : play()));
  $("#restart").addEventListener("click", () => { goTo(0); play(); });
  $("#scrub").addEventListener("input", (e) => { stop(); goTo(+e.target.value); });
  $("#speed").addEventListener("change", () => { if (timer) play(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && openDrawer) { closeDrawer(); return; }
    if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(document.activeElement.tagName) && e.key !== "Escape") return;
    if (e.key === " ") { e.preventDefault(); timer ? stop() : play(); }
    if (e.key === "ArrowRight") { stop(); goTo(i + 1); }
    if (e.key === "ArrowLeft") { stop(); goTo(i - 1); }
  });
  window.addEventListener("resize", debounce(() => { if (ep) { sizeCanvas(); drawWorld(); } }, 80));
  window.addEventListener("themechange", () => { if (ep) { drawWorld(); if (openDrawer === "charts") renderCharts(); } });
  await run();
}

init();
