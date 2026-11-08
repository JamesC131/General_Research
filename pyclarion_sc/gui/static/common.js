/* Shared helpers for Core Studio and Simulation Lab. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return n;
}

async function api(path, body, method) {
  const res = await fetch(path, {
    method: method || (body === undefined ? "GET" : "POST"),
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `The server answered ${res.status}.`;
    try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function fmt(x, d = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  return Number(x).toFixed(d);
}

function debounce(fn, ms = 250) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

function downloadCSV(name, rows, columns) {
  if (!rows.length) return;
  const cols = columns || Object.keys(rows[0]).filter((k) => typeof rows[0][k] !== "object");
  const esc = (v) => { const s = v === null || v === undefined ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  const text = [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
  const a = el("a", { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: name });
  document.body.append(a); a.click(); a.remove();
}

/* ---- theme ------------------------------------------------------------------ */
function initTheme() {
  const btn = $(".theme-toggle");
  let saved = null;
  try { saved = localStorage.getItem("pcsc-theme"); } catch (_) {}
  if (saved) document.documentElement.dataset.theme = saved;
  const isDark = () => document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const label = () => {
    const text = isDark() ? "Light mode" : "Dark mode";
    if (btn.classList.contains("icon")) { btn.setAttribute("aria-label", `Switch to ${text.toLowerCase()}`); btn.title = `Switch to ${text.toLowerCase()}`; }
    else btn.textContent = text;
  };
  label();
  btn.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("pcsc-theme", next); } catch (_) {}
    label();
    applyChartDefaults();
    window.dispatchEvent(new Event("themechange"));
  });
}

/* ---- charts ------------------------------------------------------------------ */
function applyChartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.font.family = "Public Sans, system-ui, sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = css("--ink-2");
  Chart.defaults.borderColor = css("--rule");
  Chart.defaults.animation = matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 250 };
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip.backgroundColor = css("--ink");
  Chart.defaults.plugins.tooltip.titleColor = css("--surface");
  Chart.defaults.plugins.tooltip.bodyColor = css("--surface");
  Chart.defaults.plugins.tooltip.padding = 8;
  Chart.defaults.plugins.tooltip.cornerRadius = 4;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.elements.bar.borderRadius = 4;
  Chart.defaults.elements.line.borderWidth = 2;
}

/* Small plugin: draws +-SD whiskers on a bar dataset that has `whiskers: [...]`. */
const whiskerPlugin = {
  id: "whiskers",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    chart.data.datasets.forEach((ds, i) => {
      if (!ds.whiskers) return;
      const meta = chart.getDatasetMeta(i);
      if (meta.hidden) return;
      ctx.save();
      ctx.strokeStyle = css("--ink-3");
      ctx.lineWidth = 1;
      meta.data.forEach((bar, j) => {
        const sd = ds.whiskers[j];
        if (sd === null || sd === undefined) return;
        const y = chart.scales.y;
        const top = y.getPixelForValue(ds.data[j] + sd), bot = y.getPixelForValue(Math.max(y.min, ds.data[j] - sd));
        ctx.beginPath();
        ctx.moveTo(bar.x, top); ctx.lineTo(bar.x, bot);
        ctx.moveTo(bar.x - 4, top); ctx.lineTo(bar.x + 4, top);
        ctx.moveTo(bar.x - 4, bot); ctx.lineTo(bar.x + 4, bot);
        ctx.stroke();
      });
      ctx.restore();
    });
  },
};

const charts = {};
function chart(id, config) {
  if (!window.Chart) return null;   // charts are optional: the rest of the page still works
  if (charts[id]) charts[id].destroy();
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  charts[id] = new Chart(canvas, config);
  return charts[id];
}

function prettyName(s) { return String(s).replace(/_/g, " "); }

applyChartDefaults();
