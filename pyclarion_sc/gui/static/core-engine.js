/* In-browser version of pyclarion_sc/gui/core_engine.py (same equations, same output fields).
   Used when Core Studio runs without the Python server (window.CORE_LOCAL).
   Randomness uses a seeded generator, so a seed always gives the same run in the browser,
   but not the same random draws as the Python version. */

(function () {
  const LIMITS = { steps: 500, drives: 12, goals: 12, rules: 40, actions: 20, n_levels: 101 };

  function rngFrom(seed) {                       // mulberry32
    let a = (seed >>> 0) + 0x6d2b79f5;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function boltzmann(values, temperature) {      // p_j = exp(v_j / tau) / sum_k exp(v_k / tau)
    const v = values.map((x) => x / temperature);
    const m = Math.max(...v);
    const e = v.map((x) => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  }

  function draw(p, rnd) {
    let u = rnd(), acc = 0;
    for (let i = 0; i < p.length; i++) { acc += p[i]; if (u < acc) return i; }
    return p.length - 1;
  }

  const f = (x, lo = -1e6, hi = 1e6) => Math.min(Math.max(Number(x) || 0, lo), hi);

  function validate(input) {
    const sc = JSON.parse(JSON.stringify(input));
    if (!sc.drives || !sc.drives.length) throw new Error("Add at least one drive.");
    if (!sc.goals || !sc.goals.length) throw new Error("Add at least one goal.");
    if (!sc.actions || !sc.actions.length) throw new Error("Add at least one action.");
    for (const k of ["drives", "goals", "rules", "actions"])
      if ((sc[k] || []).length > LIMITS[k]) throw new Error(`Too many ${k} (limit ${LIMITS[k]}).`);
    const names = sc.drives.map((d) => d.name);
    if (new Set(names).size !== names.length) throw new Error("Drive names must be unique.");
    for (const r of sc.rules || [])
      if (!sc.actions.includes(r.then)) throw new Error(`Rule '${r.name}' points to an action that does not exist: ${r.then}.`);
    const m = (sc.mcs = sc.mcs || {});
    m.n_levels = Math.round(Math.min(Math.max(parseInt(m.n_levels ?? 21, 10) || 21, 2), LIMITS.n_levels));
    m.temperature = f(m.temperature ?? 0.7, 1e-3, 100);
    m.goal_temperature = f(m.goal_temperature ?? 0.1, 1e-3, 100);
    sc.bottom_temperature = f(sc.bottom_temperature ?? 0.1, 1e-3, 100);
    sc.top_temperature = f(sc.top_temperature ?? 0.1, 1e-3, 100);
    sc.rules = sc.rules || []; sc.state = sc.state || {}; sc.q_values = sc.q_values || {};
    return sc;
  }

  function run(scenario, steps = 40, seed = 1) {
    const sc = validate(scenario);
    steps = Math.round(Math.min(Math.max(steps, 1), LIMITS.steps));
    const rnd = rngFrom(seed);
    const m = sc.mcs;
    const alpha = f(m.alpha ?? 1), cost = f(m.cost ?? 0.16), dt = f(m.dt ?? 0.05, 0, 10), n = m.n_levels;
    const drives = sc.drives.map((d) => ({ name: d.name, stimulus: f(d.stimulus), deficit: f(d.deficit, 0, 10),
      initial: f(d.deficit, 0, 10), k: f(d.k ?? 0, 0, 10) }));
    const actions = sc.actions.slice();
    const q = actions.map((a) => f(sc.q_values[a] ?? 0));
    const out = [];

    for (let t = 0; t < steps; t++) {
      // Eq. 6: deficit(t) = initial deficit - k * time on task (never below 0)
      drives.forEach((d) => { if (d.k > 0) d.deficit = Math.max(d.initial - d.k * (t + 1) * dt, 0); });
      const ds = Object.fromEntries(drives.map((d) => [d.name, d.stimulus * d.deficit]));   // Eq. 5
      const snapshot = Object.fromEntries(drives.map((d) => [d.name,
        { stimulus_level: d.stimulus, deficit: d.deficit, strength: d.stimulus * d.deficit }]));

      // Eq. 7: goal strength = sum over drives of relevance x drive strength
      const gs = Object.fromEntries(sc.goals.map((g) => [g.name,
        Object.entries(g.relevance || {}).reduce((s, [d, r]) => s + f(r) * (ds[d] || 0), 0)]));
      const gnames = Object.keys(gs);
      const gp = boltzmann(gnames.map((x) => gs[x]), m.goal_temperature);
      const goal = gnames[draw(gp, rnd)];
      const gdef = sc.goals.find((g) => g.name === goal);

      // value(g) = sum over drives of drive strength x satisfaction; U_j = level_j x (alpha*value - c)
      const value = Object.entries(gdef.satisfaction || {}).reduce((s, [d, sat]) => s + (ds[d] || 0) * f(sat), 0);
      const U = Array.from({ length: n }, (_, j) => j * (alpha * value - cost));
      let pLevels = null, level = -1, pExplicit;
      if ((m.mode || "utility") === "fixed") {
        pExplicit = f(m.fixed ?? 0.8, 0, 1);
      } else {
        pLevels = boltzmann(U, m.temperature);
        level = draw(pLevels, rnd);
        pExplicit = level / (n - 1);
      }

      // ACS top level: rules that apply -> Boltzmann over their supports
      const applicable = sc.rules.filter((r) => (r.goal === null || r.goal === undefined || r.goal === "" || r.goal === goal) &&
        Object.entries(r.if || {}).every(([k, v]) => Boolean(sc.state[k]) === Boolean(v)));
      let topChoice = null, topRule = null, topProbs = [];
      if (applicable.length) {
        topProbs = boltzmann(applicable.map((r) => f(r.support ?? 1)), sc.top_temperature);
        const i = draw(topProbs, rnd);
        topRule = applicable[i].name; topChoice = applicable[i].then;
      }
      // ACS bottom level: Q-values -> Boltzmann
      const bp = boltzmann(q, sc.bottom_temperature);
      const bottomChoice = actions[draw(bp, rnd)];
      // integration: the top level acts with probability = explicitness (if a rule applies)
      const useTop = topChoice !== null && rnd() < pExplicit;
      out.push({
        t: t + 1, drives: snapshot,
        goal_strengths: gs, goal_probs: Object.fromEntries(gnames.map((x, i) => [x, gp[i]])), goal,
        value, utilities: U, level_probs: pLevels, level, p_explicit: pExplicit,
        expected_explicitness: pLevels ? pLevels.reduce((s, p, j) => s + p * j, 0) / (n - 1) : null,
        applicable_rules: applicable.map((r) => r.name), top_rule_probs: topProbs, top_rule: topRule, top_choice: topChoice,
        q_values: Object.fromEntries(actions.map((a, i) => [a, q[i]])),
        bottom_probs: Object.fromEntries(actions.map((a, i) => [a, bp[i]])), bottom_choice: bottomChoice,
        level_used: useTop ? "top" : "bottom", action: useTop ? topChoice : bottomChoice,
        conflict: topChoice !== null && topChoice !== bottomChoice,
      });
    }
    const summary = {
      top_share: out.filter((s) => s.level_used === "top").length / out.length,
      conflicts: out.filter((s) => s.conflict).length,
      conflicts_won_by_top: out.filter((s) => s.conflict && s.level_used === "top").length,
      actions: Object.fromEntries(actions.map((a) => [a, out.filter((s) => s.action === a).length])),
    };
    return { scenario: sc, steps: out, summary };
  }

  const api = { run, boltzmann, validate };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.CoreEngine = api;
})();
