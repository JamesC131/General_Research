/* Start-here guide: theme toggle and the "Try it" calculator (same numbers as the Myrseth simulation). */
(function () {
  initTheme();
  const stim = $("#try-stim");
  if (!stim) return;
  const DEFICIT = 0.7, SAT = 0.5, ALPHA = 1, COST = 0.185, TAU = 0.7, N = 21;
  function update() {
    const s = +stim.value;
    const ds = s * DEFICIT, value = ds * SAT, net = ALPHA * value - COST;
    const U = Array.from({ length: N }, (_, j) => j * net);
    const p = window.CoreEngine.boltzmann(U, TAU);
    const e = p.reduce((acc, q, j) => acc + q * j, 0) / (N - 1);
    $("#try-stim-out").textContent = s.toFixed(2);
    $("#try-ds").textContent = `${s.toFixed(2)} × ${DEFICIT} = ${ds.toFixed(3)}`;
    $("#try-v").textContent = `${ds.toFixed(3)} × ${SAT} = ${value.toFixed(3)}`;
    $("#try-net").textContent = `${net >= 0 ? "+" : "−"}${Math.abs(net).toFixed(3)}`;
    $("#try-net").style.color = net >= 0 ? "var(--explicit)" : "var(--implicit)";
    $("#try-e").textContent = e.toFixed(2);
    $("#try-top").style.flexBasis = `${Math.max(4, Math.min(96, e * 100))}%`;
  }
  stim.addEventListener("input", update);
  update();
})();
