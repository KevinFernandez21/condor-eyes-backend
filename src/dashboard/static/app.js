"use strict";
// Arranque: navegación entre vistas, sondeo de /api/state, píldoras y avisos.
// Toda la lógica de vista vive en Python (src/dashboard/*.py); aquí solo se pinta.
const App = (() => {
  const { el, dash } = UI;
  const FETCH_TIMEOUT_MS = 4000;
  const POLL_MS = { agents: 1000, cams: 500 };
  let view = "agents", timer = null, inflight = false, firstSnapshot = true, last = null;
  const seenAlerts = new Set();

  function pills(s) {
    const pill = (text, cls) => el("span", { class: `pill ${cls}` }, text);
    document.getElementById("pills").replaceChildren(
      pill(s.api.connected ? "API conectada" : "API sin conexión", s.api.connected ? "ok" : "bad"),
      pill(s.ws.connected ? "Tiempo real activo" : "Tiempo real: reconectando", s.ws.connected ? "ok" : "warn"),
      pill(`Hardware: ${s.health.hardware}`, ""));
    const problems = [];
    if (!s.api.connected) problems.push(`API: ${dash(s.api.error)}.`);
    if (!s.ws.connected && s.ws.error) problems.push(`WebSocket: ${s.ws.error}.`);
    if (s.lag.dropped_total > 0) problems.push(`La API descartó ${s.lag.dropped_total} mensajes por lentitud de este cliente.`);
    const banner = document.getElementById("banner");
    banner.hidden = problems.length === 0;
    banner.textContent = problems.join(" ");
  }

  // Avisos apilados abajo a la derecha por cada decisión nueva (no en la primera carga).
  function toasts(s) {
    const box = document.getElementById("toasts");
    for (const a of [...s.alerts].reverse()) {
      const key = a.event_id || a.decision_id;
      if (!key || seenAlerts.has(key)) continue;
      seenAlerts.add(key);
      if (firstSnapshot || !["high", "medium"].includes(a.severity)) continue;
      const t = el("div", { class: `toast ${a.severity}`, role: "status", onclick: () => t.remove() },
        el("b", {}, a.toast), el("small", {}, `${UI.clock(a.evaluated_at)} · ${dash(a.stream_id)} · requiere operador`));
      box.append(t);
      setTimeout(() => t.remove(), 9000);
      while (box.children.length > 4) box.firstChild.remove();
    }
    firstSnapshot = false;
  }

  function show(name, push) {
    view = name;
    for (const v of ["agents", "cams"]) {
      document.getElementById(`view-${v}`).hidden = v !== name;
      document.getElementById(`tab-${v}`).setAttribute("aria-selected", String(v === name));
    }
    if (push) history.replaceState(null, "", name === "cams" ? "#multicamara" : "#agentes");
    if (name === "cams") Cams.activate();
    if (last) paint(last);
    tick(true);
  }
  function paint(s) {
    const fns = [() => pills(s), () => toasts(s), () => (view === "agents" ? Agents.render(s) : Cams.render(s))];
    for (const fn of fns) { try { fn(); } catch (err) { console.error(err); } }
  }

  async function tick() {
    if (inflight) return;
    clearTimeout(timer);
    inflight = true;
    const ctl = new AbortController();
    const guard = setTimeout(() => ctl.abort(), FETCH_TIMEOUT_MS);
    try {
      const params = new URLSearchParams({ limit: "100" });
      const f = Agents.filters;
      if (f.topics.size) params.set("topics", [...f.topics].join(","));
      if (f.q) params.set("q", f.q);
      if (f.correlation) params.set("correlation_id", f.correlation);
      const res = await fetch(`/api/state?${params}`, { signal: ctl.signal, cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      last = await res.json();
      paint(last);
    } catch (err) {
      const banner = document.getElementById("banner");
      banner.hidden = false;
      banner.textContent = "Sin conexión con el servidor del dashboard; reintentando. Se muestran los últimos datos.";
    } finally {
      clearTimeout(guard);
      inflight = false;
      timer = setTimeout(tick, POLL_MS[view]);
    }
  }

  function init() {
    Agents.init(() => tick());
    Cams.init(() => {});
    for (const b of document.querySelectorAll(".tab")) b.addEventListener("click", () => show(b.dataset.view, true));
    show(location.hash === "#multicamara" ? "cams" : "agents", false);
  }
  return { init };
})();
App.init();
