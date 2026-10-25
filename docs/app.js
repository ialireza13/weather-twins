"use strict";

// Weather Twins: in-browser similarity over monthly climate profiles.
// Scoring mirrors climate.find_similar() in the Python pipeline.

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DEFAULT_CITY = "new-york-united-states";
const TOP_N = 10;

const GROUPS = [
  { key: "temp", label: "Temperature", features: ["tmax", "tmin"] },
  { key: "humid", label: "Humidity", features: ["dewpoint"] },
  { key: "rain", label: "Rain", features: ["precip", "rain_days"] },
  { key: "cloud", label: "Sunshine vs. clouds", features: ["cloud"] },
  { key: "snow", label: "Snow", features: ["snow"] },
  { key: "wind", label: "Wind", features: ["wind"] },
  { key: "swing", label: "Day-to-day swings", features: ["tmax_sd"] },
];
const WEIGHT_STEPS = { 0: "Ignore", 0.5: "Less", 1: "Normal", 1.5: "More", 2: "Most" };

const PANELS = [
  { f: "tmax", title: "Daily high", kind: "temp" },
  { f: "tmin", title: "Daily low", kind: "temp" },
  { f: "dewpoint", title: "Dew point", sub: "how muggy it feels", kind: "temp" },
  { f: "precip", title: "Rain and snow", sub: "per month", kind: "precip", zero: true },
  { f: "rain_days", title: "Rainy days", sub: "per month", kind: "days", zero: true },
  { f: "cloud", title: "Cloud cover", kind: "pct", zero: true },
];

const UNITS = {
  temp: { us: [(v) => v * 1.8 + 32, "°F", 0], metric: [(v) => v, "°C", 0] },
  precip: { us: [(v) => v / 25.4, "in", 1], metric: [(v) => v, "mm", 0] },
  days: { us: [(v) => v, "days", 0], metric: [(v) => v, "days", 0] },
  pct: { us: [(v) => v, "%", 0], metric: [(v) => v, "%", 0] },
};

const state = {
  query: DEFAULT_CITY,
  twin: null,
  seasons: false,
  varied: true,
  units: "us",
  mult: Object.fromEntries(GROUPS.map((g) => [g.key, 1])),
};

let DATA, BY_ID, US, WORLD, ranking = [];
const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------- helpers

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child);
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const sum = (xs) => xs.reduce((a, b) => a + b, 0);

function convert(kind, v) {
  const [fn, unit, digits] = UNITS[kind][state.units];
  return { value: fn(v), unit, digits };
}

function fmt(kind, v, withUnit = true) {
  const { value, unit, digits } = convert(kind, v);
  const s = value.toFixed(digits).replace(/^-0(\.0+)?$/, "0");
  if (!withUnit) return s;
  return unit === "%" || unit.startsWith("°") ? `${s}${unit}` : `${s} ${unit}`;
}

function haversineKm(a, b) {
  const r = Math.PI / 180;
  const dLat = (b.lat - a.lat) * r, dLon = (b.lon - a.lon) * r;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dLon / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(h));
}

function shiftOf(city) {
  return state.seasons && city.lat < 0 ? 6 : 0;
}

function rolled(values, shift) {
  return shift ? values.map((_, i) => values[(i + shift) % 12]) : values;
}

// ---------------------------------------------------------------- scoring

// z-scores against the cities' statistics, exactly as climate.standardize() does.
function zScores(profile) {
  const z = {};
  for (const f of DATA.features) {
    const [mu, sd] = DATA.stats[f];
    const log = DATA.log_features.includes(f);
    z[f] = profile[f].map((v) => ((log ? Math.log1p(v) : v) - mu) / sd);
  }
  return z;
}

function prepare(data) {
  for (const c of data.cities) c.z = zScores(c.p);
}

function weights() {
  const w = {};
  for (const g of GROUPS) for (const f of g.features) w[f] = DATA.weights[f] * state.mult[g.key];
  return w;
}

function distance(q, c, w) {
  const sq = shiftOf(q), sc = shiftOf(c);
  let total = 0, wsum = 0;
  for (const f of DATA.features) {
    const wf = w[f];
    if (!(wf > 0)) continue;
    wsum += wf;
    const a = q.z[f], b = c.z[f];
    for (let m = 0; m < 12; m++) {
      const d = b[(m + sc) % 12] - a[(m + sq) % 12];
      total += wf * d * d;
    }
  }
  return wsum ? Math.sqrt(total / (12 * wsum)) : 0;
}

const matchPct = (d) => Math.round(100 * Math.exp(-d));

// Fixed match-score bins for coloring world cities, so colors mean the same for every
// US city (Fairbanks honestly shows no deep blue anywhere). Chosen from the spread of
// all US x world scores: median ~31%, top decile above ~51%.
const HEAT_BINS = [0, 25, 40, 50, 60];
const HEAT_RADIUS = [2.5, 3.5, 4.5, 5.5, 7];
const HEAT_LABELS = ["under 25%", "25–40%", "40–50%", "50–60%", "60%+"];
const heatBin = (pct) => HEAT_BINS.filter((b) => pct >= b).length - 1;
let distById = {};

function rank() {
  const q = BY_ID[state.query], w = weights();
  const all = WORLD.map((c) => ({ c, d: distance(q, c, w) })).sort((a, b) => a.d - b.d);
  all.forEach((r, i) => (r.overall = i + 1));
  distById = Object.fromEntries(all.map((r) => [r.c.id, r.d]));
  const perCountry = {};
  const top = [];
  for (const r of all) {
    if (state.varied) {
      perCountry[r.c.country] = (perCountry[r.c.country] || 0) + 1;
      if (perCountry[r.c.country] > 2) continue;
    }
    top.push(r);
    if (top.length === TOP_N) break;
  }
  return { all, top };
}

// Plain-language differences, same thresholds as climate.explain().
function explain(a, b) {
  const chips = [];
  const tempDiff = (d) => (state.units === "us" ? d * 1.8 : d);
  const deg = state.units === "us" ? "°F" : "°C";
  const dh = mean(b.p.tmax) - mean(a.p.tmax), dl = mean(b.p.tmin) - mean(a.p.tmin);
  const temp = (label, d) => {
    const v = Math.round(Math.abs(tempDiff(d)));
    return v < 1 ? `${label} about the same` : `${label} ${v}${deg} ${d > 0 ? "warmer" : "cooler"}`;
  };
  if (Math.abs(dh) < 1 && Math.abs(dl) < 1) chips.push("Same temperatures");
  else chips.push(temp("Highs", dh), temp("Lows", dl));

  const pa = sum(a.p.precip), pb = sum(b.p.precip);
  const ratio = pa > 0 ? pb / pa : Infinity;
  if (ratio >= 1.3) chips.push(`${ratio.toFixed(1)}× the rain`);
  else if (ratio <= 0.77) chips.push(`${Math.round((1 - ratio) * 100)}% less rain`);
  else chips.push("Similar rainfall");

  const dd = mean(b.p.dewpoint) - mean(a.p.dewpoint);
  if (dd >= 2) chips.push("Muggier");
  else if (dd <= -2) chips.push("Drier air");
  const dc = mean(b.p.cloud) - mean(a.p.cloud);
  if (dc >= 10) chips.push("Cloudier");
  else if (dc <= -10) chips.push("Sunnier");

  const sa = sum(a.p.snow), sb = sum(b.p.snow);
  if (Math.max(sa, sb) >= 10) {
    if (sb < sa * 0.5) chips.push(sb < 2 ? "Hardly any snow" : "Less snow");
    else if (sb > sa * 2) chips.push("More snow");
  }
  return chips;
}

// ---------------------------------------------------------------- map

let map, usMarkers = {}, worldMarkers = {}, rankLayer;

// Decode Natural Earth TopoJSON (world-atlas) into GeoJSON features. The basemap is
// drawn from these shapes instead of map tiles: no API key, and it follows the theme.
function topoFeatures(topo, name) {
  const [sx, sy] = topo.transform.scale, [tx, ty] = topo.transform.translate;
  const arcs = topo.arcs.map((arc) => {
    let x = 0, y = 0;
    return arc.map(([dx, dy]) => { x += dx; y += dy; return [x * sx + tx, y * sy + ty]; });
  });
  const ring = (ids) => ids.flatMap((i, k) => {
    const pts = i >= 0 ? arcs[i] : arcs[~i].slice().reverse();
    return k ? pts.slice(1) : pts;
  });
  return topo.objects[name].geometries.filter((g) => g.arcs).map((g) => ({
    type: "Feature",
    properties: g.properties || {},
    geometry: g.type === "Polygon"
      ? { type: "Polygon", coordinates: g.arcs.map(ring) }
      : { type: "MultiPolygon", coordinates: g.arcs.map((p) => p.map(ring)) },
  }));
}

function initMap(countries) {
  map = L.map("map", { worldCopyJump: true, minZoom: 1, maxZoom: 7, zoomSnap: 0.5, scrollWheelZoom: false })
    .setView([30, -30], 2);
  // Wheel-zoom only once the reader engages with the map, so page scrolling isn't hijacked.
  map.on("click focus", () => map.scrollWheelZoom.enable());
  map.getContainer().addEventListener("mouseleave", () => map.scrollWheelZoom.disable());
  map.attributionControl.setPrefix(false);
  map.attributionControl.addAttribution('<a href="https://www.naturalearthdata.com/">Natural Earth</a>');
  L.geoJSON(topoFeatures(countries, "countries"), {
    interactive: false,
    style: { className: "land", weight: 0.75 },
  }).addTo(map);

  for (const c of WORLD) {
    worldMarkers[c.id] = L.circleMarker([c.lat, c.lon], { radius: 3.5, className: "dot-world" })
      .bindTooltip(() => `${c.name}, ${c.country}: ${matchPct(distById[c.id])}% match`, { direction: "top", offset: [0, -4] })
      .on("click", (e) => { L.DomEvent.stopPropagation(e); selectTwin(c.id); })
      .addTo(map);
  }
  for (const c of US) {
    usMarkers[c.id] = L.circleMarker([c.lat, c.lon], { radius: 3.5, className: "dot-us" })
      .bindTooltip(c.name, { direction: "top", offset: [0, -5] })
      .on("click", (e) => { L.DomEvent.stopPropagation(e); selectQuery(c.id); })
      .addTo(map);
  }
  rankLayer = L.layerGroup().addTo(map);
  new ResizeObserver(() => { map.invalidateSize(); if (pendingFit) fitMap(); }).observe(map.getContainer());

  // Small dots are hard to hit: a click near a city counts, and a click elsewhere
  // in the US picks the nearest US city.
  map.on("click", (e) => {
    const pt = map.latLngToContainerPoint(e.latlng);
    let best = null;
    for (const c of DATA.cities) {
      const d = pt.distanceTo(map.latLngToContainerPoint([c.lat, c.lon]));
      if (d <= 18 && (!best || d < best.d)) best = { c, d };
    }
    if (best) return best.c.us ? selectQuery(best.c.id) : selectTwin(best.c.id);
    const here = { lat: e.latlng.lat, lon: e.latlng.lng };
    const near = nearestUS(here);
    if (near.km <= 400) selectQuery(near.city.id);
  });
}

function nearestUS(point) {
  let best = null;
  for (const c of US) {
    const km = haversineKm(point, c);
    if (!best || km < best.km) best = { city: c, km };
  }
  return best;
}

function renderMap(fit) {
  for (const [id, m] of Object.entries(usMarkers)) {
    const on = id === state.query;
    m.setRadius(on ? 7 : 3.5);
    m.getElement()?.classList.toggle("selected", on);
    if (on) m.bringToFront();
  }
  // Every world city, colored and sized by how well it matches; best matches drawn on top.
  for (const r of [...ranking.all].reverse()) {
    const m = worldMarkers[r.c.id];
    const bin = heatBin(matchPct(r.d));
    m.setRadius(HEAT_RADIUS[bin]);
    const path = m.getElement();
    if (path) {
      path.classList.remove("h1", "h2", "h3", "h4");
      if (bin > 0) path.classList.add(`h${bin}`);
    }
    m.bringToFront();
  }
  renderHeatLegend();

  rankLayer.clearLayers();
  ranking.top.forEach((r, i) => {
    const icon = L.divIcon({
      className: "",
      html: `<div class="rank-icon${r.c.id === state.twin ? " selected" : ""}" data-id="${r.c.id}">${i + 1}</div>`,
      iconSize: [18, 18],
      iconAnchor: [9, 9],
    });
    L.marker([r.c.lat, r.c.lon], { icon, zIndexOffset: 1000 - i, keyboard: false })
      .bindTooltip(`${i + 1}. ${r.c.name}: ${matchPct(r.d)}% match`, { direction: "top", offset: [0, -12] })
      .on("click", () => selectTwin(r.c.id))
      .addTo(rankLayer);
  });
  if (fit) fitMap();
}

let pendingFit = false;

function fitMap() {
  map.invalidateSize();  // the container may have changed size since Leaflet last measured it
  // A page loaded in a background tab has no size yet; fit once it does (see initMap).
  pendingFit = map.getSize().x === 0;
  if (pendingFit) return;
  const q = BY_ID[state.query];
  const pts = [[q.lat, q.lon], ...ranking.top.map((r) => [r.c.lat, r.c.lon])];
  map.fitBounds(L.latLngBounds(pts), { padding: [30, 30], maxZoom: 4 });
}

function renderHeatLegend() {
  const items = HEAT_LABELS.map((label, bin) => {
    const size = 2 * HEAT_RADIUS[bin];
    const swatch = el("i", { style: `width:${size}px;height:${size}px;background:${bin ? `var(--heat-${bin})` : "var(--world-dot)"}` });
    return el("span", { class: "swatch" }, swatch, label);
  });
  $("#heat-legend").replaceChildren(el("span", { text: `Match with ${BY_ID[state.query].name}:` }), ...items);
}

// ---------------------------------------------------------------- results list

function renderResults() {
  const q = BY_ID[state.query];
  $("#results-title").textContent = `Closest weather twins for ${q.name}`;
  const list = $("#results");
  list.replaceChildren();
  ranking.top.forEach((r, i) => {
    const btn = el("button", { type: "button", class: "result", "data-id": r.c.id },
      el("span", { class: "rank", text: String(i + 1) }),
      el("span", {}, el("span", { class: "result-name", text: r.c.name }),
        el("span", { class: "result-country", text: ` · ${r.c.country}` })),
      el("span", { class: "result-score", text: `${matchPct(r.d)}%` }),
      el("span", { class: "result-why", text: explain(q, r.c).slice(0, 3).join(" · ") }));
    if (r.c.id === state.twin) btn.setAttribute("aria-current", "true");
    btn.addEventListener("click", () => selectTwin(r.c.id));
    btn.addEventListener("mouseenter", () => highlightRank(r.c.id, true));
    btn.addEventListener("mouseleave", () => highlightRank(r.c.id, false));
    list.append(el("li", {}, btn));
  });
}

function highlightRank(id, on) {
  const icon = document.querySelector(`.rank-icon[data-id="${CSS.escape(id)}"]`);
  if (icon && id !== state.twin) icon.classList.toggle("selected", on);
}

// ---------------------------------------------------------------- detail + charts

function renderDetail() {
  const a = BY_ID[state.query], b = BY_ID[state.twin];
  const r = ranking.all.find((x) => x.c.id === b.id);
  $("#detail-title").textContent = `${a.name} vs. ${b.name}, ${b.country}`;
  const inTop = ranking.top.findIndex((x) => x.c.id === b.id);
  $("#detail-sub").textContent = inTop >= 0
    ? `#${inTop + 1} closest weather twin`
    : `#${r.overall} of ${WORLD.length} world cities`;
  const score = $("#detail-score");
  score.replaceChildren(el("strong", { text: `${matchPct(r.d)}%` }), el("span", { text: "climate match" }));

  const chips = $("#detail-chips");
  chips.replaceChildren(...explain(a, b).map((t) => el("li", { text: t })));

  const legend = $("#legend");
  legend.replaceChildren(
    el("span", {}, el("i", { style: "background: var(--series-1)" }), a.name),
    el("span", {}, el("i", { style: "background: var(--series-2)" }),
      shiftOf(b) ? `${b.name} (shifted 6 months)` : b.name));

  const notes = [];
  if (shiftOf(b)) notes.push(`${b.name}'s months are shifted by six so its seasons line up with ${a.name}'s.`);
  notes.push(`Rainfall: ${a.name} from ${a.rain}; ${b.name} from ${b.rain}.`);
  $("#detail-note").textContent = notes.join(" ");

  renderCharts();
  if (!$("#table-wrap").hidden) renderTable();
}

function niceTicks(lo, hi, count = 4) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.floor(lo / step) * step; v <= Math.ceil(hi / step) * step + step / 2; v += step) {
    ticks.push(Number(v.toFixed(10)));
  }
  return ticks;
}

function seriesFor(panel) {
  const a = BY_ID[state.query], b = BY_ID[state.twin];
  return [
    { name: a.name, color: "var(--series-1)", values: rolled(a.p[panel.f], shiftOf(a)) },
    { name: b.name, color: "var(--series-2)", values: rolled(b.p[panel.f], shiftOf(b)) },
  ].map((s) => ({ ...s, shown: s.values.map((v) => convert(panel.kind, v).value) }));
}

function renderCharts() {
  const wrap = $("#charts");
  wrap.replaceChildren();
  for (const panel of PANELS) {
    const unit = convert(panel.kind, 0).unit;
    const head = el("h3", {}, panel.title + " ");
    head.append(el("span", { text: [panel.sub, unit].filter(Boolean).join(", ") }));
    const box = el("div", { class: "chart" }, head);
    wrap.append(box);
    drawChart(box, panel);
  }
}

function drawChart(box, panel) {
  const series = seriesFor(panel);
  const W = Math.max(box.clientWidth, 240), H = 170;
  const m = { l: 36, r: 8, t: 8, b: 22 };
  const vals = series.flatMap((s) => s.shown);
  const ticks = niceTicks(panel.zero ? 0 : Math.min(...vals), Math.max(...vals));
  const y0 = ticks[0], y1 = ticks[ticks.length - 1];
  const x = (i) => m.l + (i * (W - m.l - m.r)) / 11;
  const y = (v) => m.t + ((y1 - v) / (y1 - y0)) * (H - m.t - m.b);

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, tabindex: "0", role: "img",
    "aria-label": `${panel.title} by month for ${series.map((s) => s.name).join(" and ")}. Use arrow keys to read values.` });
  root.style.height = `${H}px`;
  const grid = svg("g", { class: "grid" });
  for (const t of ticks) {
    if (t !== y0) grid.append(svg("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }));
    const label = svg("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end" });
    label.textContent = Number.isInteger(t) ? t : t.toFixed(1);
    grid.append(label);
  }
  root.append(grid, svg("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: y(y0), y2: y(y0) }));
  MONTHS.forEach((mo, i) => {
    const t = svg("text", { x: x(i), y: H - 6, "text-anchor": "middle" });
    t.textContent = W < 360 ? mo[0] : mo;
    root.append(t);
  });
  for (const s of series) {
    root.append(svg("path", {
      d: s.shown.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(""),
      fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
  }

  const cross = svg("g", { visibility: "hidden" });
  const hair = svg("line", { class: "crosshair", y1: m.t, y2: H - m.b });
  cross.append(hair);
  const dots = series.map((s) => {
    const d = svg("circle", { r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 });
    cross.append(d);
    return d;
  });
  root.append(cross);

  const show = (i, anchor) => {
    cross.setAttribute("visibility", "visible");
    hair.setAttribute("x1", x(i)); hair.setAttribute("x2", x(i));
    dots.forEach((d, k) => { d.setAttribute("cx", x(i)); d.setAttribute("cy", y(series[k].shown[i])); });
    showTooltip(anchor, MONTHS[i], series.map((s) => ({
      color: s.color, name: s.name, value: fmt(panel.kind, s.values[i]),
    })));
  };
  const hide = () => { cross.setAttribute("visibility", "hidden"); hideTooltip(); };
  const monthAt = (clientX) => {
    const r = root.getBoundingClientRect();
    const px = ((clientX - r.left) / r.width) * W;
    return Math.max(0, Math.min(11, Math.round(((px - m.l) / (W - m.l - m.r)) * 11)));
  };
  let focusMonth = 11;
  root.addEventListener("pointermove", (e) => show(monthAt(e.clientX), { x: e.clientX, y: e.clientY }));
  root.addEventListener("pointerleave", hide);
  root.addEventListener("focus", () => show(focusMonth, anchorOf(root, x(focusMonth) / W)));
  root.addEventListener("blur", hide);
  root.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    focusMonth = Math.max(0, Math.min(11, focusMonth + (e.key === "ArrowRight" ? 1 : -1)));
    show(focusMonth, anchorOf(root, x(focusMonth) / W));
  });
  box.append(root);
}

function anchorOf(node, fx) {
  const r = node.getBoundingClientRect();
  return { x: r.left + fx * r.width, y: r.top + 12 };
}

function showTooltip(anchor, title, rows) {
  const tip = $("#tooltip");
  tip.replaceChildren(el("div", { class: "tt-title", text: title }),
    ...rows.map((r) => el("div", { class: "tt-row" },
      el("i", { style: `background: ${r.color}` }), el("strong", { text: r.value }), el("span", { text: r.name }))));
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let left = anchor.x + 14, top = anchor.y - h - 10;
  if (left + w > window.innerWidth - 8) left = anchor.x - w - 14;
  if (top < 8) top = anchor.y + 16;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${top}px`;
}

function hideTooltip() { $("#tooltip").hidden = true; }

function renderTable() {
  const a = BY_ID[state.query], b = BY_ID[state.twin];
  const head1 = el("tr", {}, el("th", { text: "" }));
  const head2 = el("tr", {}, el("th", { text: "Month" }));
  for (const p of PANELS) {
    head1.append(el("th", { colspan: "2", text: `${p.title} (${convert(p.kind, 0).unit})` }));
    head2.append(el("th", { text: a.name }), el("th", { text: b.name }));
  }
  const body = el("tbody");
  const cols = PANELS.map(seriesFor);
  MONTHS.forEach((mo, i) => {
    const row = el("tr", {}, el("td", { text: mo }));
    PANELS.forEach((p, k) => cols[k].forEach((s) => row.append(el("td", { text: fmt(p.kind, s.values[i], false) }))));
    body.append(row);
  });
  $("#table-wrap").replaceChildren(el("table", {}, el("thead", {}, head1, head2), body));
}

// ---------------------------------------------------------------- twin-region map

// Each NOAA climate division gets its single closest world twin. Regions are colored by
// where that twin is (DATA.groups; 3 colors + neutral is all a map can keep apart), and
// each patch of touching regions sharing a twin gets one name label.
let REGIONS = null, regionMap, labelLayer, clusters = [], activeRegion = null;

function initRegions(geo, countries) {
  REGIONS = geo.features.map((f) => f.properties);
  for (const r of REGIONS) {
    r.z = zScores(r.p);
    r.lat = r.label[0];
    r.name = `${r.state}: ${r.division}`;
  }
  $("#regions").hidden = false;
  $("#region-info").replaceChildren(el("span", { class: "muted", text: "Point at or tap a region to see its closest twins." }));

  regionMap = L.map("region-map", { zoomSnap: 0.25, minZoom: 3, maxZoom: 8, scrollWheelZoom: false });
  regionMap.on("click focus", () => regionMap.scrollWheelZoom.enable());
  regionMap.getContainer().addEventListener("mouseleave", () => regionMap.scrollWheelZoom.disable());
  regionMap.attributionControl.setPrefix(false);
  regionMap.attributionControl.addAttribution(
    '<a href="https://www.ncei.noaa.gov/access/monitoring/climate-divisions/">NOAA climate divisions</a> · <a href="https://www.naturalearthdata.com/">Natural Earth</a>');
  L.geoJSON(topoFeatures(countries, "countries"), { interactive: false, style: { className: "context", weight: 0.75 } })
    .addTo(regionMap);
  L.geoJSON(geo, {
    style: { className: "region", weight: 0.75 },
    onEachFeature: (feature, layer) => {
      const r = feature.properties;
      r.layer = layer;
      layer.bindTooltip(() => `${r.name}: ${r.twin.name} (${matchPct(r.d)}%)`, { sticky: true, direction: "top" });
      layer.on({ mouseover: () => showRegion(r), click: () => showRegion(r) });
    },
  }).addTo(regionMap);
  labelLayer = L.layerGroup().addTo(regionMap);
  regionMap.on("zoomend moveend", placeLabels);
  // Fit once the map has a real size: a page loaded in a background tab has none yet,
  // and fitting a 0x0 box zooms all the way in.
  let fitted = false;
  const onResize = () => {
    regionMap.invalidateSize();
    if (!fitted && regionMap.getSize().x > 0) {
      regionMap.fitBounds([[24.5, -124.8], [49.4, -66.9]], { padding: [8, 8] });
      fitted = true;
    }
    placeLabels();
  };
  if (regionMap.getSize().x > 0) onResize(); else regionMap.setView([38, -96], 4);
  new ResizeObserver(onResize).observe(regionMap.getContainer());
}

function renderRegions() {
  if (!REGIONS) return;
  const w = weights();
  const counts = DATA.groups.map(() => 0);
  for (const r of REGIONS) {
    let best = null, bestD = Infinity;
    for (const c of WORLD) {
      const d = distance(r, c, w);
      if (d < bestD) { bestD = d; best = c; }
    }
    r.twin = best;
    r.d = bestD;
    r.g = DATA.groups.indexOf(best.group);
    counts[r.g]++;
    const path = r.layer.getElement();
    path.classList.remove("g0", "g1", "g2", "g3");
    path.classList.add(`g${r.g}`);
  }

  $("#region-legend").replaceChildren(...DATA.groups.map((g, i) =>
    el("span", {}, el("i", { style: `background: var(--group-${i + 1})` }), `Twin in ${g} `,
      el("span", { class: "count", text: `(${counts[i]})` }))));

  // Patches of touching regions that share a twin, labeled once, biggest first.
  const index = new Map(REGIONS.map((r, i) => [r.id, i]));
  const parent = REGIONS.map((_, i) => i);
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  REGIONS.forEach((r, i) => {
    for (const n of r.neighbors) {
      const j = index.get(n);
      if (j !== undefined && REGIONS[j].twin === r.twin) parent[find(i)] = find(j);
    }
  });
  const groups = new Map();
  REGIONS.forEach((r, i) => {
    const root = find(i);
    const g = groups.get(root) || { twin: r.twin, area: 0, anchor: r };
    g.area += r.area;
    if (r.area > g.anchor.area) g.anchor = r;
    groups.set(root, g);
  });
  clusters = [...groups.values()].sort((a, b) => b.area - a.area);
  placeLabels();
  if (activeRegion) showRegion(activeRegion);
}

function placeLabels() {
  if (!labelLayer) return;
  labelLayer.clearLayers();
  const size = regionMap.getSize();
  const pxPerDeg = (256 * 2 ** regionMap.getZoom()) / 360;
  const placed = [];
  for (const c of clusters) {
    const name = c.twin.name.replace(/ \(.*\)$/, "");
    const w = name.length * 6.6 + 16, h = 18;
    // Only label patches big enough on screen to hold their label.
    const pxArea = c.area * pxPerDeg ** 2 * Math.cos((c.anchor.label[0] * Math.PI) / 180);
    if (pxArea < w * h * 0.8) continue;
    const pt = regionMap.latLngToContainerPoint(c.anchor.label);
    const box = [pt.x - w / 2 - 3, pt.y - h / 2 - 3, pt.x + w / 2 + 3, pt.y + h / 2 + 3];
    if (box[0] < 0 || box[1] < 0 || box[2] > size.x || box[3] > size.y) continue;
    if (placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
    placed.push(box);
    const label = el("span", { class: "twin-label", text: name });
    L.marker(c.anchor.label, {
      icon: L.divIcon({ className: "twin-label-wrap", html: label, iconSize: [0, 0] }),
      interactive: false, keyboard: false,
    }).addTo(labelLayer);
  }
}

function showRegion(r) {
  if (activeRegion && activeRegion !== r) activeRegion.layer.getElement()?.classList.remove("active");
  activeRegion = r;
  r.layer.getElement()?.classList.add("active");
  r.layer.bringToFront();
  const w = weights();
  const next = WORLD.map((c) => ({ c, d: distance(r, c, w) })).sort((a, b) => a.d - b.d).slice(1, 4);
  const where = r.town ? ` (weather in ${r.town})` : "";
  $("#region-info").replaceChildren(
    el("div", {}, el("strong", { text: r.name }), where + " · closest twin: ",
      el("strong", { text: `${r.twin.name}, ${r.twin.country}` }), ` · ${matchPct(r.d)}% match`),
    el("div", { class: "alts", text: `Next closest: ${next.map((x) => `${x.c.name} ${matchPct(x.d)}%`).join(" · ")}` }));
}

// ---------------------------------------------------------------- state + controls

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (BY_ID[h.get("city")]?.us) state.query = h.get("city");
  if (BY_ID[h.get("twin")] && !BY_ID[h.get("twin")].us) state.twin = h.get("twin");
  state.seasons = h.get("seasons") === "1";
  state.varied = h.get("varied") !== "0";
  if (h.get("units") === "metric") state.units = "metric";
  for (const g of GROUPS) {
    const v = parseFloat(h.get(`w_${g.key}`));
    if (v >= 0 && v <= 2) state.mult[g.key] = v;
  }
}

function writeHash() {
  const h = new URLSearchParams({ city: state.query, twin: state.twin });
  if (state.seasons) h.set("seasons", "1");
  if (!state.varied) h.set("varied", "0");
  if (state.units !== "us") h.set("units", state.units);
  for (const g of GROUPS) if (state.mult[g.key] !== 1) h.set(`w_${g.key}`, state.mult[g.key]);
  history.replaceState(null, "", `#${h}`);
}

let regionsStale = true;  // region twins depend on weights and seasons, not on the chosen city

function update({ fit = false, keepTwin = true } = {}) {
  ranking = rank();
  if (!keepTwin || !state.twin || !BY_ID[state.twin]) state.twin = ranking.top[0].c.id;
  renderResults();
  renderMap(fit);
  renderDetail();
  if (regionsStale) { renderRegions(); regionsStale = false; }
  writeHash();
}

function selectQuery(id) {
  if (id === state.query) return;
  state.query = id;
  $("#city").value = id;
  update({ fit: true, keepTwin: false });
}

function selectTwin(id) {
  state.twin = id;
  renderResults();
  renderMap(false);
  renderDetail();
  writeHash();
  if (window.matchMedia("(max-width: 960px)").matches) {
    $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function initControls() {
  const select = $("#city");
  for (const c of [...US].sort((a, b) => a.name.localeCompare(b.name))) {
    select.append(el("option", { value: c.id, text: c.name }));
  }
  select.value = state.query;
  select.addEventListener("change", () => selectQuery(select.value));

  const seasons = $("#seasons"), varied = $("#varied");
  seasons.checked = state.seasons;
  varied.checked = state.varied;
  seasons.addEventListener("change", () => { state.seasons = seasons.checked; regionsStale = true; update({ fit: true }); });
  varied.addEventListener("change", () => { state.varied = varied.checked; update({ fit: true }); });

  const segs = document.querySelectorAll(".seg");
  const syncUnits = () => segs.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.units === state.units)));
  segs.forEach((b) => b.addEventListener("click", () => {
    state.units = b.dataset.units; syncUnits(); renderResults(); renderDetail(); writeHash();
  }));
  syncUnits();

  const grid = $("#weights");
  const inputs = {};
  for (const g of GROUPS) {
    const id = `w-${g.key}`;
    const out = el("output", { for: id });
    const input = el("input", { type: "range", id, min: "0", max: "2", step: "0.5", value: String(state.mult[g.key]) });
    const sync = () => { out.textContent = WEIGHT_STEPS[input.value] ?? input.value; };
    input.addEventListener("input", () => { state.mult[g.key] = parseFloat(input.value); sync(); regionsStale = true; update(); });
    input.addEventListener("change", () => renderMap(true));
    sync();
    inputs[g.key] = { input, sync };
    grid.append(el("label", { class: "weight", for: id }, el("span", { text: g.label }), out, input));
  }
  $("#reset-weights").addEventListener("click", () => {
    for (const g of GROUPS) { state.mult[g.key] = 1; inputs[g.key].input.value = "1"; inputs[g.key].sync(); }
    regionsStale = true;
    update({ fit: true });
  });

  const toggle = $("#table-toggle");
  toggle.addEventListener("click", () => {
    const wrap = $("#table-wrap");
    wrap.hidden = !wrap.hidden;
    toggle.setAttribute("aria-expanded", String(!wrap.hidden));
    toggle.textContent = wrap.hidden ? "Show as table" : "Hide table";
    if (!wrap.hidden) renderTable();
  });

  const locate = $("#locate");
  const msg = el("p", { class: "muted small", id: "locate-msg" });
  msg.style.margin = "4px 0 0";
  $(".city-control").append(msg);
  if (!("geolocation" in navigator)) locate.hidden = true;
  locate.addEventListener("click", () => {
    msg.textContent = "Finding you…";
    navigator.geolocation.getCurrentPosition((pos) => {
      const near = nearestUS({ lat: pos.coords.latitude, lon: pos.coords.longitude });
      msg.textContent = near.km > 150
        ? `Closest city on the list: ${near.city.name} (${Math.round(state.units === "us" ? near.km / 1.609 : near.km)} ${state.units === "us" ? "mi" : "km"} away).`
        : "";
      selectQuery(near.city.id);
    }, () => { msg.textContent = "Couldn't get your location. Pick a city from the list instead."; },
    { timeout: 10000, maximumAge: 600000 });
  });

  let pending = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(renderCharts);
  }).observe($("#charts"));
}

async function main() {
  const [cities, countries, regions] = await Promise.all([
    fetch("data/cities.json").then((r) => r.json()),
    fetch("data/countries-50m.json").then((r) => r.json()),
    fetch("data/regions.json").then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  DATA = cities;
  prepare(DATA);
  BY_ID = Object.fromEntries(DATA.cities.map((c) => [c.id, c]));
  US = DATA.cities.filter((c) => c.us);
  WORLD = DATA.cities.filter((c) => !c.us);
  $("#period").textContent = DATA.period;
  readHash();
  initControls();
  initMap(countries);
  if (regions) initRegions(regions, countries);
  update({ fit: true });
  // A pasted or edited link (only the hash changes) should show what it describes.
  window.addEventListener("hashchange", () => location.reload());
}

main().catch((err) => {
  console.error(err);
  $("#results").replaceChildren(el("li", { class: "muted", text: "Couldn't load the climate data. Try reloading the page." }));
});
