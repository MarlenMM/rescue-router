// Rescue Router web viewer (PLAN §6.4). Plain ES module + Leaflet 1.9 (global L), no build step.
// Scenes come from src/export_web_assets.py; routing runs in the browser with js/router.js.

import { sceneGrids, b64ToBytes, b64ToFloat32, TRUCK, BOAT, FOOT, ISOLATED } from './router.js';

const CELL = 4;
const KIND = ['truck', 'boat', 'foot', 'isolated'];
const WORD = ['truck', 'boat', 'on foot', 'no route'];
const TITLE = ['Truck can reach it', 'Boat needed', 'Last stretch on foot', 'No route: send an air team'];
const LEG = [{ name: 'truck', icon: '🚚', color: '#ffffff' }, { name: 'boat', icon: '🚤', color: 'var(--accent)' },
             { name: 'on foot', icon: '🚶', color: 'var(--foot)', dash: '1 9' }];
const SIDE = { pred: 'Prediction', gt: 'Labelled map' };
const SVG = {
  truck: '<svg viewBox="0 0 24 24" fill="#fff"><path d="M2 6h11v9H2zM14 9h4l3 3v3h-7z"/><circle cx="6" cy="17" r="2"/><circle cx="17" cy="17" r="2"/></svg>',
  boat: '<svg viewBox="0 0 24 24" fill="#fff"><path d="M3 14h18l-3 5H6zM11 4v9H6zM13 7l5 6h-5z"/></svg>',
  foot: '<svg viewBox="0 0 24 24" fill="#fff"><circle cx="13" cy="4" r="2.4"/><path d="M11 7.5l-3 4 1.8 1.2 2-2.4 1 4-3.3 6.4 2 1 3-5.8 2.2 5.8 2-.8-2.6-7.4-.9-3.6 2 1.4 2.4-.4-.3-2-2-.3z"/></svg>',
  isolated: '<svg viewBox="0 0 24 24" stroke="#fff" stroke-width="3.2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
};
const CLASS_LABEL = {
  floodnet: { 2: 'flooded building', 3: 'dry building' },
  rescuenet: { 2: 'intact building', 3: 'damaged building' },
};
const DATASET = {
  floodnet: { name: 'FloodNet', event: 'Hurricane Harvey, Texas 2017',
    attrib: '<a href="https://github.com/BinaLab/FloodNet-Supervised_v1.0">FloodNet</a> (CDLA-Permissive-1.0)' },
  rescuenet: { name: 'RescueNet', event: 'Hurricane Michael, Florida 2018',
    attrib: '<a href="https://springernature.figshare.com/collections/RescueNet_A_High_Resolution_UAV_Semantic_Segmentation_Benchmark_Dataset_for_Natural_Disaster_Damage_Assessment/6647354/1">RescueNet</a> (figshare CC0; GitHub release CC BY-NC-ND 4.0)' },
};
const MODEL_ATTRIB = '<a href="https://github.com/GitGyun/chameleon">Chameleon</a> (Kim et al., ECCV 2024)';

const $ = (id) => document.getElementById(id);
const state = { index: null, key: null, s: null, src: 'pred', lam: 5, swipe: 0.5, sel: null, anim: null, entry: null, picking: false };

// ---------- map and panes ----------
const map = L.map('map', { crs: L.CRS.Simple, zoomSnap: 0.25, zoomDelta: 0.5, minZoom: -4, maxZoom: 4,
  attributionControl: true, zoomControl: false, boxZoom: false });
L.control.zoom({ position: 'bottomright' }).addTo(map);
map.attributionControl.setPrefix('<a href="https://leafletjs.com">Leaflet</a>');
const pane = (name, z) => { const p = map.createPane(name); p.style.zIndex = z; return p; };
const panes = { pred: pane('pred', 410), gt: pane('gt', 411), base: pane('base', 440), dis: pane('dis', 445), route: pane('route', 460) };
const rend = Object.fromEntries(['pred', 'gt', 'dis', 'route'].map(k => [k, L.svg({ pane: k, padding: 0.5 })]));
const baseRend = L.canvas({ pane: 'base' });
const groups = Object.fromEntries(['photo', 'pred', 'gt', 'predHouses', 'gtHouses', 'base', 'dis', 'route']
  .map(k => [k, L.layerGroup().addTo(map)]));

// ---------- helpers ----------
const ll = (x, y) => L.latLng(state.s.H - y, x);                 // image px -> map
const cellLL = ([cy, cx]) => ll(cx * CELL + CELL / 2, cy * CELL + CELL / 2);
const fmt = (v) => (v == null ? '–' : v.toFixed(2).replace(/^0/, ''));

function pointInPoly([x, y], poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function badgeIcon(kind, sel) {
  return L.divIcon({ className: '', iconSize: [22, 22], iconAnchor: [11, 11],
    html: `<div class="st ${kind}${sel ? ' sel' : ''}">${SVG[kind]}</div>` });
}

// ---------- scene loading ----------
async function loadIndex() {
  state.index = await (await fetch('scenes/scenes.json')).json();
  const box = $('scenes');
  for (const sc of state.index.scenes) {
    const b = document.createElement('button');
    b.dataset.key = sc.key;
    b.title = sc.caption;
    b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label', `${sc.dataset} ${sc.id}: ${sc.caption}`);
    const fail = /failure/i.test(sc.caption) ? '<span class="warn">fail</span>' : '';
    b.innerHTML = `<img src="scenes/${sc.key}/thumb.jpg" alt="" loading="lazy"><span class="tag">${sc.dataset === 'floodnet' ? 'FN' : 'RN'} ${sc.id}</span>${fail}`;
    b.addEventListener('click', () => { location.hash = sc.key; });
    box.appendChild(b);
  }
}

async function loadScene(key) {
  const sc = state.index.scenes.find(s => s.key === key) || state.index.scenes[0];
  const dir = `scenes/${sc.key}/`;
  const [meta, grid, blds] = await Promise.all(['meta', 'grid', 'buildings'].map(f => fetch(dir + f + '.json').then(r => r.json())));
  for (const k of ['pred', 'gt']) {                                   // decode once; Grid accepts typed arrays
    grid[k] = { cls: b64ToBytes(grid[k].cls), risk: b64ToFloat32(grid[k].risk), pboat: b64ToFloat32(grid[k].pboat) };
  }
  stopAnim();
  state.key = sc.key;
  state.sel = null;
  state.entry = null;
  setPicking(false);
  const [W, H] = meta.size;
  state.s = { sc, dir, meta, grid, blds, W, H, ds: meta.dataset };
  document.querySelectorAll('#scenes button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.key === sc.key)));
  $('caption').textContent = `${DATASET[meta.dataset].event} · ${sc.caption}`;
  const shots = meta.run.match(/_(\d+)shot/)[1];
  const ious = Object.values(meta.iou).filter(v => v != null);
  $('model').innerHTML = `${DATASET[meta.dataset].name} · Chameleon adapted from <b>${shots}</b> labelled photos · ` +
    `this photo: mean IoU ${fmt(ious.reduce((a, b) => a + b, 0) / ious.length)}`;
  for (const k of Object.keys(groups)) groups[k].clearLayers();
  const bounds = [[0, 0], [H, W]];
  L.imageOverlay(dir + 'photo.jpg', bounds).addTo(groups.photo);
  state.overlays = {
    pred: L.imageOverlay(dir + 'pred.png', bounds, { pane: 'pred', opacity: opacity() }).addTo(groups.pred),
    gt: L.imageOverlay(dir + 'gt.png', bounds, { pane: 'gt', opacity: opacity() }).addTo(groups.gt),
  };
  map.attributionControl._attributions = {};
  map.attributionControl.addAttribution(`${DATASET[meta.dataset].attrib} · ${MODEL_ATTRIB}`);
  drawLegend();
  $('routeCard').hidden = true;
  solve();
  fit();
}

function fit() {
  if (!state.s) return;
  const panel = $('panel').getBoundingClientRect();
  const phone = innerWidth <= 760;
  document.documentElement.style.setProperty('--sheet-h', `${Math.round(panel.height)}px`);
  const top = $('caption').closest('.title').getBoundingClientRect().bottom + 8;
  const counter = $('counter').getBoundingClientRect().height;
  map.invalidateSize();
  map.fitBounds([[0, 0], [state.s.H, state.s.W]], {
    paddingTopLeft: [12, phone ? top : 12],
    paddingBottomRight: phone ? [12, panel.height + counter + 16] : [panel.width + 24, 12],
  });
}

// ---------- routing ----------
function solve() {
  const s = state.s;
  s.grids = sceneGrids(s.grid, state.lam);
  s.sol = { pred: s.grids.pred.solve(base()), gt: s.grids.gt.solve(base()) };
  s.status = { pred: s.blds.map(b => s.sol.pred.status(b.ring)), gt: s.blds.map(b => s.sol.gt.status(b.ring)) };
  drawHouses();
  drawBase();
  drawCounter();
  if (state.sel) select(state.sel.i, state.sel.side, false);
  else { groups.route.clearLayers(); $('routeCard').hidden = true; }
}

const base = () => state.entry || state.s.grid.base;

// ---------- entry point chosen by the user ----------
function setPicking(on) {
  state.picking = on;
  $('map').classList.toggle('picking', on);
  $('entryPick').setAttribute('aria-pressed', String(on));
  if (state.s) drawCounter();
}

function setEntry(x, y) {
  const s = state.s, g = s.grids.gt, w = s.grid.w, h = s.grid.h;
  const cx = Math.min(w - 1, Math.max(0, Math.floor(x / CELL))), cy = Math.min(h - 1, Math.max(0, Math.floor(y / CELL)));
  const c = cy * w + cx;
  if (g.wall[c] && !g.boatOk[c]) { state.entryMsg = 'That is a building: tap a road, a yard or water.'; drawCounter(); return; }
  const mode = g.truckOk[c] ? 'TRUCK' : g.boatOk[c] ? 'BOAT' : 'TRUCK';
  const cells = [];
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {      // a 3x3-cell staging area
    const yy = cy + dy, xx = cx + dx, k = yy * w + xx;
    if (yy >= 0 && yy < h && xx >= 0 && xx < w && !(mode === 'TRUCK' ? !g.truckOk[k] && k !== c : !g.boatOk[k])) cells.push(k);
  }
  state.entry = { cells, mode, at: [x, y] };
  state.entryMsg = null;
  setPicking(false);
  solve();
}

function drawHouses() {
  const s = state.s;
  groups.predHouses.clearLayers(); groups.gtHouses.clearLayers(); groups.dis.clearLayers();
  const show = $('tStatus').checked;
  let nDis = 0;
  s.blds.forEach((b, i) => {
    const poly = b.polygon.map(([x, y]) => ll(x, y));
    for (const side of ['pred', 'gt']) {
      const kind = KIND[s.status[side][i]];
      const g = side === 'pred' ? groups.predHouses : groups.gtHouses;
      if (show) {
        L.polygon(poly, { renderer: rend[side], interactive: false, color: getVar(`--${kind}`), weight: 2,
          fillColor: getVar(`--${kind}`), fillOpacity: 0.14 }).addTo(g);
        const sel = state.sel && state.sel.i === i && state.sel.side === side;
        L.marker(ll(...b.centroid), { pane: side, icon: badgeIcon(kind, sel), interactive: false, keyboard: false }).addTo(g);
      }
    }
    if (s.status.pred[i] !== s.status.gt[i]) {
      nDis++;
      if ($('tDisagree').checked) {
        L.polygon(poly, { renderer: rend.dis, interactive: false, color: getVar('--disagree'), weight: 4,
          dashArray: '6 5', fill: false }).addTo(groups.dis);
      }
    }
  });
  $('nDisagree').textContent = nDis;
}

function drawBase() {
  const s = state.s, b = base();
  groups.base.clearLayers();
  $('entryMode').textContent = state.entry ? `chosen (${b.mode === 'BOAT' ? 'boat' : 'truck'})` : 'auto: largest road or dry land at the edge';
  if (!b || !$('tBase').checked) return;
  const color = b.mode === 'BOAT' ? getVar('--accent') : '#ffffff';
  if (state.entry) {
    L.marker(ll(...state.entry.at), { pane: 'route', interactive: false, keyboard: false,
      icon: L.divIcon({ className: 'vehicle', html: '🚩', iconSize: [24, 24], iconAnchor: [5, 22] }) }).addTo(groups.base);
  }
  for (const c of b.cells) {
    const cy = Math.floor(c / s.grid.w), cx = c % s.grid.w;
    L.rectangle([ll(cx * CELL, cy * CELL + CELL), ll(cx * CELL + CELL, cy * CELL)],
      { renderer: baseRend, interactive: false, stroke: false, fillColor: color, fillOpacity: 0.95 }).addTo(groups.base);
  }
}

function drawCounter() {
  const s = state.s, M = s.blds.length;
  const block = (side, label) => {
    const st = s.status[side], n = (k) => st.filter(v => v === k).length;
    const none = n(ISOLATED);
    return `<div>${label ? `<div class="side">${label}</div>` : ''}<div class="big">${M - n(TRUCK)} <small>/ ${M} houses</small></div>` +
      `<div class="split"><span><b>${n(BOAT)}</b> by boat</span><span><b>${n(FOOT)}</b> on foot</span>` +
      `${none ? `<span><b>${none}</b> no route</span>` : ''}</div></div>`;
  };
  const hint = state.picking ? '<p class="hint pick">Tap where the rescue team enters (road, yard or water).</p>'
    : state.entryMsg ? `<p class="hint pick">${state.entryMsg}</p>`
      : state.sel ? '' : '<p class="hint">Tap a house to send a rescue team.</p>';
  $('counter').innerHTML = !base() ? '<div class="lbl">No entry point in this photo: choose one</div>' + hint
    : '<div class="lbl">Unreachable by road</div>' + (state.src === 'swipe'
      ? `<div class="duo">${block('pred', 'Prediction')}${block('gt', 'Labelled map')}</div>`
      : block(state.src, state.src === 'gt' ? 'Labelled map' : '')) + hint;
}

function select(i, side, animate = true) {
  const s = state.s, b = s.blds[i];
  state.sel = { i, side };
  stopAnim();
  groups.route.clearLayers();
  drawHouses();
  const st = s.status[side][i];
  const r = s.sol[side].route(b.ring);                              // ends in the mode of the status, as router.py
  const other = side === 'pred' ? 'gt' : 'pred', stO = s.status[other][i];
  L.polygon(b.polygon.map(([x, y]) => ll(x, y)), { renderer: rend.route, interactive: false, color: getVar('--accent'),
    weight: 3, fill: false }).addTo(groups.route);
  const card = $('routeCard');
  let body = '';
  if (r) {
    const steps = [0, 0, 0];
    for (let k = 1; k < r.modes.length; k++) steps[r.modes[k]]++;
    const parts = steps.map((n, m) => (n ? `${LEG[m].name} <b>${n}</b>` : null)).filter(Boolean);
    body += `<p>${parts.length ? parts.join(' → ') + ' cells' : 'The house is at the entry point'} · cost <b>${r.cost.toFixed(0)}</b></p>`;
    body += `<div class="legs">${steps.map((n, m) => `<span style="flex:${n};background:${LEG[m].color}"></span>`).join('')}</div>`;
  } else {
    body += '<p>No truck, boat or walking route from the entry point on this map.</p>';
  }
  const agree = st === stO;
  body += `<p class="${agree ? 'gtline' : 'mismatch'}">${SIDE[other]}: ${WORD[stO]}${agree ? ' ✓' : ` (${SIDE[side].toLowerCase()} says ${WORD[st]})`}</p>`;
  body += `<p class="gtline">Label: ${CLASS_LABEL[s.ds][b.cls]} · ${SIDE[side]}${state.lam ? ', risk-aware truck' : ', shortest truck'}</p>`;
  card.innerHTML = `<h3><span class="st ${KIND[st]}" style="width:20px;height:20px">${SVG[KIND[st]]}</span>${TITLE[st]}</h3>${body}`;
  card.hidden = false;
  drawCounter();
  if (r && r.cells.length > 1) drawRoute(r, animate);
}

function drawRoute(r, animate) {
  const pts = r.cells.map(cellLL);
  const mk = (color, w, op = 1, dash = null) => L.polyline([], { renderer: rend.route, color, weight: w, opacity: op,
    interactive: false, lineCap: 'round', lineJoin: 'round', dashArray: dash }).addTo(groups.route);
  const legs = [];
  for (let m = 0; m < 3; m++) {                                    // one leg per mode, joined at the switch cell
    const first = r.modes.indexOf(m), last = r.modes.lastIndexOf(m);
    if (first < 0) continue;
    const seg = pts.slice(Math.max(first - 1, 0), last + 1);
    if (seg.length < 2) continue;
    const color = LEG[m].color.startsWith('var') ? getVar(LEG[m].color.slice(4, -1)) : LEG[m].color;
    legs.push({ pts: seg, icon: LEG[m].icon, lines: [mk('#000', 8, 0.45), mk(color, LEG[m].dash ? 5 : 4, 1, LEG[m].dash)] });
  }
  if (!legs.length) return;
  L.circleMarker(pts[0], { renderer: rend.route, radius: 6, color: '#000', weight: 2, fillColor: '#fff', fillOpacity: 1, interactive: false })
    .addTo(groups.route);
  if (!animate || matchMedia('(prefers-reduced-motion: reduce)').matches) {
    legs.forEach(l => l.lines.forEach(p => p.setLatLngs(l.pts)));
    return;
  }
  const lenOf = (p) => p.slice(1).reduce((a, q, j) => a + map.distance(p[j], q), 0);
  const total = legs.reduce((a, l) => a + lenOf(l.pts), 0);
  const speed = total / Math.min(5000, Math.max(1800, total * 6));       // map units per ms
  const veh = L.marker(pts[0], { icon: L.divIcon({ className: 'vehicle', html: legs[0].icon, iconSize: [26, 26], iconAnchor: [13, 13] }),
    pane: 'route', interactive: false, zIndexOffset: 1000 }).addTo(groups.route);
  let li = 0, seg = 0, along = 0, last = performance.now(), pause = 0;
  const done = [];
  const step = (now) => {
    let dt = now - last; last = now;
    if (pause > 0) { pause -= dt; state.anim = requestAnimationFrame(step); return; }
    let move = dt * speed;
    while (move > 0 && li < legs.length) {
      const P = legs[li].pts;
      const d = map.distance(P[seg], P[seg + 1]) - along;
      if (move < d) { along += move; move = 0; break; }
      move -= d; along = 0; seg++;
      if (seg >= P.length - 1) {
        legs[li].lines.forEach(p => p.setLatLngs(P));
        done.push(li); li++; seg = 0;
        if (li < legs.length) { veh.setIcon(L.divIcon({ className: 'vehicle', html: legs[li].icon, iconSize: [26, 26], iconAnchor: [13, 13] })); pause = 350; break; }
      }
    }
    if (li >= legs.length) { veh.setLatLng(pts[pts.length - 1]); state.anim = null; return; }
    const P = legs[li].pts, a = P[seg], b = P[seg + 1], f = along / Math.max(map.distance(a, b), 1e-9);
    const cur = L.latLng(a.lat + (b.lat - a.lat) * f, a.lng + (b.lng - a.lng) * f);
    legs[li].lines.forEach(p => p.setLatLngs([...P.slice(0, seg + 1), cur]));
    veh.setLatLng(cur);
    state.anim = requestAnimationFrame(step);
  };
  state.anim = requestAnimationFrame(step);
}

function stopAnim() { if (state.anim) cancelAnimationFrame(state.anim); state.anim = null; }

map.on('click', (e) => {
  const s = state.s;
  if (!s) return;
  const p = [e.latlng.lng, s.H - e.latlng.lat];
  if (state.picking) { setEntry(...p); return; }
  if (!base()) return;
  let hit = s.blds.findIndex(b => pointInPoly(p, b.polygon));
  if (hit < 0) {                                                   // forgiving tap: nearest badge within 22 px
    let best = 22;
    s.blds.forEach((b, i) => {
      const d = map.latLngToContainerPoint(ll(...b.centroid)).distanceTo(e.containerPoint);
      if (d < best) { best = d; hit = i; }
    });
  }
  if (hit < 0) return;
  const side = state.src === 'swipe' ? (e.containerPoint.x < map.getSize().x * state.swipe ? 'pred' : 'gt') : state.src;
  select(hit, side);
});

// ---------- map source, swipe ----------
function setSource(src) {
  if (src === 'swipe' && state.src !== 'swipe' && state.s) {
    const c = map.latLngToContainerPoint(ll(state.s.W / 2, state.s.H / 2));
    state.swipe = Math.min(0.9, Math.max(0.1, c.x / map.getSize().x));
  }
  state.src = src;
  document.querySelectorAll('.seg button').forEach(b => b.setAttribute('aria-checked', String(b.dataset.src === src)));
  $('swipe').hidden = src !== 'swipe';
  panes.pred.style.display = src === 'gt' ? 'none' : '';
  panes.gt.style.display = src === 'pred' ? 'none' : '';
  updateClip();
  if (state.s) {
    drawCounter();
    if (state.sel) {
      const side = src === 'swipe' ? state.sel.side : src;
      if (side !== state.sel.side || src !== 'swipe') select(state.sel.i, side);
    }
  }
}

function updateClip() {
  if (state.src !== 'swipe') { panes.pred.style.clip = panes.gt.style.clip = ''; return; }
  const size = map.getSize();
  const nw = map.containerPointToLayerPoint([0, 0]), se = map.containerPointToLayerPoint(size);
  const x = map.containerPointToLayerPoint([size.x * state.swipe, 0]).x;
  panes.pred.style.clip = `rect(${nw.y}px, ${x}px, ${se.y}px, ${nw.x}px)`;
  panes.gt.style.clip = `rect(${nw.y}px, ${se.x}px, ${se.y}px, ${x}px)`;
  $('swipe').style.setProperty('--x', `${state.swipe * 100}%`);
}
map.on('move zoom resize viewreset', updateClip);

const handle = $('swipeHandle');
handle.addEventListener('pointerdown', (e) => {
  handle.setPointerCapture(e.pointerId);
  const move = (ev) => { state.swipe = Math.min(0.98, Math.max(0.02, ev.clientX / innerWidth)); updateClip(); };
  const up = () => { handle.removeEventListener('pointermove', move); handle.removeEventListener('pointerup', up); };
  handle.addEventListener('pointermove', move);
  handle.addEventListener('pointerup', up);
});
handle.addEventListener('keydown', (e) => {
  const d = { ArrowLeft: -0.05, ArrowRight: 0.05 }[e.key];
  if (d) { state.swipe = Math.min(0.98, Math.max(0.02, state.swipe + d)); updateClip(); e.preventDefault(); }
});

// ---------- legend, controls ----------
function drawLegend() {
  const m = state.s.meta;
  $('legend').innerHTML = m.class_names.map((n, c) =>
    `<li><span class="sw" style="background:rgb(${m.class_colors[c].join(',')})"></span>${n}</li>`).join('');
}

function getVar(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function opacity() { return $('tClasses').checked ? $('opacity').value / 100 : 0; }

document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => setSource(b.dataset.src)));
$('opacity').addEventListener('input', () => { $('tClasses').checked = true; Object.values(state.overlays || {}).forEach(o => o.setOpacity(opacity())); });
$('tClasses').addEventListener('change', () => Object.values(state.overlays || {}).forEach(o => o.setOpacity(opacity())));
$('tStatus').addEventListener('change', drawHouses);
$('tDisagree').addEventListener('change', drawHouses);
$('tBase').addEventListener('change', drawBase);
$('tRisk').addEventListener('change', () => { state.lam = $('tRisk').checked ? 5 : 0; solve(); });
$('entryPick').addEventListener('click', () => setPicking(!state.picking));
$('entryAuto').addEventListener('click', () => { state.entry = null; state.entryMsg = null; setPicking(false); solve(); });
$('sheetToggle').addEventListener('click', () => {
  const p = $('panel'), c = p.classList.toggle('collapsed');
  $('sheetToggle').setAttribute('aria-expanded', String(!c));
  setTimeout(fit, 50);
});
addEventListener('resize', () => { clearTimeout(state.rt); state.rt = setTimeout(fit, 150); });
addEventListener('hashchange', () => loadScene(location.hash.slice(1)));

(async () => {
  if (innerWidth <= 760) { $('panel').classList.add('collapsed'); $('sheetToggle').setAttribute('aria-expanded', 'false'); }
  await loadIndex();
  await loadScene(location.hash.slice(1));
  setSource('pred');
})();

window.rr = { state, map };                                         // for debugging in the console
