// Rescue Router web viewer (PLAN §6.4). Plain ES module + Leaflet 1.9 (global L), no build step.
// Scenes come from src/export_web_assets.py; routing runs in the browser with js/router.js.

import { sceneGrids, parseScene, eta, tier, TRUCK, BOAT, FOOT, ISOLATED } from './router.js';
import { t, LANG, DEFAULT_LANG, setLang } from './i18n.js';

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
SVG.check = SVG.truck;                                              // 'probably truck: check first' badge
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
// colour-blind palette (Okabe-Ito) for the five class slots, by meaning: clear road sky blue, flooded road dark blue /
// blocked road bluish green (pink is the on-foot status), flooded or damaged building vermillion, dry or intact building yellow, water navy
const CB_CLASS = {
  floodnet: [[86, 180, 233], [0, 114, 178], [213, 94, 0], [240, 228, 66], [0, 31, 120]],
  rescuenet: [[86, 180, 233], [0, 158, 115], [240, 228, 66], [213, 94, 0], [0, 31, 120]],
};
// the two models of the switch (scenes.json 'models'; model b's layers in scenes/<key>/b/): the submitted Chameleon
// and the same-budget baseline on the same labelled photos. Their measured false-safe rates per dataset (for the
// printed briefing) are in scenes.json: models.<a|b>.false_safe.<dataset>
const MODEL = {
  a: { name: 'Chameleon', attrib: '<a href="https://github.com/GitGyun/chameleon">Chameleon</a> (Kim et al., ECCV 2024)'
    + ' · FloodNet building outlines: <a href="https://github.com/facebookresearch/segment-anything">SAM</a>' },
  b: { name: 'DeepLabV3-ResNet50', attrib: '<a href="https://pytorch.org/vision/stable/models/deeplabv3.html">DeepLabV3-ResNet50</a>'
    + ' (Chen et al. 2017; torchvision, COCO-pretrained)' },
};

const $ = (id) => document.getElementById(id);
const state = { index: null, key: null, s: null, src: 'pred', layer: 'classes', lam: 5, swipe: 0.5, sel: null, anim: null,
  entry: null, picking: false, ds: 'floodnet', sort: 'id', cb: false, model: 'a' };

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
  // revalidated on every visit (a 304 when unchanged): a cached index from an older build would hide new scenes or
  // the model switch until the browser's heuristic cache expires
  state.index = await (await fetch('scenes/scenes.json', { cache: 'no-cache' })).json();
  const box = $('scenes');
  for (const sc of state.index.scenes.filter(s => s.showcase)) {
    const b = document.createElement('button');
    b.dataset.key = sc.key;
    b.title = sc.caption;
    b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label', `${sc.dataset} ${sc.id}: ${sc.caption}`);
    const fail = /failure/i.test(sc.caption) ? '<span class="warn">fail</span>' : '';
    b.innerHTML = `<img src="scenes/${sc.key}/thumb.jpg" alt="" loading="lazy"><span class="tag">${sc.dataset === 'floodnet' ? 'FN' : 'RN'} ${sc.id}</span>${fail}`;
    b.addEventListener('click', () => goScene(sc.key));
    box.appendChild(b);
  }
}

// ---------- model switch: 'a' Chameleon, 'b' the baseline where its layers exist (scenes.json entry has 'b') ----------
const modelOf = (sc) => (state.model === 'b' && sc && sc.b ? 'b' : 'a');
const summary = (sc) => (modelOf(sc) === 'b' ? { ...sc, ...sc.b } : sc);     // the browse badges of the shown model

// ---------- browsing all eval100 photos ----------
const cutOff = (sc) => sc.n_buildings - summary(sc).status_pred.TRUCK;    // houses not reachable by truck on the prediction
const SORTS = {
  id: (a, b) => +a.id - +b.id,
  cut: (a, b) => cutOff(b) - cutOff(a) || +a.id - +b.id,
  dis: (a, b) => summary(b).n_disagree - summary(a).n_disagree || +a.id - +b.id,
  iou: (a, b) => (summary(a).miou ?? 2) - (summary(b).miou ?? 2) || +a.id - +b.id,
};
const browseList = () => state.index.scenes.filter(s => s.dataset === state.ds).sort(SORTS[state.sort]);

function drawBrowse() {
  const box = $('allScenes');
  box.innerHTML = '';
  document.querySelectorAll('.seg.ds button').forEach(b => b.setAttribute('aria-checked', String(b.dataset.ds === state.ds)));
  state.browseModel = state.model;
  for (const s0 of browseList()) {
    const b = document.createElement('button');
    const sc = summary(s0), cut = cutOff(s0);
    b.dataset.key = sc.key;
    b.setAttribute('role', 'listitem');
    b.setAttribute('aria-pressed', String(sc.key === state.key));
    b.title = LANG === 'ko'
      ? `${DATASET[sc.dataset].name} ${sc.id}: 가구 ${sc.n_buildings}곳, 트럭 도달 불가 ${cut}곳 (예측), 라벨 지도와 불일치 ${sc.n_disagree}곳` +
        (sc.miou != null ? `, 평균 IoU ${fmt(sc.miou)}` : '')
      : `${DATASET[sc.dataset].name} ${sc.id}: ${sc.n_buildings} houses, ${cut} not reachable by truck (prediction), ` +
        `${sc.n_disagree} disagreement${sc.n_disagree === 1 ? '' : 's'} with the labelled map${sc.miou != null ? `, mean IoU ${fmt(sc.miou)}` : ''}`;
    b.setAttribute('aria-label', b.title);
    b.innerHTML = `<img src="scenes/${sc.key}/thumb.jpg" alt="" loading="lazy"><span class="tag">${sc.id}</span>` +
      (sc.n_buildings ? `<span class="cnt${cut ? ' cut' : ''}">${cut}/${sc.n_buildings}</span>` : '') +
      (sc.n_disagree ? '<span class="dis" aria-hidden="true"></span>' : '');
    b.addEventListener('click', () => goScene(sc.key));
    box.appendChild(b);
  }
}

function step(d) {
  const list = browseList(), k = list.findIndex(s => s.key === state.key);
  const next = k < 0 ? list[0] : list[(k + d + list.length) % list.length];
  goScene(next.key);
}

async function goScene(key) {
  await loadScene(key);
  writeHash(true);
}

function markPressed() {
  document.querySelectorAll('#scenes button, #allScenes button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.key === state.key)));
  const cur = document.querySelector('#allScenes button[aria-pressed="true"]');
  if (cur) {                                                       // keep it visible without scrolling the page
    const box = $('allScenes'), top = cur.offsetTop - box.offsetTop;
    if (top < box.scrollTop || top + cur.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = top - box.clientHeight / 2;
  }
}

// an uploaded photo (Gradio app, src/gradio_viewer.py): scenes/up/<token>/, not in the index, no labels
const uploadEntry = (key) => (/^up\/[\w-]+$/.test(key || '') ? { key, dataset: 'floodnet', id: 'upload', caption: '', upload: true } : null);

function setNoLabels(on) {                                         // hide everything that compares with labels
  state.nolabels = on;
  document.querySelectorAll('.seg button[data-src="gt"], .seg button[data-src="swipe"]').forEach(b => { b.style.display = on ? 'none' : ''; });
  $('tDisagree').closest('label').style.display = on ? 'none' : '';
  if (on && state.src !== 'pred') setSource('pred');
}

async function loadScene(key, keepView = false) {             // keepView: the same photo on the other model
  const sc = state.index.scenes.find(s => s.key === key) || uploadEntry(key) || state.index.scenes[0];
  const model = modelOf(sc), dir = `scenes/${sc.key}/`, mdir = model === 'b' ? `${dir}b/` : dir;   // photo and labels: dir
  const [meta, { grid, buildings: blds }] = await Promise.all([
    fetch(mdir + 'meta.json').then(r => r.json()),
    fetch(mdir + 'scene.rrb').then(r => r.arrayBuffer()).then(parseScene)]);
  stopAnim();
  state.key = sc.key;
  state.sel = null;
  state.entry = null;
  setPicking(false);
  const [W, H] = meta.size;
  if (meta.upload) sc.caption = meta.caption;
  setNoLabels(Boolean(meta.upload));                               // before state.s: setSource redraws the old scene
  // an upload names the model that made its map (Gradio: Chameleon on a GPU, or the CPU baseline); no switch there
  state.s = { sc, dir, mdir, model: meta.upload ? (meta.model === 'b' ? 'b' : 'a') : model, meta, grid, blds, W, H, ds: meta.dataset };
  if (sc.dataset !== state.ds || state.model !== state.browseModel) { state.ds = sc.dataset; drawBrowse(); }
  markPressed();
  drawModelSeg();
  drawCaption();
  for (const k of Object.keys(groups)) groups[k].clearLayers();
  const bounds = [[0, 0], [H, W]];
  L.imageOverlay(dir + 'photo.jpg', bounds).addTo(groups.photo);
  state.overlays = {
    pred: L.imageOverlay(mdir + 'pred.png', bounds, { pane: 'pred', opacity: opacity() }).addTo(groups.pred),
    gt: L.imageOverlay(dir + 'gt.png', bounds, { pane: 'gt', opacity: opacity() }).addTo(groups.gt),
    pwet: L.imageOverlay(pwetImage(grid), bounds, { pane: 'pred', opacity: 0, className: 'pwet' }).addTo(groups.pred),
  };
  setOpacities();
  drawPwetLegend();
  if (state.cb) applyPalette();
  map.attributionControl._attributions = {};
  map.attributionControl.addAttribution(`${DATASET[meta.dataset].attrib} · ${MODEL[state.s.model].attrib}`);
  drawLegend();
  $('routeCard').hidden = true;
  solve();
  if (!keepView) fit();
  loadOther(state.s);
}

// ---------- the other model's statuses: where the two models disagree about a house ----------
// Both models route the same houses (they come from the labelled map) from the same entry point; a house that one
// model sends a truck to and the other a boat is worth a second look. Loaded after the scene is shown.
async function loadOther(s) {
  if (!s.sc.b || s.meta.upload) return;
  const other = s.model === 'b' ? 'a' : 'b', odir = other === 'b' ? `${s.dir}b/` : s.dir;
  const { grid, buildings } = await fetch(odir + 'scene.rrb').then(r => r.arrayBuffer()).then(parseScene);
  if (state.s !== s || buildings.length !== s.blds.length || buildings.some((b, i) => b.id !== s.blds[i].id)) return;
  s.other = { model: other, grid, blds: buildings };
  solveOther();
  drawHouseList();
  drawCounter();
}

function solveOther() {                                            // the other model's statuses, this view's entry
  const s = state.s, o = s.other;
  if (!o) return;
  if (o.lam !== state.lam) { o.grids = sceneGrids(o.grid, state.lam); o.lam = state.lam; }
  const sol = o.grids.pred.solve(state.entry || o.grid.base);
  s.status.other = sol ? o.blds.map(b => sol.status(b.goals)) : null;
}

const otherDiffers = (i) => Boolean(state.s.status.other) && state.s.status.other[i] !== state.s.status.pred[i];

function drawCaption() {
  const { meta, model } = state.s, sc = summary(state.s.sc);
  const ko = LANG === 'ko';
  // the showcase captions describe Chameleon's map: the baseline gets the generic one
  $('caption').textContent = `${t(DATASET[meta.dataset].event)} · ${((ko && !meta.upload) || model === 'b' ? '' : sc.caption) || (ko ? `테스트 사진 ${sc.id}: ` +
    (!sc.n_buildings ? '라벨 지도에 가구 없음' : sc.n_disagree ? `가구 ${sc.n_buildings}곳 중 ${sc.n_disagree}곳이 라벨 지도와 다름`
      : `가구 ${sc.n_buildings}곳 모두 라벨 지도와 일치`) : `Test photo ${sc.id}: ` +
    (!sc.n_buildings ? 'no houses in the labelled map' : sc.n_disagree
      ? `${sc.n_disagree} of ${sc.n_buildings} houses differ from the labelled map` : `all ${sc.n_buildings} houses agree with the labelled map`))}`;
  const shots = meta.run.match(/_(\d+)shot/)[1];
  const ious = Object.values(meta.iou).filter(v => v != null);
  const miouTxt = ious.length ? fmt(ious.reduce((a, b) => a + b, 0) / ious.length) : null;
  const what = model === 'b'
    ? (LANG === 'ko' ? `같은 라벨 사진 <b>${shots}</b>장으로 미세 조정한 DeepLabV3-ResNet50 (조정용 사진 없음)`
      : `DeepLabV3-ResNet50 fine-tuned on the same <b>${shots}</b> labelled photos (no tuning photos)`)
    : LANG === 'ko' ? `라벨 사진 <b>${shots}</b>장(+조정용 30장)으로 적응한 Chameleon` : `Chameleon adapted from <b>${shots}</b> labelled photos (+30 for tuning)`;
  $('model').innerHTML = `${DATASET[meta.dataset].name} · ${what} · ` + (LANG === 'ko'
    ? (miouTxt ? `이 사진: 평균 IoU ${miouTxt}` : '업로드 사진: 채점할 라벨 없음')
    : (miouTxt ? `this photo: mean IoU ${miouTxt}` : 'uploaded photo: no labels to score it'));
  $('modelNote').textContent = meta.upload ? '' : LANG === 'ko' ? `둘 다 같은 라벨 사진 ${shots}장` : `both from the same ${shots} labelled photos`;
}

function drawModelSeg() {                                          // the switch shows the model on screen
  const has = Boolean(state.s.sc.b), up = Boolean(state.s.meta.upload);
  document.querySelectorAll('.seg.models button').forEach(b => {
    b.setAttribute('aria-checked', String(b.dataset.model === state.s.model));
    b.disabled = up;                                               // an upload has the one model that served it
    if (b.dataset.model === 'b' && !up) {
      b.disabled = !has;
      b.title = has ? t('DeepLabV3-ResNet50 fine-tuned on the same labelled photos (same-budget baseline)') : t('Not available for this photo');
    }
  });
}

function setModel(m) {                                             // the same view (house, entry, map) on the other model
  if (m === state.model && m === state.s.model) return;
  state.model = m;
  writeHash();
  applyHash();
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
  }, { animate: false });
  if (phone) {                                                     // a wide photo on a tall screen: no gap above it
    const gap = map.latLngToContainerPoint([state.s.H, 0]).y - top;
    if (gap > 1) map.panBy([0, gap], { animate: false });
  }
}

// ---------- routing ----------
function solve() {
  const s = state.s;
  s.grids = sceneGrids(s.grid, state.lam);
  s.sol = { pred: s.grids.pred.solve(base()), gt: s.grids.gt.solve(base()), cautious: s.grids.pred_cautious.solve(base()) };
  s.status = { pred: s.blds.map(b => s.sol.pred.status(b.goals)), gt: s.blds.map(b => s.sol.gt.status(b.goals)) };
  s.tier = s.blds.map((b, i) => tier(s.status.pred[i], s.sol.cautious.status(b.goals)));
  solveOther();
  drawHouses();
  drawHouseList();
  s.stops = { pred: stopPoints('pred'), gt: stopPoints('gt') };
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
  writeHash();
}

function drawHouses() {
  const s = state.s;
  groups.predHouses.clearLayers(); groups.gtHouses.clearLayers(); groups.dis.clearLayers();
  const show = $('tStatus').checked;
  let nDis = 0;
  s.blds.forEach((b, i) => {
    const poly = b.polygon.map(([x, y]) => ll(x, y));
    for (const side of ['pred', 'gt']) {
      const kind = side === 'pred' && s.tier[i] === 'PROBABLE_TRUCK' ? 'check' : KIND[s.status[side][i]];
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

// ---------- house list (keyboard access to every house) ----------
function drawHouseList() {
  const s = state.s, side = state.src === 'gt' ? 'gt' : 'pred', box = $('houses');
  box.dataset.empty = t('No houses in this photo.');
  $('housesNote').textContent = s.blds.length ? `${s.blds.length} · ${t(SIDE[side]).toLowerCase()}` : '';
  box.innerHTML = '';
  s.blds.forEach((b, i) => {
    const st = s.status[side][i], tr = side === 'pred' ? s.tier[i] : null;
    const kind = tr === 'PROBABLE_TRUCK' ? 'check' : KIND[st];
    const word = t(tr === 'PROBABLE_TRUCK' ? 'truck, check first' : tr === 'CONFIDENT_TRUCK' ? 'truck (confident)' : WORD[st]);
    const dis = s.status.pred[i] !== s.status.gt[i];
    const el = document.createElement('button');
    el.setAttribute('role', 'option');
    el.dataset.i = i;
    const oth = side === 'pred' && otherDiffers(i) ? `${MODEL[s.other.model].name.split('-')[0]}: ${t(WORD[s.status.other[i]])}` : '';
    el.innerHTML = `<span class="st ${kind}" aria-hidden="true">${SVG[kind]}</span><span class="what">${t('House')} ${b.id}: ${word}` +
      `<br><span class="sub">${t(CLASS_LABEL[s.ds][b.cls])}${oth ? ` · <span class="oth">${oth}</span>` : ''}</span></span>` +
      (dis ? `<span class="dis" title="${t('Prediction and labelled map disagree')}">≠ ${t(WORD[s.status[side === 'pred' ? 'gt' : 'pred'][i]])}</span>` : '');
    const otherWord = WORD[s.status[side === 'pred' ? 'gt' : 'pred'][i]];
    el.setAttribute('aria-label', LANG === 'ko'
      ? `${t('House')} ${b.id}: ${word}, ${t(CLASS_LABEL[s.ds][b.cls])}` + (dis ? `; ${t(side === 'pred' ? 'Labelled map' : 'Prediction')}: ${t(otherWord)}` : '')
      : `House ${b.id}: ${word}, ${CLASS_LABEL[s.ds][b.cls]}` + (dis ? `; the ${side === 'pred' ? 'labelled map' : 'prediction'} says ${otherWord}` : '')
        + (oth ? `; ${MODEL[s.other.model].name} says ${WORD[s.status.other[i]]}` : ''));
    el.addEventListener('click', () => { if (base()) { select(i, side); writeHash(); } });
    box.appendChild(el);
  });
  markHouse();
}

function markHouse() {                                             // aria-selected + roving tabindex
  const items = [...$('houses').children], cur = state.sel ? state.sel.i : 0;
  items.forEach((el, k) => {
    el.setAttribute('aria-selected', String(Boolean(state.sel) && k === cur));
    el.tabIndex = k === cur ? 0 : -1;
  });
}

$('houses').addEventListener('keydown', (e) => {
  const items = [...$('houses').children], k = items.indexOf(document.activeElement);
  if (k < 0) return;
  const to = { ArrowDown: k + 1, ArrowUp: k - 1, Home: 0, End: items.length - 1 }[e.key];
  if (to === undefined) return;
  e.preventDefault();
  const el = items[Math.max(0, Math.min(items.length - 1, to))];
  items.forEach(x => { x.tabIndex = -1; });
  el.tabIndex = 0;
  el.focus();
});

function drawBase() {
  const s = state.s, b = base();
  groups.base.clearLayers();
  $('entryMode').textContent = LANG === 'ko' ? (state.entry ? `직접 선택 (${b.mode === 'BOAT' ? '보트' : '트럭'})` : '자동: 가장자리의 가장 큰 도로 또는 마른 땅')
    : state.entry ? `chosen (${b.mode === 'BOAT' ? 'boat' : 'truck'})` : 'auto: largest road or dry land at the edge';
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
  drawStops();
}

// ---------- dispatch: where the truck stops for the houses it cannot reach (boat launch or walk-on point) ----------
const STOP_GROUP_M = 20;                                           // stop points closer than this share one staging spot
function stopPoints(side) {
  const s = state.s, solR = s.sol[side], mpc = s.meta.m_per_cell || s.grid.cellM || 1, groups = [];
  s.blds.forEach((b, i) => {
    if (s.status[side][i] === TRUCK || s.status[side][i] === ISOLATED) return;
    const r = solR.route(b.goals);
    if (!r) return;
    const k = r.modes.findIndex((m, j) => j > 0 && m !== TRUCK);
    const cell = r.cells[Math.max(k - 1, 0)];                     // last truck cell (the entry if it starts by boat)
    const g = groups.find(x => Math.hypot(x.cell[0] - cell[0], x.cell[1] - cell[1]) * mpc <= STOP_GROUP_M);
    if (g) g.houses.push(i); else groups.push({ cell, houses: [i], boat: r.modes.includes(BOAT) });
    if (g && r.modes.includes(BOAT)) g.boat = true;
  });
  return groups;
}

function drawStops() {                                            // called by drawBase (same 'Entry points' toggle)
  const s = state.s, side = state.src === 'gt' ? 'gt' : 'pred';
  if (!s.stops) return;
  s.stops[side].forEach((g, n) => {
    L.marker(cellLL(g.cell), { pane: 'base', interactive: false, keyboard: false, icon: L.divIcon({ className: '',
      iconSize: [20, 20], iconAnchor: [10, 10], html: `<div class="stop" title="${LANG === 'ko' ? `트럭 정차: ${g.boat ? '보트 출발' : '도보 출발'}, 가구 ${g.houses.length}곳` : `Truck stops here: ${g.boat ? 'boat launch' : 'teams walk on'} for ${g.houses.length} house${g.houses.length > 1 ? 's' : ''}`}">${String.fromCharCode(65 + n)}</div>` }) }).addTo(groups.base);
  });
}

function drawCounter() {
  const s = state.s, M = s.blds.length;
  const block = (side, label) => {
    const st = s.status[side], n = (k) => st.filter(v => v === k).length;
    const none = n(ISOLATED);
    const check = side === 'pred' ? s.tier.filter(t => t === 'PROBABLE_TRUCK').length : 0;
    const ns = s.stops ? s.stops[side].length : 0, list = ns ? s.stops[side].map((g, k) => `${String.fromCharCode(65 + k)}: ${g.houses.length}`).join(', ') : '';
    return `<div>${label ? `<div class="side">${t(label)}</div>` : ''}<div class="big">${M - n(TRUCK)} <small>/ ${M} ${t('houses')}</small></div>` +
      `<div class="split"><span><b>${n(BOAT)}</b> ${t('by boat')}</span><span><b>${n(FOOT)}</b> ${t('on foot')}</span>` +
      `${none ? `<span><b>${none}</b> ${t('no route')}</span>` : ''}</div>` +
      (check ? `<div class="split check"><span><b>${check}</b> ${t('more look truck-reachable: check first')}</span></div>` : '') +
      (ns ? `<div class="split stops"><span>${LANG === 'ko' ? `트럭 정차 지점 <b>${ns}</b>곳 (${list} 가구)`
        : `truck stops at <b>${ns}</b> point${ns > 1 ? 's' : ''} (${list} houses)`}</span></div>` : '') +
      (side === 'pred' && nOther ? `<div class="split other"><span>${LANG === 'ko' ? `${MODEL[s.other.model].name}의 판단이 다른 가구 <b>${nOther}</b>곳`
        : `${MODEL[s.other.model].name} differs on <b>${nOther}</b> house${nOther > 1 ? 's' : ''}`}</span></div>` : '') + '</div>';
  };
  const nOther = s.status.other ? s.blds.filter((_, i) => otherDiffers(i)).length : 0;
  const hint = state.picking ? `<p class="hint pick">${t('Tap where the rescue team enters (road, yard or water).')}</p>`
    : state.entryMsg ? `<p class="hint pick">${t(state.entryMsg)}</p>`
      : state.sel ? '' : `<p class="hint">${t('Tap a house to send a rescue team.')}</p>`;
  $('counter').innerHTML = !M ? `<div class="lbl">${t('No houses in this photo')}</div><p class="hint">${t('Try the next photo: › (or the N key).')}</p>`
    : !base() ? `<div class="lbl">${t('No entry point in this photo: choose one')}</div>` + hint
    : `<div class="lbl">${t('Unreachable by road')}</div>` + (state.src === 'swipe'
      ? `<div class="duo">${block('pred', 'Prediction')}${block('gt', 'Labelled map')}</div>`
      : block(state.src, state.src === 'gt' ? 'Labelled map' : '')) + hint;
}

function select(i, side, animate = true) {
  const s = state.s, b = s.blds[i];
  state.sel = { i, side };
  markHouse();
  stopAnim();
  groups.route.clearLayers();
  drawHouses();
  const st = s.status[side][i];
  const tr = side === 'pred' ? s.tier[i] : null;
  // a confident-truck house gets its route on the cautious map (it avoids land that may be wet)
  const solR = tr === 'CONFIDENT_TRUCK' ? s.sol.cautious : s.sol[side];
  const r = solR.route(b.goals);                                     // ends in the mode of the status, as router.py
  const other = side === 'pred' ? 'gt' : 'pred', stO = s.status[other][i];
  L.polygon(b.polygon.map(([x, y]) => ll(x, y)), { renderer: rend.route, interactive: false, color: getVar('--accent'),
    weight: 3, fill: false }).addTo(groups.route);
  const card = $('routeCard');
  let body = '';
  if (r) {
    const steps = [0, 0, 0];
    for (let k = 1; k < r.modes.length; k++) steps[r.modes[k]]++;
    const parts = steps.map((n, m) => (n ? `${t(LEG[m].name)} <b>${n}</b>` : null)).filter(Boolean);
    const e = eta(r, solR.grid, s.grid.eta, s.meta.m_per_cell);
    const km = (v) => (v >= 1000 ? `${(v / 1000).toFixed(1)} km` : `${Math.round(v / 10) * 10} m`);
    const legs = e.metres.map((v, m) => (v > 0 ? `${t(LEG[m].name)} ${km(v)}` : null)).filter(Boolean);
    body += `<p class="eta">≈ <b>${Math.max(1, Math.round(e.minutes))} ${LANG === 'ko' ? '분' : 'min'}</b> · ${legs.length ? legs.join(' → ') : t('at the entry point')}</p>`;
    body += `<div class="legs">${steps.map((n, m) => `<span style="flex:${n};background:${LEG[m].color}"></span>`).join('')}</div>`;
    body += `<p class="gtline">${t('Route cost')} ${r.cost.toFixed(0)} (${parts.length ? parts.join(' → ') + ' ' + t('cells') : t('no steps')})</p>`;
  } else {
    body += `<p>${t('No truck, boat or walking route from the entry point on this map.')}</p>`;
  }
  const agree = st === stO || state.nolabels;
  if (!state.nolabels) body += `<p class="${agree ? 'gtline' : 'mismatch'}">${t(SIDE[other])}: ${t(WORD[stO])}${agree ? ' ✓' : LANG === 'ko' ? ` (${t(SIDE[side])}: ${t(WORD[st])})` : ` (${SIDE[side].toLowerCase()} says ${WORD[st]})`}</p>`;
  body += `<p class="gtline">${t(state.nolabels ? 'Predicted' : 'Label')}: ${t(CLASS_LABEL[s.ds][b.cls])} · ${t(SIDE[side])}, ${t(state.lam ? 'risk-aware truck' : 'shortest truck')}</p>`;
  const ts = s.meta.tier_stats, pct = (v) => `${Math.round(100 * v)} %`;
  let kind = KIND[st], title = t(TITLE[st]);
  const ko = LANG === 'ko';
  if (tr === 'CONFIDENT_TRUCK') {
    title = t('Truck can reach it: confident');
    if (ts) body = (ko ? `<p class="tierline">안전 경로: 젖었을 수 있는 땅을 피합니다. 테스트 사진에서 이런 가구의 ${pct(ts.confident)}가 실제로 트럭 도달 가능했습니다.</p>`
      : `<p class="tierline">Safe route: avoids land that may be wet. On the test photos ${pct(ts.confident)} of such houses were truly truck-reachable.</p>`) + body;
  } else if (tr === 'PROBABLE_TRUCK') {
    kind = 'check'; title = t('Probably reachable by truck: check first');
    if (ts) body = (ko ? `<p class="tierline warn">유일한 트럭 경로가 젖었을 수 있는 땅을 지납니다. 테스트 사진에서 이런 가구는 ${pct(ts.probable)}만 실제로 트럭 도달 가능했습니다(확실 등급: ${pct(ts.confident)}). 트럭을 보내기 전에 드론이나 무전으로 확인하세요.</p>`
      : `<p class="tierline warn">The only truck route crosses land that may be wet. On the test photos only ${pct(ts.probable)} of such houses were truly truck-reachable (confident tier: ${pct(ts.confident)}). Confirm by drone or radio before sending a truck.</p>`) + body;
  }
  // on disagreement, the other map's route as a dashed ghost (under this map's route)
  const rO = agree ? null : s.sol[other].route(b.goals);
  if (rO && rO.cells.length > 1) {
    L.polyline(rO.cells.map(cellLL), { renderer: rend.route, color: '#000', weight: 6, opacity: 0.35, interactive: false,
      lineCap: 'round', lineJoin: 'round' }).addTo(groups.route);
    L.polyline(rO.cells.map(cellLL), { renderer: rend.route, color: getVar('--disagree'), weight: 3, opacity: 0.95,
      dashArray: '7 6', interactive: false, lineCap: 'butt', lineJoin: 'round' }).addTo(groups.route);
    const mO = [...new Set(rO.modes.slice(1))].map(m => t(LEG[m].name)).join(' → ');
    body += `<p class="gtline"><span class="line ghostline"></span> ${t(SIDE[other])} ${LANG === 'ko' ? '경로' : 'route'}: ${mO || t('at the entry point')}</p>`;
  }
  card.innerHTML = `<h3><span class="st ${kind}" style="width:20px;height:20px">${SVG[KIND[st]]}</span>${title}</h3>${body}`;
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
  writeHash();
});

// ---------- map source, swipe ----------
function setSource(src) {
  if (state.nolabels && src !== 'pred') return;                     // an upload has no labelled map
  if (src === 'swipe' && state.src !== 'swipe' && state.s) {
    const c = map.latLngToContainerPoint(ll(state.s.W / 2, state.s.H / 2));
    state.swipe = Math.min(0.9, Math.max(0.1, c.x / map.getSize().x));
  }
  state.src = src;
  document.querySelectorAll('.seg button[data-src]').forEach(b => b.setAttribute('aria-checked', String(b.dataset.src === src)));
  $('swipe').hidden = src !== 'swipe';
  panes.pred.style.display = src === 'gt' ? 'none' : '';
  panes.gt.style.display = src === 'pred' ? 'none' : '';
  updateClip();
  if (state.s) {
    drawBase();
    drawCounter();
    drawHouseList();
    if (state.sel) {
      const side = src === 'swipe' ? state.sel.side : src;
      if (side !== state.sel.side || src !== 'swipe') select(state.sel.i, side);
    }
  }
}

function updateClip() {
  if (state.src !== 'swipe') { panes.pred.style.clipPath = panes.gt.style.clipPath = ''; return; }
  const size = map.getSize();
  const nw = map.containerPointToLayerPoint([0, 0]), se = map.containerPointToLayerPoint(size);
  const x = map.containerPointToLayerPoint([size.x * state.swipe, 0]).x;
  const box = (l, r) => `polygon(${l}px ${nw.y}px, ${r}px ${nw.y}px, ${r}px ${se.y}px, ${l}px ${se.y}px)`;
  panes.pred.style.clipPath = box(nw.x, x);                        // clip-path, not the deprecated clip: rect()
  panes.gt.style.clipPath = box(x, se.x);
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
  const cols = classColors();
  $('legend').innerHTML = m.class_names.map((n, c) =>
    `<li><span class="sw" style="background:rgb(${cols[c].join(',')})"></span>${t(n)}</li>`).join('');
}

// ---------- colour-blind palette ----------
const classColors = () => (state.cb ? CB_CLASS[state.s.ds] : state.s.meta.class_colors);

async function recolored(url, from, to) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height), a = d.data;
  const key = (r, g, b) => (r << 16) | (g << 8) | b, lut = new Map(from.map((f, k) => [key(...f), to[k]]));
  for (let i = 0; i < a.length; i += 4) {                          // overlays are exact colours, alpha 0 or 255
    if (!a[i + 3]) continue;
    const q = lut.get(key(a[i], a[i + 1], a[i + 2]));
    if (q) { a[i] = q[0]; a[i + 1] = q[1]; a[i + 2] = q[2]; }
  }
  ctx.putImageData(d, 0, 0);
  return c.toDataURL();
}

async function applyPalette() {
  document.documentElement.classList.toggle('cb', state.cb);
  const s = state.s;
  if (!s) return;
  if (state.cb && !s.cbUrls) {
    const [p, g] = await Promise.all(['pred', 'gt'].map(k => recolored(layerUrl(s, k), s.meta.class_colors, CB_CLASS[s.ds])));
    if (state.s !== s) return;                                     // the scene changed meanwhile
    s.cbUrls = { pred: p, gt: g };
  }
  for (const k of ['pred', 'gt']) state.overlays[k].setUrl(state.cb ? s.cbUrls[k] : layerUrl(s, k));
  drawLegend();
  drawHouses();
  drawHouseList();
  if (state.sel) select(state.sel.i, state.sel.side, false);
}

function layerUrl(s, k) { return (k === 'pred' ? s.mdir : s.dir) + k + '.png'; }   // the prediction is the model's
function getVar(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function opacity() { return $('tClasses').checked ? $('opacity').value / 100 : 0; }
function setOpacities() {
  const o = state.overlays;
  if (!o) return;
  o.pred.setOpacity(state.layer === 'classes' ? opacity() : 0);
  o.pwet.setOpacity(state.layer === 'pwet' ? Math.min(1, opacity() * 1.4) : 0);   // its colours carry their own alpha
  o.gt.setOpacity(opacity());
}

// ---------- water probability layer (prediction): p_wet = p1 + p4 per routing cell, coloured by the gates ----------
// solid yellow: a truck may cross the cell on the normal map but not on the cautious one (what makes "check first");
// faint yellow: other cells between the two gates; blue: at or above the normal gate. Buildings are left clear.
function pwetColor(p, cautious, tuned, gateCell = false) {
  if (gateCell) return [250, 204, 21, 190];
  if (p < cautious) return [0, 0, 0, 0];
  if (p < tuned) return [250, 204, 21, 80];
  const t = Math.min(1, (p - tuned) / (1 - tuned));             // light -> deep blue
  return [56 + (30 - 56) * t, 189 + (64 - 189) * t, 248 + (175 - 248) * t, 150 + 90 * t];
}

function pwetImage(grid) {
  const c = document.createElement('canvas');
  c.width = grid.w; c.height = grid.h;
  const ctx = c.getContext('2d'), img = ctx.createImageData(grid.w, grid.h), r = grid.pred.risk;
  const f = grid.pred.flags, fc = grid.pred_cautious.flags;
  const { offroad_maxwet_cautious: cg, offroad_maxwet_pred: tg } = grid.params;
  for (let i = 0; i < r.length; i++) {
    if (f[i] & 64) continue;                                       // building cell
    img.data.set(pwetColor(r[i], cg, tg, (f[i] & 1) && !(fc[i] & 1)).map(Math.round), 4 * i);
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL();
}

function drawPwetLegend() {
  const s = state.s, { offroad_maxwet_cautious: cg, offroad_maxwet_pred: tg } = s.grid.params;
  const stops = [];
  for (let k = 0; k <= 100; k += 2) {
    const [r, g, b, a] = pwetColor(k / 100, cg, tg);
    stops.push(`rgba(${Math.round(r)},${Math.round(g)},${Math.round(b)},${(a / 255).toFixed(2)}) ${k}%`);
  }
  $('pwetRamp').style.background = `linear-gradient(90deg, ${stops.join(',')}), #1e293b`;
  $('pwetRamp').innerHTML = [0, cg, tg, 1].map(v => `<span style="left:${v * 100}%">${v === 0 || v === 1 ? v : fmt(v)}</span>`).join('');
  const what = s.ds === 'rescuenet' ? 'water or blocked road' : 'water';
  $('pwetText').innerHTML = LANG === 'ko'
    ? `셀별 모델의 ${s.ds === 'rescuenet' ? '물 또는 막힌 도로' : '물'} 확률. <b class="b">파랑</b> (≥ ${fmt(tg)}): 젖은 것으로 보고 트럭이 ` +
      `도로 밖으로 지나가지 않습니다. <b class="y">노랑</b> (${fmt(cg)}–${fmt(tg)}): 보통 지도에서만 트럭이 지날 수 있고 신중한 지도에서는 ` +
      '지날 수 없는 곳: 이곳을 지나야 닿는 가구는 "먼저 확인"입니다. 예측만 해당: 라벨 지도에는 확률이 없습니다.'
    : `Model's probability of ${what} per cell. <b class="b">Blue</b> (≥ ${fmt(tg)}): treated as wet; ` +
      `trucks keep off it off-road. <b class="y">Yellow</b> (${fmt(cg)}–${fmt(tg)}), solid where a truck may cross only on the ` +
      'normal map, not the cautious one: houses reached through it are "check first". Prediction only: the labelled map has no probabilities.';
  $('pwetLegend').hidden = state.layer !== 'pwet';
}

function setLayer(layer) {
  state.layer = layer === 'pwet' ? 'pwet' : 'classes';
  document.querySelectorAll('.seg button[data-layer]').forEach(b => b.setAttribute('aria-checked', String(b.dataset.layer === state.layer)));
  $('pwetLegend').hidden = state.layer !== 'pwet';
  setOpacities();
}

document.querySelectorAll('.seg button[data-src]').forEach(b => b.addEventListener('click', () => { setSource(b.dataset.src); writeHash(); }));
document.querySelectorAll('.seg button[data-layer]').forEach(b => b.addEventListener('click', () => { setLayer(b.dataset.layer); writeHash(); }));
document.querySelectorAll('.seg.ds button').forEach(b => b.addEventListener('click', () => { state.ds = b.dataset.ds; drawBrowse(); markPressed(); }));
document.querySelectorAll('.seg.models button').forEach(b => b.addEventListener('click', () => setModel(b.dataset.model)));
$('sortScenes').addEventListener('change', (e) => { state.sort = e.target.value; drawBrowse(); markPressed(); });
$('prevScene').addEventListener('click', () => step(-1));
$('nextScene').addEventListener('click', () => step(1));

// the segmented controls are ARIA radio groups: one tab stop per group (the checked button), arrow keys move the
// choice (WAI-ARIA radio group pattern); tabindex follows aria-checked wherever the code sets it
const radioTab = (r) => { r.tabIndex = r.getAttribute('aria-checked') === 'true' ? 0 : -1; };
document.querySelectorAll('[role="radio"]').forEach(radioTab);
new MutationObserver(ms => ms.forEach(m => radioTab(m.target)))
  .observe(document.body, { subtree: true, attributeFilter: ['aria-checked'] });
document.querySelectorAll('[role="radiogroup"]').forEach(g => g.addEventListener('keydown', (e) => {
  const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
  const radios = [...g.querySelectorAll('[role="radio"]:not(:disabled)')], k = radios.indexOf(document.activeElement);
  if (!d || k < 0 || radios.length < 2) return;
  e.preventDefault();
  const to = radios[(k + d + radios.length) % radios.length];
  to.click();
  to.focus();
}));
addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  if (e.key === 'n' || e.key === 'N') step(1);
  else if (e.key === 'p' || e.key === 'P') step(-1);
  else if (e.key === 'Escape' && state.picking) setPicking(false);
  else if (e.key === 'Escape' && state.sel) { clearSelection(); writeHash(); }
});
// ---------- house list as CSV (for a field team's sheet; same routes and ETAs as the route card) ----------
function houseRows() {                                           // prediction: what a field team would have
  const s = state.s;
  const stopOf = (i) => { const n = (s.stops ? s.stops.pred : []).findIndex(g => g.houses.includes(i)); return n < 0 ? '' : String.fromCharCode(65 + n); };
  return s.blds.map((b, i) => {
    const st = s.status.pred[i], tr = s.tier[i];
    const solR = tr === 'CONFIDENT_TRUCK' ? s.sol.cautious : s.sol.pred;
    const r = solR.route(b.goals), e = r ? eta(r, solR.grid, s.grid.eta, s.meta.m_per_cell) : null;
    return { i, id: b.id, cls: CLASS_LABEL[s.ds][b.cls], st, tier: tr === 'CONFIDENT_TRUCK' ? 'confident' : tr === 'PROBABLE_TRUCK' ? 'check first' : '',
      minutes: e ? Math.max(1, Math.round(e.minutes)) : '', metres: e ? e.metres.map(v => Math.round(v)) : ['', '', ''],
      x: Math.round(b.centroid[0]), y: Math.round(b.centroid[1]), stop: stopOf(i), gt: s.status.gt[i],
      other: s.status.other ? s.status.other[i] : null };
  });
}

function houseCsv() {
  const s = state.s, q = (v) => `"${String(v).replace(/"/g, '""')}"`;
  const rows = [['photo', 'model', 'house', 'class', 'status', 'tier', 'minutes', 'truck_m', 'boat_m', 'foot_m', 'x_px', 'y_px',
    'truck_stop', ...(s.status.other ? [`${MODEL[s.other.model].name}_status`] : []), ...(state.nolabels ? [] : ['labelled_map_status'])]];
  for (const h of houseRows()) {
    rows.push([`${DATASET[s.ds].name} ${s.sc.id}`, MODEL[s.model].name, h.id, h.cls, WORD[h.st], h.tier, h.minutes, ...h.metres, h.x, h.y, h.stop,
      ...(s.status.other ? [WORD[h.other]] : []), ...(state.nolabels ? [] : [WORD[h.gt]])]);
  }
  return rows.map(r => r.map(q).join(',')).join('\r\n') + '\r\n';
}

// ---------- printed briefing: one page per photo for a field team (same rows as the CSV, in the reader's language) ----------
function buildBriefing() {
  const s = state.s, ko = LANG === 'ko', esc = (v) => String(v).replace(/[&<>"]/g, c => `&#${c.charCodeAt(0)};`);
  const rows = houseRows(), n = (k) => s.status.pred.filter(v => v === k).length;
  const conf = s.tier.filter(x => x === 'CONFIDENT_TRUCK').length, check = s.tier.filter(x => x === 'PROBABLE_TRUCK').length;
  const word = (h) => t(h.tier === 'confident' ? 'truck (confident)' : h.tier === 'check first' ? 'truck, check first' : WORD[h.st]);
  const shots = s.meta.run.match(/_(\d+)shot/)[1], mo = MODEL[s.model];
  // the measured false-safe rate of this model on this dataset's test photos (none for uploads: no labels, other place)
  const fs = !s.meta.upload && state.index.models ? ((state.index.models[s.model] || {}).false_safe || {})[s.ds] : null;
  const pct = fs ? Math.round(fs.rate * 100) : null, nPh = fs ? fs.n_photos : 0;
  const fsTxt = !fs ? '' : ko ? `${DATASET[s.ds].name} 테스트 사진 ${nPh}장에서 이 모델이 트럭 도달 가능으로 표시한 가구의 약 ${pct} %는 실제로 그렇지 않았습니다. `
    : `on ${nPh} ${DATASET[s.ds].name} test photos about ${pct} % of the houses this model showed as truck-reachable were not. `;
  const view = t(state.src === 'gt' ? 'Labelled map' : state.src === 'swipe' ? 'Swipe' : 'Prediction');
  const diff = rows.filter(h => h.other != null && h.other !== h.st);       // the other model's word on these houses
  const head = ko ? ['가구', '건물', '상태', '분', '트럭 m', '보트 m', '도보 m', '트럭 정차']
    : ['House', 'Building', 'Status', 'Min', 'Truck m', 'Boat m', 'Foot m', 'Truck stop'];
  const stops = (s.stops ? s.stops.pred : []).map((g, k) => `${String.fromCharCode(65 + k)}: ${t(g.boat ? 'boat launch' : 'walk-on point')}, ${g.houses.length}`).join(' · ');
  $('briefing').innerHTML = (ko
    ? `<h2>가구 목록: ${esc(DATASET[s.ds].name)} ${esc(s.sc.id)} (${mo.name} 예측)</h2><p>가구 ${s.blds.length}곳: 트럭 ${n(TRUCK)}곳 (확실 ${conf}, 먼저 확인 ${check}), 보트 ${n(BOAT)}, 도보 ${n(FOOT)}, 경로 없음 ${n(ISOLATED)}. 지도: ${esc(view)}. ` +
      `진입: ${esc($('entryMode').textContent)}. 모델: ${s.model === 'b' ? `DeepLabV3-ResNet50, 라벨 사진 ${shots}장(조정용 없음)` : `Chameleon, 라벨 사진 ${shots}장(+조정용 30장)`}.</p>`
    : `<h2>Houses: ${esc(DATASET[s.ds].name)} ${esc(s.sc.id)} (${mo.name} prediction)</h2><p>${s.blds.length} houses: ${n(TRUCK)} by truck (${conf} confident, ${check} check first), ` +
      `${n(BOAT)} boat, ${n(FOOT)} on foot, ${n(ISOLATED)} no route. Map shown: ${esc(view)}. Entry: ${esc($('entryMode').textContent)}. ` +
      `Model: ${s.model === 'b' ? `DeepLabV3-ResNet50 fine-tuned on ${shots} labelled photos (no tuning photos)` : `Chameleon adapted from ${shots} labelled photos (+30 for tuning)`}.</p>`) +
    (stops ? `<p>${t('Truck stops')}: ${esc(stops)} ${t('houses')}</p>` : '') +
    (diff.length ? `<p>${ko ? `${s.other ? MODEL[s.other.model].name : ''}의 판단이 다른 가구 ${diff.length}곳 (먼저 확인): `
      : `${s.other ? MODEL[s.other.model].name : ''} differs on ${diff.length} house${diff.length > 1 ? 's' : ''} (check first): `}` +
      `${esc(diff.map(h => `${h.id} ${t(WORD[h.other])}`).join(', '))}</p>` : '') +
    `<table><thead><tr>${head.map(x => `<th>${x}</th>`).join('')}</tr></thead><tbody>` +
    rows.map(h => `<tr${h.tier === 'check first' ? ' class="warn"' : ''}><td>${h.id}</td><td>${esc(t(h.cls))}</td><td>${esc(word(h))}</td>` +
      `<td class="n">${h.minutes}</td>${h.metres.map(v => `<td class="n">${v}</td>`).join('')}<td>${h.stop}</td></tr>`).join('') +
    '</tbody></table>' +
    `<p class="fine">${ko ? `운영용이 아닙니다: ${fsTxt}"먼저 확인" 가구는 트럭을 보내기 전에 드론이나 무전으로 확인하세요. 시간과 거리는 대략적인 추정입니다.`
      : `Not for operational use: ${fsTxt}Confirm "check first" houses by drone or radio before sending a truck. Minutes and metres are rough estimates.`} ` +
    `${esc(new Date().toLocaleString(ko ? 'ko-KR' : 'en-GB'))} · ${esc(location.href)}</p>`;
}

addEventListener('beforeprint', () => {
  if (!state.s) return;
  buildBriefing();
  const w = 680, h = Math.round(w * state.s.H / state.s.W);           // A4 width at 96 dpi, the photo's aspect
  Object.assign($('map').style, { width: `${w}px`, height: `${h}px` });
  map.invalidateSize({ animate: false });
  map.fitBounds([[0, 0], [state.s.H, state.s.W]], { padding: [0, 0], animate: false });
  if (state.sel) select(state.sel.i, state.sel.side, false);           // the whole route, not a frame of the animation
});
addEventListener('afterprint', () => { $('map').style.width = $('map').style.height = ''; fit(); });
$('printBrief').addEventListener('click', () => { if (state.s) window.print(); });
$('csvHouses').addEventListener('click', () => {
  if (!state.s || !state.s.blds.length) return;
  const url = URL.createObjectURL(new Blob([houseCsv()], { type: 'text/csv' }));
  const a = Object.assign(document.createElement('a'), { href: url,
    download: `rescue_router_${state.s.sc.key}_${state.s.model === 'b' ? 'deeplabv3' : 'chameleon'}_houses.csv` });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

$('copyLink').addEventListener('click', async () => {
  writeHash();
  let ok = true;
  try { await navigator.clipboard.writeText(location.href); } catch { ok = false; }
  $('copyLink').textContent = t(ok ? 'Link copied' : 'Copy the address bar');
  setTimeout(() => { $('copyLink').textContent = t('Copy link to this view'); }, 1800);
});
$('opacity').addEventListener('input', () => { $('tClasses').checked = true; setOpacities(); });
$('tClasses').addEventListener('change', setOpacities);
$('tStatus').addEventListener('change', drawHouses);
$('tDisagree').addEventListener('change', drawHouses);
$('tBase').addEventListener('change', drawBase);
$('tCb').addEventListener('change', () => { state.cb = $('tCb').checked; applyPalette(); writeHash(); });
$('tRisk').addEventListener('change', () => { state.lam = $('tRisk').checked ? 5 : 0; solve(); writeHash(); });
$('entryPick').addEventListener('click', () => setPicking(!state.picking));
$('entryAuto').addEventListener('click', () => { state.entry = null; state.entryMsg = null; setPicking(false); solve(); writeHash(); });
$('sheetToggle').addEventListener('click', () => {
  const p = $('panel'), c = p.classList.toggle('collapsed');
  $('sheetToggle').setAttribute('aria-expanded', String(!c));
  if (!c) markPressed();                                           // the photo list can only scroll once visible
  setTimeout(fit, 50);
});
addEventListener('resize', () => { clearTimeout(state.rt); state.rt = setTimeout(fit, 150); });
addEventListener('hashchange', applyHash);                          // back/forward, or a pasted link

// ---------- the view in the URL: #s=<scene>&mo=b&h=<house id>[gt]&e=<x>,<y>&m=gt|swipe&l=pwet&r=0&cb=1 ----------
function readHash() {
  const h = decodeURIComponent(location.hash.slice(1));
  const q = new URLSearchParams(h.includes('=') ? h : `s=${h}`);   // old links: #floodnet_7241
  return Object.fromEntries(['s', 'mo', 'h', 'e', 'm', 'l', 'r', 'cb', 'lang'].map(k => [k, q.get(k)]));
}

function writeHash(push = false) {
  if (!state.s) return;
  const q = [`s=${state.key}`];
  if (state.model === 'b') q.push('mo=b');
  if (state.sel) q.push(`h=${state.s.blds[state.sel.i].id}${state.src === 'swipe' && state.sel.side === 'gt' ? 'gt' : ''}`);
  if (state.entry) q.push(`e=${state.entry.at.map(Math.round).join(',')}`);
  if (state.src !== 'pred') q.push(`m=${state.src}`);
  if (state.layer !== 'classes') q.push(`l=${state.layer}`);
  if (!state.lam) q.push('r=0');
  if (state.cb) q.push('cb=1');
  if (LANG !== DEFAULT_LANG) q.push(`lang=${LANG}`);
  const url = `#${q.join('&')}`;
  if (url !== location.hash) history[push ? 'pushState' : 'replaceState'](null, '', url);
}

// A link names the whole view: a key it leaves out means that key's default (model Chameleon, map 'pred', layer
// 'classes', risk-aware truck, auto entry, no house), also when it is pasted into an open viewer. Colour palette and
// language are the reader's preferences and change only when the link names them.
async function applyHash() {
  const p = readHash();
  state.model = p.mo === 'b' ? 'b' : 'a';
  if (!state.s || (p.s && p.s !== state.key)) await loadScene(p.s);
  else if (modelOf(state.s.sc) !== state.s.model) await loadScene(state.key, true);
  else if (state.model !== state.browseModel) { drawBrowse(); markPressed(); }
  const lam = p.r === '0' ? 0 : 5;
  if (lam !== state.lam) { $('tRisk').checked = Boolean(lam); state.lam = lam; solve(); }
  setLayer(p.l || 'classes');
  if (p.lang && p.lang !== LANG && ['en', 'ko'].includes(p.lang)) relabel(p.lang);
  if (p.cb != null && (p.cb === '1') !== state.cb) { state.cb = p.cb === '1'; $('tCb').checked = state.cb; await applyPalette(); }
  const m = ['pred', 'gt', 'swipe'].includes(p.m) ? p.m : 'pred';
  if (m !== state.src) setSource(m);
  const [x, y] = (p.e || '').split(',').map(Number);
  if (p.e && Number.isFinite(x) && Number.isFinite(y)) setEntry(Math.min(state.s.W - 1, Math.max(0, x)), Math.min(state.s.H - 1, Math.max(0, y)));
  else if (state.entry) { state.entry = null; state.entryMsg = null; solve(); }
  const i = p.h ? state.s.blds.findIndex(b => b.id === parseInt(p.h, 10)) : -1;
  if (i >= 0 && base()) select(i, state.src === 'swipe' ? (p.h.endsWith('gt') ? 'gt' : 'pred') : state.src);
  else if (state.sel) clearSelection();
  writeHash();
}

function clearSelection() {
  state.sel = null;
  stopAnim();
  groups.route.clearLayers();
  $('routeCard').hidden = true;
  markHouse();
  drawHouses();
  drawCounter();
}

// ---------- language (English / 한국어) ----------
function relabel(lang) {
  setLang(lang);
  $('langBtn').textContent = LANG === 'ko' ? 'English' : '한국어';
  if (!state.s) return;
  drawCaption(); drawModelSeg(); drawLegend(); drawPwetLegend(); drawBase(); drawCounter(); drawHouseList();
  if (state.sel) select(state.sel.i, state.sel.side, false);
}
$('langBtn').addEventListener('click', () => { relabel(LANG === 'ko' ? 'en' : 'ko'); writeHash(); });

(async () => {
  relabel(LANG);
  if (innerWidth <= 760) { $('panel').classList.add('collapsed'); $('sheetToggle').setAttribute('aria-expanded', 'false'); }
  await loadIndex();
  drawBrowse();
  setSource('pred');
  await applyHash();
})();

window.rr = { state, map };                                         // for debugging in the console
