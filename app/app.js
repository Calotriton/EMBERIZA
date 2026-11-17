// EMBERIZA's page: it collects the recordings and bacpipe's options, asks the worker to run bacpipe, and draws
// what bacpipe wrote. Nothing here computes a result.
import { CONFIG_FIELDS, SETTINGS_GROUPS, HELP } from './options.js';

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of children.flat()) if (child != null) node.append(child);
  return node;
};
const fmtBytes = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`);
const fmtTime = (s) => (s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${s.toFixed(s < 10 ? 1 : 0)} s`);
const store = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(`emberiza.${key}`)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(`emberiza.${key}`, JSON.stringify(value)); } catch { /* private window */ } },
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const deep = (obj, path) => path.split('.').reduce((o, k) => o?.[k], obj);
const setDeep = (obj, path, value) => { const keys = path.split('.'); const last = keys.pop(); keys.reduce((o, k) => (o[k] ??= {}), obj)[last] = value; };
const clone = (x) => JSON.parse(JSON.stringify(x));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ---------------------------------------------------------------------------------------- the worker

const worker = new Worker('worker.js', { type: 'module' });
const waiting = new Map();
let nextId = 1;
const call = (action, args = {}) => new Promise((resolve, reject) => {
  const id = nextId++;
  waiting.set(id, { resolve, reject });
  worker.postMessage({ id, do: action, ...args });
});

const state = {
  ready: false, busy: false, defaults: null, models: {}, settings: null, config: null, gpu: false,
  dataset: null, files: [], chosenModels: store.get('models', ['perch_v2']),
  summary: null, points: {}, selection: new Set(), existing: {}, runStarted: 0,
};

worker.onmessage = ({ data }) => {
  if (data.id) {
    const w = waiting.get(data.id);
    waiting.delete(data.id);
    if (data.ok) w.resolve(data.result); else w.reject(Object.assign(new Error(data.error), { detail: data.detail }));
    return;
  }
  if (data.type === 'ready') onReady(data);
  else if (data.type === 'failed') { setEngine('failed', 'Could not start'); showAlert(`EMBERIZA could not start: ${data.error}. Reload the page. It needs a current Chrome, Edge or Firefox.`); }
  else if (data.type === 'status' && (state.busy || !state.ready)) progress(data.text, data.done, data.total);
  else if (data.type === 'progress' && state.busy) progress(null, data.done, data.total, data.text);
  else if (data.type === 'file') onFile(data);
  else if (data.type === 'backend') log('info', `${state.models[data.model].title} runs on ${data.backend}.`);
  else if (data.type === 'log') log(data.level, data.text);
};
worker.onerror = (e) => { setEngine('failed', 'Stopped'); showAlert(`The engine stopped: ${e.message}. Reload the page.`); };

function setEngine(kind, text) { $('engine').className = `pill ${kind}`; $('engine-text').textContent = text; }
function showAlert(text) { $('alert').textContent = text; $('alert').hidden = !text; }

// ---------------------------------------------------------------------------------------- log and progress

let lastLog = '';
function log(level, text) {
  if (!text || text === lastLog) return;
  lastLog = text;
  const list = $('log');
  list.append(el('li', { className: level, textContent: text }));
  while (list.children.length > 1500) list.firstChild.remove();
}

let lastTitle = '';
function progress(title, done, total, detail) {
  $('progress').hidden = false;
  if (title) { lastTitle = title; $('progress-title').textContent = title; $('progress-detail').textContent = ''; }
  if (detail != null) $('progress-detail').textContent = detail;
  const bar = $('progress-bar').parentElement;
  if (total > 0) {
    bar.classList.remove('unknown');
    $('progress-bar').style.transform = `scaleX(${Math.min(1, done / total)})`;
    $('progress-count').textContent = total > 1e5 ? `${Math.round((100 * done) / total)}%  ${fmtBytes(done)} of ${fmtBytes(total)}` : `${done} of ${total}`;
  } else if (title) {
    bar.classList.add('unknown');
    $('progress-count').textContent = '';
  }
  if (state.busy) document.title = `${total > 0 ? `${Math.round((100 * done) / total)}% ` : ''}${lastTitle} · EMBERIZA`;
}

function onFile({ title, index, total, name, windows, seconds }) {
  const rate = seconds > 2 && windows ? ` · ${(windows / seconds).toFixed(1)} windows a second` : '';
  progress(`${title}: embedding recording ${index + 1} of ${total}`, index, total, `${name}${rate}`);
}

// ---------------------------------------------------------------------------------------- options (both YAML files)

function fieldValue(field, source) { return deep(source, field.key); }

function control(field, values, onChange, defaults) {
  const id = `opt-${field.key.replaceAll('.', '-')}`;
  const value = fieldValue(field, values);
  const off = Boolean(field.off);
  let input;
  if (field.type === 'bool') {
    input = el('input', { type: 'checkbox', id, checked: Boolean(value), disabled: off });
    input.onchange = () => onChange(field.key, input.checked);
  } else if (field.type === 'select') {
    input = el('select', { id, disabled: off }, field.options.map((o) => el('option', { value: String(o.value ?? o), textContent: o.label ?? o, selected: String(o.value ?? o) === String(value) })));
    input.onchange = () => { const picked = field.options.find((o) => String(o.value ?? o) === input.value); onChange(field.key, picked?.value ?? picked); };
  } else if (field.type === 'multi') {
    input = el('div', { className: 'multi', id }, field.options.map((o) => {
      const box = el('input', { type: 'checkbox', checked: (value ?? []).includes(o), disabled: off });
      box.onchange = () => onChange(field.key, field.options.filter((opt, i) => input.querySelectorAll('input')[i].checked));
      return el('label', {}, box, o);
    }));
  } else if (field.type === 'list') {
    input = el('input', { type: 'text', id, value: (value ?? []).join(' '), disabled: off });
    input.onchange = () => onChange(field.key, input.value.split(/[\s,]+/).filter(Boolean));
  } else if (field.type === 'text') {
    input = el('input', { type: 'text', id, value: value ?? '', disabled: off });
    input.onchange = () => onChange(field.key, field.parse ? field.parse(input.value) : input.value);
  } else {
    input = el('input', { type: 'number', id, value: value ?? '', disabled: off, step: field.step ?? (field.type === 'int' ? 1 : 'any') });
    if (field.min != null) input.min = field.min;
    if (field.max != null) input.max = field.max;
    input.onchange = () => {
      let v = Number(input.value);
      if (!Number.isFinite(v)) v = fieldValue(field, defaults);
      if (field.type === 'int') v = Math.round(v);
      if (field.min != null) v = Math.max(field.min, v);
      if (field.max != null) v = Math.min(field.max, v);
      input.value = v;
      onChange(field.key, v);
    };
  }
  const help = el('button', { type: 'button', className: 'help', textContent: '?', title: field.help });
  help.setAttribute('aria-label', `About ${field.label}`);
  help.onclick = () => showHelp(field.label, `${field.help}${field.off ? `\n\n${field.off}` : ''}`, field.key);
  const label = el(field.type === 'multi' ? 'span' : 'label', { className: field.type === 'multi' ? 'label' : '', textContent: field.label });
  if (field.type !== 'multi') label.htmlFor = id;
  label.append(help);
  const changed = !off && !same(value, fieldValue(field, defaults));
  return el('div', { className: `field${off ? ' off' : ''}${changed ? ' changed' : ''}` }, label, input,
    el('span', { className: 'key', textContent: field.key + (field.off ? ' · no effect in a browser' : '') }));
}

function renderOptions() {
  const cfg = $('config-form');
  cfg.replaceChildren(...CONFIG_FIELDS.map((f) => control(f, state.config, (key, v) => { setDeep(state.config, key, v); saveOptions(); }, state.defaults.config)));
  const open = store.get('open-groups', []);
  $('settings-form').replaceChildren(...SETTINGS_GROUPS.map((group) => {
    const changed = group.fields.filter((f) => !f.off && !f.heading && !same(fieldValue(f, state.settings), fieldValue(f, state.defaults.settings))).length;
    const details = el('details', { className: 'group', open: open.includes(group.title) },
      el('summary', {}, group.title, el('span', { className: 'count', textContent: changed ? `${changed} changed` : `${group.fields.filter((f) => !f.heading).length}` })),
      el('div', { className: 'form' }, group.note ? el('p', { className: 'hint', textContent: group.note }) : null,
        group.fields.map((f) => (f.heading ? el('p', { className: 'subhead', textContent: f.heading })
          : control(f, f.key === '__backend' ? { __backend: state.gpu ? 'gpu' : 'cpu' } : state.settings,
            (key, v) => { if (key === '__backend') { state.gpu = v === 'gpu'; store.set('gpu', state.gpu); } else setDeep(state.settings, key, v); saveOptions(); },
            f.key === '__backend' ? { __backend: 'cpu' } : state.defaults.settings)))));
    details.ontoggle = () => store.set('open-groups', [...document.querySelectorAll('details.group[open] > summary')].map((s) => s.firstChild.textContent));
    return details;
  }));
}

function saveOptions() {
  store.set('config', state.config);
  store.set('settings', state.settings);
  renderOptions();
  if (state.summary) redrawAll();
}

function showHelp(title, text, key) {
  $('help-title').textContent = title;
  $('help-text').textContent = key ? `${text}\n\nIn bacpipe: ${key}` : text;
  $('help-dialog').showModal();
}

function toYaml(obj, indent = '') {
  return Object.entries(obj).map(([k, v]) => {
    if (Array.isArray(v)) return `${indent}${k}: [${v.map((x) => JSON.stringify(x)).join(', ')}]`;
    if (v && typeof v === 'object') return `${indent}${k}:\n${toYaml(v, `${indent}  `)}`;
    return `${indent}${k}: ${typeof v === 'string' ? JSON.stringify(v) : v === true ? 'True' : v === false ? 'False' : v}`;
  }).join('\n');
}

function saveBlob(blob, name) {
  const a = el('a', { href: URL.createObjectURL(blob), download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30000);
}

// ---------------------------------------------------------------------------------------- recordings

const AUDIO = /\.(wav|aif|aiff|mp3|flac|ogg)$/i;
const TABLE = /\.(csv)$/i;

function setFiles(entries) {                       // entries: [{path, file}] with path relative to what was dropped
  const kept = entries.filter(({ path }) => (AUDIO.test(path) || TABLE.test(path)) && !path.split('/').some((p) => p.startsWith('.')));
  if (!kept.some(({ path }) => AUDIO.test(path))) { showAlert('No recordings were found there. EMBERIZA reads WAV, FLAC, OGG and MP3 files.'); return; }
  showAlert('');
  const tops = new Set(kept.map(({ path }) => path.split('/')[0]));
  const oneFolder = tops.size === 1 && kept.every(({ path }) => path.includes('/'));
  state.dataset = (oneFolder ? [...tops][0] : 'recordings').replace(/[^\w.-]+/g, '_');
  state.files = kept.map(({ path, file }) => ({ path: oneFolder ? path.slice(path.indexOf('/') + 1) : path, file }));
  const audio = state.files.filter(({ path }) => AUDIO.test(path));
  const folders = new Set(audio.map(({ path }) => path.split('/').slice(0, -1).join('/')));
  const annotations = state.files.find(({ path }) => path === (state.settings?.annotations_filename ?? 'annotations.csv'));
  const lossy = audio.filter(({ path }) => /\.mp3$/i.test(path)).length;
  const facts = [['Dataset', state.dataset], ['Recordings', `${audio.length} in ${folders.size} folder${folders.size === 1 ? '' : 's'}`],
    ['Size', fmtBytes(audio.reduce((s, { file }) => s + file.size, 0))],
    ['Annotations', annotations ? `${annotations.path} found` : 'none (optional)']];
  if (lossy) facts.push(['Note', `${lossy} MP3: lossy sound changes embeddings`]);
  const big = audio.filter(({ file }) => file.size > 4e8).length;
  if (big) facts.push(['Note', `${big} over 400 MB may not fit in memory`]);
  $('recordings-facts').replaceChildren(...facts.flatMap(([k, v]) => [el('dt', { textContent: k }), el('dd', { textContent: v })]));
  $('recordings-facts').hidden = false;
  $('drop').querySelector('.drop-title').textContent = 'Drop another folder to replace these';
  updateRun();
}

async function readDropped(items) {
  const out = [];
  const walk = async (entry, prefix) => {
    if (entry.isFile) out.push({ path: prefix + entry.name, file: await new Promise((res, rej) => entry.file(res, rej)) });
    else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
      }
    }
  };
  await Promise.all([...items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean).map((e) => walk(e, '')));
  return out;
}

const fromInput = (input) => [...input.files].map((file) => ({ path: file.webkitRelativePath || file.name, file }));

function updateRun() {
  const hasAudio = state.files.some(({ path }) => AUDIO.test(path));
  const ok = state.ready && !state.busy && hasAudio && state.chosenModels.length > 0;
  $('run').disabled = !ok;
  const bytes = state.chosenModels.reduce((s, m) => s + (state.models[m]?.bytes ?? 0), 0);
  $('run-hint').textContent = !state.ready ? 'The engine is starting.' : state.busy ? 'Running. The tab must stay open.'
    : !hasAudio ? 'Choose recordings and at least one model.' : !state.chosenModels.length ? 'Choose at least one model.'
    : `Ready. First use downloads ${fmtBytes(bytes)} of models.`;
}

function renderModels() {
  const notes = { perch_v2: '5 s windows · 32 kHz · 1,536 numbers · 14,795 classes', birdnet_v3: '3 s windows · 32 kHz · 1,280 numbers · 11,560 classes' };
  $('models').replaceChildren(...Object.entries(state.models).map(([name, m]) => {
    const box = el('input', { type: 'checkbox', checked: state.chosenModels.includes(name), value: name });
    box.onchange = () => {
      state.chosenModels = Object.keys(state.models).filter((k) => $('models').querySelector(`input[value="${k}"]`).checked);
      store.set('models', state.chosenModels);
      updateRun();
    };
    return el('label', { className: 'choice' }, box, el('strong', { textContent: m.title }), el('span', { textContent: `${notes[name]} · ${fmtBytes(m.bytes)}` }));
  }));
}

// ---------------------------------------------------------------------------------------- start

function onReady(data) {
  state.ready = true;
  state.defaults = data.defaults;
  state.models = data.models;
  state.existing = data.existing;
  const savedCfg = store.get('config', {}), savedSet = store.get('settings', {});
  state.config = { ...clone(data.defaults.config), ...savedCfg };
  state.settings = { ...clone(data.defaults.settings), ...savedSet };
  state.gpu = store.get('gpu', false) && data.gpu;
  state.chosenModels = state.chosenModels.filter((m) => data.models[m]);
  setEngine('ready', 'Ready');
  $('progress').hidden = true;
  $('about-bacpipe').textContent = `1.3.5 (commit ${data.versions.bacpipe.slice(0, 7)})`;
  $('about-versions').textContent = `Pyodide ${data.versions.pyodide} · ONNX Runtime Web ${data.versions.onnxruntimeWeb} · results kept in ${data.storage === 'browser' ? 'this browser' : 'memory only'} · ${data.isolated ? 'several processor threads' : 'one processor thread'}`;
  renderModels();
  renderOptions();
  renderSaved();
  updateRun();
  const last = store.get('last');
  if (last && state.existing[last] && !location.hash.includes('fresh')) openSaved(last);
}

function renderSaved() {
  const names = Object.keys(state.existing);
  $('saved').hidden = names.length === 0;
  $('saved-list').replaceChildren(...names.map((name) => {
    const open = el('button', { type: 'button', className: 'button quiet-button', textContent: 'Open' });
    open.onclick = () => openSaved(name);
    return el('li', {}, el('span', { className: 'name', textContent: name }), el('span', { className: 'models', textContent: state.existing[name].map((m) => state.models[m]?.title ?? m).join(', ') }), open);
  }));
}

async function busy(title, fn) {
  state.busy = true;
  setEngine('busy', 'Working');
  progress(title);
  updateRun();
  try { return await fn(); } finally {
    state.busy = false;
    setEngine('ready', 'Ready');
    $('progress').hidden = true;
    document.title = 'EMBERIZA';
    updateRun();
  }
}

async function openSaved(dataset) {
  const saved = store.get(`run.${dataset}`, {});
  try {
    const out = await busy(`Opening ${dataset}`, () => call('open', { dataset, config: { ...state.config, ...(saved.config ?? {}), models: state.existing[dataset], dataset },
      settings: { ...state.settings, ...(saved.settings ?? {}) } }));
    state.dataset = dataset;
    showResults(out.summary);
    store.set('last', dataset);
  } catch (e) { showAlert(`${dataset} could not be opened: ${e.message}`); }
}

async function runPipeline() {
  showAlert('');
  $('log').replaceChildren();
  $('stop').hidden = false;
  $('run').hidden = true;
  state.runStarted = performance.now();
  const config = { ...state.config, models: state.chosenModels, dataset: state.dataset };
  try {
    const report = await busy('Preparing', async () => {
      await call('recordings', { dataset: state.dataset, files: state.files });
      return call('run', { config, settings: state.settings, gpu: state.gpu });
    });
    store.set(`run.${state.dataset}`, { config: state.config, settings: state.settings });
    store.set('last', state.dataset);
    state.existing = await call('existing');
    renderSaved();
    showResults(report.summary);
    log('info', `Done in ${fmtTime(report.seconds)}: ${report.windows} windows embedded${report.backend ? ` on ${report.backend}` : ''}.`);
    if (document.hidden && 'Notification' in window && Notification.permission === 'granted') new Notification('EMBERIZA has finished', { body: state.dataset });
  } catch (e) {
    if (e.message === 'stopped') { showAlert('Stopped. The recordings embedded so far are kept: run again to carry on from there.'); }
    else { showAlert(`The run stopped: ${e.message}`); log('error', e.detail ?? e.message); }
  } finally {
    $('stop').hidden = true;
    $('run').hidden = false;
  }
}

// ---------------------------------------------------------------------------------------- results

const PALETTE = ['#ff8a4c', '#5cc8e8', '#b48cf2', '#5fd39a', '#f3c04f', '#f2749c', '#7c9cf5', '#c9d65c', '#4fd1c5', '#e8a0d8', '#e5534b', '#d98f5c',
  '#8fd0ff', '#9be08a', '#f5a3a3', '#c2a5ff', '#ffd27a', '#6ee0d0', '#e07bd0', '#9a7bd6'];   // no greys: grey is for "noise" and "below_thresh"
const TWILIGHT = [[0, '#e2d9e2'], [0.125, '#9ebbc9'], [0.25, '#6785be'], [0.375, '#5e43a5'], [0.5, '#2f1436'], [0.625, '#8e2c50'], [0.75, '#ba6657'], [0.875, '#ceac94'], [1, '#e2d9e2']];
const scale = (name) => (String(name).toLowerCase() === 'twilight' ? TWILIGHT : name);
const nice = (key) => key.replaceAll('_', ' ');

function plotLayout(extra = {}) {
  const axis = { gridcolor: css('--line'), zerolinecolor: css('--line'), linecolor: css('--line'), tickfont: { color: css('--muted'), size: 11 }, title: { font: { color: css('--muted'), size: 12 } } };
  return { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: css('--font'), color: css('--text'), size: 12 },
    margin: { l: 44, r: 12, t: 12, b: 40 }, xaxis: { ...axis }, yaxis: { ...axis }, legend: { font: { size: 11 }, bgcolor: 'rgba(0,0,0,0)', itemsizing: 'constant' },
    hoverlabel: { bgcolor: css('--raised'), bordercolor: css('--line-strong'), font: { color: css('--text'), family: css('--mono'), size: 12 } }, ...extra };
}
const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['autoScale2d', 'toggleSpikelines'], toImageButtonOptions: { format: 'png', scale: 2, filename: 'emberiza' } };

async function getPoints(model) {
  state.points[model] ??= await call('points', { model });
  return state.points[model];
}

function labelOrder(values) {
  const unique = [...new Set(values)];
  const numeric = unique.every((v) => v !== '' && !Number.isNaN(Number(v)));
  return unique.sort((a, b) => (numeric ? Number(a) - Number(b) : a.localeCompare(b, undefined, { numeric: true })));
}

const mapLabels = (pts, labelKey) => pts.labels[labelKey] ?? pts.labels.audio_file_name;

// The traces of one map: a trace per label, or one trace on a colour scale when there are too many labels.
// `key` lists the labels in colour order. Maps drawn side by side share one, so a colour means the same label in each.
function mapTraces(pts, labelKey, onlyAnnotated, three, key) {
  const labels = mapLabels(pts, labelKey);
  const keep = pts.x.map((_, i) => !onlyAnnotated || pts.annotated[i]);
  const present = labelOrder(labels.filter((_, i) => keep[i]));
  const order = key ?? present;
  const type = three ? 'scatter3d' : 'scattergl';
  const hover = (i) => `${pts.file[i].replaceAll('\\', '/')}<br>${pts.start[i].toFixed(1)}–${pts.end[i].toFixed(1)} s<br>${nice(labelKey)}: ${labels[i]}`;
  const size = three ? 3.5 : Math.max(4, Math.min(9, 60 / Math.sqrt(pts.n) + 3));
  const base = (idx) => ({ type, mode: 'markers', x: idx.map((i) => pts.x[i]), y: idx.map((i) => pts.y[i]), ...(three ? { z: idx.map((i) => pts.z[i]) } : {}),
    customdata: idx, text: idx.map(hover), hovertemplate: '%{text}<extra></extra>' });
  const all = pts.x.map((_, i) => i).filter((i) => keep[i]);
  if (order.length > (state.settings.max_nr_categories ?? 55)) {
    const rank = new Map(order.map((v, i) => [v, i]));
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * (order.length - 1)));
    return [{ ...base(all), marker: { size, color: all.map((i) => rank.get(labels[i])), colorscale: scale(state.settings.color_continuous ?? 'Twilight'), opacity: 0.9,
      colorbar: { thickness: 10, tickvals: ticks, ticktext: ticks.map((t) => order[t]), tickfont: { size: 10, color: css('--muted') }, outlinewidth: 0 } }, showlegend: false }];
  }
  return present.map((value) => {
    const idx = all.filter((i) => labels[i] === value);
    const n = order.indexOf(value);
    const quiet = value === 'noise' || value === 'below_thresh';
    return { ...base(idx), name: String(value).replaceAll('\\', '/').slice(-38), marker: { size, color: quiet ? css('--faint') : PALETTE[n % PALETTE.length], opacity: quiet ? 0.45 : 0.9,
      symbol: three ? undefined : ['circle', 'diamond', 'square', 'cross', 'triangle-up'][Math.floor(n / PALETTE.length) % 5] } };
  });
}

function fillSelect(select, options, wanted, text = (o) => o) {
  select.replaceChildren(...options.map((o) => el('option', { value: o, textContent: text(o) })));
  select.value = options.includes(wanted) ? wanted : options[0] ?? '';
}

async function drawExplore() {
  const model = $('ex-model').value;
  if (!model) return;
  let pts;
  try { pts = await getPoints(model); } catch (e) { Plotly.purge('ex-plot'); $('ex-plot').textContent = `No map for this model: ${e.message}`; return; }
  const keys = Object.keys(pts.labels);
  if (![...$('ex-label').options].length || !keys.includes($('ex-label').value)) fillSelect($('ex-label'), keys, store.get('label', 'default_classifier'), nice);
  const hasNoise = pts.annotated.some((a) => !a);
  $('ex-annotated-wrap').hidden = !hasNoise;
  const three = Array.isArray(pts.z);
  const traces = mapTraces(pts, $('ex-label').value, hasNoise && $('ex-annotated').checked, three);
  const reduction = state.summary.dim_reduction_model;
  const layout = plotLayout(three ? { scene: { xaxis: { title: `${reduction} 1` }, yaxis: { title: `${reduction} 2` }, zaxis: { title: `${reduction} 3` } }, margin: { l: 0, r: 0, t: 0, b: 0 } }
    : { dragmode: 'pan', xaxis: { ...plotLayout().xaxis, title: { text: `${reduction} 1` } }, yaxis: { ...plotLayout().yaxis, title: { text: `${reduction} 2` } } });
  layout.height = state.settings.embed_fig_height ? Math.min(900, Math.max(360, state.settings.embed_fig_height - 100)) : undefined;
  await Plotly.react('ex-plot', traces, layout, { ...PLOT_CONFIG, scrollZoom: true });
  const plot = $('ex-plot');
  plot.removeAllListeners?.('plotly_click');
  plot.removeAllListeners?.('plotly_selected');
  plot.on('plotly_click', (ev) => showSegment(model, ev.points[0].customdata));
  plot.on('plotly_selected', (ev) => {
    state.selection = new Set((ev?.points ?? []).map((p) => p.customdata));
    $('ex-save-selection').disabled = state.selection.size === 0;
    $('ex-save-selection').textContent = state.selection.size ? `Save selection (${state.selection.size})` : 'Save selection';
  });
}

function wavBlob(samples, sr) {
  const n = samples.length, view = new DataView(new ArrayBuffer(44 + 2 * n));
  const text = (at, s) => [...s].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + 2 * n, true); text(8, 'WAVEfmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sr, true); view.setUint32(28, 2 * sr, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, 2 * n, true);
  let peak = 0;
  for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(samples[i]));
  const gain = peak > 1 ? 1 / peak : 1;
  for (let i = 0; i < n; i++) view.setInt16(44 + 2 * i, Math.round(Math.max(-1, Math.min(1, samples[i] * gain)) * 32767), true);
  return new Blob([view], { type: 'audio/wav' });
}

async function showSegment(model, index) {
  $('seg-empty').hidden = true;
  $('seg').hidden = false;
  $('seg-title').textContent = 'Loading';
  let seg;
  try { seg = await call('segment', { model, index }); } catch (e) { $('seg-title').textContent = `Could not load this window: ${e.message}`; return; }
  if (seg.missing) {
    $('seg-title').textContent = `Choose the recordings again (step 1) to hear and see ${String(seg.missing).replaceAll('\\', '/')}.`;
    Plotly.purge('seg-spec'); $('seg-audio').removeAttribute('src'); $('seg-facts').replaceChildren();
    return;
  }
  $('seg-title').textContent = `${String(seg.file).replaceAll('\\', '/')} · ${seg.start.toFixed(1)}–${seg.end.toFixed(1)} s`;
  const step = Math.max(1, Math.ceil(seg.rows / 512));                 // every row is drawn up to 512, then every other one
  const z = [];
  for (let r = 0; r < seg.rows; r += step) z.push(seg.spec.subarray(r * seg.cols, (r + 1) * seg.cols));
  const layout = plotLayout({ margin: { l: 46, r: 8, t: 8, b: 34 }, height: Math.min(520, Math.max(220, (state.settings.spectrogram_plot_height ?? 550) - 250)) });
  layout.xaxis.title = { text: 'time (s)' }; layout.yaxis.title = { text: 'frequency (kHz)' };
  Plotly.react('seg-spec', [{ type: 'heatmap', z, x0: 0, dx: (seg.end - seg.start) / (seg.cols - 1), y0: 0, dy: (seg.f_max / 1000) / (z.length - 1),
    colorscale: scale(state.settings.spec_colorscale ?? 'Viridis'), showscale: false, hovertemplate: '%{x:.2f} s · %{y:.1f} kHz · %{z:.0f} dB<extra></extra>' }], layout, { ...PLOT_CONFIG, displayModeBar: false });
  const audio = $('seg-audio');
  if (audio.src) URL.revokeObjectURL(audio.src);
  audio.src = URL.createObjectURL(wavBlob(seg.audio, seg.sr));
  if ($('ex-autoplay').checked) audio.play().catch(() => {});
  const facts = [...seg.predictions.map(([name, score], i) => [i ? '' : 'Predictions', `${name} ${score.toFixed(3)}`]),
    ...(seg.predictions.length ? [] : [['Predictions', `none above ${state.settings.classifier_threshold}`]]),
    ...Object.entries(seg.labels).filter(([k]) => !['audio_file_name', 'parent_directory'].includes(k)).map(([k, v]) => [nice(k), v])];
  $('seg-facts').replaceChildren(...facts.flatMap(([k, v]) => [el('dt', { textContent: k }), el('dd', { textContent: v })]));
}

async function drawCompare() {
  const models = state.summary.models.map((m) => m.name);
  const all = await Promise.all(models.map((m) => getPoints(m).catch(() => null)));
  const usable = models.filter((_, i) => all[i]);
  const common = usable.length ? Object.keys(all[models.indexOf(usable[0])].labels).filter((k) => usable.every((m) => k in state.points[m].labels)) : [];
  if (!common.includes($('cmp-label').value)) fillSelect($('cmp-label'), common, store.get('label', 'default_classifier'), nice);
  $('cmp-annotated-wrap').hidden = !usable.some((m) => state.points[m].annotated.some((a) => !a));
  const grid = $('cmp-grid');
  if (grid.children.length !== usable.length) grid.replaceChildren(...usable.map((m) => el('div', { className: 'card' }, el('h3', { textContent: state.models[m]?.title ?? m }), el('div', { className: 'plot', id: `cmp-${m}` }))));
  const onlyAnnotated = $('cmp-annotated').checked;
  const key = labelOrder(usable.flatMap((m) => mapLabels(state.points[m], $('cmp-label').value).filter((_, i) => !onlyAnnotated || state.points[m].annotated[i])));
  for (const m of usable) {
    const pts = state.points[m];
    const traces = mapTraces(pts, $('cmp-label').value, onlyAnnotated, Array.isArray(pts.z), key);
    await Plotly.react(`cmp-${m}`, traces, plotLayout({ dragmode: 'lasso', margin: { l: 36, r: 8, t: 8, b: 30 },
      legend: { ...plotLayout().legend, orientation: 'h', yanchor: 'top', y: -0.12, x: 0 } }), { ...PLOT_CONFIG, scrollZoom: true });
    const plot = $(`cmp-${m}`);
    plot.removeAllListeners?.('plotly_selected');
    plot.on('plotly_selected', (ev) => {
      const chosen = new Set((ev?.points ?? []).map((p) => `${p.text.split('<br>')[0]}|${p.text.split('<br>')[1]}`));
      for (const other of usable.filter((o) => o !== m)) {
        const data = $(`cmp-${other}`).data;
        Plotly.restyle(`cmp-${other}`, { selectedpoints: data.map((t) => (chosen.size ? t.text.map((txt, i) => (overlaps(txt, chosen) ? i : -1)).filter((i) => i >= 0) : null)) });
      }
    });
  }
}
// Windows of two models differ in length (5 s and 3 s), so "the same sound" means the same recording and overlapping times
function overlaps(text, chosen) {
  const [file, span] = text.split('<br>');
  const [a, b] = span.replace(' s', '').split('–').map(Number);
  for (const key of chosen) {
    const [f, s] = key.split('|');
    if (f !== file) continue;
    const [c, d] = s.replace(' s', '').split('–').map(Number);
    if (a < d && c < b) return true;
  }
  return false;
}

const GROUPERS = {
  file: (t, f) => f.replaceAll('\\', '/'),
  hour: (t) => `${t.slice(11, 13)} h`,
  day: (t) => t.slice(0, 10),
  month: (t) => t.slice(0, 7),
  week: (t) => { const d = new Date(`${t.slice(0, 10)}T00:00:00Z`); const day = (d.getUTCDay() + 6) % 7; d.setUTCDate(d.getUTCDate() - day + 3); const first = new Date(Date.UTC(d.getUTCFullYear(), 0, 4));
    return `${d.getUTCFullYear()}-W${String(1 + Math.round(((d - first) / 864e5 - 3 + ((first.getUTCDay() + 6) % 7)) / 7)).padStart(2, '0')}`; },
};
const predictions = {};
async function drawPredictions() {
  const model = $('pr-model').value;
  if (!model) return;
  predictions[model] ??= await call('predictions', { model });
  const p = predictions[model];
  if (!p || !p.bin.length) { Plotly.purge('pr-plot'); $('pr-note').textContent = 'No predictions were saved for this model. Switch on "Run the built-in classifier" and run again.'; return; }
  if (!$('pr-threshold').value || Number($('pr-threshold').value) < p.threshold) $('pr-threshold').value = p.threshold;
  $('pr-threshold').min = p.threshold;
  const threshold = Number($('pr-threshold').value), group = GROUPERS[$('pr-by').value];
  const groups = labelOrder(p.time.map((t, i) => group(t, p.file[i])));
  const col = new Map(groups.map((g, i) => [g, i]));
  const counts = new Map();
  for (let i = 0; i < p.bin.length; i++) {
    if (p.score[i] < threshold) continue;
    const row = counts.get(p.class[i]) ?? counts.set(p.class[i], new Float64Array(groups.length)).get(p.class[i]);
    row[col.get(group(p.time[p.bin[i]], p.file[p.bin[i]]))] += 1;
  }
  const top = [...counts.entries()].map(([c, row]) => [c, row, row.reduce((a, b) => a + b, 0)]).sort((a, b) => b[2] - a[2]).slice(0, Number($('pr-top').value) || 25).reverse();
  const layout = plotLayout({ margin: { l: 220, r: 12, t: 12, b: 90 }, height: Math.max(320, Math.min(1400, 60 + 22 * top.length + 90)) });
  layout.xaxis.tickangle = -40;
  layout.xaxis.type = 'category';
  layout.yaxis.type = 'category';
  Plotly.react('pr-plot', [{ type: 'heatmap', z: top.map(([, row]) => [...row]), x: groups, y: top.map(([c, , total]) => `${p.species[c]} (${total})`),
    colorscale: [[0, 'rgba(0,0,0,0)'], [0.001, '#3a1c10'], [0.5, '#c2480c'], [1, '#ffd0a8']], xgap: 2, ygap: 2,
    colorbar: { thickness: 10, title: { text: 'windows', font: { size: 11 } }, outlinewidth: 0 }, hovertemplate: '%{y}<br>%{x}: %{z} windows<extra></extra>' }], layout, PLOT_CONFIG);
  $('pr-note').textContent = `Windows in which the model's own classifier gave a confidence score of ${threshold} or more, out of ${p.bins} windows. bacpipe saved the scores above ${p.threshold}. A confidence score is not a probability, and a detection is not proof of presence.`;
}

function drawEvaluation() {
  const models = state.summary.models;
  const withClust = models.filter((m) => m.clustering);
  if (!withClust.length) { Plotly.purge('ev-clust-plot'); $('ev-clust-plot').textContent = ''; $('ev-clust-note').textContent = 'No clustering yet. Tick "clustering" under Pipeline, evaluation tasks, and run.'; }
  else {
    // bacpipe's keys are "<grouping>-<label>": k-means against each label, and each annotated label against the others.
    // Shown here: k-means against every label. Its score against an annotated label is stored as "<label>-kmeans".
    const metric = $('ev-metric').value, noise = $('ev-annotated').checked ? '_no_noise' : '';
    const scoreOf = (scores, label) => scores[`kmeans${noise}-${label}`] ?? scores[`${label}${noise}-kmeans`];
    const labels = [...new Set(withClust.flatMap((m) => Object.keys(m.clustering[metric] ?? {}).flatMap((k) => {
      const [group, label] = [k.slice(0, k.indexOf('-')), k.slice(k.indexOf('-') + 1)];
      return group.startsWith('kmeans') ? [label] : label === 'kmeans' ? [group.replace(/_no_noise$/, '')] : [];
    })))].filter((l) => !['start', 'end', 'kmeans', 'kmeans_no_noise'].includes(l));
    const annotated = new Set(withClust.flatMap((m) => Object.keys(m.clustering[metric] ?? {}).filter((k) => k.endsWith('-kmeans')).map((k) => k.split('-')[0].replace(/_no_noise$/, ''))));
    labels.sort((a, b) => Number(annotated.has(b)) - Number(annotated.has(a)));
    $('ev-annotated-wrap').hidden = annotated.size === 0;
    Plotly.react('ev-clust-plot', withClust.map((m, i) => ({ type: 'bar', name: state.models[m.name]?.title ?? m.name, x: labels.map((l) => (annotated.has(l) ? `${nice(l)} (annotated)` : nice(l))),
      y: labels.map((l) => scoreOf(m.clustering[metric], l) ?? null), marker: { color: PALETTE[i] }, hovertemplate: `k-means against %{x}: ${metric} %{y:.3f}<extra>%{fullData.name}</extra>` })),
    plotLayout({ barmode: 'group', margin: { l: 50, r: 8, t: 8, b: 110 }, yaxis: { ...plotLayout().yaxis, title: { text: metric === 'AMI' ? 'adjusted mutual information' : 'adjusted Rand index' }, range: [-0.1, 1.02] } }), PLOT_CONFIG);
    $('ev-clust-note').textContent = `How well the k-means clusters of the embeddings match each label: 1 is the same grouping, 0 is what chance gives.${annotated.size ? ' An annotated label is the strongest test. Labels from file names and folders show what else the embeddings are organised by.' : ''} bacpipe does not seed k-means, so the scores change a little between runs.`;
  }
  const kind = $('ev-probe').value;
  const withProbe = models.filter((m) => m.probing?.[kind]);
  if (!withProbe.length) { Plotly.purge('ev-probe-plot'); $('ev-probe-plot').textContent = ''; $('ev-probe-note').textContent = 'No probing yet. It needs an annotations.csv file in the recordings folder, and "probing" ticked under Pipeline.'; return; }
  const classes = [...new Set(withProbe.flatMap((m) => Object.keys(m.probing[kind].per_class_accuracy ?? {})))];
  Plotly.react('ev-probe-plot', withProbe.map((m, i) => ({ type: 'bar', name: state.models[m.name]?.title ?? m.name, x: classes,
    y: classes.map((c) => m.probing[kind].per_class_accuracy?.[c]), marker: { color: PALETTE[i] }, customdata: classes.map((c) => m.probing[kind].items_per_class?.[c] ?? 0),
    hovertemplate: '%{x}: %{y:.2f} on %{customdata} test windows<extra>%{fullData.name}</extra>' })),
  plotLayout({ barmode: 'group', margin: { l: 50, r: 8, t: 8, b: 90 }, yaxis: { ...plotLayout().yaxis, title: { text: 'accuracy per class' }, range: [0, 1.02] } }), PLOT_CONFIG);
  $('ev-probe-note').replaceChildren(...withProbe.map((m) => {
    const o = m.probing[kind].overall ?? {}, n = Object.values(m.probing[kind].items_per_class ?? {}).reduce((a, b) => a + b, 0);
    return el('p', { textContent: `${state.models[m.name]?.title ?? m.name}: ${Object.entries(o).map(([k, v]) => `${nice(k)} ${Number(v).toFixed(3)}`).join(' · ')} on ${n} test windows.` });
  }), el('p', { textContent: 'With few test windows these numbers are noisy. Windows from one recording are in both the training and the test part, which flatters the scores.' }));
}

async function drawFiles() {
  const files = await call('listing', { dataset: state.dataset });
  const total = files.reduce((s, f) => s + f.bytes, 0);
  $('files-note').textContent = `${files.length} files, ${fmtBytes(total)}, in bacpipe's layout: bacpipe_results/${state.dataset}/. bacpipe can open this folder with already_computed: True.`;
  $('files-table').tBodies[0].replaceChildren(...files.map((f) => el('tr', {}, el('td', { textContent: f.path }), el('td', { className: 'num', textContent: fmtBytes(f.bytes) }))));
}

const drawers = { explore: drawExplore, compare: drawCompare, predictions: drawPredictions, evaluation: drawEvaluation, files: drawFiles, log: () => {}, about: () => {} };
let currentTab = 'explore';
function showTab(name) {
  currentTab = name;
  for (const tab of document.querySelectorAll('.tabs [role="tab"]')) tab.setAttribute('aria-selected', String(tab.dataset.tab === name));
  for (const panel of document.querySelectorAll('.panel')) panel.hidden = panel.id !== `panel-${name}`;
  $('welcome').hidden = Boolean(state.summary) || name === 'about';
  if (state.summary || name === 'about') Promise.resolve(drawers[name]()).catch((e) => showAlert(`Could not draw this: ${e.message}`));
}
function redrawAll() {
  const spec = $('seg-spec');
  if (spec?.data) {                  // the selected window keeps its spectrogram: only the theme's colours change
    const old = spec.layout, layout = plotLayout({ margin: old.margin, height: old.height });
    layout.xaxis.title = { text: old.xaxis.title?.text }; layout.yaxis.title = { text: old.yaxis.title?.text };
    Plotly.react(spec, spec.data, layout, { ...PLOT_CONFIG, displayModeBar: false });
  }
  for (const id of ['ex-plot', 'pr-plot', 'ev-clust-plot', 'ev-probe-plot']) if ($(id)?.data) Plotly.purge(id);
  $('cmp-grid').replaceChildren();
  showTab(currentTab);
}

function showResults(summary) {
  state.summary = summary;
  state.points = {};
  for (const k of Object.keys(predictions)) delete predictions[k];
  $('results').hidden = false;
  $('welcome').hidden = true;
  const windows = summary.models.reduce((s, m) => s + m.embeddings, 0);
  const tiles = [[summary.dataset, 'dataset'], [String(Math.max(...summary.models.map((m) => m.files))), 'recordings'],
    [fmtTime(Math.max(...summary.models.map((m) => m.seconds))), 'of sound'], [windows.toLocaleString('en'), 'embeddings'],
    ...summary.models.map((m) => [`${m.embeddings.toLocaleString('en')} × ${m.embedding_size}`, `${state.models[m.name]?.title ?? m.name}, ${m.segment_s} s windows`])];
  $('stats').replaceChildren(...tiles.map(([v, l]) => el('div', { className: 'stat' }, el('div', { className: 'value', textContent: v }), el('div', { className: 'label', textContent: l }))));
  $('failed').hidden = !summary.failed.length;
  if (summary.failed.length) $('failed').replaceChildren(el('strong', { textContent: `${summary.failed.length} recording${summary.failed.length === 1 ? ' was' : 's were'} skipped: ` }),
    summary.failed.slice(0, 20).map((f) => `${f.file} (${f.reason})`).join(' · '));
  const names = summary.models.map((m) => m.name);
  fillSelect($('ex-model'), names, $('ex-model').value, (m) => state.models[m]?.title ?? m);
  fillSelect($('pr-model'), names.filter((n) => summary.models.find((m) => m.name === n).has_predictions), $('pr-model').value, (m) => state.models[m]?.title ?? m);
  $('ex-label').replaceChildren();
  $('cmp-label').replaceChildren();
  $('tab-compare').hidden = names.length < 2;
  showTab(names.length < 2 && currentTab === 'compare' ? 'explore' : (currentTab === 'about' ? 'explore' : currentTab));
}

// ---------------------------------------------------------------------------------------- the welcome picture

function constellation() {
  const canvas = $('constellation'), ctx = canvas.getContext('2d');
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let points = [], frame = 0;
  const seed = (() => { let s = 42; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); })();
  const make = () => {
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * devicePixelRatio; canvas.height = height * devicePixelRatio;
    const centres = Array.from({ length: 7 }, () => [0.45 + seed() * 0.55, seed()]);
    points = Array.from({ length: 420 }, () => { const [cx, cy] = centres[Math.floor(seed() * centres.length)]; const r = Math.sqrt(-2 * Math.log(seed() + 1e-9)) * 0.055, a = seed() * 6.283;
      return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) * 1.4, phase: seed() * 6.283, c: Math.floor(seed() * 3) }; });
  };
  const draw = () => {
    const w = canvas.width, h = canvas.height, colours = [css('--accent'), css('--cool'), css('--faint')];
    ctx.clearRect(0, 0, w, h);
    for (const p of points) {
      ctx.globalAlpha = 0.25 + 0.3 * (0.5 + 0.5 * Math.sin(p.phase + frame * 0.012)) * (p.x > 0.5 ? 1 : 0.35);
      ctx.fillStyle = colours[p.c];
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, (p.c === 0 ? 2.1 : 1.5) * devicePixelRatio, 0, 6.283);
      ctx.fill();
    }
    frame += 1;
    if (!still && !$('welcome').hidden) requestAnimationFrame(draw);
  };
  make(); draw();
  addEventListener('resize', () => { make(); if (still) draw(); });
  return () => { if (!still) requestAnimationFrame(draw); else draw(); };
}

// ---------------------------------------------------------------------------------------- wiring

const ask = (text, yes, no = 'Cancel') => new Promise((resolve) => {
  $('ask-text').textContent = text; $('ask-yes').textContent = yes; $('ask-no').textContent = no;
  $('ask').onclose = () => resolve($('ask').returnValue === 'yes');
  $('ask').showModal();
});

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  store.set('theme', theme);
  if (state.summary) redrawAll();
}

document.documentElement.dataset.theme = store.get('theme', 'dark');
const restart = constellation();
$('theme').onclick = () => { applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'); restart(); };
$('pick-folder').onclick = () => $('folder-input').click();
$('pick-files').onclick = () => $('files-input').click();
$('folder-input').onchange = (e) => setFiles(fromInput(e.target));
$('files-input').onchange = (e) => setFiles(fromInput(e.target));
const drop = $('drop');
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = async (e) => { e.preventDefault(); drop.classList.remove('over'); setFiles(await readDropped(e.dataTransfer.items)); };
drop.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('folder-input').click(); } };
$('run').onclick = () => { if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission().catch(() => {}); runPipeline(); };
$('stop').onclick = () => { worker.postMessage({ do: 'stop' }); progress('Stopping after this batch'); };
$('reset-settings').onclick = async () => {
  if (!(await ask("Put every option back to bacpipe's defaults?", 'Reset', 'Keep mine'))) return;
  state.config = clone(state.defaults.config); state.settings = clone(state.defaults.settings); state.gpu = false; store.set('gpu', false); saveOptions();
};
$('export-settings').onclick = () => {
  const config = { ...state.config, models: state.chosenModels, audio_dir: state.dataset ?? '' };
  saveBlob(new Blob([`# EMBERIZA: bacpipe's config.yaml\n${toYaml(config)}\n\n# bacpipe's settings.yaml\n${toYaml(state.settings)}\n`], { type: 'text/yaml' }), 'emberiza_bacpipe_options.yaml');
};
for (const tab of document.querySelectorAll('[data-tab]')) tab.addEventListener('click', (e) => { e.preventDefault(); showTab(tab.dataset.tab); if (tab.dataset.tab === 'about') $('panel-about').scrollIntoView({ block: 'start' }); });
for (const button of document.querySelectorAll('[data-help]')) button.onclick = () => showHelp(button.dataset.help === 'clustering' ? 'Clustering scores' : 'Probing', HELP[button.dataset.help]);
$('ex-model').onchange = () => { $('ex-label').replaceChildren(); drawExplore(); };
$('ex-label').onchange = () => { store.set('label', $('ex-label').value); drawExplore(); };
$('ex-annotated').onchange = drawExplore;
$('cmp-label').onchange = drawCompare;
$('cmp-annotated').onchange = drawCompare;
for (const id of ['pr-model', 'pr-by', 'pr-threshold', 'pr-top']) $(id).onchange = drawPredictions;
$('ev-probe').onchange = drawEvaluation;
$('ev-metric').onchange = drawEvaluation;
$('ev-annotated').onchange = drawEvaluation;
$('ex-save-selection').onclick = async () => {
  const pts = await getPoints($('ex-model').value), label = $('ex-label').value;
  const rows = [...state.selection].sort((a, b) => a - b).map((i) => [pts.file[i].replaceAll('\\', '/'), pts.start[i], pts.end[i], pts.labels[label][i], pts.x[i], pts.y[i]].join(','));
  saveBlob(new Blob([`audiofilename,start,end,label:${label},x,y\n${rows.join('\n')}\n`], { type: 'text/csv' }), `selection_${$('ex-model').value}_${state.dataset}.csv`);
};
$('download').onclick = async () => {
  try { const { blob } = await busy('Packing the results', () => call('download', { dataset: state.dataset })); saveBlob(blob, `bacpipe_results_${state.dataset}.zip`); } catch (e) { showAlert(`The ZIP could not be made: ${e.message}`); }
};
$('delete-results').onclick = async () => {
  if (!(await ask(`Delete the results of "${state.dataset}" stored in this browser? Files you downloaded are kept.`, 'Delete', 'Keep'))) return;
  state.existing = await call('remove', { dataset: state.dataset });
  state.summary = null; store.set('last', null);
  $('results').hidden = true; $('welcome').hidden = false; renderSaved(); restart();
};
$('open-results').onclick = () => $('results-input').click();
$('results-input').onchange = async (e) => {
  const files = fromInput(e.target).filter(({ path }) => /\/(embeddings|dim_reduced_embeddings|evaluations)\//.test(`/${path}`));
  if (!files.length) { showAlert('That folder has no bacpipe results. Choose the folder of one dataset inside bacpipe_results: it holds "embeddings" and "evaluations".'); return; }
  const marker = files[0].path.split('/').findIndex((p) => ['embeddings', 'dim_reduced_embeddings', 'evaluations'].includes(p));
  const dataset = files[0].path.split('/')[marker - 1] ?? 'results';
  try {
    state.existing = await busy(`Copying ${dataset}`, () => call('importResults', { dataset, files: files.map(({ path, file }) => ({ path: path.split('/').slice(marker).join('/'), file })) }));
    renderSaved();
    await openSaved(dataset);
  } catch (err) { showAlert(`The results could not be opened: ${err.message}`); }
};
addEventListener('beforeunload', (e) => { if (state.busy) e.preventDefault(); });
updateRun();
