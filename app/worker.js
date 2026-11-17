// EMBERIZA's worker: Python (Pyodide) runs bacpipe, ONNX Runtime Web runs the two networks.
// The page sends one request at a time, {id, do, ...}, and gets {id, ok, result} or {id, error} back.
// Progress, log lines and state changes arrive as messages without an id. "stop" is read between batches.
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';

const VERSIONS = { pyodide: '314.0.7', onnxruntimeWeb: '1.30.0' };
const FFLATE = 'https://cdn.jsdelivr.net/npm/fflate@0.8.3/esm/browser.js';
const PACKAGES = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'soundfile', 'soxr', 'pyyaml'];
// The model files are bacpipe's own, from its repository on Hugging Face, at one fixed commit. On a development
// server (tools/serve.py) they are read from /model/ instead, where tools/bacpipe_reference.py put them.
const LOCAL = ['127.0.0.1', 'localhost'].includes(self.location.hostname);
const MODEL_COMMIT = 'f10c46a68eb9fcb18abe89394893400988248a70';
const MODEL_BASE = LOCAL ? new URL('../model/', self.location.href).href
  : `https://huggingface.co/datasets/vskode/bacpipe_models/resolve/${MODEL_COMMIT}/`;
const GPU_BATCH = 4;                 // WebGPU compiles its programs for one batch shape, so every batch has this size
// Windows per call of the network on the processor. bacpipe's own batch size (global_batch_size) still decides how
// bacpipe groups the windows: it reads the results back window by window, so the numbers do not depend on this.
// Larger batches made BirdNET+ V3.0 far slower in WebAssembly (16 windows: about 48 min for 61 windows; 1 window: 3 a second).
const CPU_BATCH = 2;

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSIONS.onnxruntimeWeb}/dist/`;

let py, engine, views, models, stopping = false, storage = 'memory', recordings = new Map(), audioRoot = null;
let loadedAudio = null;              // the one recording whose bytes are in Python's file system at the moment
const post = (message, transfer) => self.postMessage(message, transfer ?? []);
const status = (text, done, total) => post({ type: 'status', text, done, total });
const timing = (what, since) => post({ type: 'log', level: 'info', text: `Timing: ${what} in ${((performance.now() - since) / 1000).toFixed(1)} s` });
const lastLine = (e) => String(e?.message ?? e).trim().split('\n').at(-1).replace(/^\w*(Error|Exception): /, '');
const toJs = (value) => (value?.toJs ? value.toJs({ dict_converter: Object.fromEntries, create_proxies: false }) : value);

// ---------------------------------------------------------------------------------------- models

async function sha256(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
}

// A model file from the browser's cache, or downloaded once with progress. Refused unless its SHA-256 matches.
async function modelFile(spec) {
  const url = MODEL_BASE + spec.file;
  let cache = null, cached = null;
  try { cache = await caches.open('emberiza-model'); cached = await cache.match(url); } catch { /* private window */ }
  let bytes;
  if (cached) {
    status(`Reading ${spec.title} from this browser`);
    bytes = new Uint8Array(await cached.arrayBuffer());
  } else {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not download ${spec.title} (HTTP ${res.status}).`);
    const total = Number(res.headers.get('Content-Length')) || spec.bytes;
    const reader = res.body.getReader();
    bytes = new Uint8Array(total);
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (got + value.length > bytes.length) throw new Error(`${spec.title}: the download is larger than expected.`);
      bytes.set(value, got);
      got += value.length;
      status(`Downloading ${spec.title}`, got, total);
      if (stopping) { await reader.cancel(); throw new Error('stopped'); }
    }
    if (got !== total) bytes = bytes.subarray(0, got);
  }
  status(`Checking ${spec.title}`);
  if ((await sha256(bytes)) !== spec.sha256) {
    if (cache) await cache.delete(url);
    throw new Error(`${spec.title} is not the expected file. Run again to download it.`);
  }
  if (cache && !cached) {
    try { await cache.put(url, new Response(bytes)); } catch { /* no room: it downloads again next time */ }
  }
  return bytes;
}

async function openSession(spec, wantGpu) {
  const t0 = performance.now();
  const bytes = await modelFile(spec);
  timing(`${spec.title}: file ready`, t0);
  status(`Starting ${spec.title}`);
  if (wantGpu && self.navigator.gpu) {
    try {
      const adapter = await self.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        ort.env.webgpu.adapter = adapter;
        const session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
        const info = adapter.info ?? {};
        return { session, backend: `GPU (${[info.vendor, info.architecture || info.device].filter(Boolean).join(' ') || 'WebGPU'})`, gpu: true };
      }
    } catch (e) {
      post({ type: 'log', level: 'warning', text: `The graphics card could not run ${spec.title} (${lastLine(e)}). Using the processor.` });
    }
  }
  const t1 = performance.now();
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  timing(`${spec.title}: network started`, t1);
  const threads = ort.env.wasm.numThreads || 1;
  return { session, backend: `CPU (${self.crossOriginIsolated ? threads : 1} thread${threads === 1 || !self.crossOriginIsolated ? '' : 's'})`, gpu: false };
}

// The network on the windows of one recording: {output name: Float32Array of all rows}
async function infer(run, info, frames, n, length) {
  const size = run.gpu ? GPU_BATCH : CPU_BATCH;
  const outputs = {};
  for (let start = 0; start < n; start += size) {
    const rows = Math.min(size, n - start);
    let data = frames.subarray(start * length, (start + rows) * length);
    let batch = rows;
    if (run.gpu && rows < size) {                      // a full batch for the GPU: the missing rows are zeros, dropped below
      const padded = new Float32Array(size * length);
      padded.set(data);
      data = padded;
      batch = size;
    }
    const result = await run.session.run({ [info.input]: new ort.Tensor('float32', data, [batch, length]) }, info.wanted);
    for (const name of info.wanted) {
      const width = result[name].dims[1];
      outputs[name] ??= new Float32Array(n * width);
      outputs[name].set(result[name].data.subarray(0, rows * width), start * width);
      result[name].dispose?.();
    }
    if (stopping) throw new Error('stopped');
  }
  return outputs;
}

// ---------------------------------------------------------------------------------------- UMAP's update loop

// emberiza/umap_port.py optimize_layout, statement for statement (see there for why each rounding is where it is).
// states: for each vertex three 64-bit integers, as low and high 32-bit halves.
function layoutKernel(embedding, dim, head, tail, nEpochs, nVertices, epochsPerSample, a, b, lo, hi, gamma, initialAlpha, negativeRate, tell) {
  const emb = embedding;                               // Float32Array, changed in place
  const nEdges = epochsPerSample.length;
  const epsNeg = new Float64Array(nEdges), nextNeg = new Float64Array(nEdges), nextSample = new Float64Array(nEdges);
  for (let i = 0; i < nEdges; i++) { epsNeg[i] = epochsPerSample[i] / negativeRate; nextNeg[i] = epsNeg[i]; nextSample[i] = epochsPerSample[i]; }
  const fround = Math.fround;
  const tau = (j) => {                                 // umap's tau_rand_int on 64-bit states, returned as int32
    const p = 3 * j;
    let l = lo[p], h = hi[p], t = ((l & 0xFFFFFFFE) << 12), B = ((l << 13) ^ l) >>> 0;
    lo[p] = (t ^ ((B >>> 19) | (h << 13))) | 0; hi[p] = h >> 19;
    l = lo[p + 1]; h = hi[p + 1]; t = ((l & 0xFFFFFFF8) << 4); B = ((l << 2) ^ l) >>> 0;
    lo[p + 1] = (t ^ ((B >>> 25) | (h << 7))) | 0; hi[p + 1] = h >> 25;
    l = lo[p + 2]; h = hi[p + 2]; t = ((l & 0xFFFFFFF0) << 17); B = ((l << 3) ^ l) >>> 0;
    lo[p + 2] = (t ^ ((B >>> 11) | (h << 21))) | 0; hi[p + 2] = h >> 11;
    return (lo[p] ^ lo[p + 1] ^ lo[p + 2]) | 0;
  };
  let alpha = initialAlpha;
  for (let n = 0; n < nEpochs; n++) {
    for (let i = 0; i < nEdges; i++) {
      if (nextSample[i] > n) continue;
      const j = head[i] * dim;
      let k = tail[i] * dim;
      let dist = 0;
      for (let d = 0; d < dim; d++) { const diff = fround(emb[j + d] - emb[k + d]); dist = fround(dist + diff * diff); }
      let coeff = 0;
      if (dist > 0) coeff = (-2 * a * b * Math.pow(dist, b - 1)) / (a * Math.pow(dist, b) + 1);
      for (let d = 0; d < dim; d++) {
        let g = coeff * fround(emb[j + d] - emb[k + d]);
        g = g > 4 ? 4 : (g < -4 ? -4 : g);
        emb[j + d] = emb[j + d] + g * alpha;
        emb[k + d] = emb[k + d] + -g * alpha;
      }
      nextSample[i] += epochsPerSample[i];
      const nNeg = Math.trunc((n - nextNeg[i]) / epsNeg[i]);
      for (let p = 0; p < nNeg; p++) {
        const r = tau(head[i]);
        const v = ((r % nVertices) + nVertices) % nVertices;
        k = v * dim;
        dist = 0;
        for (let d = 0; d < dim; d++) { const diff = fround(emb[j + d] - emb[k + d]); dist = fround(dist + diff * diff); }
        if (dist > 0) coeff = (2 * gamma * b) / ((0.001 + dist) * (a * Math.pow(dist, b) + 1));
        else if (j === k) continue;
        else coeff = 0;
        for (let d = 0; d < dim; d++) {
          let g = 0;
          if (coeff > 0) { g = coeff * fround(emb[j + d] - emb[k + d]); g = g > 4 ? 4 : (g < -4 ? -4 : g); }
          emb[j + d] = emb[j + d] + g * alpha;
        }
      }
      nextNeg[i] += nNeg * epsNeg[i];
    }
    alpha = initialAlpha * (1 - n / nEpochs);
    if (tell && n % 25 === 0) tell(n + 1, nEpochs);
  }
  return emb;
}

// ---------------------------------------------------------------------------------------- start

async function mountResults() {
  py.FS.mkdirTree('/results');
  try {
    const root = await self.navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle('emberiza-results', { create: true });
    const fs = await py.mountNativeFS('/results', dir);
    storage = 'browser';
    return fs;
  } catch (e) {
    post({ type: 'log', level: 'warning', text: `Results stay in memory until the page closes (${lastLine(e)}). Download them before leaving.` });
    return null;
  }
}

let resultsFs = null;
const saveResults = async () => { if (resultsFs) { try { await resultsFs.syncfs(); } catch (e) { post({ type: 'log', level: 'warning', text: `Could not save to the browser: ${lastLine(e)}` }); } } };

async function start() {
  status('Starting Python');
  py = await loadPyodide();
  await py.loadPackage(PACKAGES);
  status('Loading bacpipe');
  const manifest = await (await fetch('py/manifest.json')).json();
  await Promise.all(manifest.files.map(async (f) => {
    const res = await fetch(`py/${f}`);
    if (!res.ok) throw new Error(`py/${f}: HTTP ${res.status}`);
    const path = `/app/py/${f}`;
    py.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
    py.FS.writeFile(path, new Uint8Array(await res.arrayBuffer()));
  }));
  resultsFs = await mountResults();
  self.emberizaLayoutKernel = layoutKernel;
  self.emberizaTell = (kind, text, done, total) => post({ type: kind, text, done, total });
  self.emberizaLog = (level, text) => post({ type: 'log', level, text });
  const api = py.runPython(`
import sys, logging, json
sys.path.insert(0, "/app/py")
import numpy as np
from js import emberizaLayoutKernel, emberizaTell, emberizaLog
from pyodide.ffi import to_js
from emberiza.engine import Engine, MODELS
from emberiza.views import Views
from emberiza import umap_port
import tqdm, bacpipe

class _PageLog(logging.Handler):
    def emit(self, record):
        try:
            emberizaLog(record.levelname.lower(), self.format(record).strip()[:2000])
        except Exception:
            pass
logging.getLogger("bacpipe").addHandler(_PageLog())
for _h in logging.getLogger("bacpipe").handlers:
    if type(_h) is logging.StreamHandler:
        logging.getLogger("bacpipe").removeHandler(_h)

tqdm.tqdm.listener = lambda desc, done, total: emberizaTell("progress", desc, done, total)
umap_port.progress = lambda done, total: emberizaTell("progress", "Laying out the embeddings", done, total)

def _kernel(embedding, head, tail, n_epochs, n_vertices, epochs_per_sample, a, b, states, gamma, alpha, negative_rate):
    flat = to_js(np.ascontiguousarray(embedding, dtype=np.float32).reshape(-1))
    lo = to_js((states & 0xFFFFFFFF).astype(np.uint32).astype(np.int32).reshape(-1))
    hi = to_js((states >> 32).astype(np.int32).reshape(-1))
    out = emberizaLayoutKernel(flat, embedding.shape[1], to_js(head.astype(np.int32)), to_js(tail.astype(np.int32)),
                               n_epochs, n_vertices, to_js(np.asarray(epochs_per_sample, dtype=np.float64)), a, b, lo, hi,
                               gamma, alpha, negative_rate, umap_port.progress)
    embedding[:] = np.asarray(out.to_py(), dtype=np.float32).reshape(embedding.shape)
    return embedding
umap_port.layout_kernel = _kernel

engine = Engine("/models")
views = Views(engine)

def kernel_selftest():
    """The JavaScript loop against the Python loop it copies, on a small fixed case: the largest difference."""
    rng = np.random.RandomState(7)
    X = np.vstack([c + rng.normal(size=(12, 8)) for c in rng.normal(size=(3, 8)) * 5]).astype(np.float32)
    fast = umap_port.UMAP(n_neighbors=8, random_state=42, n_epochs=60).fit_transform(X)
    umap_port.layout_kernel = None
    try:
        slow = umap_port.UMAP(n_neighbors=8, random_state=42, n_epochs=60).fit_transform(X)
    finally:
        umap_port.layout_kernel = _kernel
    return float(np.abs(fast - slow).max())

def existing():
    """The datasets with results in /results: {name: [models]}."""
    from pathlib import Path
    out = {}
    for d in sorted(Path("/results").iterdir()) if Path("/results").exists() else []:
        emb = d / "embeddings"
        found = sorted({p.name.split("___")[-1].split("-")[0] for p in emb.iterdir() if (p / "metadata.yml").exists()}) if emb.exists() else []
        if found:
            out[d.name] = found
    return json.dumps(out)

{"engine": engine, "views": views, "models": json.dumps(MODELS), "defaults": json.dumps(Engine.defaults(), default=str),
 "kernel_selftest": kernel_selftest, "existing": existing, "bacpipe": json.loads(open("/app/py/bacpipe/VENDORED.json").read())["commit"]}
`);
  const got = toJs(api);
  engine = api.get('engine');
  views = api.get('views');
  models = JSON.parse(got.models);
  self.kernelSelftest = api.get('kernel_selftest');
  self.existing = api.get('existing');
  post({ type: 'ready', defaults: JSON.parse(got.defaults), models, storage, versions: { ...VERSIONS, bacpipe: got.bacpipe },
         isolated: self.crossOriginIsolated, gpu: Boolean(self.navigator.gpu), existing: JSON.parse(self.existing()) });
}

// ---------------------------------------------------------------------------------------- recordings

const AUDIO = /\.(wav|WAV|aif|mp3|MP3|flac|ogg)$/;     // bacpipe's audio_suffixes default; the settings can change it

// The chosen files become a folder in Python's file system. Recordings are empty files until bacpipe needs them
// (their bytes come and go one at a time), tables such as annotations.csv are written whole.
async function setRecordings(dataset, files) {
  if (audioRoot) py.runPython(`import shutil; shutil.rmtree(${JSON.stringify(audioRoot)}, ignore_errors=True)`);
  recordings = new Map();
  loadedAudio = null;
  audioRoot = `/audio/${dataset}`;
  for (const { path, file } of files) {
    const full = `${audioRoot}/${path}`;
    py.FS.mkdirTree(full.slice(0, full.lastIndexOf('/')));
    if (/\.(csv|txt|json|ya?ml)$/i.test(path) && file.size < 64e6) {
      py.FS.writeFile(full, new Uint8Array(await file.arrayBuffer()));
    } else {
      py.FS.writeFile(full, new Uint8Array(0));
      recordings.set(path, file);
    }
  }
  return { dataset, recordings: recordings.size };
}

async function bringAudio(path) {
  if (loadedAudio === path) return true;
  if (loadedAudio) py.FS.writeFile(`${audioRoot}/${loadedAudio}`, new Uint8Array(0));
  loadedAudio = null;
  const file = recordings.get(path);
  if (!file) return false;
  py.FS.writeFile(`${audioRoot}/${path}`, new Uint8Array(await file.arrayBuffer()));
  loadedAudio = path;
  return true;
}

// ---------------------------------------------------------------------------------------- the run

async function run({ config, settings, gpu }) {
  stopping = false;
  const t0 = performance.now();
  const cfg = { ...config, audio_dir: audioRoot ?? `/audio/${config.dataset}` };
  delete cfg.dataset;
  if (!audioRoot) py.FS.mkdirTree(cfg.audio_dir);
  const plan = toJs(engine.configure.callKwargs(py.toPy(cfg), py.toPy({ ...settings, main_results_dir: '/results', device: 'cpu' }), {}));
  views.forget();
  const report = { models: [], backend: null, windows: 0, seconds: 0 };
  for (const model of plan.models) {
    const info = toJs(engine.start_model(model));
    if (!info.already_computed) {
      const run_ = await openSession({ ...models[model] }, gpu);
      report.backend = run_.backend;
      post({ type: 'backend', model, backend: run_.backend });
      let windows = 0, prepMs = 0, inferMs = 0, storeMs = 0;
      const tModel = performance.now();
      try {
        for (let i = 0; i < info.files.length; i++) {
          if (stopping) throw new Error('stopped');
          post({ type: 'file', model, title: models[model].title, index: i, total: info.files.length, name: info.files[i], windows,
                 seconds: (performance.now() - tModel) / 1000 });
          if (!(await bringAudio(info.files[i]))) { engine.skip(i, 'the recording is no longer available to the page'); continue; }
          const tPrep = performance.now();
          const frames = engine.prepare(i);
          if (!frames) continue;
          prepMs += performance.now() - tPrep;
          const view = frames.getBuffer('f32');
          let outputs;
          try {
            const [n, length] = view.shape;
            const tInfer = performance.now();
            outputs = await infer(run_, info, view.data, n, length);
            inferMs += performance.now() - tInfer;
            const tStore = performance.now();
            windows += engine.store_flat(i, info.wanted, info.wanted.map((name) => outputs[name]), n);
            storeMs += performance.now() - tStore;
          } catch (e) {
            if (lastLine(e) === 'stopped') throw e;
            engine.skip(i, `the model failed: ${lastLine(e)}`);
          } finally {
            view.release();
            frames.destroy();
          }
        }
      } finally {
        await run_.session.release();
      }
      post({ type: 'log', level: 'info', text: `Timing: ${models[model].title}: reading audio ${(prepMs / 1000).toFixed(1)} s, network ${(inferMs / 1000).toFixed(1)} s, saving ${(storeMs / 1000).toFixed(1)} s` });
      timing(`${models[model].title}: ${windows} windows embedded (${(windows / ((performance.now() - tModel) / 1000)).toFixed(2)} a second)`, tModel);
      toJs(engine.finish_model());
      report.windows += windows;
      report.models.push({ model, windows, seconds: (performance.now() - tModel) / 1000 });
      await saveResults();
    } else {
      toJs(engine.finish_model());
      post({ type: 'log', level: 'info', text: `${models[model].title}: these recordings were already embedded. Using the saved embeddings.` });
    }
    if (stopping) throw new Error('stopped');
    status(`Reducing ${models[model].title} embeddings to ${settings.visualization_dimensions ?? 2} dimensions`);
    await new Promise((r) => setTimeout(r));            // let the status out before Python holds the thread
    const tReduce = performance.now();
    engine.reduce(model);
    timing(`${models[model].title}: map`, tReduce);
    const tSave = performance.now();
    await saveResults();
    timing('saved to the browser', tSave);
  }
  if (stopping) throw new Error('stopped');
  status('Evaluating');
  await new Promise((r) => setTimeout(r));
  const tEval = performance.now();
  toJs(engine.evaluate());
  timing('evaluation', tEval);
  if (loadedAudio) { py.FS.writeFile(`${audioRoot}/${loadedAudio}`, new Uint8Array(0)); loadedAudio = null; }
  await saveResults();
  report.seconds = (performance.now() - t0) / 1000;
  report.summary = JSON.parse(views.summary());
  return report;
}

// Results that exist already (from an earlier visit, or copied in): the same calls with nothing left to compute
async function open({ dataset, config, settings }) {
  stopping = false;
  const audio = audioRoot && audioRoot.endsWith(`/${dataset}`) ? audioRoot : `/audio/${dataset}`;
  py.FS.mkdirTree(audio);
  const cfg = { ...config, audio_dir: audio, already_computed: true, overwrite: false };
  delete cfg.dataset;
  const plan = toJs(engine.configure.callKwargs(py.toPy(cfg), py.toPy({ ...settings, main_results_dir: '/results', device: 'cpu',
    check_if_already_processed: true, check_if_already_dim_reduced: true }), { save_logs_too: false }));
  views.forget();
  for (const model of plan.models) {
    const info = toJs(engine.start_model(model));
    if (!info.already_computed) throw new Error(`The saved embeddings of ${model} are incomplete. Choose the recordings and run again.`);
    engine.finish_model();
  }
  return { summary: JSON.parse(views.summary()), models: plan.models };
}

// A results folder made elsewhere (by bacpipe itself, for example) is copied into /results
async function importResults({ dataset, files }) {
  for (const { path, file } of files) {
    const full = `/results/${dataset}/${path}`;
    py.FS.mkdirTree(full.slice(0, full.lastIndexOf('/')));
    py.FS.writeFile(full, new Uint8Array(await file.arrayBuffer()));
  }
  await saveResults();
  return JSON.parse(self.existing());
}

function walk(dir, out = []) {
  for (const name of py.FS.readdir(dir)) {
    if (name === '.' || name === '..') continue;
    const path = `${dir}/${name}`;
    if (py.FS.isDir(py.FS.stat(path).mode)) walk(path, out); else out.push(path);
  }
  return out;
}

async function download({ dataset }) {
  const { zipSync } = await import(FFLATE);
  const root = `/results/${dataset}`;
  const entries = {};
  let bytes = 0;
  for (const path of walk(root)) {
    const data = py.FS.readFile(path);
    bytes += data.length;
    entries[`bacpipe_results/${dataset}${path.slice(root.length)}`] = data;
  }
  const zip = zipSync(entries, { level: 3 });
  return { blob: new Blob([zip], { type: 'application/zip' }), files: Object.keys(entries).length, bytes };
}

function listing({ dataset }) {
  const root = `/results/${dataset}`;
  return walk(root).map((path) => ({ path: path.slice(root.length + 1), bytes: py.FS.stat(path).size }));
}

async function segment({ model, index }) {
  const pts = JSON.parse(views.points(model));
  await bringAudio(String(pts.file[index]).replaceAll('\\', '/'));
  const seg = toJs(views.segment(model, index));
  const transfer = [seg.audio?.buffer, seg.spec?.buffer].filter(Boolean);
  return [seg, transfer];
}

const actions = {
  recordings: ({ dataset, files }) => setRecordings(dataset, files),
  run,
  open,
  importResults,
  download,
  listing,
  existing: () => JSON.parse(self.existing()),
  remove: async ({ dataset }) => { py.runPython(`import shutil; shutil.rmtree(${JSON.stringify(`/results/${dataset}`)}, ignore_errors=True)`); await saveResults(); return JSON.parse(self.existing()); },
  summary: () => JSON.parse(views.summary()),
  points: ({ model }) => JSON.parse(views.points(model)),
  predictions: ({ model }) => JSON.parse(views.predictions(model)),
  segment,
  readText: ({ dataset, path }) => py.FS.readFile(`/results/${dataset}/${path}`, { encoding: 'utf8' }),
  kernelSelftest: () => self.kernelSelftest(),
};

let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  if (data.do === 'stop') { stopping = true; return; }
  queue = queue.then(async () => {
    try {
      let result = await actions[data.do](data);
      let transfer = [];
      if (data.do === 'segment') [result, transfer] = result;
      post({ id: data.id, ok: true, result }, transfer);
    } catch (e) {
      const stopped = lastLine(e) === 'stopped';
      if (!stopped) console.error(e);
      post({ id: data.id, error: stopped ? 'stopped' : lastLine(e), detail: String(e?.message ?? e) });
    }
  });
};

start().catch((e) => { console.error(e); post({ type: 'failed', error: lastLine(e), detail: String(e?.message ?? e) }); });
